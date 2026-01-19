#!/usr/bin/env python3
"""
RealSense + Open3D 기반 직사각형 박스 인식 시스템
- 포인트클라우드에서 평면(테이블) 제거 후 박스 감지
- Oriented Bounding Box (OBB)로 박스 피팅
- 8개 꼭지점 좌표 실시간 출력
- 로봇팔 강화학습용 좌표 제공

Author: Claude (Anthropic)
Date: 2025-12-16
"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import threading
from queue import Queue


@dataclass
class BoxInfo:
    """감지된 박스 정보"""
    center: np.ndarray          # 박스 중심 좌표 (x, y, z)
    extent: np.ndarray          # 박스 크기 (width, height, depth)
    rotation: np.ndarray        # 3x3 회전 행렬
    corners: np.ndarray         # 8개 꼭지점 좌표 (8, 3)
    euler_angles: np.ndarray    # 오일러 각도 (roll, pitch, yaw) in degrees
    

class RealSenseCamera:
    """Intel RealSense 카메라 관리 클래스"""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = None
        self.intrinsics = None
        
    def start(self):
        """카메라 스트림 시작"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 스트림 설정
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        # 파이프라인 시작
        profile = self.pipeline.start(self.config)
        
        # Depth와 Color를 정렬
        self.align = rs.align(rs.stream.color)
        
        # 깊이 카메라 intrinsic 파라미터 획득
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.intrinsics = depth_profile.get_intrinsics()
        
        # 카메라 워밍업
        for _ in range(30):
            self.pipeline.wait_for_frames()
            
        print(f"[RealSense] 카메라 시작: {self.width}x{self.height} @ {self.fps}fps")
        
    def stop(self):
        """카메라 스트림 정지"""
        if self.pipeline:
            self.pipeline.stop()
            print("[RealSense] 카메라 정지")
            
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """정렬된 컬러/깊이 프레임 획득"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
    
    def get_open3d_intrinsic(self) -> o3d.camera.PinholeCameraIntrinsic:
        """Open3D 카메라 내부 파라미터 반환"""
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height,
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.ppx, self.intrinsics.ppy
        )


class BoxDetector:
    """3D 박스 감지기"""
    
    def __init__(self, 
                 voxel_size: float = 0.005,
                 plane_distance_threshold: float = 0.02,
                 min_cluster_points: int = 100,
                 depth_min: float = 0.1,
                 depth_max: float = 1.5):
        """
        Args:
            voxel_size: 다운샘플링 복셀 크기 (m)
            plane_distance_threshold: 평면 세그멘테이션 임계값 (m)
            min_cluster_points: 최소 클러스터 포인트 수
            depth_min: 최소 깊이 (m)
            depth_max: 최대 깊이 (m)
        """
        self.voxel_size = voxel_size
        self.plane_distance_threshold = plane_distance_threshold
        self.min_cluster_points = min_cluster_points
        self.depth_min = depth_min
        self.depth_max = depth_max
        
    def create_point_cloud(self, 
                           color_image: np.ndarray, 
                           depth_image: np.ndarray,
                           intrinsic: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.PointCloud:
        """RGB-D 이미지로부터 포인트클라우드 생성"""
        # Open3D 이미지로 변환
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image)
        
        # RGBD 이미지 생성
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,  # RealSense 깊이 스케일 (mm -> m)
            depth_trunc=self.depth_max,
            convert_rgb_to_intensity=False
        )
        
        # 포인트클라우드 생성
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 깊이 범위 필터링
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Z축(깊이) 기준 필터링
        mask = (points[:, 2] > self.depth_min) & (points[:, 2] < self.depth_max)
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
        if len(colors) > 0:
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            
        return filtered_pcd
    
    def remove_plane(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """RANSAC을 사용하여 평면(테이블) 제거"""
        if len(pcd.points) < 100:
            return pcd
            
        # RANSAC 평면 세그멘테이션
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        # 평면 외의 포인트 (박스 후보)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        
        return outlier_cloud
    
    def cluster_objects(self, pcd: o3d.geometry.PointCloud) -> List[o3d.geometry.PointCloud]:
        """DBSCAN 클러스터링으로 개별 객체 분리"""
        if len(pcd.points) < self.min_cluster_points:
            return []
            
        # DBSCAN 클러스터링
        labels = np.array(pcd.cluster_dbscan(
            eps=0.02,  # 이웃 탐색 반경
            min_points=10,
            print_progress=False
        ))
        
        if len(labels) == 0 or labels.max() < 0:
            return []
            
        clusters = []
        for i in range(labels.max() + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) >= self.min_cluster_points:
                cluster = pcd.select_by_index(cluster_indices)
                clusters.append(cluster)
                
        return clusters
    
    def fit_oriented_box(self, pcd: o3d.geometry.PointCloud) -> Optional[BoxInfo]:
        """포인트클라우드에 Oriented Bounding Box 피팅"""
        if len(pcd.points) < 10:
            return None
            
        try:
            # Oriented Bounding Box 계산
            obb = pcd.get_oriented_bounding_box()
            obb.color = (1, 0, 0)  # 빨간색
            
            # 박스 정보 추출
            center = np.asarray(obb.center)
            extent = np.asarray(obb.extent)
            rotation = np.asarray(obb.R)
            
            # 8개 꼭지점 좌표
            corners = np.asarray(obb.get_box_points())
            
            # 회전 행렬에서 오일러 각도 계산 (ZYX 순서)
            euler_angles = self._rotation_matrix_to_euler(rotation)
            
            return BoxInfo(
                center=center,
                extent=extent,
                rotation=rotation,
                corners=corners,
                euler_angles=euler_angles
            )
        except Exception as e:
            print(f"[BoxDetector] OBB 피팅 실패: {e}")
            return None
            
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """회전 행렬을 오일러 각도(roll, pitch, yaw)로 변환 (degrees)"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
            
        return np.degrees(np.array([roll, pitch, yaw]))
    
    def detect(self, 
               color_image: np.ndarray, 
               depth_image: np.ndarray,
               intrinsic: o3d.camera.PinholeCameraIntrinsic) -> Tuple[List[BoxInfo], o3d.geometry.PointCloud, List[o3d.geometry.OrientedBoundingBox]]:
        """
        박스 감지 메인 함수
        
        Returns:
            boxes: 감지된 박스 정보 리스트
            pcd: 처리된 포인트클라우드
            obbs: Oriented Bounding Box 객체 리스트 (시각화용)
        """
        # 1. 포인트클라우드 생성
        pcd = self.create_point_cloud(color_image, depth_image, intrinsic)
        
        # 2. 다운샘플링
        pcd = pcd.voxel_down_sample(self.voxel_size)
        
        # 3. 노이즈 제거 (Statistical Outlier Removal)
        if len(pcd.points) > 100:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 4. 평면(테이블) 제거
        objects_pcd = self.remove_plane(pcd)
        
        # 5. 클러스터링
        clusters = self.cluster_objects(objects_pcd)
        
        # 6. 각 클러스터에 OBB 피팅
        boxes = []
        obbs = []
        
        for cluster in clusters:
            box_info = self.fit_oriented_box(cluster)
            if box_info is not None:
                boxes.append(box_info)
                
                # 시각화용 OBB 생성
                obb = cluster.get_oriented_bounding_box()
                obb.color = (1, 0, 0)
                obbs.append(obb)
                
        return boxes, objects_pcd, obbs


class BoxDetectionVisualizer:
    """실시간 박스 감지 시각화"""
    
    def __init__(self, camera: RealSenseCamera, detector: BoxDetector):
        self.camera = camera
        self.detector = detector
        self.running = False
        self.vis = None
        self.pcd = o3d.geometry.PointCloud()
        self.obbs = []
        self.geometry_added = False
        
    def _create_coordinate_frame(self, size=0.1) -> o3d.geometry.TriangleMesh:
        """좌표계 프레임 생성"""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    def _draw_box_info_on_image(self, 
                                 image: np.ndarray, 
                                 boxes: List[BoxInfo],
                                 intrinsic: o3d.camera.PinholeCameraIntrinsic) -> np.ndarray:
        """2D 이미지에 박스 정보 오버레이"""
        img = image.copy()
        
        # 카메라 파라미터
        fx, fy = intrinsic.intrinsic_matrix[0, 0], intrinsic.intrinsic_matrix[1, 1]
        cx, cy = intrinsic.intrinsic_matrix[0, 2], intrinsic.intrinsic_matrix[1, 2]
        
        for idx, box in enumerate(boxes):
            # 3D 꼭지점을 2D로 투영
            corners_2d = []
            for corner in box.corners:
                if corner[2] > 0:  # Z > 0 체크
                    u = int(fx * corner[0] / corner[2] + cx)
                    v = int(fy * corner[1] / corner[2] + cy)
                    corners_2d.append((u, v))
                    
            if len(corners_2d) == 8:
                # 박스 엣지 그리기
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # 하단 면
                    (4, 5), (5, 6), (6, 7), (7, 4),  # 상단 면
                    (0, 4), (1, 5), (2, 6), (3, 7)   # 수직 엣지
                ]
                
                for e1, e2 in edges:
                    pt1 = corners_2d[e1]
                    pt2 = corners_2d[e2]
                    if 0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0]:
                        if 0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]:
                            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                            
                # 꼭지점 표시
                for i, pt in enumerate(corners_2d):
                    if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                        cv2.circle(img, pt, 4, (0, 0, 255), -1)
                        cv2.putText(img, str(i), (pt[0]+5, pt[1]+5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 중심점 투영
            center = box.center
            if center[2] > 0:
                cu = int(fx * center[0] / center[2] + cx)
                cv_pt = int(fy * center[1] / center[2] + cy)
                
                if 0 <= cu < img.shape[1] and 0 <= cv_pt < img.shape[0]:
                    cv2.circle(img, (cu, cv_pt), 6, (255, 0, 0), -1)
                    
                    # 박스 정보 텍스트
                    info_text = f"Box {idx}: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
                    cv2.putText(img, info_text, (cu + 10, cv_pt), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                               
        return img
    
    def _print_box_coordinates(self, boxes: List[BoxInfo]):
        """콘솔에 박스 좌표 출력"""
        print("\n" + "="*80)
        print(f"감지된 박스 수: {len(boxes)}")
        print("="*80)
        
        for idx, box in enumerate(boxes):
            print(f"\n[Box {idx}]")
            print(f"  중심 좌표 (m): X={box.center[0]:.4f}, Y={box.center[1]:.4f}, Z={box.center[2]:.4f}")
            print(f"  크기 (m): W={box.extent[0]:.4f}, H={box.extent[1]:.4f}, D={box.extent[2]:.4f}")
            print(f"  회전 (deg): Roll={box.euler_angles[0]:.2f}, Pitch={box.euler_angles[1]:.2f}, Yaw={box.euler_angles[2]:.2f}")
            print(f"  꼭지점 좌표 (m):")
            for i, corner in enumerate(box.corners):
                print(f"    V{i}: ({corner[0]:.4f}, {corner[1]:.4f}, {corner[2]:.4f})")
                
            # 로봇팔용 좌표 출력 (mm 단위로 변환)
            print(f"\n  [로봇팔 좌표 (mm)]")
            print(f"    중심: X={box.center[0]*1000:.1f}, Y={box.center[1]*1000:.1f}, Z={box.center[2]*1000:.1f}")
            print(f"    크기: W={box.extent[0]*1000:.1f}, H={box.extent[1]*1000:.1f}, D={box.extent[2]*1000:.1f}")
            
    def run_2d_visualization(self):
        """2D OpenCV 기반 실시간 시각화 (간단 버전)"""
        print("\n[BoxDetector] 2D 시각화 시작...")
        print("종료: 'q' 키")
        
        intrinsic = self.camera.get_open3d_intrinsic()
        
        while True:
            # 프레임 획득
            color_image, depth_image = self.camera.get_frames()
            if color_image is None:
                continue
                
            # 박스 감지
            start_time = time.time()
            boxes, _, _ = self.detector.detect(color_image, depth_image, intrinsic)
            detect_time = (time.time() - start_time) * 1000
            
            # 2D 이미지에 오버레이
            vis_image = self._draw_box_info_on_image(color_image, boxes, intrinsic)
            
            # FPS 및 감지 시간 표시
            cv2.putText(vis_image, f"Detection: {detect_time:.1f}ms", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_image, f"Boxes: {len(boxes)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 박스 좌표 출력 (0.5초마다)
            if len(boxes) > 0:
                self._print_box_coordinates(boxes)
                
            # 깊이 이미지 컬러맵
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 나란히 표시
            combined = np.hstack([vis_image, depth_colormap])
            cv2.imshow("Box Detection (Color | Depth)", combined)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
    def run_3d_visualization(self):
        """Open3D 기반 실시간 3D 시각화"""
        print("\n[BoxDetector] 3D 시각화 시작...")
        print("종료: ESC 키 또는 창 닫기")
        
        # Visualizer 초기화
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("3D Box Detection", width=1280, height=720)
        
        # 렌더링 옵션
        render_opt = self.vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.1, 0.1, 0.1])
        
        # 좌표계 프레임 추가
        coord_frame = self._create_coordinate_frame(0.1)
        self.vis.add_geometry(coord_frame)
        
        # 초기 포인트클라우드 추가
        self.vis.add_geometry(self.pcd)
        
        intrinsic = self.camera.get_open3d_intrinsic()
        self.running = True
        
        def update_scene():
            # 프레임 획득
            color_image, depth_image = self.camera.get_frames()
            if color_image is None:
                return True
                
            # 박스 감지
            boxes, objects_pcd, obbs = self.detector.detect(color_image, depth_image, intrinsic)
            
            # 포인트클라우드 업데이트
            self.pcd.points = objects_pcd.points
            self.pcd.colors = objects_pcd.colors
            self.vis.update_geometry(self.pcd)
            
            # 기존 OBB 제거
            for obb in self.obbs:
                self.vis.remove_geometry(obb, reset_bounding_box=False)
            self.obbs.clear()
            
            # 새 OBB 추가
            for obb in obbs:
                self.vis.add_geometry(obb, reset_bounding_box=False)
                self.obbs.append(obb)
                
            # 박스 정보 출력
            if len(boxes) > 0:
                self._print_box_coordinates(boxes)
                
            return True
            
        # ESC 키로 종료
        def close_callback(vis):
            self.running = False
            return False
            
        self.vis.register_key_callback(256, close_callback)  # ESC
        
        # 메인 루프
        while self.running:
            update_scene()
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.033)  # ~30 FPS
            
        self.vis.destroy_window()


class RobotCoordinateProvider:
    """로봇팔 강화학습을 위한 좌표 제공 클래스"""
    
    def __init__(self, camera: RealSenseCamera, detector: BoxDetector):
        self.camera = camera
        self.detector = detector
        self.latest_boxes: List[BoxInfo] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        
    def start(self):
        """백그라운드 감지 스레드 시작"""
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        print("[RobotCoordinateProvider] 감지 스레드 시작")
        
    def stop(self):
        """감지 스레드 정지"""
        self._running = False
        if self._thread:
            self._thread.join()
        print("[RobotCoordinateProvider] 감지 스레드 정지")
        
    def _detection_loop(self):
        """감지 루프 (백그라운드)"""
        intrinsic = self.camera.get_open3d_intrinsic()
        
        while self._running:
            color_image, depth_image = self.camera.get_frames()
            if color_image is None:
                continue
                
            boxes, _, _ = self.detector.detect(color_image, depth_image, intrinsic)
            
            with self._lock:
                self.latest_boxes = boxes
                
            time.sleep(0.05)  # 20 Hz
            
    def get_target_position(self, box_index: int = 0) -> Optional[dict]:
        """
        로봇 엔드이펙터 타겟 위치 반환
        
        Returns:
            dict: {
                'position': [x, y, z] in meters,
                'orientation': [roll, pitch, yaw] in radians,
                'corners': [[x,y,z], ...] 8개 꼭지점,
                'size': [w, h, d] in meters
            }
        """
        with self._lock:
            if box_index >= len(self.latest_boxes):
                return None
                
            box = self.latest_boxes[box_index]
            
            return {
                'position': box.center.tolist(),
                'orientation': np.radians(box.euler_angles).tolist(),
                'corners': box.corners.tolist(),
                'size': box.extent.tolist(),
                'rotation_matrix': box.rotation.tolist()
            }
            
    def get_all_boxes(self) -> List[dict]:
        """모든 감지된 박스 정보 반환"""
        with self._lock:
            results = []
            for box in self.latest_boxes:
                results.append({
                    'position': box.center.tolist(),
                    'orientation': np.radians(box.euler_angles).tolist(),
                    'corners': box.corners.tolist(),
                    'size': box.extent.tolist(),
                    'rotation_matrix': box.rotation.tolist()
                })
            return results


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RealSense + Open3D 박스 감지')
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d', 'api'],
                       help='실행 모드: 2d (2D 시각화), 3d (3D 시각화), api (좌표 API)')
    parser.add_argument('--width', type=int, default=640, help='이미지 너비')
    parser.add_argument('--height', type=int, default=480, help='이미지 높이')
    parser.add_argument('--fps', type=int, default=30, help='프레임 레이트')
    parser.add_argument('--depth-min', type=float, default=0.1, help='최소 깊이 (m)')
    parser.add_argument('--depth-max', type=float, default=1.5, help='최대 깊이 (m)')
    
    args = parser.parse_args()
    
    # 카메라 초기화
    camera = RealSenseCamera(args.width, args.height, args.fps)
    
    # 박스 감지기 초기화
    detector = BoxDetector(
        voxel_size=0.005,
        plane_distance_threshold=0.02,
        min_cluster_points=100,
        depth_min=args.depth_min,
        depth_max=args.depth_max
    )
    
    try:
        camera.start()
        
        if args.mode == '2d':
            # 2D OpenCV 시각화
            visualizer = BoxDetectionVisualizer(camera, detector)
            visualizer.run_2d_visualization()
            
        elif args.mode == '3d':
            # 3D Open3D 시각화
            visualizer = BoxDetectionVisualizer(camera, detector)
            visualizer.run_3d_visualization()
            
        elif args.mode == 'api':
            # API 모드 - 강화학습 통합용
            provider = RobotCoordinateProvider(camera, detector)
            provider.start()
            
            print("\n[API 모드] 실시간 박스 좌표 제공 중...")
            print("종료: Ctrl+C")
            
            try:
                while True:
                    boxes = provider.get_all_boxes()
                    if boxes:
                        print(f"\n감지된 박스 수: {len(boxes)}")
                        for i, box in enumerate(boxes):
                            pos = box['position']
                            print(f"  Box {i}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                    time.sleep(0.5)
            except KeyboardInterrupt:
                provider.stop()
                
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
