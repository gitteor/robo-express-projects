"""
Vision Service - RealSense 카메라 기반 박스 감지 서비스

pallet_stack_planner를 사용하여 박스 위치를 감지하고,
특정 박스가 Y 범위 내에 들어오면 알림을 제공합니다.

Usage:
    from vision_service import VisionService
    
    vision = VisionService()
    vision.start()
    
    # 박스 감지 상태 확인
    status = vision.get_detection_status("Box_Kancho")
    if status['in_range']:
        print(f"칸쵸 감지! 위치: {status['position']}")
"""

import sys
import threading
import time
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum

# 로컬 모듈 import (같은 backend 폴더 내)
try:
    from pallet_stack_planner import PalletStackPlanner
    from box_detector import RealSenseCamera
    VISION_AVAILABLE = True
    print("[VisionService] Vision modules loaded successfully")
except ImportError as e:
    print(f"[VisionService] Warning: Could not import vision modules: {e}")
    VISION_AVAILABLE = False


class BoxStatus(Enum):
    """박스 상태"""
    NOT_DETECTED = "not_detected"
    DETECTED = "detected"
    IN_RANGE = "in_range"


@dataclass
class DetectedBox:
    """감지된 박스 정보"""
    box_type: str
    position: List[float]  # [x, y, z] 로봇 좌표계
    position_cam: List[float]  # [x, y, z] 카메라 좌표계
    size: List[float]  # [width, length, height]
    angle: float
    status: BoxStatus


class VisionService:
    """RealSense 기반 비전 감지 서비스"""
    
    # 박스 타입 매핑 (웹 product_id -> pallet_stack_planner box_type)
    PRODUCT_TO_BOX = {
        'cube_a': 'Box_Kancho',    # 칸쵸
        'cube_b': 'Box_Choco',     # 초코송이
        'cube_c': 'Box_Whalebob',  # 고래밥
    }
    
    # Y좌표 범위 (로봇 좌표계 기준)
    Y_RANGE_MIN = -0.04  # -4cm
    Y_RANGE_MAX = 0.04   # +4cm
    
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # 카메라 및 플래너
        self.camera: Optional[RealSenseCamera] = None
        self.planner: Optional[PalletStackPlanner] = None
        
        # 감지 상태
        self.detected_boxes: Dict[str, DetectedBox] = {}
        self.pallet_position: Optional[List[float]] = None
        
        # 콜백 함수들
        self.on_box_in_range: Optional[Callable[[str, DetectedBox], None]] = None
        
        # 마지막 업데이트 시간
        self.last_update_time = 0
        
    def start(self) -> bool:
        """비전 서비스 시작"""
        if not VISION_AVAILABLE:
            print("[VisionService] Vision modules not available, running in simulation mode")
            return False
            
        if self.running:
            print("[VisionService] Already running")
            return True
            
        try:
            # 카메라 초기화
            self.camera = RealSenseCamera(width=640, height=480, fps=30)
            self.camera.start()
            
            # 플래너 초기화
            self.planner = PalletStackPlanner(self.camera)
            
            # 백그라운드 스레드 시작
            self.running = True
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            
            print("[VisionService] Started successfully")
            return True
            
        except Exception as e:
            print(f"[VisionService] Failed to start: {e}")
            self.running = False
            return False
    
    def stop(self):
        """비전 서비스 정지"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        if self.camera:
            self.camera.stop()
            self.camera = None
            
        self.planner = None
        print("[VisionService] Stopped")
    
    def _detection_loop(self):
        """백그라운드 감지 루프"""
        while self.running:
            try:
                if self.planner:
                    # 센서 데이터 업데이트
                    success = self.planner.update()
                    
                    if success:
                        self._process_detections()
                        
                self.last_update_time = time.time()
                time.sleep(0.05)  # 20Hz
                
            except Exception as e:
                print(f"[VisionService] Detection error: {e}")
                time.sleep(0.1)
    
    def _process_detections(self):
        """감지 결과 처리"""
        if not self.planner:
            return
            
        with self.lock:
            # 기존 감지 결과 초기화
            self.detected_boxes.clear()
            
            # 색상 감지된 박스 처리
            for box_data in self.planner.color_detected_boxes:
                center_cam = box_data['center']
                size = box_data['size']
                angle = box_data.get('angle', 0)
                
                # 박스 크기로 종류 분류 (cm 단위)
                width_cm = size[0] * 100
                length_cm = size[1] * 100
                box_type = PalletStackPlanner.classify_box(width_cm, length_cm)
                
                if box_type == 'Unknown':
                    continue
                
                # 카메라 → 로봇 좌표 변환
                pickup_z = PalletStackPlanner.CONVEYOR_HEIGHT + size[2] / 2
                robot_pos = PalletStackPlanner.camera_to_robot(
                    center_cam, use_fixed_z=True, fixed_z=pickup_z
                )
                
                # Y 범위 확인
                y_robot = robot_pos[1]
                if self.Y_RANGE_MIN <= y_robot <= self.Y_RANGE_MAX:
                    status = BoxStatus.IN_RANGE
                else:
                    status = BoxStatus.DETECTED
                
                detected_box = DetectedBox(
                    box_type=box_type,
                    position=robot_pos.tolist(),
                    position_cam=center_cam.tolist(),
                    size=size.tolist(),
                    angle=angle,
                    status=status
                )
                
                self.detected_boxes[box_type] = detected_box
                
                # IN_RANGE 콜백
                if status == BoxStatus.IN_RANGE and self.on_box_in_range:
                    self.on_box_in_range(box_type, detected_box)
            
            # 팔레트 위치 업데이트
            if self.planner.pallet_info and self.planner.pallet_info.detected:
                pallet_center = self.planner.pallet_info.center
                robot_pallet = PalletStackPlanner.camera_to_robot(
                    pallet_center, use_fixed_z=True, fixed_z=PalletStackPlanner.PALLET_HEIGHT
                )
                self.pallet_position = robot_pallet.tolist()
    
    def get_detection_status(self, product_id: str = None, box_type: str = None) -> Dict:
        """
        박스 감지 상태 조회
        
        Args:
            product_id: 웹 상품 ID (cube_a, cube_b, etc.)
            box_type: 박스 타입 (Box_Kancho, Box_Choco, etc.)
            
        Returns:
            감지 상태 딕셔너리
        """
        # product_id를 box_type으로 변환
        if product_id and not box_type:
            box_type = self.PRODUCT_TO_BOX.get(product_id)
            
        if not box_type:
            return {
                'detected': False,
                'in_range': False,
                'position': None,
                'error': 'Invalid product_id or box_type'
            }
        
        with self.lock:
            if box_type in self.detected_boxes:
                box = self.detected_boxes[box_type]
                return {
                    'detected': True,
                    'in_range': box.status == BoxStatus.IN_RANGE,
                    'position': box.position,
                    'position_cam': box.position_cam,
                    'size': box.size,
                    'angle': box.angle,
                    'box_type': box.box_type
                }
            else:
                return {
                    'detected': False,
                    'in_range': False,
                    'position': None,
                    'box_type': box_type
                }
    
    def get_all_detections(self) -> Dict:
        """모든 감지 결과 반환"""
        with self.lock:
            result = {
                'boxes': {},
                'pallet': self.pallet_position,
                'last_update': self.last_update_time,
                'running': self.running
            }
            
            for box_type, box in self.detected_boxes.items():
                result['boxes'][box_type] = {
                    'detected': True,
                    'in_range': box.status == BoxStatus.IN_RANGE,
                    'position': box.position,
                    'position_cam': box.position_cam,
                    'size': box.size,
                    'angle': box.angle
                }
                
            return result
    
    def get_pallet_position(self) -> Optional[List[float]]:
        """팔레트 위치 반환"""
        with self.lock:
            return self.pallet_position
    
    def wait_for_box_in_range(self, product_id: str, timeout: float = 30.0) -> Optional[Dict]:
        """
        박스가 Y 범위에 들어올 때까지 대기
        
        Args:
            product_id: 웹 상품 ID
            timeout: 최대 대기 시간 (초)
            
        Returns:
            박스 정보 또는 None (타임아웃)
        """
        box_type = self.PRODUCT_TO_BOX.get(product_id)
        if not box_type:
            return None
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_detection_status(box_type=box_type)
            
            if status.get('in_range'):
                return status
                
            time.sleep(0.1)
            
        return None


# 테스트 코드
if __name__ == "__main__":
    print("Testing VisionService...")
    
    vision = VisionService()
    
    # 콜백 설정
    def on_box_detected(box_type, box):
        print(f"[CALLBACK] {box_type} in range! Position: {box.position}")
    
    vision.on_box_in_range = on_box_detected
    
    if vision.start():
        try:
            print("Press Ctrl+C to stop...")
            while True:
                # 모든 감지 결과 출력
                detections = vision.get_all_detections()
                
                print(f"\n--- Detections ---")
                print(f"Running: {detections['running']}")
                print(f"Pallet: {detections['pallet']}")
                
                for box_type, info in detections['boxes'].items():
                    status = "IN_RANGE" if info['in_range'] else "detected"
                    print(f"  {box_type}: {status} at {info['position']}")
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\nStopping...")
    else:
        print("Running in simulation mode (no camera)")
        
    vision.stop()
