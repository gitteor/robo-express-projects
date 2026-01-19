#!/usr/bin/env python3
"""
팔레트 빈 공간 감지 및 적재 위치 계산 시스템

RealSense 카메라로 팔레트와 기존 박스를 감지하고,
새 박스를 쌓을 최적 위치를 계산합니다.

Usage:
    python3 pallet_stack_planner.py --mode visualize
    python3 pallet_stack_planner.py --mode api

Author: Claude (Anthropic)
Date: 2025-01-02
"""

import numpy as np
import cv2
import open3d as o3d
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time
from enum import Enum

# 기존 box_detector 임포트
from box_detector import RealSenseCamera, BoxDetector, BoxInfo


class BoxType(Enum):
    """박스 타입 정의"""
    TYPE_A = "A"  # 8.5cm x 12cm x 3.5cm
    TYPE_B = "B"  # 8.5cm x 12cm x 3.5cm (동일)
    TYPE_C = "C"  # 20cm x 5cm x 5cm


@dataclass
class BoxSize:
    """박스 크기 정의 (단위: m)"""
    width: float   # X축
    length: float  # Y축
    height: float  # Z축
    
    def to_array(self) -> np.ndarray:
        return np.array([self.width, self.length, self.height])


# 박스 크기 정의 (단위: m)
BOX_SIZES = {
    BoxType.TYPE_A: BoxSize(0.085, 0.12, 0.035),
    BoxType.TYPE_B: BoxSize(0.085, 0.12, 0.035),
    BoxType.TYPE_C: BoxSize(0.20, 0.05, 0.05),
}


@dataclass
class PalletInfo:
    """팔레트 정보"""
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 중심 좌표 (x, y, z)
    size: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.20]))  # 크기 (width, length)
    corners_2d: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))  # 2D 꼭지점
    height: float = 0.0  # 팔레트 상단 높이
    detected: bool = False


@dataclass 
class PlacementCandidate:
    """적재 위치 후보"""
    position: np.ndarray  # (x, y, z) 적재 위치 (박스 중심)
    rotation: float = 0.0  # Z축 회전 (0 또는 90도)
    score: float = 0.0  # 적재 점수 (높을수록 좋음)
    valid: bool = True


@dataclass
class StackPlan:
    """적재 계획"""
    target_position: np.ndarray  # 목표 위치 (박스 중심)
    target_rotation: float  # 목표 회전 (도)
    box_type: BoxType
    confidence: float
    candidates: List[PlacementCandidate] = field(default_factory=list)


class PalletDetector:
    """팔레트 감지기 (색상 기반)"""
    
    def __init__(self,
                 pallet_color_hsv_low: Tuple[int, int, int] = (100, 100, 50),
                 pallet_color_hsv_high: Tuple[int, int, int] = (130, 255, 255),
                 pallet_size: Tuple[float, float] = (0.20, 0.20),
                 min_area_ratio: float = 0.3):
        """
        Args:
            pallet_color_hsv_low: 팔레트 색상 HSV 하한 (파란색)
            pallet_color_hsv_high: 팔레트 색상 HSV 상한
            pallet_size: 팔레트 크기 (width, length) in meters
            min_area_ratio: 최소 검출 면적 비율
        """
        self.hsv_low = np.array(pallet_color_hsv_low)
        self.hsv_high = np.array(pallet_color_hsv_high)
        self.pallet_size = np.array(pallet_size)
        self.min_area_ratio = min_area_ratio
        
    def detect(self, 
               color_image: np.ndarray, 
               depth_image: np.ndarray,
               intrinsic: o3d.camera.PinholeCameraIntrinsic) -> PalletInfo:
        """
        팔레트 감지
        
        Returns:
            PalletInfo: 감지된 팔레트 정보
        """
        pallet = PalletInfo()
        
        # BGR -> HSV 변환
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # 파란색 마스크 생성
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return pallet
            
        # 가장 큰 컨투어 선택
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # 최소 면적 체크
        min_area = color_image.shape[0] * color_image.shape[1] * self.min_area_ratio * 0.01
        if area < min_area:
            return pallet
            
        # 최소 외접 사각형
        rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rect)
        box_points = np.int32(box_points)
        
        # 중심점의 깊이 값으로 3D 좌표 계산
        center_2d = np.array(rect[0])
        cx, cy = int(center_2d[0]), int(center_2d[1])
        
        # 깊이 값 획득 (중심 주변 평균)
        roi_size = 10
        y_min = max(0, cy - roi_size)
        y_max = min(depth_image.shape[0], cy + roi_size)
        x_min = max(0, cx - roi_size)
        x_max = min(depth_image.shape[1], cx + roi_size)
        
        depth_roi = depth_image[y_min:y_max, x_min:x_max]
        valid_depths = depth_roi[depth_roi > 0]
        
        if len(valid_depths) == 0:
            return pallet
            
        depth_m = np.median(valid_depths) / 1000.0  # mm -> m
        
        # 2D -> 3D 변환
        fx = intrinsic.intrinsic_matrix[0, 0]
        fy = intrinsic.intrinsic_matrix[1, 1]
        cx_intr = intrinsic.intrinsic_matrix[0, 2]
        cy_intr = intrinsic.intrinsic_matrix[1, 2]
        
        x_3d = (center_2d[0] - cx_intr) * depth_m / fx
        y_3d = (center_2d[1] - cy_intr) * depth_m / fy
        z_3d = depth_m
        
        pallet.center = np.array([x_3d, y_3d, z_3d])
        pallet.size = self.pallet_size
        pallet.corners_2d = box_points.astype(float)
        pallet.height = z_3d  # 카메라 기준 깊이 (나중에 테이블 높이로 변환)
        pallet.detected = True
        
        return pallet
    
    def get_mask(self, color_image: np.ndarray) -> np.ndarray:
        """팔레트 영역만 마스크 반환 (컨투어 기반)"""
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 가장 큰 컨투어(팔레트)만 마스킹
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask_result = np.zeros_like(mask)
            cv2.drawContours(mask_result, [largest], 0, 255, -1)
            return mask_result
        
        return mask


class ColorBoxDetector:
    """색상 기반 박스 감지기"""
    
    def __init__(self,
                 min_area: int = 3000,
                 max_area: int = 80000,
                 saturation_threshold: int = 30,
                 value_threshold: int = 100):
        """
        Args:
            min_area: 최소 컨투어 면적 (픽셀)
            max_area: 최대 컨투어 면적 (픽셀)
            saturation_threshold: 최소 채도 (컨베이어 제외용)
            value_threshold: 최소 밝기 (어두운 부분 제외)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.saturation_threshold = saturation_threshold
        self.value_threshold = value_threshold
        
    def detect(self, 
               color_image: np.ndarray, 
               depth_image: np.ndarray,
               intrinsic: o3d.camera.PinholeCameraIntrinsic,
               exclude_mask: np.ndarray = None) -> List[Dict]:
        """
        색상 기반으로 박스 감지 - 밝고 채도 있는 물체 찾기
        
        Args:
            exclude_mask: 제외할 영역 마스크 (팔레트 등)
        
        Returns:
            감지된 박스 리스트 [{'center': [x,y,z], 'size': [w,l,h], 'contour': ...}, ...]
        """
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # 방법: 채도와 밝기가 있는 모든 색상 감지 (검정 컨베이어 제외)
        # S > threshold AND V > threshold
        mask_s = hsv[:, :, 1] > self.saturation_threshold  # 채도
        mask_v = hsv[:, :, 2] > self.value_threshold       # 밝기
        combined_mask = (mask_s & mask_v).astype(np.uint8) * 255
        
        # 팔레트 등 제외할 영역 마스킹
        if exclude_mask is not None:
            combined_mask[exclude_mask > 0] = 0
        
        # 모폴로지 연산으로 노이즈 제거 및 영역 연결
        kernel = np.ones((7, 7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 카메라 파라미터
        fx = intrinsic.intrinsic_matrix[0, 0]
        fy = intrinsic.intrinsic_matrix[1, 1]
        cx = intrinsic.intrinsic_matrix[0, 2]
        cy = intrinsic.intrinsic_matrix[1, 2]
        
        detected_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 면적 필터링
            if area < self.min_area or area > self.max_area:
                continue
            
            # 최소 외접 사각형
            rect = cv2.minAreaRect(contour)
            center_2d = rect[0]
            size_2d = rect[1]  # (width, height) in pixels
            angle = rect[2]
            
            # 중심점의 깊이 값
            cx_int, cy_int = int(center_2d[0]), int(center_2d[1])
            
            # 깊이 값 획득 (중심 주변 평균)
            roi_size = 15
            y_min = max(0, cy_int - roi_size)
            y_max = min(depth_image.shape[0], cy_int + roi_size)
            x_min = max(0, cx_int - roi_size)
            x_max = min(depth_image.shape[1], cx_int + roi_size)
            
            depth_roi = depth_image[y_min:y_max, x_min:x_max]
            valid_depths = depth_roi[depth_roi > 0]
            
            if len(valid_depths) == 0:
                continue
                
            depth_m = np.median(valid_depths) / 1000.0  # mm -> m
            
            # 2D -> 3D 변환 (카메라 좌표계)
            x_3d = (center_2d[0] - cx) * depth_m / fx
            y_3d = (center_2d[1] - cy) * depth_m / fy
            z_3d = depth_m
            
            # 픽셀 크기 -> 미터 변환
            width_m = size_2d[0] * depth_m / fx
            length_m = size_2d[1] * depth_m / fy
            
            # 박스 높이는 고정값 사용 (위에서 내려다보므로 측정 어려움)
            height_m = 0.04  # 4cm 기본값
            
            detected_boxes.append({
                'center': np.array([x_3d, y_3d, z_3d]),
                'size': np.array([width_m, length_m, height_m]),
                'angle': angle,
                'contour': contour,
                'rect': rect,
                'area': area
            })
        
        return detected_boxes, combined_mask


class OccupancyGrid:
    """2D 점유 그리드"""
    
    def __init__(self, 
                 width: float, 
                 length: float, 
                 resolution: float = 0.01,
                 margin: float = 0.02,
                 min_box_size: float = 0.04):
        """
        Args:
            width: 팔레트 너비 (m)
            length: 팔레트 길이 (m)
            resolution: 그리드 해상도 (m/cell)
            margin: 박스 간 여유 간격 (m)
            min_box_size: 최소 박스 크기 (m) - 이보다 작은 빈 공간은 무시
        """
        self.width = width
        self.length = length
        self.resolution = resolution
        self.margin = margin
        self.min_box_size = min_box_size
        
        # 그리드 크기 계산
        self.grid_width = int(np.ceil(width / resolution))
        self.grid_length = int(np.ceil(length / resolution))
        
        # 점유 그리드 (0: 빈 공간, 1: 점유)
        self.grid = np.zeros((self.grid_length, self.grid_width), dtype=np.uint8)
        
        # 팔레트 중심 (로컬 좌표계 기준)
        self.origin = np.array([0.0, 0.0])
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """월드 좌표 -> 그리드 인덱스"""
        # 팔레트 중심 기준 로컬 좌표
        local_x = x - self.origin[0] + self.width / 2
        local_y = y - self.origin[1] + self.length / 2
        
        grid_x = int(local_x / self.resolution)
        grid_y = int(local_y / self.resolution)
        
        return grid_x, grid_y
        
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """그리드 인덱스 -> 월드 좌표"""
        local_x = grid_x * self.resolution
        local_y = grid_y * self.resolution
        
        world_x = local_x - self.width / 2 + self.origin[0]
        world_y = local_y - self.length / 2 + self.origin[1]
        
        return world_x, world_y
        
    def mark_occupied(self, box_center: np.ndarray, box_size: np.ndarray):
        """박스 영역을 점유로 표시"""
        # 박스 경계 (마진 포함)
        half_w = box_size[0] / 2 + self.margin
        half_l = box_size[1] / 2 + self.margin
        
        x_min, y_min = self.world_to_grid(box_center[0] - half_w, box_center[1] - half_l)
        x_max, y_max = self.world_to_grid(box_center[0] + half_w, box_center[1] + half_l)
        
        # 그리드 범위 클리핑
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.grid_width - 1, x_max)
        y_max = min(self.grid_length - 1, y_max)
        
        self.grid[y_min:y_max+1, x_min:x_max+1] = 1
        
    def can_place(self, 
                  position: np.ndarray, 
                  box_size: np.ndarray,
                  min_overlap_ratio: float = 0.8) -> Tuple[bool, float]:
        """
        해당 위치에 박스를 놓을 수 있는지 확인
        
        Args:
            position: 박스 중심 위치 (x, y)
            box_size: 박스 크기 (width, length)
            min_overlap_ratio: 최소 팔레트 겹침 비율
            
        Returns:
            (가능 여부, 점수)
        """
        half_w = box_size[0] / 2
        half_l = box_size[1] / 2
        
        # 박스 경계
        x_min_w = position[0] - half_w
        x_max_w = position[0] + half_w
        y_min_w = position[1] - half_l
        y_max_w = position[1] + half_l
        
        # 팔레트 경계
        pallet_x_min = self.origin[0] - self.width / 2
        pallet_x_max = self.origin[0] + self.width / 2
        pallet_y_min = self.origin[1] - self.length / 2
        pallet_y_max = self.origin[1] + self.length / 2
        
        # 팔레트와 겹치는 영역 계산
        overlap_x_min = max(x_min_w, pallet_x_min)
        overlap_x_max = min(x_max_w, pallet_x_max)
        overlap_y_min = max(y_min_w, pallet_y_min)
        overlap_y_max = min(y_max_w, pallet_y_max)
        
        if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
            return False, 0.0
            
        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        box_area = box_size[0] * box_size[1]
        overlap_ratio = overlap_area / box_area
        
        if overlap_ratio < min_overlap_ratio:
            return False, 0.0
        
        # 4x4cm 격자 검증: 박스가 실제로 들어갈 수 있는 충분한 공간인지 확인
        # 박스 영역을 4x4cm 격자로 나누어 각 격자가 빈 공간인지 확인
        grid_size = self.min_box_size  # 4cm
        num_grids_x = int(np.ceil(box_size[0] / grid_size))
        num_grids_y = int(np.ceil(box_size[1] / grid_size))
        
        # 각 격자 중심이 모두 빈 공간에 있어야 함
        for i in range(num_grids_x):
            for j in range(num_grids_y):
                grid_x = x_min_w + (i + 0.5) * grid_size
                grid_y = y_min_w + (j + 0.5) * grid_size
                
                # 이 격자가 팔레트 내부에 있는지 확인
                if not (pallet_x_min <= grid_x <= pallet_x_max and 
                        pallet_y_min <= grid_y <= pallet_y_max):
                    continue
                
                # 격자 주변 영역이 점유되어 있는지 확인
                gx, gy = self.world_to_grid(grid_x, grid_y)
                
                # 격자 크기를 그리드 셀 단위로 변환
                grid_cells = int(grid_size / self.resolution)
                gx_min = max(0, gx - grid_cells // 2)
                gy_min = max(0, gy - grid_cells // 2)
                gx_max = min(self.grid_width - 1, gx + grid_cells // 2)
                gy_max = min(self.grid_length - 1, gy + grid_cells // 2)
                
                region = self.grid[gy_min:gy_max+1, gx_min:gx_max+1]
                if np.any(region > 0):
                    return False, 0.0
            
        # 그리드 좌표로 변환 (마진 포함)
        gx_min, gy_min = self.world_to_grid(x_min_w - self.margin, y_min_w - self.margin)
        gx_max, gy_max = self.world_to_grid(x_max_w + self.margin, y_max_w + self.margin)
        
        # 범위 클리핑
        gx_min = max(0, gx_min)
        gy_min = max(0, gy_min)
        gx_max = min(self.grid_width - 1, gx_max)
        gy_max = min(self.grid_length - 1, gy_max)
        
        # 점유 체크
        region = self.grid[gy_min:gy_max+1, gx_min:gx_max+1]
        if np.any(region > 0):
            return False, 0.0
            
        # 점수 계산: 팔레트 모서리(왼쪽 위)에 붙을수록 높은 점수
        # 박스가 팔레트 경계에 얼마나 가까운지 계산
        
        # 박스 왼쪽 끝이 팔레트 왼쪽 끝에 가까울수록 좋음
        dist_to_left = abs(x_min_w - pallet_x_min)
        # 박스 위쪽 끝이 팔레트 위쪽 끝에 가까울수록 좋음  
        dist_to_top = abs(y_min_w - pallet_y_min)
        
        # 모서리까지의 거리 (작을수록 좋음)
        corner_dist = np.sqrt(dist_to_left**2 + dist_to_top**2)
        max_dist = np.sqrt(self.width**2 + self.length**2)
        
        # 점수: 모서리에 가까울수록 1에 가까움
        score = 1.0 - (corner_dist / max_dist)
        
        # 추가 보너스: 팔레트 경계에 완전히 붙어있으면 보너스
        edge_bonus = 0.0
        if dist_to_left < 0.005:  # 5mm 이내면 붙은 것으로 간주
            edge_bonus += 0.1
        if dist_to_top < 0.005:
            edge_bonus += 0.1
            
        score = min(1.0, score + edge_bonus)
        
        return True, score
        
    def find_placement_candidates(self, 
                                   box_size: np.ndarray,
                                   step: float = 0.01,
                                   allow_rotation: bool = True) -> List[PlacementCandidate]:
        """
        적재 가능한 위치 후보 찾기
        
        Args:
            box_size: 박스 크기 (width, length, height)
            step: 탐색 스텝 (m)
            allow_rotation: 90도 회전 허용 여부
            
        Returns:
            적재 후보 리스트 (점수 내림차순)
        """
        candidates = []
        
        rotations = [0.0]
        if allow_rotation:
            rotations.append(90.0)
            
        for rotation in rotations:
            # 회전에 따른 박스 크기
            if rotation == 90.0:
                size_2d = np.array([box_size[1], box_size[0]])  # swap width/length
            else:
                size_2d = np.array([box_size[0], box_size[1]])
                
            # 팔레트 내 탐색
            pallet_x_min = self.origin[0] - self.width / 2
            pallet_y_min = self.origin[1] - self.length / 2
            
            x = pallet_x_min + size_2d[0] / 2
            while x < self.origin[0] + self.width / 2 - size_2d[0] / 2 + step:
                y = pallet_y_min + size_2d[1] / 2
                while y < self.origin[1] + self.length / 2 - size_2d[1] / 2 + step:
                    can_place, score = self.can_place(np.array([x, y]), size_2d)
                    
                    if can_place:
                        candidates.append(PlacementCandidate(
                            position=np.array([x, y, 0.0]),  # Z는 나중에 설정
                            rotation=rotation,
                            score=score,
                            valid=True
                        ))
                    y += step
                x += step
                
        # 점수 내림차순 정렬
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        return candidates
        
    def visualize(self) -> np.ndarray:
        """그리드 시각화 이미지 반환"""
        # 그리드를 이미지로 변환 (확대)
        scale = 4
        vis = np.zeros((self.grid_length * scale, self.grid_width * scale, 3), dtype=np.uint8)
        
        for y in range(self.grid_length):
            for x in range(self.grid_width):
                color = (0, 100, 0) if self.grid[y, x] == 0 else (0, 0, 200)
                cv2.rectangle(vis, 
                             (x * scale, y * scale), 
                             ((x + 1) * scale - 1, (y + 1) * scale - 1),
                             color, -1)
                             
        return vis


class PalletStackPlanner:
    """팔레트 적재 계획기"""
    
    # 카메라 → 로봇 좌표 변환 파라미터
    CAMERA_HEIGHT = 0.80  # 카메라 설치 높이 (m)
    OFFSET_X = 0.24  # X축 오프셋 (m)
    OFFSET_Y = 0.03  # Y축 오프셋 (m)
    OFFSET_Z = -0.02  # Z축 오프셋 (m)
    
    # 고정 높이 (로봇 좌표계 기준)
    PALLET_HEIGHT = 0.05  # 팔레트 높이 (m)
    CONVEYOR_HEIGHT = 0.05  # 컨베이어 높이 (m)
    
    # 박스 종류 정의 (크기: cm, 오차 허용: ±1.0cm)
    BOX_TYPES = {
        'Box_Kancho': {'size': (8.5, 14.5), 'tolerance': 1.0},    # 8.5 x 14.5 cm
        'Box_Choco': {'size': (12.5, 9.0), 'tolerance': 1.0},     # 12.5 x 9.0 cm
        'Box_Whalebob': {'size': (9.5, 16.5), 'tolerance': 1.0},  # 9.5 x 16.5 cm (고래밥)
    }
    
    @staticmethod
    def classify_box(width_cm: float, length_cm: float) -> str:
        """
        박스 크기로 종류 분류
        
        Args:
            width_cm: 박스 너비 (cm)
            length_cm: 박스 길이 (cm)
        Returns:
            박스 종류 이름 또는 'Unknown'
        """
        # 크기를 정렬해서 비교 (가로/세로 방향 무관)
        detected = sorted([width_cm, length_cm])
        
        for box_name, info in PalletStackPlanner.BOX_TYPES.items():
            expected = sorted(info['size'])
            tolerance = info['tolerance']
            
            # 두 축 모두 허용 오차 내인지 확인
            if (abs(detected[0] - expected[0]) <= tolerance and
                abs(detected[1] - expected[1]) <= tolerance):
                return box_name
        
        return 'Unknown'
    
    @staticmethod
    def camera_to_robot(cam_pos: np.ndarray, use_fixed_z: bool = False, fixed_z: float = None) -> np.ndarray:
        """
        카메라 좌표 → 로봇 좌표 변환
        
        카메라: 위에서 아래를 바라봄 (top-down)
        - X축: 오프셋 (반전 없음)
        - Y축: 반전 + 오프셋
        - Z축: 고정값 사용 또는 계산
        
        Args:
            cam_pos: 카메라 좌표 [x, y, z] (m)
            use_fixed_z: True면 fixed_z 값 사용
            fixed_z: 고정 Z값 (m)
        Returns:
            로봇 좌표 [x, y, z] (m)
        """
        robot_x = cam_pos[0] + PalletStackPlanner.OFFSET_X  # X 반전 제거
        robot_y = -cam_pos[1] + PalletStackPlanner.OFFSET_Y
        
        if use_fixed_z and fixed_z is not None:
            robot_z = fixed_z
        else:
            robot_z = PalletStackPlanner.CAMERA_HEIGHT - cam_pos[2] + PalletStackPlanner.OFFSET_Z
        
        return np.array([robot_x, robot_y, robot_z])
    
    def __init__(self,
                 camera: RealSenseCamera,
                 pallet_size: Tuple[float, float] = (0.20, 0.20),
                 grid_resolution: float = 0.01,
                 box_margin: float = 0.02,
                 min_overlap_ratio: float = 0.8,
                 detection_x_range: Tuple[float, float] = (-0.36, -0.16),
                 detection_y_range: Tuple[float, float] = (-0.3, 0.3)):
        """
        Args:
            camera: RealSense 카메라
            pallet_size: 팔레트 크기 (width, length) in meters
            grid_resolution: 점유 그리드 해상도 (m)
            box_margin: 박스 간 여유 간격 (m)
            min_overlap_ratio: 최소 팔레트 겹침 비율
            detection_x_range: 박스 인식 X 범위 (min, max) in meters
            detection_y_range: 박스 인식 Y 범위 (min, max) in meters
        """
        self.camera = camera
        self.pallet_size = pallet_size
        self.grid_resolution = grid_resolution
        self.box_margin = box_margin
        self.min_overlap_ratio = min_overlap_ratio
        self.detection_x_range = detection_x_range
        self.detection_y_range = detection_y_range
        
        # 감지기 초기화
        self.pallet_detector = PalletDetector(pallet_size=pallet_size)
        self.box_detector = BoxDetector(
            voxel_size=0.003,              # 더 세밀한 복셀
            plane_distance_threshold=0.01,  # 평면 제거 더 엄격하게
            min_cluster_points=50,          # 작은 클러스터도 감지
            depth_min=0.3,
            depth_max=1.0
        )
        # 색상 기반 박스 감지기 추가
        self.color_box_detector = ColorBoxDetector()
        
        # 상태
        self.pallet_info: Optional[PalletInfo] = None
        self.existing_boxes: List[BoxInfo] = []  # 팔레트 위 박스
        self.pickup_boxes: List[BoxInfo] = []    # 팔레트 밖 박스 (Cube A 후보)
        self.color_detected_boxes: List[Dict] = []  # 색상 기반 감지 박스
        self.all_boxes: List[BoxInfo] = []       # 모든 감지된 박스
        self.occupancy_grid: Optional[OccupancyGrid] = None
        
    def update(self) -> bool:
        """
        센서 데이터 업데이트
        
        Returns:
            업데이트 성공 여부
        """
        color_image, depth_image = self.camera.get_frames()
        if color_image is None:
            return False
            
        intrinsic = self.camera.get_open3d_intrinsic()
        
        # 1. 팔레트 감지
        self.pallet_info = self.pallet_detector.detect(color_image, depth_image, intrinsic)
        
        # 1-1. 팔레트 마스크 생성 (색상 감지에서 제외용)
        pallet_mask = None
        if self.pallet_info.detected:
            pallet_mask = self.pallet_detector.get_mask(color_image)
        
        # 2. 색상 기반 박스 감지 (팔레트 영역 제외)
        color_boxes, color_mask = self.color_box_detector.detect(
            color_image, depth_image, intrinsic, exclude_mask=pallet_mask
        )
        
        # 디버깅: 색상 기반 감지 결과
        if color_boxes:
            print(f"\n[Color] 감지된 박스 수: {len(color_boxes)}")
            for i, box in enumerate(color_boxes):
                width_cm = box['size'][0] * 100
                length_cm = box['size'][1] * 100
                box_type = self.classify_box(width_cm, length_cm)
                print(f"  Box {i} [{box_type}]: pos=({box['center'][0]:.3f}, {box['center'][1]:.3f}, {box['center'][2]:.3f}), "
                      f"size=({width_cm:.1f}x{length_cm:.1f})cm")
        
        # 2-1. 인식 범위 필터링
        filtered_boxes = []
        for box in color_boxes:
            x, y = box['center'][0], box['center'][1]
            
            # 위치 범위 체크
            in_x_range = self.detection_x_range[0] <= x <= self.detection_x_range[1]
            in_y_range = self.detection_y_range[0] <= y <= self.detection_y_range[1]
            
            if in_x_range and in_y_range:
                print(f"    → 통과!")
                filtered_boxes.append(box)
            else:
                reasons = []
                if not in_x_range: reasons.append(f"X범위({x:.2f})")
                if not in_y_range: reasons.append(f"Y범위({y:.2f})")
                print(f"    → 제외: {', '.join(reasons)}")
        
        self.color_detected_boxes = filtered_boxes
        
        # 3. 팔레트 위/밖 박스 분류
        self.existing_boxes = []
        self.pickup_boxes = filtered_boxes  # 색상 감지된 박스는 pickup 대상
        
        # 4. 점유 그리드 생성 (팔레트가 있을 때만)
        if self.pallet_info.detected:
            self._update_occupancy_grid()
        
        return True
    
    def _classify_boxes(self, boxes: List[BoxInfo]) -> Tuple[List[BoxInfo], List[BoxInfo]]:
        """박스를 팔레트 위/밖으로 분류"""
        on_pallet = []
        off_pallet = []
        
        if not self.pallet_info or not self.pallet_info.detected:
            return [], boxes
            
        pallet_center = self.pallet_info.center[:2]  # XY 평면
        pallet_height = self.pallet_info.center[2]   # Z 높이
        pallet_half_size = self.pallet_size[0] / 2 + 0.02  # 약간의 여유
        height_threshold = 0.01  # 팔레트 상단보다 최소 1cm 위에 있어야 함
        
        for box in boxes:
            box_center_2d = box.center[:2]
            box_height = box.center[2]
            
            # XY 평면상 거리 체크
            dist = np.linalg.norm(box_center_2d - pallet_center)
            
            # 팔레트 영역 내에 있고, 팔레트보다 위에 있으면 "팔레트 위"
            if dist < pallet_half_size * np.sqrt(2) and box_height > pallet_height + height_threshold:
                on_pallet.append(box)
            else:
                off_pallet.append(box)
                
        return on_pallet, off_pallet
        
    def get_pickup_target(self) -> Optional[Dict]:
        """
        집어올릴 박스(Cube A) 좌표 반환 (로봇 좌표계)
        
        Returns:
            가장 가까운 색상 감지 박스 정보 (로봇 좌표계)
        """
        if not self.color_detected_boxes:
            return None
            
        # 카메라에서 가장 가까운 박스 선택
        closest = min(self.color_detected_boxes, key=lambda b: b['center'][2])
        
        # Z값: 컨베이어 높이 + 박스 높이/2
        pickup_z = self.CONVEYOR_HEIGHT + closest['size'][2] / 2
        
        # 카메라 좌표 → 로봇 좌표 변환
        robot_pos = self.camera_to_robot(closest['center'], use_fixed_z=True, fixed_z=pickup_z)
        
        return {
            'position': robot_pos.tolist(),
            'position_cam': closest['center'].tolist(),  # 디버깅용 카메라 좌표
            'size': closest['size'].tolist(),
            'rotation': closest['angle'],  # 회전 각도
        }
        
    def _update_occupancy_grid(self):
        """점유 그리드 업데이트"""
        self.occupancy_grid = OccupancyGrid(
            width=self.pallet_size[0],
            length=self.pallet_size[1],
            resolution=self.grid_resolution,
            margin=self.box_margin,
            min_box_size=0.04  # 4cm x 4cm 최소 격자 크기
        )
        
        # 팔레트 중심을 그리드 원점으로
        if self.pallet_info:
            self.occupancy_grid.origin = self.pallet_info.center[:2]
            
        # 기존 박스 점유 표시
        for box in self.existing_boxes:
            self.occupancy_grid.mark_occupied(box.center[:2], box.extent[:2])
            
    def plan_placement(self, 
                       box_type: BoxType,
                       allow_rotation: bool = True) -> Optional[StackPlan]:
        """
        박스 적재 위치 계획
        
        Args:
            box_type: 박스 타입
            allow_rotation: 90도 회전 허용 여부
            
        Returns:
            적재 계획 또는 None (적재 불가)
        """
        if not self.occupancy_grid or not self.pallet_info:
            return None
            
        box_size = BOX_SIZES[box_type]
        
        # 적재 후보 찾기
        candidates = self.occupancy_grid.find_placement_candidates(
            box_size=box_size.to_array(),
            step=self.grid_resolution,
            allow_rotation=allow_rotation
        )
        
        if not candidates:
            return None
            
        # 최적 후보 선택
        best = candidates[0]
        
        # Z 좌표 설정 (팔레트 높이 + 박스 높이/2)
        target_z = self.pallet_info.height + box_size.height / 2
        target_position = np.array([best.position[0], best.position[1], target_z])
        
        return StackPlan(
            target_position=target_position,
            target_rotation=best.rotation,
            box_type=box_type,
            confidence=best.score,
            candidates=candidates[:10]  # 상위 10개 후보
        )
        
    def get_stack_target(self, box_type: BoxType) -> Optional[Dict]:
        """
        Sim2Real용 적재 목표 좌표 반환
        
        Returns:
            {
                'position': [x, y, z],  # 목표 위치 (m)
                'rotation': float,      # Z축 회전 (도)
                'box_type': str,
                'confidence': float
            }
        """
        plan = self.plan_placement(box_type)
        
        if plan is None:
            return None
            
        return {
            'position': plan.target_position.tolist(),
            'rotation': plan.target_rotation,
            'box_type': plan.box_type.value,
            'confidence': plan.confidence
        }
        
    def visualize(self, color_image: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic = None) -> np.ndarray:
        """시각화 이미지 생성"""
        vis = color_image.copy()
        
        # 카메라 파라미터 (2D 투영용)
        if intrinsic is None:
            intrinsic = self.camera.get_open3d_intrinsic()
        fx = intrinsic.intrinsic_matrix[0, 0]
        fy = intrinsic.intrinsic_matrix[1, 1]
        cx = intrinsic.intrinsic_matrix[0, 2]
        cy = intrinsic.intrinsic_matrix[1, 2]
        
        # 팔레트 표시
        if self.pallet_info and self.pallet_info.detected:
            corners = self.pallet_info.corners_2d.astype(np.int32)
            cv2.drawContours(vis, [corners], 0, (255, 0, 0), 2)
            
            # 중심점
            pcx = int(np.mean(corners[:, 0]))
            pcy = int(np.mean(corners[:, 1]))
            cv2.circle(vis, (pcx, pcy), 5, (0, 255, 0), -1)
            
            # 팔레트 좌표 표시 (로봇 좌표계)
            pallet_pos_cam = self.pallet_info.center
            pallet_pos_robot = self.camera_to_robot(pallet_pos_cam, use_fixed_z=True, fixed_z=self.PALLET_HEIGHT)
            pallet_label = f"Pallet R({pallet_pos_robot[0]:.2f},{pallet_pos_robot[1]:.2f},{pallet_pos_robot[2]:.2f})m"
            cv2.putText(vis, pallet_label, (pcx + 10, pcy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 팔레트 위 박스 표시 (노란색) - 현재 비활성화
        # for i, box in enumerate(self.existing_boxes):
        #     ...
        
        # 색상 기반 감지 박스 표시
        for i, box in enumerate(self.color_detected_boxes):
            # 박스 종류 분류
            width_cm = box['size'][0] * 100
            length_cm = box['size'][1] * 100
            box_type = self.classify_box(width_cm, length_cm)
            
            # 박스 종류에 따라 색상 지정
            if box_type == 'Box_Kancho':
                color = (0, 165, 255)  # 주황색
            elif box_type == 'Box_Choco':
                color = (203, 192, 255)  # 핑크색
            else:
                color = (0, 0, 255)  # 빨간색 (Unknown)
            
            # 컨투어 그리기
            if 'contour' in box:
                cv2.drawContours(vis, [box['contour']], 0, color, 2)
            
            # 중심점에 원 표시
            if 'rect' in box:
                center_2d = box['rect'][0]
                u, v = int(center_2d[0]), int(center_2d[1])
                cv2.circle(vis, (u, v), 5, color, -1)
                
                # 로봇 좌표로 변환 (Z는 컨베이어 높이 + 박스 높이/2)
                pickup_z = self.CONVEYOR_HEIGHT + box['size'][2] / 2
                robot_pos = self.camera_to_robot(box['center'], use_fixed_z=True, fixed_z=pickup_z)
                # 두 줄로 표시: 박스 종류 + 좌표
                cv2.putText(vis, box_type, (u + 15, v - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis, f"({robot_pos[0]:.2f},{robot_pos[1]:.2f},{robot_pos[2]:.2f})", (u + 15, v + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        # 상태 표시
        cv2.putText(vis, f"Boxes on pallet: {len(self.existing_boxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Pickup candidates: {len(self.color_detected_boxes)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        return vis
        
    def visualize_grid(self) -> Optional[np.ndarray]:
        """점유 그리드 시각화"""
        if self.occupancy_grid is None:
            return None
        return self.occupancy_grid.visualize()


class PalletStackPlannerROS2:
    """ROS2 연동 버전 (별도 파일로 분리 가능)"""
    pass  # TODO: 필요시 구현


def run_visualization(planner: PalletStackPlanner):
    """시각화 모드 실행"""
    print("\n[PalletStackPlanner] 시각화 모드 시작")
    print("  'a': Type A 박스 적재 위치 계산")
    print("  'b': Type B 박스 적재 위치 계산")  
    print("  'c': Type C 박스 적재 위치 계산")
    print("  'p': Pickup 박스(Cube A) 좌표 출력")
    print("  'g': 점유 그리드 표시 토글")
    print("  'q': 종료")
    
    show_grid = False
    current_plan: Optional[StackPlan] = None
    frame_error_count = 0
    max_frame_errors = 10  # 연속 에러 허용 횟수
    
    while True:
        try:
            # 센서 업데이트
            planner.update()
            
            # 프레임 획득
            color_image, _ = planner.camera.get_frames()
            if color_image is None:
                frame_error_count += 1
                if frame_error_count > max_frame_errors:
                    print("\n[Warning] 프레임 수신 실패가 계속됩니다. 카메라 연결을 확인하세요.")
                    frame_error_count = 0
                continue
            
            frame_error_count = 0  # 성공하면 카운터 리셋
                
            # 시각화
            vis = planner.visualize(color_image)
            
            # 적재 계획 표시
            if current_plan:
                pos = current_plan.target_position
                cv2.putText(vis, f"Stack Target: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(vis, f"Rotation: {current_plan.target_rotation:.0f} deg, Conf: {current_plan.confidence:.2f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                           
            cv2.imshow("Pallet Stack Planner", vis)
            
            # 점유 그리드 표시
            if show_grid:
                grid_vis = planner.visualize_grid()
                if grid_vis is not None:
                    cv2.imshow("Occupancy Grid", grid_vis)
            else:
                try:
                    cv2.destroyWindow("Occupancy Grid")
                except:
                    pass
                    
        except RuntimeError as e:
            if "Frame didn't arrive" in str(e):
                print(f"\n[Warning] 프레임 타임아웃 - 카메라 연결 확인 중...")
                time.sleep(0.5)  # 잠시 대기 후 재시도
                continue
            else:
                raise e
            
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            pickup = planner.get_pickup_target()
            if pickup:
                pos = pickup['position']
                size = pickup['size']
                print(f"\n[Pickup] Cube A:")
                print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
                print(f"  Size: ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}) m")
                print(f"  Rotation: {pickup['rotation']:.1f} deg")
            else:
                print("\n[Pickup] Cube A 없음")
        elif key == ord('a'):
            current_plan = planner.plan_placement(BoxType.TYPE_A)
            if current_plan:
                pos = current_plan.target_position
                print(f"\n[Plan] Type A Stack Target:")
                print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
                print(f"  Rotation: {current_plan.target_rotation:.0f} deg")
                print(f"  Confidence: {current_plan.confidence:.2f}")
            else:
                print("\n[Plan] Type A: 적재 불가 (공간 없음)")
        elif key == ord('b'):
            current_plan = planner.plan_placement(BoxType.TYPE_B)
            if current_plan:
                pos = current_plan.target_position
                print(f"\n[Plan] Type B Stack Target:")
                print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
                print(f"  Rotation: {current_plan.target_rotation:.0f} deg")
            else:
                print("\n[Plan] Type B: 적재 불가")
        elif key == ord('c'):
            current_plan = planner.plan_placement(BoxType.TYPE_C)
            if current_plan:
                pos = current_plan.target_position
                print(f"\n[Plan] Type C Stack Target:")
                print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
                print(f"  Rotation: {current_plan.target_rotation:.0f} deg")
            else:
                print("\n[Plan] Type C: 적재 불가")
        elif key == ord('g'):
            show_grid = not show_grid
            
    cv2.destroyAllWindows()


def run_api_mode(planner: PalletStackPlanner):
    """API 모드 실행"""
    print("\n[PalletStackPlanner] API 모드 시작")
    print("Ctrl+C로 종료")
    
    try:
        while True:
            if planner.update():
                # 각 박스 타입에 대해 적재 위치 계산
                for box_type in BoxType:
                    target = planner.get_stack_target(box_type)
                    if target:
                        print(f"\n[{box_type.value}] pos=({target['position'][0]:.3f}, "
                              f"{target['position'][1]:.3f}, {target['position'][2]:.3f}), "
                              f"rot={target['rotation']:.0f}°, conf={target['confidence']:.2f}")
            else:
                print(".", end="", flush=True)
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n종료")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='팔레트 적재 위치 계획기')
    parser.add_argument('--mode', type=str, default='visualize',
                       choices=['visualize', 'api'],
                       help='실행 모드')
    parser.add_argument('--pallet-width', type=float, default=0.20,
                       help='팔레트 너비 (m)')
    parser.add_argument('--pallet-length', type=float, default=0.20,
                       help='팔레트 길이 (m)')
    
    args = parser.parse_args()
    
    # 카메라 초기화
    camera = RealSenseCamera(width=640, height=480, fps=30)
    
    try:
        camera.start()
        
        # 플래너 초기화
        planner = PalletStackPlanner(
            camera=camera,
            pallet_size=(args.pallet_width, args.pallet_length),
            grid_resolution=0.01,
            box_margin=0.02,
            min_overlap_ratio=0.8
        )
        
        if args.mode == 'visualize':
            run_visualization(planner)
        else:
            run_api_mode(planner)
            
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
