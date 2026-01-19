#!/usr/bin/env python3
"""
=============================================================================
Robo Express - Web Client & Robot Controller
=============================================================================

Inference Server에서 받은 궤적을 실제 로봇에 전송합니다.
기존 real_robot_player_spline.py 로직을 통합했습니다.

=============================================================================
Usage
=============================================================================

1. Inference Server가 실행 중인 상태에서:

   # 단순 테스트 (궤적만 확인)
   python3 robo_express_client.py --cubeA_x -0.05 --cubeA_y 0.0 \
                                   --cubeB_x 0.25 --cubeB_y 0.0

   # 실제 로봇 실행
   python3 robo_express_client.py --cubeA_x -0.05 --cubeA_y 0.0 \
                                   --cubeB_x 0.25 --cubeB_y 0.0 \
                                   --execute --speed 0.5

2. Robo Express 웹에서 호출:
   
   from robo_express_client import RoboExpressClient
   
   client = RoboExpressClient(server_url="http://localhost:5000")
   result = client.pick_and_place(
       pickup_x=-0.05, pickup_y=0.0,
       place_x=0.25, place_y=0.0,
       execute=True
   )

=============================================================================
"""

import requests
import argparse
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PickAndPlaceResult:
    """Pick & Place 결과"""
    success: bool
    trajectory: List[Dict]
    total_reward: float
    inference_time: float
    execution_time: float
    message: str


class RoboExpressClient:
    """
    Robo Express 클라이언트
    
    Inference Server와 통신하고, 실제 로봇에 명령을 전송합니다.
    """
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url.rstrip('/')
        self.robot_controller = None
    
    def check_server(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            data = response.json()
            return data.get('status') == 'ready'
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")
            return False
    
    def get_trajectory(self, cubeA_x: float, cubeA_y: float,
                       cubeB_x: float, cubeB_y: float) -> Dict:
        """
        Inference Server에서 궤적 요청
        
        Args:
            cubeA_x, cubeA_y: 픽업 위치
            cubeB_x, cubeB_y: 배치 위치
        
        Returns:
            서버 응답 (trajectory, success 등)
        """
        payload = {
            "cubeA": {"x": cubeA_x, "y": cubeA_y},
            "cubeB": {"x": cubeB_x, "y": cubeB_y}
        }
        
        response = requests.post(
            f"{self.server_url}/infer",
            json=payload,
            timeout=60  # 추론에 시간이 걸릴 수 있음
        )
        
        return response.json()
    
    def execute_on_robot(self, trajectory: List[Dict], speed: float = 0.5) -> float:
        """
        실제 로봇에서 궤적 실행
        
        Args:
            trajectory: 궤적 데이터
            speed: 실행 속도 (0.1 ~ 1.0)
        
        Returns:
            실행 시간 (초)
        """
        if self.robot_controller is None:
            self._init_robot_controller()
        
        start_time = time.time()
        self.robot_controller.play_trajectory(trajectory, speed)
        return time.time() - start_time
    
    def _init_robot_controller(self):
        """ROS2 로봇 컨트롤러 초기화"""
        try:
            import rclpy
            from .robot_controller import SplineTrajectoryController
            
            if not rclpy.ok():
                rclpy.init()
            
            self.robot_controller = SplineTrajectoryController()
        except ImportError:
            print("⚠️ ROS2 모듈을 찾을 수 없습니다. 로봇 실행이 비활성화됩니다.")
            self.robot_controller = None
    
    def pick_and_place(self, pickup_x: float, pickup_y: float,
                       place_x: float, place_y: float,
                       execute: bool = False, speed: float = 0.5) -> PickAndPlaceResult:
        """
        전체 Pick & Place 작업 수행
        
        Args:
            pickup_x, pickup_y: 픽업 위치 (CubeA)
            place_x, place_y: 배치 위치 (CubeB)
            execute: 실제 로봇에서 실행할지 여부
            speed: 로봇 실행 속도
        
        Returns:
            PickAndPlaceResult
        """
        execution_time = 0.0
        
        # 1. 서버 상태 확인
        if not self.check_server():
            return PickAndPlaceResult(
                success=False,
                trajectory=[],
                total_reward=0.0,
                inference_time=0.0,
                execution_time=0.0,
                message="Inference Server에 연결할 수 없습니다."
            )
        
        # 2. 궤적 요청
        print(f"\n🎯 Pick & Place 요청")
        print(f"   픽업: ({pickup_x:.3f}, {pickup_y:.3f})")
        print(f"   배치: ({place_x:.3f}, {place_y:.3f})")
        
        result = self.get_trajectory(pickup_x, pickup_y, place_x, place_y)
        
        if not result.get('success', False):
            return PickAndPlaceResult(
                success=False,
                trajectory=[],
                total_reward=result.get('total_reward', 0.0),
                inference_time=result.get('inference_time', 0.0),
                execution_time=0.0,
                message=result.get('message', '추론 실패')
            )
        
        trajectory = result.get('trajectory', [])
        print(f"   ✓ 궤적 생성 완료: {len(trajectory)}개 포인트")
        print(f"   ✓ 추론 시간: {result.get('inference_time', 0):.2f}초")
        
        # 3. 로봇 실행 (선택적)
        if execute and trajectory:
            print(f"\n🤖 로봇 실행 중... (속도: {speed}x)")
            try:
                execution_time = self.execute_on_robot(trajectory, speed)
                print(f"   ✓ 실행 완료: {execution_time:.1f}초")
            except Exception as e:
                return PickAndPlaceResult(
                    success=False,
                    trajectory=trajectory,
                    total_reward=result.get('total_reward', 0.0),
                    inference_time=result.get('inference_time', 0.0),
                    execution_time=0.0,
                    message=f"로봇 실행 오류: {str(e)}"
                )
        
        return PickAndPlaceResult(
            success=True,
            trajectory=trajectory,
            total_reward=result.get('total_reward', 0.0),
            inference_time=result.get('inference_time', 0.0),
            execution_time=execution_time,
            message=result.get('message', '성공')
        )


class SplineTrajectoryController:
    """
    실제 로봇 제어 (기존 real_robot_player_spline.py 통합)
    """
    
    def __init__(self, gripper_open: int = 0, gripper_close: int = 290):
        import rclpy
        from rclpy.node import Node
        from dsr_msgs2.srv import MoveJoint, MoveSplineJoint, MoveLine
        from std_msgs.msg import Int32, Float64MultiArray
        
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.last_gripper_state = None
        
        # ROS2 노드 생성
        self.node = rclpy.create_node('robo_express_controller')
        
        # 서비스 클라이언트
        self.move_joint_client = self.node.create_client(
            MoveJoint, '/dsr01/motion/move_joint'
        )
        self.move_spline_client = self.node.create_client(
            MoveSplineJoint, '/dsr01/motion/move_spline_joint'
        )
        self.move_line_client = self.node.create_client(
            MoveLine, '/dsr01/motion/move_line'
        )
        
        # 그리퍼 퍼블리셔
        self.gripper_pub = self.node.create_publisher(
            Int32, '/dsr01/gripper/position_cmd', 10
        )
        
        # 서비스 연결 대기
        self._wait_for_services()
    
    def _wait_for_services(self):
        """ROS2 서비스 연결 대기"""
        import rclpy
        
        services = [
            (self.move_joint_client, 'MoveJoint'),
            (self.move_spline_client, 'MoveSplineJoint'),
            (self.move_line_client, 'MoveLine'),
        ]
        
        for client, name in services:
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f'{name} 서비스를 찾을 수 없습니다!')
            print(f"✓ {name} 서비스 연결 완료")
    
    def send_gripper(self, position: int):
        """그리퍼 명령 전송"""
        from std_msgs.msg import Int32
        msg = Int32()
        msg.data = position
        self.gripper_pub.publish(msg)
    
    def open_gripper(self):
        if self.last_gripper_state != 0:
            self.send_gripper(self.gripper_open)
            self.last_gripper_state = 0
            print(f'🤚 그리퍼 OPEN')
    
    def close_gripper(self):
        if self.last_gripper_state != 1:
            self.send_gripper(self.gripper_close)
            self.last_gripper_state = 1
            print(f'✊ 그리퍼 CLOSE')
    
    def move_joint_sync(self, joint_pos_deg: list, vel: float = 30.0, acc: float = 30.0):
        """MoveJoint 동기 호출"""
        import rclpy
        from dsr_msgs2.srv import MoveJoint
        
        request = MoveJoint.Request()
        request.pos = joint_pos_deg
        request.vel = vel
        request.acc = acc
        request.time = 0.0
        request.radius = 0.0
        request.mode = 0
        request.blend_type = 0
        request.sync_type = 0
        
        future = self.move_joint_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        return future.result()
    
    def move_spline_sync(self, waypoints: list, vel: list, acc: list):
        """MoveSplineJoint 동기 호출"""
        import rclpy
        from dsr_msgs2.srv import MoveSplineJoint
        from std_msgs.msg import Float64MultiArray
        
        if len(waypoints) < 2:
            return self.move_joint_sync(waypoints[0], vel=vel[0], acc=acc[0])
        
        request = MoveSplineJoint.Request()
        
        pos_list = []
        for wp in waypoints:
            msg = Float64MultiArray()
            msg.data = wp
            pos_list.append(msg)
        
        request.pos = pos_list
        request.pos_cnt = len(waypoints)
        request.vel = vel
        request.acc = acc
        request.time = 0.0
        request.mode = 0
        request.sync_type = 0
        
        future = self.move_spline_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=120.0)
        return future.result()
    
    def move_line_relative(self, dz: float = 0.0, vel: float = 100.0, acc: float = 100.0):
        """MoveLine 상대 이동 (수직 상승용)"""
        import rclpy
        from dsr_msgs2.srv import MoveLine
        
        request = MoveLine.Request()
        request.pos = [0.0, 0.0, dz, 0.0, 0.0, 0.0]
        request.vel = [vel, 30.0]
        request.acc = [acc, 30.0]
        request.time = 0.0
        request.radius = 0.0
        request.ref = 0  # DR_BASE
        request.mode = 1  # DR_MV_MOD_REL
        request.blend_type = 0
        request.sync_type = 0
        
        future = self.move_line_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=30.0)
        return future.result()
    
    def play_trajectory(self, trajectory: list, speed: float = 0.5):
        """
        궤적 재생 (기존 real_robot_player_spline.py 로직)
        """
        # 키포인트 추출
        keypoints = self._extract_keypoints(trajectory, interval=3)
        
        # 그룹화
        groups = self._group_by_gripper(keypoints)
        
        print(f"  총 포인트: {len(trajectory)}")
        print(f"  키포인트: {len(keypoints)}")
        print(f"  그룹: {len(groups)}")
        
        # 속도 설정
        base_vel = max(20.0, 60.0 * speed)
        base_acc = max(20.0, 60.0 * speed)
        vel = [base_vel] * 6
        acc = [base_acc] * 6
        
        # 초기 위치로 이동
        first_wp = groups[0]['waypoints'][0]
        print(f'[초기화] 시작 위치로 이동 중...')
        self.move_joint_sync(first_wp, vel=20.0, acc=20.0)
        
        # 그리퍼 초기 상태
        time.sleep(0.5)
        if trajectory[0]['gripper_cmd'] == 1:
            self.close_gripper()
        else:
            self.open_gripper()
        time.sleep(1.5)
        
        # 각 그룹 재생
        for group_idx, group in enumerate(groups):
            waypoints = group['waypoints']
            
            print(f'  Group {group_idx + 1}/{len(groups)} ({len(waypoints)} waypoints)')
            
            # MoveSpline으로 이동
            if len(waypoints) >= 2:
                result = self.move_spline_sync(waypoints, vel, acc)
                if not (result and result.success):
                    self.move_joint_sync(waypoints[-1], vel=base_vel, acc=base_acc)
            else:
                self.move_joint_sync(waypoints[0], vel=base_vel, acc=base_acc)
            
            # 그리퍼 상태 변화 처리
            if group['gripper_change']:
                time.sleep(0.3)
                if group['next_gripper'] == 1:
                    self.close_gripper()
                else:
                    self.open_gripper()
                    time.sleep(0.5)
                    self.move_line_relative(dz=50.0)  # 수직 상승
                    print('  📦 Release 완료')
                    
                    # 초기 위치로 복귀
                    self.move_joint_sync(first_wp, vel=30.0, acc=30.0)
                    break
                time.sleep(1.0)
        
        print('✅ 궤적 재생 완료')
    
    def _extract_keypoints(self, trajectory: list, interval: int = 3) -> list:
        """키포인트 추출"""
        keypoints = [trajectory[0]]
        prev_gripper = trajectory[0]['gripper_cmd']
        
        for i, wp in enumerate(trajectory[1:-1], 1):
            if wp['gripper_cmd'] != prev_gripper:
                keypoints.append(wp)
                prev_gripper = wp['gripper_cmd']
            elif i % interval == 0:
                keypoints.append(wp)
        
        keypoints.append(trajectory[-1])
        return keypoints
    
    def _group_by_gripper(self, keypoints: list) -> list:
        """그리퍼 상태 변화 기준으로 그룹화"""
        groups = []
        current_group = {
            'waypoints': [],
            'gripper_change': False,
            'next_gripper': keypoints[0]['gripper_cmd']
        }
        
        prev_gripper = keypoints[0]['gripper_cmd']
        
        for kp in keypoints:
            if kp['gripper_cmd'] != prev_gripper:
                current_group['gripper_change'] = True
                current_group['next_gripper'] = kp['gripper_cmd']
                groups.append(current_group)
                
                current_group = {
                    'waypoints': [kp['joint_pos_deg']],
                    'gripper_change': False,
                    'next_gripper': kp['gripper_cmd']
                }
                prev_gripper = kp['gripper_cmd']
            else:
                current_group['waypoints'].append(kp['joint_pos_deg'])
                
                if len(current_group['waypoints']) >= 90:
                    groups.append(current_group)
                    current_group = {
                        'waypoints': [kp['joint_pos_deg']],
                        'gripper_change': False,
                        'next_gripper': kp['gripper_cmd']
                    }
        
        if current_group['waypoints']:
            groups.append(current_group)
        
        return groups


def main():
    parser = argparse.ArgumentParser(description="Robo Express Client")
    parser.add_argument("--server", type=str, default="http://localhost:5000")
    parser.add_argument("--cubeA_x", type=float, default=-0.05)
    parser.add_argument("--cubeA_y", type=float, default=0.0)
    parser.add_argument("--cubeB_x", type=float, default=0.25)
    parser.add_argument("--cubeB_y", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true", help="실제 로봇에서 실행")
    parser.add_argument("--speed", type=float, default=0.5, help="로봇 실행 속도")
    parser.add_argument("--save_csv", type=str, default=None, help="CSV로 저장")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🤖 Robo Express Client")
    print("=" * 60)
    
    client = RoboExpressClient(server_url=args.server)
    
    result = client.pick_and_place(
        pickup_x=args.cubeA_x,
        pickup_y=args.cubeA_y,
        place_x=args.cubeB_x,
        place_y=args.cubeB_y,
        execute=args.execute,
        speed=args.speed
    )
    
    print("\n" + "=" * 60)
    print("📊 결과")
    print("=" * 60)
    print(f"  성공: {'✅' if result.success else '❌'}")
    print(f"  메시지: {result.message}")
    print(f"  궤적 포인트: {len(result.trajectory)}")
    print(f"  총 리워드: {result.total_reward:.2f}")
    print(f"  추론 시간: {result.inference_time:.2f}초")
    if result.execution_time > 0:
        print(f"  실행 시간: {result.execution_time:.1f}초")
    
    # CSV 저장
    if args.save_csv and result.trajectory:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.trajectory[0].keys())
            writer.writeheader()
            writer.writerows(result.trajectory)
        print(f"\n💾 CSV 저장: {args.save_csv}")


if __name__ == "__main__":
    main()
