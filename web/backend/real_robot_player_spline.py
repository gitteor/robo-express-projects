#!/usr/bin/env python3
"""
Real Robot Trajectory Player - MoveSpline + 기존 그리퍼 방식

=============================================================================
개선 사항
=============================================================================

1. 부드러운 움직임 (MoveSpline)
   - 그리퍼 변화 없는 구간: MoveSplineJoint로 부드럽게 이동
   - 최대 100개 waypoint를 한 번에 전달

2. 그리퍼 타이밍 (기존 코드 방식 완전 복원)
   - 그리퍼 상태 변화 시: MoveJoint 동기 호출 → 완료 후 그리퍼 동작
   - 원래 코드의 키포인트 방식 + 그리퍼 처리 로직 유지

=============================================================================
Usage
=============================================================================

    conda deactivate
    cd ~/ROS2
    source install/setup.bash
    python3 real_robot_player_spline.py --csv episode_trajectory_processed.csv --speed 0.5

=============================================================================
"""

import rclpy
from rclpy.node import Node
from dsr_msgs2.srv import MoveJoint, MoveSplineJoint, MoveLine
from std_msgs.msg import Int32, Float64MultiArray

import csv
import argparse
import time
from typing import List, Dict, Optional


class SplineTrajectoryPlayerNode(Node):
    def __init__(self, gripper_open: int = 0, gripper_close: int = 290):
        super().__init__('spline_trajectory_player')
        
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.last_gripper_state: Optional[int] = None
        
        # MoveJoint 서비스 클라이언트
        self.move_joint_client = self.create_client(
            MoveJoint, '/dsr01/motion/move_joint'
        )
        
        # MoveSplineJoint 서비스 클라이언트
        self.move_spline_client = self.create_client(
            MoveSplineJoint, '/dsr01/motion/move_spline_joint'
        )
        
        # MoveLine 서비스 클라이언트 (수직 상승용)
        self.move_line_client = self.create_client(
            MoveLine, '/dsr01/motion/move_line'
        )
        
        # 그리퍼 토픽 퍼블리셔
        self.gripper_pub = self.create_publisher(
            Int32, '/dsr01/gripper/position_cmd', 10
        )
        
        # 서비스 연결 대기
        self.get_logger().info('서비스 연결 중...')
        
        if not self.move_joint_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('MoveJoint 서비스를 찾을 수 없습니다!')
            raise RuntimeError('MoveJoint service not available')
        self.get_logger().info('✓ MoveJoint 서비스 연결 완료')
        
        if not self.move_spline_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('MoveSplineJoint 서비스를 찾을 수 없습니다!')
            raise RuntimeError('MoveSplineJoint service not available')
        self.get_logger().info('✓ MoveSplineJoint 서비스 연결 완료')
        
        if not self.move_line_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('MoveLine 서비스를 찾을 수 없습니다!')
            raise RuntimeError('MoveLine service not available')
        self.get_logger().info('✓ MoveLine 서비스 연결 완료')
        
        self.get_logger().info('✓ 그리퍼 토픽 준비 완료')
    
    def send_gripper(self, position: int):
        """그리퍼 명령 전송 (토픽 퍼블리시 - non-blocking)"""
        msg = Int32()
        msg.data = position
        self.gripper_pub.publish(msg)
    
    def open_gripper(self):
        """그리퍼 열기"""
        if self.last_gripper_state != 0:
            self.send_gripper(self.gripper_open)
            self.last_gripper_state = 0
            self.get_logger().info(f'🤚 그리퍼 OPEN ({self.gripper_open})')
    
    def close_gripper(self):
        """그리퍼 닫기"""
        if self.last_gripper_state != 1:
            self.send_gripper(self.gripper_close)
            self.last_gripper_state = 1
            self.get_logger().info(f'✊ 그리퍼 CLOSE ({self.gripper_close})')
    
    def move_joint_sync(self, joint_pos_deg: List[float], vel: float = 30.0, acc: float = 30.0):
        """
        MoveJoint 동기 호출 (blocking - 완료까지 대기)
        """
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        return future.result()
    
    def move_line_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                           drx: float = 0.0, dry: float = 0.0, drz: float = 0.0,
                           vel: float = 100.0, acc: float = 100.0):
        """
        MoveLine 상대 이동 (Base 좌표계 기준 - 수직 상승용)
        
        Args:
            dx, dy, dz: 이동 거리 [mm]
            drx, dry, drz: 회전 [deg]
            vel: 속도 [mm/sec]
            acc: 가속도 [mm/sec^2]
        """
        request = MoveLine.Request()
        request.pos = [dx, dy, dz, drx, dry, drz]
        request.vel = [vel, 30.0]  # [mm/sec, deg/sec]
        request.acc = [acc, 30.0]  # [mm/sec^2, deg/sec^2]
        request.time = 0.0
        request.radius = 0.0
        request.ref = 0  # DR_BASE: Base 좌표계 기준 (Z+가 위쪽)
        request.mode = 1  # DR_MV_MOD_REL: 상대 이동
        request.blend_type = 0
        request.sync_type = 0  # SYNC
        
        self.get_logger().info(f'📐 MoveLine 상대이동 (Base): Z+{dz}mm')
        future = self.move_line_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        return future.result()
    
    def move_spline_sync(self, waypoints: List[List[float]], vel: List[float], acc: List[float]):
        """
        MoveSplineJoint 동기 호출 - 여러 waypoint를 부드럽게 연속 이동
        """
        if len(waypoints) < 2:
            # waypoint가 1개면 MoveJoint 사용
            return self.move_joint_sync(waypoints[0], vel=vel[0], acc=acc[0])
        
        request = MoveSplineJoint.Request()
        
        # waypoints를 Float64MultiArray 리스트로 변환
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
        request.mode = 0  # ABSOLUTE
        request.sync_type = 0  # SYNC
        
        future = self.move_spline_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)
        return future.result()
    
    def load_csv(self, csv_path: str) -> List[Dict]:
        """CSV 파일 로드"""
        trajectory = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoint = {
                    'timestamp': float(row['timestamp']),
                    'joint_pos_deg': [
                        float(row[f'joint_{i}_pos_deg']) for i in range(6)
                    ],
                    'gripper_cmd': int(row['gripper_cmd']),
                }
                trajectory.append(waypoint)
        return trajectory
    
    def extract_keypoints(self, trajectory: List[Dict], interval: int = 3) -> List[Dict]:
        """
        키포인트 추출 (기존 방식)
        - 매 interval 번째 waypoint
        - 그리퍼 상태 변화 시점
        - 첫/마지막 waypoint
        """
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
    
    def group_keypoints_by_gripper(self, keypoints: List[Dict]) -> List[Dict]:
        """
        그리퍼 상태 변화 기준으로 키포인트 그룹화
        
        Returns:
            [{'waypoints': [...], 'gripper_change': bool, 'next_gripper': int}, ...]
        """
        groups = []
        current_group = {
            'waypoints': [],
            'gripper_change': False,
            'next_gripper': keypoints[0]['gripper_cmd']
        }
        
        prev_gripper = keypoints[0]['gripper_cmd']
        
        for kp in keypoints:
            if kp['gripper_cmd'] != prev_gripper:
                # 그리퍼 변화 시점 - 현재 그룹 종료
                current_group['gripper_change'] = True
                current_group['next_gripper'] = kp['gripper_cmd']
                groups.append(current_group)
                
                # 새 그룹 시작
                current_group = {
                    'waypoints': [kp['joint_pos_deg']],
                    'gripper_change': False,
                    'next_gripper': kp['gripper_cmd']
                }
                prev_gripper = kp['gripper_cmd']
            else:
                current_group['waypoints'].append(kp['joint_pos_deg'])
                
                # 최대 90개로 제한
                if len(current_group['waypoints']) >= 90:
                    groups.append(current_group)
                    current_group = {
                        'waypoints': [kp['joint_pos_deg']],
                        'gripper_change': False,
                        'next_gripper': kp['gripper_cmd']
                    }
        
        # 마지막 그룹 추가
        if current_group['waypoints']:
            groups.append(current_group)
        
        return groups
    
    def play(self, csv_path: str, speed: float = 0.5, start_delay: float = 3.0):
        """
        Trajectory 재생
        """
        
        # CSV 로드
        trajectory = self.load_csv(csv_path)
        self.get_logger().info(f'CSV 로드: {len(trajectory)} waypoints')
        
        # 키포인트 추출
        keypoints = self.extract_keypoints(trajectory, interval=3)
        
        # 그룹화
        groups = self.group_keypoints_by_gripper(keypoints)
        
        total_waypoints = sum(len(g['waypoints']) for g in groups)
        
        print('=' * 60)
        print('🤖 Real Robot Trajectory Player - MoveSpline')
        print('=' * 60)
        print(f'  Total waypoints: {len(trajectory)}')
        print(f'  Keypoints: {len(keypoints)}')
        print(f'  Groups: {len(groups)}')
        print(f'  Speed: {speed}x')
        print('=' * 60)
        
        # 그룹 정보 출력
        print('\n📋 Motion Groups:')
        for i, group in enumerate(groups):
            gripper_info = ""
            if group['gripper_change']:
                gripper_info = f" → Gripper {'CLOSE' if group['next_gripper'] else 'OPEN'}"
            print(f'  Group {i+1}: {len(group["waypoints"])} waypoints{gripper_info}')
        
        # 시작 전 대기
        print(f'\n⚠️  {int(start_delay)}초 후 로봇이 움직입니다!')
        print('⚠️  로봇 주변을 확인하세요!\n')
        for i in range(int(start_delay), 0, -1):
            print(f'  시작까지 {i}초...')
            time.sleep(1.0)
        
        print('\n🚀 재생 시작!\n')
        
        # 속도 설정
        base_vel = max(20.0, 60.0 * speed)
        base_acc = max(20.0, 60.0 * speed)
        vel = [base_vel] * 6
        acc = [base_acc] * 6
        
        start_real_time = time.time()
        
        # 초기 위치로 이동 (MoveJoint 동기)
        first_wp = groups[0]['waypoints'][0]
        print(f'[초기화] 시작 위치로 이동 중...')
        self.move_joint_sync(first_wp, vel=20.0, acc=20.0)
        print(f'  위치: {[f"{j:.1f}" for j in first_wp]}')
        
        # 그리퍼 초기 상태
        time.sleep(0.5)
        if trajectory[0]['gripper_cmd'] == 1:
            self.close_gripper()
        else:
            self.open_gripper()
        time.sleep(1.5)
        
        print(f'\n[모션 재생 시작]')
        
        # 각 그룹 재생
        for group_idx, group in enumerate(groups):
            waypoints = group['waypoints']
            
            print(f'\n--- Group {group_idx + 1}/{len(groups)} ({len(waypoints)} waypoints) ---')
            
            # MoveSpline으로 부드럽게 이동
            if len(waypoints) >= 2:
                print(f'  MoveSpline 실행 중...')
                result = self.move_spline_sync(waypoints, vel, acc)
                if result and result.success:
                    print(f'  ✓ 이동 완료')
                else:
                    print(f'  ⚠ MoveSpline 실패, MoveJoint로 대체')
                    self.move_joint_sync(waypoints[-1], vel=base_vel, acc=base_acc)
            else:
                # waypoint가 1개면 MoveJoint 사용
                self.move_joint_sync(waypoints[0], vel=base_vel, acc=base_acc)
                print(f'  ✓ MoveJoint 완료')
            
            # 그리퍼 상태 변화 처리 (기존 방식: 이동 완료 후 그리퍼 동작)
            if group['gripper_change']:
                time.sleep(0.3)  # 안정화 대기
                if group['next_gripper'] == 1:
                    self.close_gripper()
                else:
                    self.open_gripper()
                    # Release 후 수직 상승 (충돌 방지)
                    time.sleep(0.5)
                    self.move_line_relative(dz=50.0, vel=100.0, acc=100.0)  # Z+50mm
                    print('  📦 박스 release 완료 - 수직 상승')
                    
                    # 초기 위치로 복귀
                    print('  🏠 초기 위치로 복귀 중...')
                    self.move_joint_sync(first_wp, vel=30.0, acc=30.0)
                    print('  ✓ 복귀 완료')
                    
                    # 남은 그룹 스킵
                    if group_idx < len(groups) - 1:
                        print('  ⏭️  남은 경로 스킵 - 재생 종료')
                        break
                time.sleep(1.0)  # 그리퍼 동작 완료 대기
            
            # 진행 상황 출력
            elapsed = time.time() - start_real_time
            progress = (group_idx + 1) / len(groups) * 100
            print(f'  Progress: {progress:.0f}% | 경과: {elapsed:.1f}s')
        
        # 완료 대기
        print(f'\n[완료 대기 중...]')
        time.sleep(2.0)
        
        total_time = time.time() - start_real_time
        print('=' * 60)
        print(f'✅ 재생 완료! (총 {total_time:.1f}초)')
        print('=' * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Real Robot Trajectory Player - MoveSpline Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
    python3 real_robot_player_spline.py --csv episode_trajectory_processed.csv --speed 0.5
    python3 real_robot_player_spline.py --csv trajectory.csv --speed 0.3 --gripper_close 400
        """
    )
    parser.add_argument('--csv', type=str, required=True, help='CSV 파일 경로')
    parser.add_argument('--speed', type=float, default=0.5, help='재생 속도 (default: 0.5)')
    parser.add_argument('--start_delay', type=float, default=3.0, help='시작 전 대기 시간 (default: 3.0)')
    parser.add_argument('--gripper_open', type=int, default=0, help='그리퍼 열림 위치 (default: 0)')
    parser.add_argument('--gripper_close', type=int, default=290, help='그리퍼 닫힘 위치 (default: 290)')
    
    args = parser.parse_args()
    
    # ROS2 초기화
    rclpy.init()
    
    try:
        node = SplineTrajectoryPlayerNode(
            gripper_open=args.gripper_open,
            gripper_close=args.gripper_close
        )
        node.play(
            csv_path=args.csv,
            speed=args.speed,
            start_delay=args.start_delay
        )
    except KeyboardInterrupt:
        print('\n\n⚠️ 사용자에 의해 중단됨')
    except FileNotFoundError:
        print(f'❌ Error: CSV 파일을 찾을 수 없습니다: {args.csv}')
    except RuntimeError as e:
        print(f'❌ Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
