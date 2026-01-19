"""
Arduino 컨베이어 벨트 제어 모듈

사용법:
    from conveyor_controller import ConveyorController
    
    conveyor = ConveyorController('/dev/ttyUSB0')
    conveyor.connect()
    conveyor.start()
    conveyor.stop()
    conveyor.disconnect()
"""

import serial
import time
from typing import Optional
import threading


class ConveyorController:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 9600):
        """
        컨베이어 컨트롤러 초기화
        
        Args:
            port: 시리얼 포트 (예: '/dev/ttyUSB0', '/dev/ttyACM0', 'COM3')
            baudrate: 통신 속도 (Arduino와 동일하게 9600)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self.connected = False
        self.current_state = "UNKNOWN"
        self.current_speed = 200
        self._lock = threading.Lock()
    
    def connect(self, timeout: float = 2.0) -> bool:
        """
        Arduino 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=timeout
            )
            # Arduino 리셋 대기
            time.sleep(2)
            
            # 초기 메시지 읽기
            if self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8').strip()
                print(f"[Conveyor] {response}")
            
            self.connected = True
            print(f"[Conveyor] Connected to {self.port}")
            return True
            
        except serial.SerialException as e:
            print(f"[Conveyor] Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Arduino 연결 해제"""
        if self.serial and self.serial.is_open:
            self.stop()  # 정지 후 연결 해제
            self.serial.close()
            self.connected = False
            print("[Conveyor] Disconnected")
    
    def _send_command(self, command: str) -> Optional[str]:
        """
        명령 전송 및 응답 수신
        
        Args:
            command: 전송할 명령어
            
        Returns:
            Arduino 응답 문자열
        """
        if not self.connected or not self.serial:
            print("[Conveyor] Not connected")
            return None
        
        with self._lock:
            try:
                # 명령 전송
                self.serial.write(f"{command}\n".encode('utf-8'))
                self.serial.flush()
                
                # 응답 대기
                time.sleep(0.1)
                if self.serial.in_waiting:
                    response = self.serial.readline().decode('utf-8').strip()
                    return response
                return None
                
            except serial.SerialException as e:
                print(f"[Conveyor] Command failed: {e}")
                return None
    
    def start(self) -> bool:
        """
        컨베이어 시작 (정방향)
        
        Returns:
            성공 여부
        """
        response = self._send_command("START")
        if response and response.startswith("OK"):
            self.current_state = "FORWARD"
            print("[Conveyor] Started (Forward)")
            return True
        return False
    
    def stop(self) -> bool:
        """
        컨베이어 정지
        
        Returns:
            성공 여부
        """
        response = self._send_command("STOP")
        if response and response.startswith("OK"):
            self.current_state = "STOPPED"
            print("[Conveyor] Stopped")
            return True
        return False
    
    def reverse(self) -> bool:
        """
        컨베이어 역방향
        
        Returns:
            성공 여부
        """
        response = self._send_command("REVERSE")
        if response and response.startswith("OK"):
            self.current_state = "REVERSE"
            print("[Conveyor] Reversed")
            return True
        return False
    
    def set_speed(self, speed: int) -> bool:
        """
        속도 설정
        
        Args:
            speed: 속도 값 (0-255)
            
        Returns:
            성공 여부
        """
        if not 0 <= speed <= 255:
            print("[Conveyor] Invalid speed (0-255)")
            return False
        
        response = self._send_command(f"SPEED:{speed}")
        if response and response.startswith("OK"):
            self.current_speed = speed
            print(f"[Conveyor] Speed set to {speed}")
            return True
        return False
    
    def get_status(self) -> dict:
        """
        현재 상태 조회
        
        Returns:
            상태 딕셔너리 {'state': str, 'speed': int}
        """
        response = self._send_command("STATUS")
        if response and response.startswith("STATUS:"):
            parts = response.split(":")[1].split(",")
            if len(parts) >= 2:
                self.current_state = parts[0]
                self.current_speed = int(parts[1])
        
        return {
            "state": self.current_state,
            "speed": self.current_speed,
            "connected": self.connected
        }


# ==================== 테스트 코드 ====================
if __name__ == "__main__":
    import sys
    
    # 포트 자동 감지
    port = '/dev/ttyUSB0'
    if len(sys.argv) > 1:
        port = sys.argv[1]
    
    print(f"Testing ConveyorController on {port}")
    print("=" * 50)
    
    conveyor = ConveyorController(port)
    
    if not conveyor.connect():
        print("Failed to connect. Check the port and try again.")
        print("\nAvailable ports:")
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        sys.exit(1)
    
    try:
        print("\n[Test 1] Get Status")
        status = conveyor.get_status()
        print(f"  Status: {status}")
        
        print("\n[Test 2] Start Conveyor")
        conveyor.start()
        time.sleep(2)
        
        print("\n[Test 3] Get Status")
        status = conveyor.get_status()
        print(f"  Status: {status}")
        
        print("\n[Test 4] Stop Conveyor")
        conveyor.stop()
        time.sleep(1)
        
        print("\n[Test 5] Set Speed to 150")
        conveyor.set_speed(150)
        
        print("\n[Test 6] Start with new speed")
        conveyor.start()
        time.sleep(2)
        
        print("\n[Test 7] Reverse")
        conveyor.reverse()
        time.sleep(2)
        
        print("\n[Test 8] Final Stop")
        conveyor.stop()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        conveyor.disconnect()
