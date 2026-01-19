/*
 * 컨베이어 벨트 제어 - 시리얼 통신 버전
 * 
 * 시리얼 명령어:
 *   START    - 컨베이어 정방향 시작
 *   STOP     - 컨베이어 정지
 *   REVERSE  - 컨베이어 역방향
 *   SPEED:값 - 속도 설정 (0-255), 예: SPEED:150
 *   STATUS   - 현재 상태 반환
 * 
 * 응답:
 *   OK:명령어    - 명령 성공
 *   ERROR:메시지 - 명령 실패
 *   STATUS:상태,속도 - 상태 응답
 */

const int ENA = 5;  // PWM 속도 제어
const int IN1 = 8;  // 방향 제어 1
const int IN2 = 9;  // 방향 제어 2

// 상태 변수
int currentSpeed = 150;      // 기본 속도 (0-255)
String currentState = "STOPPED";  // STOPPED, FORWARD, REVERSE

void setup() {
  // 시리얼 통신 시작
  Serial.begin(9600);
  
  // 핀 모드 설정
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  
  // 초기 상태: 정지
  stopConveyor();
  
  Serial.println("READY:Conveyor Controller");
}

void loop() {
  // 시리얼 명령 처리
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();  // 공백 제거
    processCommand(command);
  }
}

void processCommand(String command) {
  command.toUpperCase();  // 대문자로 변환
  
  if (command == "START" || command == "FORWARD") {
    startForward();
    Serial.println("OK:START");
  }
  else if (command == "STOP") {
    stopConveyor();
    Serial.println("OK:STOP");
  }
  else if (command == "REVERSE") {
    startReverse();
    Serial.println("OK:REVERSE");
  }
  else if (command.startsWith("SPEED:")) {
    int speed = command.substring(6).toInt();
    if (speed >= 0 && speed <= 255) {
      setSpeed(speed);
      Serial.println("OK:SPEED:" + String(speed));
    } else {
      Serial.println("ERROR:Invalid speed (0-255)");
    }
  }
  else if (command == "STATUS") {
    Serial.println("STATUS:" + currentState + "," + String(currentSpeed));
  }
  else {
    Serial.println("ERROR:Unknown command");
  }
}

void startForward() {
  analogWrite(ENA, currentSpeed);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  currentState = "FORWARD";
}

void startReverse() {
  analogWrite(ENA, currentSpeed);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  currentState = "REVERSE";
}

void stopConveyor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
  currentState = "STOPPED";
}

void setSpeed(int speed) {
  currentSpeed = speed;
  // 이미 동작 중이면 속도 즉시 적용
  if (currentState != "STOPPED") {
    analogWrite(ENA, currentSpeed);
  }
}
