"""
=============================================================================
Robo Express - 웹 통합 예시
=============================================================================

기존 Robo Express Flask 웹앱에 통합하는 예시 코드입니다.

=============================================================================
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Inference Server URL
INFERENCE_SERVER = "http://localhost:5000"


# ============================================================================
# 기존 Robo Express 라우트에 추가할 엔드포인트
# ============================================================================

@app.route('/api/robot/pick_and_place', methods=['POST'])
def pick_and_place():
    """
    웹에서 호출하는 Pick & Place API
    
    Request:
    {
        "pickup": {"x": -0.05, "y": 0.0},
        "place": {"x": 0.25, "y": 0.0},
        "execute": true,
        "speed": 0.5
    }
    
    Response:
    {
        "success": true,
        "trajectory_points": 234,
        "message": "작업 완료",
        "inference_time": 3.5,
        "execution_time": 15.2
    }
    """
    try:
        data = request.get_json()
        
        pickup = data.get('pickup', {})
        place = data.get('place', {})
        execute = data.get('execute', False)
        speed = data.get('speed', 0.5)
        
        # 1. Inference Server에 궤적 요청
        infer_response = requests.post(
            f"{INFERENCE_SERVER}/infer",
            json={
                "cubeA": {"x": pickup.get('x', -0.05), "y": pickup.get('y', 0.0)},
                "cubeB": {"x": place.get('x', 0.25), "y": place.get('y', 0.0)}
            },
            timeout=60
        )
        
        result = infer_response.json()
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'message': result.get('message', '궤적 생성 실패'),
                'trajectory_points': 0
            })
        
        trajectory = result.get('trajectory', [])
        
        # 2. 로봇 실행 (execute=True인 경우)
        execution_time = 0.0
        if execute and trajectory:
            # 실제로는 ROS2를 통해 로봇에 명령 전송
            # 여기서는 별도의 로봇 컨트롤러 호출 필요
            pass
        
        return jsonify({
            'success': True,
            'message': '작업 완료',
            'trajectory_points': len(trajectory),
            'inference_time': result.get('inference_time', 0),
            'execution_time': execution_time,
            'total_reward': result.get('total_reward', 0)
        })
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'message': 'Inference Server에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.',
            'trajectory_points': 0
        }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'오류: {str(e)}',
            'trajectory_points': 0
        }), 500


@app.route('/api/robot/status', methods=['GET'])
def robot_status():
    """로봇 시스템 상태 확인"""
    try:
        response = requests.get(f"{INFERENCE_SERVER}/health", timeout=5)
        data = response.json()
        
        return jsonify({
            'inference_server': data.get('status') == 'ready',
            'message': data.get('message', '')
        })
    except:
        return jsonify({
            'inference_server': False,
            'message': 'Inference Server 연결 실패'
        })


@app.route('/api/robot/preview', methods=['POST'])
def preview_trajectory():
    """
    로봇 실행 없이 궤적만 미리보기
    
    웹에서 3D 시각화용으로 사용
    """
    try:
        data = request.get_json()
        
        pickup = data.get('pickup', {})
        place = data.get('place', {})
        
        response = requests.post(
            f"{INFERENCE_SERVER}/infer",
            json={
                "cubeA": {"x": pickup.get('x', -0.05), "y": pickup.get('y', 0.0)},
                "cubeB": {"x": place.get('x', 0.25), "y": place.get('y', 0.0)}
            },
            timeout=60
        )
        
        result = response.json()
        
        # 시각화용으로 간소화된 궤적 반환 (매 10번째 포인트만)
        trajectory = result.get('trajectory', [])
        simplified = trajectory[::10] if len(trajectory) > 10 else trajectory
        
        return jsonify({
            'success': result.get('success', False),
            'trajectory': simplified,
            'total_points': len(trajectory),
            'inference_time': result.get('inference_time', 0)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'trajectory': [],
            'message': str(e)
        }), 500


# ============================================================================
# JavaScript 클라이언트 예시 (웹 프론트엔드용)
# ============================================================================

JAVASCRIPT_EXAMPLE = """
// Robo Express 웹 클라이언트 (JavaScript)

class RoboExpressClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }
    
    async checkStatus() {
        const response = await fetch(`${this.baseUrl}/api/robot/status`);
        return await response.json();
    }
    
    async pickAndPlace(pickupX, pickupY, placeX, placeY, execute = false, speed = 0.5) {
        const response = await fetch(`${this.baseUrl}/api/robot/pick_and_place`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                pickup: {x: pickupX, y: pickupY},
                place: {x: placeX, y: placeY},
                execute: execute,
                speed: speed
            })
        });
        return await response.json();
    }
    
    async previewTrajectory(pickupX, pickupY, placeX, placeY) {
        const response = await fetch(`${this.baseUrl}/api/robot/preview`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                pickup: {x: pickupX, y: pickupY},
                place: {x: placeX, y: placeY}
            })
        });
        return await response.json();
    }
}

// 사용 예시
const client = new RoboExpressClient();

// 1. 서버 상태 확인
const status = await client.checkStatus();
console.log('서버 상태:', status);

// 2. 궤적 미리보기
const preview = await client.previewTrajectory(-0.05, 0.0, 0.25, 0.0);
console.log('궤적 포인트:', preview.total_points);

// 3. 실제 실행
if (preview.success) {
    const result = await client.pickAndPlace(-0.05, 0.0, 0.25, 0.0, true, 0.5);
    console.log('실행 결과:', result);
}
"""


# ============================================================================
# 시스템 아키텍처 설명
# ============================================================================

ARCHITECTURE_DOC = """
=============================================================================
Robo Express 시스템 아키텍처
=============================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│                         Robo Express 웹 서비스                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   웹 프론트엔드  │    │  Flask 백엔드   │    │    데이터베이스  │     │
│  │   (React/Vue)   │───▶│   (기존 앱)     │◀───│    (주문 정보)   │     │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘     │
└──────────────────────────────────│──────────────────────────────────────┘
                                   │
                                   │ HTTP (REST API)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Inference Server (새로 추가)                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   Isaac Lab     │    │   학습된 모델    │    │   Flask API     │     │
│  │   환경 (1회)    │───▶│   (.pt 파일)    │───▶│   /infer        │     │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘     │
│                                                         │              │
│  ⏱️ 초기화: 10-20초                                     │              │
│  ⚡ 추론: 요청당 3-5초                                   │              │
└─────────────────────────────────────────────────────────│──────────────┘
                                                          │
                                   ┌──────────────────────┘
                                   │ 궤적 데이터 (JSON)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         실제 로봇 제어                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  로봇 컨트롤러   │    │   ROS2 Bridge   │    │  Doosan E0509   │     │
│  │  (Python)       │───▶│  MoveSpline     │───▶│  + RH-P12-RN    │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘

=============================================================================
데이터 흐름
=============================================================================

1. 웹 주문 접수
   └─▶ 픽업 위치 (카메라로 감지) + 배치 위치 (주문 정보)

2. Inference Server 요청
   └─▶ POST /infer {"cubeA": {x, y}, "cubeB": {x, y}}

3. 정책 추론 (3-5초)
   └─▶ Isaac Lab 시뮬레이션 실행
   └─▶ 성공 시 궤적 반환 (후처리 포함)

4. 로봇 실행
   └─▶ MoveSplineJoint로 부드러운 이동
   └─▶ 그리퍼 OPEN/CLOSE 동기화

=============================================================================
장점
=============================================================================

✅ 환경 1회 로드 → 이후 요청당 3-5초 응답
✅ REST API로 웹 통합 용이
✅ 기존 sim2real 코드 재사용
✅ 기존 robot player 코드 재사용
✅ 확장 가능한 구조

=============================================================================
"""

if __name__ == "__main__":
    print(ARCHITECTURE_DOC)
    app.run(host='0.0.0.0', port=8080, debug=True)
