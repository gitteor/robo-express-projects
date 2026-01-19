"""
Robo Express Web 물류 시스템 - FastAPI 백엔드
RealSense 카메라 연동 버전

주요 기능:
- 웹 주문 처리
- RealSense로 박스 감지
- 박스가 Y 범위(-0.04~0.04)에 들어오면 컨베이어 정지
- 로봇 픽앤플레이스 제어
- AI 비서 (GPT + TTS)
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import json
import uuid
import threading
import subprocess
import os

# OpenAI
from openai import OpenAI

# 컨베이어 컨트롤러 import
from conveyor_controller import ConveyorController

# 비전 서비스 import
from vision_service import VisionService, VISION_AVAILABLE

app = FastAPI(title="Robo Express Web 물류 시스템 API")

# CORS 설정 (모든 origin 허용 - 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 설정 ====================
CONVEYOR_PORT = '/dev/ttyACM0'  # Arduino 포트
CONVEYOR_ENABLED = True         # 실제 컨베이어 제어
VISION_ENABLED = True           # RealSense 카메라 사용
ROBOT_ENABLED = True            # 실제 로봇 제어

# OpenAI 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # 환경변수에서 가져오기
openai_client = None

# 로봇 CSV 경로 설정
CSV_DIR = os.path.expanduser("~/robo_express_web/csv")
CSV_FILES = [
    os.path.join(CSV_DIR, "conveyor_to_pallet_position1.csv"),
    os.path.join(CSV_DIR, "conveyor_to_pallet_position2.csv"),
]

# 로봇 플레이어 스크립트 경로
ROBOT_PLAYER_SCRIPT = os.path.expanduser("~/robo_express_web/backend/real_robot_player_spline.py")
ROS2_SETUP_SCRIPT = os.path.expanduser("~/ROS2/install/setup.bash")

conveyor: Optional[ConveyorController] = None
vision: Optional[VisionService] = None

# ==================== Data Models ====================
class OrderItem(BaseModel):
    product_id: str
    quantity: int

class OrderRequest(BaseModel):
    items: List[OrderItem]

class Order:
    def __init__(self, order_id: str, items: List[OrderItem]):
        self.order_id = order_id
        self.items = items
        self.status = "PENDING"
        self.created_at = datetime.now()
        self.completed_items = 0
        self.total_items = sum(item.quantity for item in items)
        self.current_item = None
        self.steps = []
        self._build_steps()
    
    def _build_steps(self):
        """주문 항목별 단계 생성"""
        for item in self.items:
            product_name = PRODUCTS.get(item.product_id, {}).get('name', item.product_id)
            for i in range(item.quantity):
                self.steps.append({
                    "step": f"{product_name} #{i+1} 픽업",
                    "status": "pending",
                    "product_id": item.product_id
                })
    
    def to_dict(self):
        return {
            "order_id": self.order_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "progress": {
                "total_items": self.total_items,
                "completed_items": self.completed_items,
                "current_item": self.current_item,
                "percent": int((self.completed_items / self.total_items) * 100) if self.total_items > 0 else 0
            },
            "steps": self.steps,
            "items": [{"product_id": item.product_id, "quantity": item.quantity} for item in self.items]
        }

# ==================== Product Data ====================
PRODUCTS = {
    "cube_a": {"id": "cube_a", "name": "칸쵸", "description": "8.5x14.5cm 박스", "price": 1500, "box_type": "Box_Kancho"},
    "cube_b": {"id": "cube_b", "name": "초코송이", "description": "12.5x9cm 박스", "price": 1200, "box_type": "Box_Choco"},
    "cube_c": {"id": "cube_c", "name": "고래밥", "description": "9.5x16.5cm 박스", "price": 800, "box_type": "Box_Whalebob"},
}

# ==================== State ====================
orders: Dict[str, Order] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# ==================== WebSocket Manager ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, order_id: str):
        await websocket.accept()
        if order_id not in self.active_connections:
            self.active_connections[order_id] = []
        self.active_connections[order_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, order_id: str):
        if order_id in self.active_connections:
            if websocket in self.active_connections[order_id]:
                self.active_connections[order_id].remove(websocket)
    
    async def send_to_order(self, order_id: str, message: dict):
        if order_id in self.active_connections:
            for connection in self.active_connections[order_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()

# ==================== Startup/Shutdown ====================
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 하드웨어 연결"""
    global conveyor, vision, openai_client
    
    # 컨베이어 연결
    if CONVEYOR_ENABLED:
        conveyor = ConveyorController(CONVEYOR_PORT)
        if conveyor.connect():
            print(f"[System] Conveyor connected on {CONVEYOR_PORT}")
        else:
            print(f"[System] Conveyor connection failed")
            conveyor = None
    else:
        print("[System] Conveyor disabled (simulation mode)")
    
    # 비전 서비스 시작
    if VISION_ENABLED and VISION_AVAILABLE:
        vision = VisionService()
        if vision.start():
            print("[System] Vision service started")
        else:
            print("[System] Vision service failed to start")
            vision = None
    else:
        print("[System] Vision service disabled")
    
    # OpenAI 클라이언트 초기화
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("[System] OpenAI client initialized")
    else:
        print("[System] OpenAI API key not set - AI assistant disabled")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 하드웨어 연결 해제"""
    global conveyor, vision
    
    if conveyor:
        conveyor.disconnect()
        print("[System] Conveyor disconnected")
    
    if vision:
        vision.stop()
        print("[System] Vision service stopped")

# ==================== API Routes ====================
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.get("/index.html")
async def index_page():
    return FileResponse("frontend/index.html")

@app.get("/api/products")
async def get_products():
    """상품 목록 조회"""
    return {"products": list(PRODUCTS.values())}

@app.post("/api/order")
async def create_order(request: OrderRequest):
    """주문 생성"""
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    
    order = Order(order_id, request.items)
    orders[order_id] = order
    
    print(f"[주문 생성] {order_id}: {[(item.product_id, item.quantity) for item in request.items]}")
    
    # 비동기로 주문 처리 시작
    asyncio.create_task(process_order_with_vision(order_id))
    
    return {
        "order_id": order_id,
        "status": order.status,
        "message": "주문이 접수되었습니다.",
        "created_at": order.created_at.isoformat()
    }

@app.get("/api/order/{order_id}/status")
async def get_order_status(order_id: str):
    """주문 상태 조회"""
    if order_id not in orders:
        raise HTTPException(status_code=404, detail="주문을 찾을 수 없습니다.")
    
    return orders[order_id].to_dict()

# ==================== WebSocket ====================
@app.websocket("/ws/order/{order_id}")
async def websocket_endpoint(websocket: WebSocket, order_id: str):
    await manager.connect(websocket, order_id)
    try:
        if order_id in orders:
            await websocket.send_json({
                "type": "status_update",
                "data": orders[order_id].to_dict()
            })
        
        while True:
            data = await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, order_id)

# ==================== Conveyor API ====================
@app.get("/api/conveyor/status")
async def get_conveyor_status():
    """컨베이어 상태 조회"""
    if conveyor and CONVEYOR_ENABLED:
        return conveyor.get_status()
    return {"state": "SIMULATION", "speed": 200, "connected": False}

@app.post("/api/conveyor/start")
async def start_conveyor():
    """컨베이어 시작"""
    if conveyor and CONVEYOR_ENABLED:
        success = conveyor.start()
        return {"success": success, "state": "FORWARD" if success else "ERROR"}
    return {"success": True, "state": "SIMULATION", "message": "Simulation mode"}

@app.post("/api/conveyor/stop")
async def stop_conveyor():
    """컨베이어 정지"""
    if conveyor and CONVEYOR_ENABLED:
        success = conveyor.stop()
        return {"success": success, "state": "STOPPED" if success else "ERROR"}
    return {"success": True, "state": "SIMULATION", "message": "Simulation mode"}

@app.post("/api/conveyor/speed/{speed}")
async def set_conveyor_speed(speed: int):
    """컨베이어 속도 설정"""
    if not 0 <= speed <= 255:
        raise HTTPException(status_code=400, detail="Speed must be 0-255")
    if conveyor and CONVEYOR_ENABLED:
        success = conveyor.set_speed(speed)
        return {"success": success, "speed": speed}
    return {"success": True, "speed": speed, "message": "Simulation mode"}

# ==================== Vision API ====================
@app.get("/api/vision/status")
async def get_vision_status():
    """비전 서비스 상태 조회"""
    if vision:
        return vision.get_all_detections()
    return {
        "boxes": {},
        "pallet": None,
        "running": False,
        "message": "Vision service not available"
    }

@app.get("/api/vision/detect/{product_id}")
async def get_box_detection(product_id: str):
    """특정 박스 감지 상태 조회"""
    if vision:
        return vision.get_detection_status(product_id=product_id)
    return {
        "detected": False,
        "in_range": False,
        "position": None,
        "message": "Vision service not available"
    }

# ==================== Robot Control ====================
async def execute_robot_trajectory(csv_path: str, order_id: str, speed: float = 0.5) -> bool:
    """
    로봇 궤적 실행 (subprocess로 real_robot_player_spline.py 호출)
    
    Args:
        csv_path: CSV 파일 경로
        order_id: 주문 ID (로그용)
        speed: 재생 속도
        
    Returns:
        성공 여부
    """
    if not ROBOT_ENABLED:
        await send_log(order_id, "⚠ 로봇 비활성화 - 시뮬레이션 모드")
        await asyncio.sleep(3)
        return True
    
    if not os.path.exists(csv_path):
        await send_log(order_id, f"❌ CSV 파일 없음: {csv_path}", "error")
        return False
    
    if not os.path.exists(ROBOT_PLAYER_SCRIPT):
        await send_log(order_id, f"❌ 로봇 플레이어 스크립트 없음: {ROBOT_PLAYER_SCRIPT}", "error")
        return False
    
    await send_log(order_id, f"🤖 로봇 궤적 실행: {os.path.basename(csv_path)}")
    
    # subprocess로 실행 (ROS2 환경에서)
    cmd = f"""
        source {ROS2_SETUP_SCRIPT} && \
        python3 {ROBOT_PLAYER_SCRIPT} \
            --csv {csv_path} \
            --speed {speed} \
            --start_delay 1
    """
    
    try:
        # 비동기로 subprocess 실행
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable='/bin/bash'
        )
        
        # 실시간 로그 출력
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode().strip()
            if decoded:
                # 주요 로그만 웹으로 전송
                if any(k in decoded for k in ['✓', '✊', '🤚', '📦', '🏠', '완료', 'Progress']):
                    await send_log(order_id, decoded)
        
        await process.wait()
        
        if process.returncode == 0:
            await send_log(order_id, "✅ 로봇 동작 완료")
            return True
        else:
            stderr = await process.stderr.read()
            await send_log(order_id, f"❌ 로봇 오류: {stderr.decode()}", "error")
            return False
            
    except Exception as e:
        await send_log(order_id, f"❌ 로봇 실행 실패: {str(e)}", "error")
        return False


# ==================== Order Processing with Vision ====================
async def process_order_with_vision(order_id: str):
    """
    비전 시스템을 사용한 주문 처리
    
    1. 컨베이어 시작
    2. 주문한 박스 중 아무거나 Y 범위에 들어올 때까지 대기
    3. 컨베이어 정지
    4. 해당 박스 로봇 픽앤플레이스
    5. 완료 후 남은 상품 반복
    """
    if order_id not in orders:
        return
    
    order = orders[order_id]
    
    # 상태: PROCESSING
    order.status = "PROCESSING"
    await send_status_update(order_id, "PROCESSING", "주문 처리를 시작합니다.")
    await send_log(order_id, "주문 처리 시작")
    await asyncio.sleep(1)
    
    # 배치 위치 인덱스 (0부터 시작, CSV 파일 순환)
    placement_index = 0
    
    # 남은 주문 상품 목록 생성 (product_id와 step 인덱스)
    remaining_items = []
    for step_idx, step in enumerate(order.steps):
        remaining_items.append({
            'step_idx': step_idx,
            'product_id': step['product_id'],
            'step': step
        })
    
    # 남은 상품이 있을 때까지 반복
    while remaining_items:
        # 남은 상품의 box_type 목록
        remaining_box_types = []
        for item in remaining_items:
            product_info = PRODUCTS.get(item['product_id'], {})
            box_type = product_info.get('box_type')
            if box_type:
                remaining_box_types.append(box_type)
        
        await send_log(order_id, f"대기 중인 상품: {', '.join(remaining_box_types)}")
        
        # 1. 컨베이어 작동
        order.status = "CONVEYOR_RUNNING"
        await send_status_update(order_id, "CONVEYOR_RUNNING", f"상품을 찾는 중... ({len(remaining_items)}개 남음)")
        await send_log(order_id, "컨베이어 시작")
        
        if conveyor and CONVEYOR_ENABLED:
            conveyor.start()
        
        # 2. 주문한 박스 중 아무거나 감지 대기
        detected_item = None
        detected_box_type = None
        box_position = None
        
        if vision and vision.running:
            max_wait_time = 120
            start_time = asyncio.get_event_loop().time()
            last_log_time = 0
            
            while asyncio.get_event_loop().time() - start_time < max_wait_time:
                current_time = asyncio.get_event_loop().time()
                
                # 남은 상품 중 범위에 들어온 박스 확인
                for item in remaining_items:
                    status = vision.get_detection_status(product_id=item['product_id'])
                    
                    if status.get('in_range'):
                        detected_item = item
                        detected_box_type = status.get('box_type')
                        box_position = status.get('position')
                        break
                
                if detected_item:
                    product_info = PRODUCTS.get(detected_item['product_id'], {})
                    product_name = product_info.get('name', detected_item['product_id'])
                    await send_log(order_id, f"✓ {product_name} 범위 진입! Y={box_position[1]:.3f}m")
                    break
                
                # 감지 상태 로그 (5초마다)
                if current_time - last_log_time > 5.0:
                    detected_any = False
                    for item in remaining_items:
                        status = vision.get_detection_status(product_id=item['product_id'])
                        if status.get('detected'):
                            detected_any = True
                            y_pos = status.get('position', [0, 0, 0])[1]
                            product_info = PRODUCTS.get(item['product_id'], {})
                            await send_log(order_id, f"{product_info.get('box_type')} 감지됨 (Y={y_pos:.3f}m)")
                    
                    if not detected_any:
                        await send_log(order_id, "상품 탐색 중... (컨베이어 작동 중)")
                    last_log_time = current_time
                
                await asyncio.sleep(0.1)
            
            if not detected_item:
                await send_log(order_id, "⚠ 감지 타임아웃 (120초)", "warning")
                if conveyor and CONVEYOR_ENABLED:
                    conveyor.stop()
                continue  # 다시 시도
        else:
            # 비전 서비스 없음 - 시뮬레이션 모드
            await send_log(order_id, "⚠ 비전 서비스 없음 - 시뮬레이션 모드")
            await asyncio.sleep(2)
            detected_item = remaining_items[0]
            box_position = [0.1, 0.0, 0.05]
        
        # 3. 컨베이어 정지
        product_info = PRODUCTS.get(detected_item['product_id'], {})
        product_name = product_info.get('name', detected_item['product_id'])
        
        order.status = "CONVEYOR_STOPPED"
        order.current_item = product_name
        await send_status_update(order_id, "CONVEYOR_STOPPED", f"{product_name} 감지됨!")
        await send_log(order_id, f"{product_name} 감지 - 컨베이어 정지")
        
        if conveyor and CONVEYOR_ENABLED:
            conveyor.stop()
        
        await asyncio.sleep(0.5)
        
        # step 상태 업데이트
        detected_item['step']['status'] = 'active'
        await send_step_update(order_id, order.steps)
        
        # 4. 로봇 픽앤플레이스 실행
        order.status = "ROBOT_MOVING"
        
        csv_index = placement_index % len(CSV_FILES)
        csv_path = CSV_FILES[csv_index]
        
        await send_status_update(order_id, "ROBOT_MOVING", f"로봇이 {product_name}를 위치 {csv_index + 1}에 배치합니다.")
        await send_log(order_id, f"🤖 로봇 시작 - CSV: {os.path.basename(csv_path)} (위치 {csv_index + 1})")
        
        robot_success = await execute_robot_trajectory(csv_path, order_id, speed=1.0)
        
        if robot_success:
            await send_log(order_id, f"✅ {product_name} 배치 완료 (위치 {csv_index + 1})")
        else:
            await send_log(order_id, f"⚠ 로봇 동작 실패 - 다음 상품으로 진행", "warning")
        
        placement_index += 1
        
        # 5. 완료 처리
        detected_item['step']['status'] = 'completed'
        order.completed_items += 1
        await send_step_update(order_id, order.steps)
        await send_progress_update(order_id, order.completed_items, order.total_items)
        await send_log(order_id, f"📦 {product_name} 완료 ({order.completed_items}/{order.total_items})")
        
        # 처리된 항목 제거
        remaining_items.remove(detected_item)
        
        await asyncio.sleep(0.5)
    
    # 주문 완료
    order.status = "COMPLETED"
    await send_status_update(order_id, "COMPLETED", "모든 상품이 준비되었습니다!")
    await manager.send_to_order(order_id, {
        "type": "completed",
        "data": {
            "message": "주문이 완료되었습니다!",
            "total_time": "완료"
        }
    })

# ==================== Helper Functions ====================
async def send_status_update(order_id: str, status: str, message: str):
    await manager.send_to_order(order_id, {
        "type": "status_update",
        "data": {"status": status, "message": message}
    })

async def send_progress_update(order_id: str, completed: int, total: int):
    percent = int((completed / total) * 100) if total > 0 else 0
    await manager.send_to_order(order_id, {
        "type": "progress_update",
        "data": {"completed_items": completed, "total_items": total, "percent": percent}
    })

async def send_step_update(order_id: str, steps: list):
    await manager.send_to_order(order_id, {
        "type": "step_update",
        "data": {"steps": steps}
    })

async def send_log(order_id: str, message: str, level: str = "info"):
    await manager.send_to_order(order_id, {
        "type": "log",
        "data": {"message": message, "level": level}
    })

# ==================== AI Assistant API ====================
SYSTEM_PROMPT = """당신은 "로보"라는 이름의 친절한 AI 주문 비서입니다.
로봇 물류 시스템 "Robo Express"에서 고객의 주문을 도와줍니다.

## 판매 상품
1. 칸쵸 - 1,500원 (달콤한 초콜릿이 들어간 비스킷, 아이들에게 인기)
2. 초코송이 - 1,200원 (귀여운 버섯 모양, 바삭한 과자에 초콜릿 코팅)
3. 고래밥 - 800원 (짭짤한 볶음 양념맛, 여러 모양을 찾는 재미, 최고의 가성비)

## 추천 가이드
- 달콤한 거 원하면: 칸쵸, 초코송이
- 짭짤한 거 원하면: 고래밥
- 가성비/저렴한 거 원하면: 고래밥 (800원)
- 아이들 선물: 칸쵸, 고래밥

## 역할
- 고객에게 상품을 추천하고 주문을 받습니다
- 친근하고 밝은 톤으로 대화합니다
- 주문 내용을 명확히 확인합니다

## 주문 확정 규칙
고객이 주문을 확정하면 (예: "네", "주문할게요", "그걸로 할게요" 등)
반드시 다음 JSON 형식을 응답 끝에 포함하세요:

[ORDER_CONFIRMED]
{"items": [{"product_id": "cube_a", "quantity": 2}]}
[/ORDER_CONFIRMED]

상품 ID 매핑:
- 칸쵸 → cube_a
- 초코송이 → cube_b  
- 고래밥 → cube_c

## 응답 스타일
- 짧고 명확하게 답변 (2-3문장)
- 이모지 사용 가능
- 자연스러운 한국어 사용
"""

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict] = []

class TTSRequest(BaseModel):
    text: str

@app.post("/api/assistant/chat")
async def assistant_chat(request: ChatRequest):
    """GPT 대화 API"""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured")
    
    # 메시지 구성
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(request.conversation_history[-10:])  # 최근 10개만
    messages.append({"role": "user", "content": request.message})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        
        assistant_response = response.choices[0].message.content
        
        # 주문 확정 여부 확인
        order_confirmed = False
        order_items = None
        
        if "[ORDER_CONFIRMED]" in assistant_response:
            order_confirmed = True
            # JSON 추출
            try:
                start = assistant_response.index("[ORDER_CONFIRMED]") + len("[ORDER_CONFIRMED]")
                end = assistant_response.index("[/ORDER_CONFIRMED]")
                json_str = assistant_response[start:end].strip()
                order_data = json.loads(json_str)
                order_items = order_data.get("items", [])
                # 응답에서 JSON 부분 제거
                assistant_response = assistant_response[:assistant_response.index("[ORDER_CONFIRMED]")].strip()
            except:
                order_confirmed = False
        
        return {
            "response": assistant_response,
            "order_confirmed": order_confirmed,
            "order_items": order_items
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/assistant/tts")
async def assistant_tts(request: TTSRequest):
    """OpenAI TTS API"""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI not configured")
    
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=request.text,
        )
        
        # 오디오 바이트 반환
        audio_bytes = response.content
        return Response(content=audio_bytes, media_type="audio/mpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Static Files ====================
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/images", StaticFiles(directory="frontend/images"), name="images")
app.mount("/audio", StaticFiles(directory="frontend/audio"), name="audio")

@app.get("/products.html")
async def products_page():
    return FileResponse("frontend/products.html")

@app.get("/order.html")
async def order_page():
    return FileResponse("frontend/order.html")

@app.get("/admin.html")
async def admin_page():
    return FileResponse("frontend/admin.html")

@app.get("/assistant.html")
async def assistant_page():
    return FileResponse("frontend/assistant.html")


# ==================== Run ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
