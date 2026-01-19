// ==================== Configuration ====================
const CONFIG = {
  API_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000',
};

// ==================== Audio ====================
const AUDIO = {
  convey: new Audio('audio/convey.mp3'),
  move: new Audio('audio/move.mp3'),
  finish: new Audio('audio/finish.mp3'),
};

// Preload all audio
Object.values(AUDIO).forEach(audio => {
  audio.preload = 'auto';
});

// Track last played to prevent duplicates
let lastPlayedAudio = null;

// AI 비서 모드 확인
const isAssistantMode = localStorage.getItem('assistantMode') === 'true';

// ==================== State ====================
let orderId = null;
let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT = 5;

// ==================== DOM Elements ====================
const orderNumber = document.getElementById('orderNumber');
const orderTime = document.getElementById('orderTime');
const statusIcon = document.getElementById('statusIcon');
const statusTitle = document.getElementById('statusTitle');
const statusMessage = document.getElementById('statusMessage');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const stepsContainer = document.getElementById('stepsContainer');
const logContainer = document.getElementById('logContainer');
const completeFooter = document.getElementById('completeFooter');

// ==================== Status Mapping ====================
const STATUS_INFO = {
  PENDING: {
    icon: '📋',
    title: '주문 접수됨',
    message: '주문이 접수되었습니다. 잠시만 기다려주세요.',
  },
  PROCESSING: {
    icon: '💻',
    title: '처리 중',
    message: '주문을 처리하고 있습니다.',
  },
  CONVEYOR_RUNNING: {
    icon: '⚙️',
    title: '컨베이어 작동 중',
    message: '상품을 찾고 있습니다...',
  },
  CONVEYOR_STOPPED: {
    icon: '🔍',
    title: '상품 감지됨',
    message: '상품을 발견했습니다!',
  },
  ROBOT_PICKING: {
    icon: '🦾',
    title: '로봇 픽업 중',
    message: '로봇이 상품을 픽업하고 있습니다.',
  },
  ROBOT_PLACING: {
    icon: '📦',
    title: '상품 배치 중',
    message: '로봇이 상품을 팔레트에 배치하고 있습니다.',
  },
  COMPLETED: {
    icon: '✅',
    title: '주문 완료',
    message: '모든 상품이 준비되었습니다!',
  },
  ERROR: {
    icon: '❌',
    title: '오류 발생',
    message: '처리 중 문제가 발생했습니다.',
  },
};

// ==================== Initialize ====================
async function init() {
  // Get order ID from URL
  const urlParams = new URLSearchParams(window.location.search);
  orderId = urlParams.get('id') || localStorage.getItem('currentOrderId');
  
  if (!orderId) {
    alert('주문 정보를 찾을 수 없습니다.');
    window.location.href = 'index.html';
    return;
  }
  
  orderNumber.textContent = orderId;
  
  // Fetch initial order status
  await fetchOrderStatus();
  
  // Connect WebSocket
  connectWebSocket();
}

// ==================== Fetch Order Status ====================
async function fetchOrderStatus() {
  try {
    const response = await fetch(`${CONFIG.API_URL}/api/order/${orderId}/status`);
    
    if (!response.ok) {
      throw new Error('주문 조회 실패');
    }
    
    const data = await response.json();
    updateOrderDisplay(data);
    
  } catch (error) {
    console.error('Fetch error:', error);
    addLog('서버 연결 실패', 'error');
  }
}

// ==================== WebSocket ====================
function connectWebSocket() {
  ws = new WebSocket(`${CONFIG.WS_URL}/ws/order/${orderId}`);
  
  ws.onopen = () => {
    console.log('WebSocket connected');
    addLog('실시간 연결됨');
    reconnectAttempts = 0;
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleWebSocketMessage(data);
  };
  
  ws.onclose = () => {
    console.log('WebSocket closed');
    
    if (reconnectAttempts < MAX_RECONNECT) {
      reconnectAttempts++;
      addLog(`연결 재시도 중... (${reconnectAttempts}/${MAX_RECONNECT})`, 'warning');
      setTimeout(connectWebSocket, 3000);
    } else {
      addLog('연결 실패. 페이지를 새로고침해주세요.', 'error');
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
}

function handleWebSocketMessage(data) {
  switch (data.type) {
    case 'status_update':
      updateStatus(data.data);
      break;
    case 'progress_update':
      updateProgress(data.data);
      break;
    case 'step_update':
      updateSteps(data.data);
      break;
    case 'log':
      addLog(data.data.message, data.data.level);
      break;
    case 'completed':
      handleComplete(data.data);
      break;
  }
}

// ==================== Update Display ====================
function updateOrderDisplay(data) {
  // Update order time
  if (data.created_at) {
    const date = new Date(data.created_at);
    orderTime.textContent = date.toLocaleString('ko-KR');
  }
  
  // Update status
  updateStatus({ status: data.status, message: data.message });
  
  // Update progress
  if (data.progress) {
    updateProgress(data.progress);
  }
  
  // Update steps
  if (data.steps) {
    updateSteps({ steps: data.steps });
  }
}

function updateStatus(data) {
  const info = STATUS_INFO[data.status] || STATUS_INFO.PROCESSING;
  
  document.querySelector('.status-icon-inner').textContent = info.icon;
  statusTitle.textContent = info.title;
  statusMessage.textContent = data.message || info.message;
  
  // 상태별 음성 재생 (AI 비서 모드가 아닐 때만)
  if (!isAssistantMode) {
    if (data.status === 'CONVEYOR_RUNNING' && lastPlayedAudio !== 'convey_' + Date.now()) {
      AUDIO.convey.play();
      lastPlayedAudio = 'convey_' + Date.now();
    } else if (data.status === 'CONVEYOR_STOPPED' && lastPlayedAudio !== 'move') {
      AUDIO.move.play();
      lastPlayedAudio = 'move';
    }
  }
  
  // Update status card style based on status
  const statusCard = document.querySelector('.status-card');
  statusCard.className = 'status-card';
  
  if (data.status === 'COMPLETED') {
    statusCard.style.borderColor = 'var(--success)';
    completeFooter.style.display = 'block';
  } else if (data.status === 'ERROR') {
    statusCard.style.borderColor = 'var(--error)';
  }
  
  addLog(`상태 변경: ${info.title}`);
}

function updateProgress(data) {
  const percent = data.percent || 
    (data.total_items > 0 ? Math.round((data.completed_items / data.total_items) * 100) : 0);
  
  progressFill.style.width = `${percent}%`;
  progressText.textContent = `${percent}%`;
  
  if (data.current_item) {
    statusMessage.textContent = `현재 처리 중: ${data.current_item}`;
  }
}

function updateSteps(data) {
  stepsContainer.innerHTML = '';
  
  data.steps.forEach((step, index) => {
    const stepEl = document.createElement('div');
    stepEl.className = `step-item ${step.status}`;
    
    let icon = '⏳';
    if (step.status === 'completed') icon = '✓';
    else if (step.status === 'active') icon = '▶';
    
    stepEl.innerHTML = `
      <div class="step-icon">${icon}</div>
      <span class="step-text">${step.step}</span>
    `;
    
    stepsContainer.appendChild(stepEl);
  });
}

function handleComplete(data) {
  // finish 음성 재생 (AI 비서 모드가 아닐 때만)
  if (!isAssistantMode) {
    AUDIO.finish.play();
    lastPlayedAudio = 'finish';
  }
  
  updateStatus({ status: 'COMPLETED', message: data.message });
  progressFill.style.width = '100%';
  progressText.textContent = '100%';
  
  if (data.total_time) {
    addLog(`총 소요 시간: ${data.total_time}`);
  }
  
  addLog('주문이 완료되었습니다! 🎉');
}

// ==================== Logging ====================
function addLog(message, level = 'info') {
  const now = new Date();
  const time = now.toLocaleTimeString('ko-KR', { hour12: false });
  
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="log-message ${level}">${message}</span>
  `;
  
  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;
}

// ==================== Start ====================
document.addEventListener('DOMContentLoaded', init);
