// ==================== Configuration ====================
const CONFIG = {
  API_URL: 'http://localhost:8000',
  WS_URL: 'ws://localhost:8000',
  WAKE_WORD: '로보야',
};

// ==================== State ====================
let isListening = false;
let isProcessing = false;
let isSpeaking = false;
let recognition = null;
let currentOrderId = null;
let ws = null;
let conversationHistory = [];

// ==================== DOM Elements ====================
const robotFace = document.getElementById('robotFace');
const pulseRing = document.getElementById('pulseRing');
const statusText = document.getElementById('statusText');
const chatMessages = document.getElementById('chatMessages');
const chatContainer = document.getElementById('chatContainer');
const micButton = document.getElementById('micButton');
const textInput = document.getElementById('textInput');
const sendButton = document.getElementById('sendButton');

// ==================== Initialize ====================
function init() {
  // Web Speech API 지원 확인
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    setupSpeechRecognition();
  } else {
    statusText.textContent = '음성 인식이 지원되지 않는 브라우저입니다';
  }

  // 이벤트 리스너
  micButton.addEventListener('click', toggleListening);
  sendButton.addEventListener('click', sendTextMessage);
  textInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendTextMessage();
  });

  // 항상 듣기 모드 시작 (호출어 감지용)
  startWakeWordDetection();
}

// ==================== Speech Recognition ====================
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = 'ko-KR';
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onresult = (event) => {
    const last = event.results.length - 1;
    const transcript = event.results[last][0].transcript.trim();
    
    if (event.results[last].isFinal) {
      handleSpeechResult(transcript);
    }
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    if (event.error !== 'no-speech') {
      setTimeout(() => {
        if (!isSpeaking) startWakeWordDetection();
      }, 1000);
    }
  };

  recognition.onend = () => {
    if (!isSpeaking && !isProcessing) {
      startWakeWordDetection();
    }
  };
}

function startWakeWordDetection() {
  if (recognition && !isListening && !isSpeaking) {
    try {
      recognition.start();
      isListening = true;
    } catch (e) {
      console.log('Recognition already started');
    }
  }
}

function stopListening() {
  if (recognition && isListening) {
    recognition.stop();
    isListening = false;
  }
}

function toggleListening() {
  if (isListening) {
    stopListening();
    micButton.classList.remove('recording');
    setFaceState('default');
  } else {
    startWakeWordDetection();
    micButton.classList.add('recording');
    setFaceState('listening');
    statusText.textContent = '듣고 있어요...';
  }
}

function handleSpeechResult(transcript) {
  console.log('Heard:', transcript);
  
  // 호출어 감지
  if (transcript.includes(CONFIG.WAKE_WORD) || transcript.includes('로보아') || transcript.includes('로봐')) {
    activateAssistant();
    return;
  }
  
  // 대화 모드에서 입력 처리
  if (micButton.classList.contains('recording')) {
    addMessage(transcript, 'user');
    processUserInput(transcript);
  }
}

function activateAssistant() {
  micButton.classList.add('recording');
  setFaceState('happy');
  statusText.textContent = '네, 말씀하세요!';
  
  // 인사 응답
  speakAndRespond('안녕하세요! 무엇을 도와드릴까요? 과자를 주문하시려면 말씀해주세요.');
}

// ==================== Process User Input ====================
async function processUserInput(input) {
  if (isProcessing) return;
  isProcessing = true;
  
  setFaceState('searching');
  statusText.textContent = '생각하고 있어요...';
  
  try {
    const response = await fetch(`${CONFIG.API_URL}/api/assistant/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message: input,
        conversation_history: conversationHistory
      }),
    });
    
    const data = await response.json();
    
    // 대화 히스토리 업데이트
    conversationHistory.push({ role: 'user', content: input });
    conversationHistory.push({ role: 'assistant', content: data.response });
    
    // 응답 표시 및 음성 출력
    addMessage(data.response, 'assistant');
    await speakAndRespond(data.response);
    
    // 주문 확정 시 처리
    if (data.order_confirmed && data.order_items) {
      await processOrder(data.order_items);
    }
    
  } catch (error) {
    console.error('Error:', error);
    addMessage('죄송해요, 문제가 발생했어요.', 'assistant');
    setFaceState('default');
  }
  
  isProcessing = false;
}

// ==================== Order Processing ====================
async function processOrder(items) {
  // AI 비서 모드 플래그 설정
  localStorage.setItem('assistantMode', 'true');
  
  setFaceState('working');
  statusText.textContent = '주문을 처리하고 있어요...';
  
  try {
    const response = await fetch(`${CONFIG.API_URL}/api/order`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items }),
    });
    
    const data = await response.json();
    currentOrderId = data.order_id;
    
    // WebSocket 연결
    connectOrderWebSocket(currentOrderId);
    
    await speakAndRespond('주문이 접수되었어요! 이제 상품을 준비할게요.');
    
  } catch (error) {
    console.error('Order error:', error);
    await speakAndRespond('주문 처리 중 문제가 발생했어요.');
    setFaceState('default');
  }
}

function connectOrderWebSocket(orderId) {
  ws = new WebSocket(`${CONFIG.WS_URL}/ws/order/${orderId}`);
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleOrderUpdate(data);
  };
  
  ws.onclose = () => {
    console.log('Order WebSocket closed');
  };
}

function handleOrderUpdate(data) {
  switch (data.type) {
    case 'status_update':
      handleStatusChange(data.data.status, data.data.message);
      break;
    case 'completed':
      handleOrderComplete();
      break;
  }
}

function handleStatusChange(status, message) {
  switch (status) {
    case 'CONVEYOR_RUNNING':
      setFaceState('searching');
      statusText.textContent = '상품을 찾고 있어요...';
      break;
    case 'CONVEYOR_STOPPED':
      setFaceState('happy');
      statusText.textContent = '상품을 찾았어요!';
      break;
    case 'ROBOT_MOVING':
      setFaceState('working');
      statusText.textContent = '상품을 옮기는 중이에요...';
      break;
  }
}

function handleOrderComplete() {
  // AI 비서 모드 플래그 해제
  localStorage.removeItem('assistantMode');
  
  setFaceState('happy');
  statusText.textContent = '주문이 완료되었어요!';
  speakAndRespond('주문이 완료되었습니다! 이용해주셔서 감사합니다. 또 필요한 게 있으시면 불러주세요.');
  
  // 대화 히스토리 초기화
  conversationHistory = [];
  currentOrderId = null;
  
  if (ws) {
    ws.close();
    ws = null;
  }
}

// ==================== Text-to-Speech ====================
async function speakAndRespond(text) {
  isSpeaking = true;
  stopListening();
  setFaceState('speaking');
  pulseRing.classList.add('active');
  
  try {
    const response = await fetch(`${CONFIG.API_URL}/api/assistant/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    
    if (response.ok) {
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.onended = () => {
        isSpeaking = false;
        pulseRing.classList.remove('active');
        setFaceState('default');
        statusText.textContent = '말씀해주세요...';
        micButton.classList.add('recording');
        startWakeWordDetection();
      };
      
      await audio.play();
    }
  } catch (error) {
    console.error('TTS error:', error);
    isSpeaking = false;
    pulseRing.classList.remove('active');
    setFaceState('default');
    startWakeWordDetection();
  }
}

// ==================== UI Functions ====================
function setFaceState(state) {
  robotFace.className = 'robot-face';
  if (state !== 'default') {
    robotFace.classList.add(state);
  }
}

function addMessage(text, type) {
  const messageEl = document.createElement('div');
  messageEl.className = `chat-message ${type}`;
  messageEl.textContent = text;
  chatMessages.appendChild(messageEl);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function sendTextMessage() {
  const text = textInput.value.trim();
  if (!text) return;
  
  addMessage(text, 'user');
  textInput.value = '';
  processUserInput(text);
}

// ==================== Start ====================
document.addEventListener('DOMContentLoaded', init);
