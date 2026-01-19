// ==================== Configuration ====================
const CONFIG = {
  API_URL: 'http://localhost:8000',  // FastAPI 서버 주소
};

// ==================== Audio ====================
const thanksAudio = new Audio('audio/thanks.mp3');
thanksAudio.preload = 'auto';

// ==================== Product Data ====================
const PRODUCTS = {
  cube_a: { id: 'cube_a', name: '칸쵸', desc: '265kcal', price: 1500 },
  cube_b: { id: 'cube_b', name: '초코송이', desc: '266kcal', price: 1200 },
  cube_c: { id: 'cube_c', name: '고래밥', desc: '206kcal', price: 800 },
};

// ==================== State ====================
const cart = {
  cube_a: 0,
  cube_b: 0,
  cube_c: 0,
};

// ==================== DOM Elements ====================
const productGrid = document.getElementById('productGrid');
const cartBadge = document.getElementById('cartBadge');
const totalCount = document.getElementById('totalCount');
const totalPrice = document.getElementById('totalPrice');
const cartItems = document.getElementById('cartItems');
const btnOrder = document.getElementById('btnOrder');
const modalOverlay = document.getElementById('modalOverlay');
const orderSummary = document.getElementById('orderSummary');
const btnCancel = document.getElementById('btnCancel');
const btnConfirm = document.getElementById('btnConfirm');
const modalClose = document.getElementById('modalClose');
const loadingOverlay = document.getElementById('loadingOverlay');

// ==================== Initialize ====================
function init() {
  // Quantity button event listeners
  productGrid.addEventListener('click', (e) => {
    const btn = e.target.closest('.qty-btn');
    if (!btn) return;
    
    const card = btn.closest('.product-card');
    const productId = card.dataset.productId;
    const action = btn.dataset.action;
    
    if (action === 'plus') {
      cart[productId]++;
    } else if (action === 'minus' && cart[productId] > 0) {
      cart[productId]--;
    }
    
    updateUI();
  });

  // Order button
  btnOrder.addEventListener('click', showOrderModal);
  
  // Modal controls
  btnCancel.addEventListener('click', hideOrderModal);
  modalClose.addEventListener('click', hideOrderModal);
  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) hideOrderModal();
  });
  
  // Confirm order
  btnConfirm.addEventListener('click', submitOrder);
  
  updateUI();
}

// ==================== UI Updates ====================
function updateUI() {
  let total = 0;
  let count = 0;
  
  // Update quantity displays and card states
  Object.keys(cart).forEach(productId => {
    const qty = cart[productId];
    const card = document.querySelector(`[data-product-id="${productId}"]`);
    const qtyDisplay = card.querySelector('.qty-value');
    
    qtyDisplay.textContent = qty;
    qtyDisplay.dataset.qty = qty;
    
    // Toggle selected state
    if (qty > 0) {
      card.classList.add('selected');
    } else {
      card.classList.remove('selected');
    }
    
    count += qty;
    total += qty * PRODUCTS[productId].price;
  });
  
  // Update cart badge
  cartBadge.textContent = count;
  
  // Update footer summary
  totalCount.textContent = count;
  totalPrice.textContent = total.toLocaleString();
  
  // Update cart items tags
  cartItems.innerHTML = '';
  Object.keys(cart).forEach(productId => {
    if (cart[productId] > 0) {
      const tag = document.createElement('span');
      tag.className = 'cart-item-tag';
      tag.textContent = `${PRODUCTS[productId].name} ×${cart[productId]}`;
      cartItems.appendChild(tag);
    }
  });
  
  // Enable/disable order button
  btnOrder.disabled = count === 0;
}

// ==================== Modal ====================
function showOrderModal() {
  // Build order summary
  let summaryHTML = '';
  let total = 0;
  
  Object.keys(cart).forEach(productId => {
    const qty = cart[productId];
    if (qty > 0) {
      const product = PRODUCTS[productId];
      const subtotal = qty * product.price;
      total += subtotal;
      
      summaryHTML += `
        <div class="order-summary-item">
          <span>${product.name} × ${qty}</span>
          <span>₩${subtotal.toLocaleString()}</span>
        </div>
      `;
    }
  });
  
  summaryHTML += `
    <div class="order-summary-total">
      <span>총 금액</span>
      <span>₩${total.toLocaleString()}</span>
    </div>
  `;
  
  orderSummary.innerHTML = summaryHTML;
  modalOverlay.classList.add('show');
}

function hideOrderModal() {
  modalOverlay.classList.remove('show');
}

// ==================== Submit Order ====================
async function submitOrder() {
  hideOrderModal();
  
  // thanks 음성 재생
  thanksAudio.play();
  
  // 음성 재생 완료 후 로딩 표시 (2초)
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  loadingOverlay.classList.add('show');
  
  // Build order items
  const items = [];
  Object.keys(cart).forEach(productId => {
    if (cart[productId] > 0) {
      items.push({
        product_id: productId,
        quantity: cart[productId]
      });
    }
  });
  
  try {
    const response = await fetch(`${CONFIG.API_URL}/api/order`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ items }),
    });
    
    if (!response.ok) {
      throw new Error('주문 생성 실패');
    }
    
    const data = await response.json();
    
    // Store order ID and redirect
    localStorage.setItem('currentOrderId', data.order_id);
    window.location.href = `order.html?id=${data.order_id}`;
    
  } catch (error) {
    console.error('Order error:', error);
    loadingOverlay.classList.remove('show');
    alert('주문 처리 중 오류가 발생했습니다.\n서버 연결을 확인해주세요.');
  }
}

// ==================== Start ====================
document.addEventListener('DOMContentLoaded', init);
