/**
 * HealthPipe AI - Shared app utilities and API helpers
 */

// API base URL (adapt for your deployment)
const API_BASE = 'http://localhost:8501/api';

/**
 * API call helper with error handling
 */
async function api(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const {
    method = 'GET',
    body = null,
    headers = {}
  } = options;

  try {
    const fetchOptions = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers
      }
    };

    if (body) {
      fetchOptions.body = JSON.stringify(body);
    }

    const response = await fetch(url, fetchOptions);

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API call failed: ${endpoint}`, error);
    throw error;
  }
}

/**
 * Tab switching utility
 */
function setupTabs(containerSelector) {
  const container = document.querySelector(containerSelector);
  if (!container) return;

  const buttons = container.querySelectorAll('.tab-button');
  const contents = container.querySelectorAll('.tab-content');

  buttons.forEach((button, index) => {
    button.addEventListener('click', () => {
      // Remove active class from all
      buttons.forEach(b => b.classList.remove('active'));
      contents.forEach(c => c.classList.remove('active'));

      // Add active class to clicked button and corresponding content
      button.classList.add('active');
      if (contents[index]) {
        contents[index].classList.add('active');
      }
    });
  });

  // Set first tab as active by default
  if (buttons.length > 0) {
    buttons[0].classList.add('active');
    if (contents[0]) {
      contents[0].classList.add('active');
    }
  }
}

/**
 * Sidebar toggle for mobile
 */
function setupSidebarToggle() {
  const sidebar = document.querySelector('.sidebar');
  const toggleBtn = document.querySelector('[data-sidebar-toggle]');

  if (!toggleBtn || !sidebar) return;

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // Close sidebar when clicking outside
  document.addEventListener('click', (e) => {
    if (!sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
      sidebar.classList.remove('open');
    }
  });
}

/**
 * Format numbers with thousand separators
 */
function formatNumber(num) {
  if (num === null || num === undefined) return '—';
  return num.toLocaleString();
}

/**
 * Format percentage with 1 decimal
 */
function formatPercent(num) {
  if (num === null || num === undefined) return '—';
  return `${(num * 100).toFixed(1)}%`;
}

/**
 * Show toast message
 */
function showToast(message, type = 'info', duration = 3000) {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  const style = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : '#0284C7'};
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    z-index: 9999;
    animation: slideInUp 0.3s ease;
  `;
  toast.style.cssText = style;

  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'slideOutDown 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

/**
 * Styled confirm dialog — replaces the native window.confirm() browser popup.
 * Returns a Promise<boolean> that resolves true on confirm, false on cancel/escape.
 *
 * Usage: if (await hpConfirm('Delete "foo" permanently?', { title: 'Delete dataset', danger: true })) { ... }
 */
function hpConfirm(message, opts = {}) {
  const { title = 'Are you sure?', confirmLabel = 'Delete', cancelLabel = 'Cancel', danger = true } = opts;

  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.className = 'hp-confirm-overlay';

    const card = document.createElement('div');
    card.className = 'hp-confirm-card';
    card.setAttribute('role', 'alertdialog');
    card.setAttribute('aria-modal', 'true');

    const icon = document.createElement('div');
    icon.className = 'hp-confirm-icon';
    icon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>';

    const titleEl = document.createElement('div');
    titleEl.className = 'hp-confirm-title';
    titleEl.textContent = title;

    const bodyEl = document.createElement('div');
    bodyEl.className = 'hp-confirm-body';
    bodyEl.textContent = message;

    const actions = document.createElement('div');
    actions.className = 'hp-confirm-actions';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn btn-ghost';
    cancelBtn.type = 'button';
    cancelBtn.textContent = cancelLabel;

    const okBtn = document.createElement('button');
    okBtn.className = danger ? 'btn btn-danger' : 'btn btn-primary';
    okBtn.type = 'button';
    okBtn.textContent = confirmLabel;

    actions.append(cancelBtn, okBtn);
    card.append(icon, titleEl, bodyEl, actions);
    overlay.appendChild(card);
    document.body.appendChild(overlay);

    const cleanup = (result) => {
      document.removeEventListener('keydown', onKeydown);
      overlay.remove();
      resolve(result);
    };
    const onKeydown = (e) => { if (e.key === 'Escape') cleanup(false); };

    cancelBtn.addEventListener('click', () => cleanup(false));
    okBtn.addEventListener('click', () => cleanup(true));
    overlay.addEventListener('click', (e) => { if (e.target === overlay) cleanup(false); });
    document.addEventListener('keydown', onKeydown);

    okBtn.focus();
  });
}

/**
 * Loading state management
 */
function setLoading(element, isLoading) {
  if (isLoading) {
    element.disabled = true;
    element.setAttribute('data-loading', 'true');
    element.textContent = '⏳ Loading...';
  } else {
    element.disabled = false;
    element.removeAttribute('data-loading');
    element.textContent = element.getAttribute('data-original-text') || 'Submit';
  }
}

/**
 * Initialize common functionality on page load
 */
document.addEventListener('DOMContentLoaded', () => {
  // Setup sidebar toggle
  setupSidebarToggle();

  // Setup all tabs on the page
  document.querySelectorAll('[data-tabs]').forEach(container => {
    setupTabs(`[data-tabs="${container.getAttribute('data-tabs')}"]`);
  });

  // Add animations CSS if not already present
  if (!document.querySelector('style[data-app-animations]')) {
    const style = document.createElement('style');
    style.setAttribute('data-app-animations', 'true');
    style.textContent = `
      @keyframes slideInUp {
        from {
          opacity: 0;
          transform: translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @keyframes slideOutDown {
        from {
          opacity: 1;
          transform: translateY(0);
        }
        to {
          opacity: 0;
          transform: translateY(20px);
        }
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }

      .spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid rgba(0, 0, 0, 0.1);
        border-top-color: currentColor;
        border-radius: 50%;
        animation: spin 0.6s linear infinite;
      }

      @media (prefers-reduced-motion: reduce) {
        * {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
        }
      }
    `;
    document.head.appendChild(style);
  }
});

/* ===== Sidebar AI models panel (shared across pages) ===== */
const MODE_HELP = {
  hybrid: 'Cloud handles schema-only tasks; anything with your data runs locally.',
  local_only: 'Everything runs on this device. Nothing goes to the cloud.',
};
function setModeActive(mode) {
  document.querySelectorAll('#model-panel .mp-mode').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode));
  const help = document.getElementById('mp-help');
  if (help) help.textContent = MODE_HELP[mode] || '';
}

async function initModelPanel() {
  const panel = document.getElementById('model-panel');
  if (!panel) return;
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const list = document.getElementById('mp-list');
    list.innerHTML = (data.providers || []).map(p => {
      const model = String(p.model || p.label).replace(/[<>&"]/g, '');
      const friendly = p.kind === 'cloud' ? 'Gemini' : 'Gemma';
      const kind = p.available ? p.kind : 'offline';
      const kindLabel = p.available ? (p.kind === 'cloud' ? 'Cloud' : 'Local') : 'Offline';
      return `<div class="mp-model${p.available ? '' : ' unavailable'}" title="${model}">
        <span class="mp-dot${p.available ? ' on' : ''}"></span>
        <span class="mp-name">${friendly}</span>
        <span class="mp-kind ${kind}">${kindLabel}</span>
      </div>`;
    }).join('');
    setModeActive(data.mode || 'hybrid');
    panel.style.display = 'block';
  } catch (e) { /* leave hidden if the API isn't reachable */ }

  panel.querySelectorAll('.mp-mode').forEach(btn => {
    btn.addEventListener('click', async () => {
      const mode = btn.dataset.mode;
      const prev = document.querySelector('#model-panel .mp-mode.active');
      setModeActive(mode);  // optimistic
      try {
        const res = await fetch('/api/models/mode', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode }),
        });
        const d = await res.json();
        if (d.ok) {
          // Let the current page react live (e.g. refresh the chat provider badge).
          document.dispatchEvent(new CustomEvent('hp:modechange', { detail: { mode } }));
          if (typeof showToast === 'function') {
            showToast(mode === 'local_only'
              ? 'On-device only — cloud is now disabled' : 'Cloud + on-device enabled', 'success');
          }
        } else if (prev) {
          setModeActive(prev.dataset.mode);  // revert on failure
        }
      } catch (e) {
        if (prev) setModeActive(prev.dataset.mode);
        if (typeof showToast === 'function') showToast('Could not change mode', 'error');
      }
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initModelPanel);
} else {
  initModelPanel();
}

/* ===== Topbar avatar menu + logout (shared across pages) ===== */
function goLogout(e) {
  if (e) e.preventDefault();
  // Single-operator demo: no server session to clear — return to the login gate.
  window.location.href = '/';
}

async function initUserMenu() {
  // Wire the sidebar "Log out" link on every page.
  document.querySelectorAll('.logout').forEach(a => a.addEventListener('click', goLogout));

  const btn = document.querySelector('.avatar-btn');
  if (!btn) return;

  let username = 'operator', initials = 'OP';
  try {
    const d = await fetch('/api/whoami').then(r => r.json());
    username = d.username || username;
    initials = d.initials || initials;
  } catch (e) { /* keep defaults if offline */ }

  const safe = s => String(s).replace(/[<>&"]/g, '');
  const av = btn.querySelector('.avatar');
  if (av) av.textContent = initials;
  btn.setAttribute('aria-label', 'Signed in as ' + username);
  btn.setAttribute('title', 'Signed in as ' + username);
  btn.setAttribute('aria-haspopup', 'menu');
  btn.setAttribute('aria-expanded', 'false');

  const menu = document.createElement('div');
  menu.className = 'user-menu';
  menu.setAttribute('role', 'menu');
  menu.innerHTML = `
    <div class="um-head">
      <span class="um-avatar">${safe(initials)}</span>
      <div>
        <div class="um-name">${safe(username)}</div>
        <div class="um-sub">Signed in · single operator</div>
      </div>
    </div>
    <a class="um-item" href="/pages/dashboard.html" role="menuitem">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12l9-9 9 9"/><path d="M5 10v10h14V10"/></svg>Dashboard</a>
    <button class="um-item danger" type="button" role="menuitem" data-logout>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5M21 12H9"/></svg>Log out</button>`;
  btn.parentElement.appendChild(menu);

  const close = () => { menu.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); };
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = menu.classList.toggle('open');
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  menu.querySelector('[data-logout]').addEventListener('click', goLogout);
  document.addEventListener('click', (e) => {
    if (!menu.contains(e.target) && !btn.contains(e.target)) close();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initUserMenu);
} else {
  initUserMenu();
}
