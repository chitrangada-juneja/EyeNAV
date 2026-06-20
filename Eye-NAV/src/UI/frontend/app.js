/**
 * Eye-NAV – app.js (v2)
 *
 * Features:
 *  - Screen router (splash → home → running ↔ settings ↔ dev)
 *  - 2-second polling of /api/scene while running
 *  - TTS via browser SpeechSynthesis (reads LLM responses aloud)
 *  - Beep + "Stop!" voice alert on immediate_danger
 *  - Danger banner + full-screen red overlay
 *  - Audio / haptic toggle state
 *  - Settings save/discard via /api/settings
 *  - Developer mode with live detection list + depth bar
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────
const API_BASE       = window.location.origin;  // auto-resolves for any device on the network

const SPLASH_MS      = 2400;
const IS_REMOTE      = /ngrok|loca\.lt|cloudflare|zrok|localxpose|trycloudflare/.test(window.location.hostname);
const MAX_IN_FLIGHT  = 3;     // PIPELINING: hides upload/download latency

// --- ADAPTIVE QUALITY ---
// Starts at full quality for LAN, reduced for tunnels.
// Automatically adjusts based on measured round-trip time.
const QUALITY_HIGH   = IS_REMOTE ? 0.65 : 1.0;  // Best quality
const QUALITY_MED    = IS_REMOTE ? 0.45 : 0.6;  // Reduced when latency climbs
const QUALITY_LOW    = IS_REMOTE ? 0.25 : 0.35; // Fallback for bad connections
const RTT_GOOD_MS    = 150;   // Below this => full quality
const RTT_POOR_MS    = 400;   // Above this => low quality

// ── State ─────────────────────────────────────────────────────────
const state = {
  screen:      'splash',
  running:     false,
  audioOn:     true,
  hapticOn:    true,
  videoFeedOn: true,  // Video feed in dev screen ON by default
  devMode:     false,
  settings:    {},
  lastLLM:     '',    // de-dupe TTS
  inDanger:    false, // suppress repeated danger alerts
  audioCtx:    null,  // Web Audio API (lazy-init)
  speaking:    false, // prevent TTS overlap
  inFlight:    0,     // active fetch requests
  frameSeq:    0,     // monotonically increasing; used to label outgoing frames
  lastRenderedSeq: 0, // highest sequence number submitted to the UI
  lastFrameTs: 0,     // timestamp of last dispatched frame (for rate-limiting)
  uiFpsHistory: [],   // timestamps of successfully rendered frames
  lastSpokenTime: 0,  // timestamp of last TTS announcement
  lastBeepTime: 0,    // Cooldown for warning beeps (NEW)
  videoDevices: [],   // list of available cameras
  cameraIndex:  0,    // current camera being used
  captureQuality: IS_REMOTE ? 0.65 : 1.0, // Adaptive: starts at best for connection type
  rttHistory:   [],   // rolling window of round-trip times (ms)
  reconnectAttempts: 0,        // consecutive reconnect failures
  userStopped:  false,         // true when user pressed Stop — disables auto-reconnect
  lastFrameReceived: 0,        // timestamp of last successful frame from server
};

// ── DOM cache (built in init after DOMContentLoaded) ──────────────
let $s = {};

const $id = id => document.getElementById(id);

// ── Screen routing ────────────────────────────────────────────────
function goTo(name) {
  Object.values($s.screens).forEach(s => s.classList.remove('active'));
  const target = $s.screens[name];
  if (!target) { console.warn('No screen:', name); return; }
  target.classList.add('active');
  state.screen = name;

  if (name === 'running' || name === 'dev') startNavCapture();
  else if (name !== 'running' && name !== 'dev') stopNavCapture();

  if (name === 'settings') loadSettingsUI();
}

// ── Toast ─────────────────────────────────────────────────────────
let _toastTimer;
function toast(msg, ms = 2400) {
  const el = $id('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), ms);
}

// ── Audio: Beep via Web Audio API ─────────────────────────────────
function ensureAudioCtx() {
  if (!state.audioCtx) {
    try { state.audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
    catch(e) { console.warn('AudioContext not available', e); }
  }
  return state.audioCtx;
}

function playBeep({ freq = 880, duration = 0.18, type = 'square', vol = 0.6, times = 3 } = {}) {
  const ctx = ensureAudioCtx();
  if (!ctx) return;
  for (let i = 0; i < times; i++) {
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = type;
    osc.frequency.value = freq;
    gain.gain.value = vol;
    const t = ctx.currentTime + i * (duration + 0.06);
    osc.start(t);
    osc.stop(t + duration);
    gain.gain.setValueAtTime(vol, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
  }
}

function playWarningBeep() {
  playBeep({ freq: 660, duration: 0.14, type: 'sawtooth', vol: 0.45, times: 1 });
}

function playDangerBeep() {
  playBeep({ freq: 880, duration: 0.2, type: 'square', vol: 0.7, times: 3 });
}

// ── TTS via browser SpeechSynthesis ───────────────────────────────
function primeTTS() {
  if (!window.speechSynthesis) return;
  console.log('[TTS] Priming SpeechSynthesis...');
  const utt = new SpeechSynthesisUtterance(' '); // Speak a silent space
  utt.volume = 0;
  window.speechSynthesis.speak(utt);
}

function getDeltaMessage(current, next) {
  if (!current) return next;
  if (!next) return '';
  
  // If next starts with current, return only the unique suffix
  if (next.startsWith(current)) {
    const delta = next.slice(current.length).trim();
    return delta.startsWith('.') || delta.startsWith(',') ? delta.slice(1).trim() : delta;
  }
  
  // Otherwise, it's a completely different instruction (e.g. different hazard)
  return next;
}

function speak(text, { rate = 1.0, pitch = 1.0, priority = false } = {}) {
  if (!state.audioOn) return;
  if (!window.speechSynthesis) { console.warn('SpeechSynthesis not supported'); return; }

  // --- TELEMETRY HELPER ---
  const sendLog = (evt, msg) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: 'tts_log', event: evt, text: msg })); } catch(e){}
    }
  };

  // --- MANAGED QUEUE LOGIC ---
  if (priority) {
    // Interrupt: Clear everything and speak NOW
    state.pendingMessage = null;
    state.currentSpeechText = text;
    sendLog('REQUEST_INTERRUPT', text);
    window.speechSynthesis.cancel();
    state.speaking = false;
  } else {
    // Standard: Determine if this is NEW info or just a repeat
    if (state.speaking) {
      const delta = getDeltaMessage(state.currentSpeechText, text);
      
      if (!delta || delta.length < 2) {
        return; // No significant new info to append
      }

      state.pendingMessage = { text: delta, rate, pitch };
      console.log('[TTS] Delta Appended:', delta);
      sendLog('QUEUED_DELTA', delta);
      
      const logEl = document.getElementById('dev-speech-log');
      if (logEl) {
        const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const entry = document.createElement('div');
        entry.style.color = '#10b981';
        entry.innerHTML = `[${time}] <span style="color:#34c759;">[APPENDED]</span> ${delta}`;
        logEl.prepend(entry);
        if (logEl.children.length > 20) logEl.lastElementChild.remove();
      }
      return;
    }
  }

  state.currentSpeechText = text;

  state.speaking = true;
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = rate;
  utt.pitch = pitch;
  utt.volume = 1.0;

  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find(v => v.lang === 'en-US' && v.name.includes('Google'))
                 || voices.find(v => v.lang.startsWith('en'))
                 || voices[0];
  
  if (preferred) utt.voice = preferred;

  utt.onstart = () => { sendLog('AUDIO_START', text); };
  utt.onend = () => { 
    state.speaking = false; 
    sendLog('AUDIO_END', text);
    // Check if there is a pending message that was waiting for the AI to finish talking
    if (state.pendingMessage) {
      const { text: pText, rate: pRate, pitch: pPitch } = state.pendingMessage;
      state.pendingMessage = null;
      speak(pText, { rate: pRate, pitch: pPitch, priority: false });
    }
  };
  utt.onerror = (e) => { 
    console.error('[TTS] Error:', e);
    state.speaking = false; 
    sendLog('AUDIO_ERROR', text);
  };
  
  console.log('[TTS] Speaking:', text);
  window.speechSynthesis.speak(utt);

  // LOG TO DEV UI
  const logEl = document.getElementById('dev-speech-log');
  if (logEl) {
    if (logEl.innerHTML.includes('Awaiting speech')) logEl.innerHTML = '';
    const time = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const status = priority ? '<span style="color:#ff3b30;">[INTERRUPT]</span>' : '<span style="color:#34c759;">[SPEAK]</span>';
    const entry = document.createElement('div');
    entry.style.marginBottom = '6px';
    entry.style.borderBottom = '1px solid #222';
    entry.style.paddingBottom = '4px';
    entry.innerHTML = `[${time}] ${status} ${text}`;
    logEl.prepend(entry);
    if (logEl.children.length > 20) logEl.lastElementChild.remove();
  }
}

function speakDanger(text) {
  speak(text, { rate: 1.15, pitch: 1.0, priority: true });
}

// ── Haptic ────────────────────────────────────────────────────────
function vibrate(pattern) {
  if (!state.hapticOn) return;
  if (navigator.vibrate) navigator.vibrate(pattern);
}

// ── API ───────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  try {
    const r = await fetch(API_BASE + path, opts);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  } catch (e) {
    console.warn('API:', path, e.message);
    return null;
  }
}
const apiGet  = path => apiFetch(path);
const apiPost = (path, body) => apiFetch(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

// ── Camera & Nav Loop ─────────────────────────────────────────────
// Decoupled architecture:
//   navCaptureLoop  – runs at requestAnimationFrame speed (≤60fps)
//   captureAndSend  – fires a fetch without awaiting it (fire-and-forget)
//   MAX_IN_FLIGHT   – caps concurrent requests so server isn't overwhelmed
// This maximises FPS: the server is always processing something, and the
// frontend captures the freshest possible frame for each new request slot.

let ws = null;
let wsPreFetchQueue = [];
let pendingResult = null; // Stores last JSON metadata while waiting for the matching binary blob

function initWebSocket() {
  if (ws && ws.readyState !== WebSocket.CLOSED) return;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/api/ws_stream`);
  
  ws.binaryType = 'blob'; // Ensure binary messages are received as Blobs

  ws.onopen = () => { 
    console.log('[WebSocket] Connected binary-stream mode'); 
    wsPreFetchQueue = [];
  };
  ws.onmessage = async (event) => {
    // 1. Handle JSON Metadata
    if (typeof event.data === 'string') {
      try {
        const result = JSON.parse(event.data);
        if (result.error) {
          console.warn('[WebSocket] Server Error:', result.error);
          return;
        }
        pendingResult = result; // Store metadata
        
        // If we're not expecting a binary image, we must update the UI now
        if (!state.videoFeedOn || !state.devMode || !result.has_render) {
          // Track UI FPS since this is a complete "frame" response cycle
          const now = performance.now();
          state.uiFpsHistory.push(now);
          while (state.uiFpsHistory.length > 0 && now - state.uiFpsHistory[0] > 1000) {
            state.uiFpsHistory.shift();
          }
          const uiFps = state.uiFpsHistory.length;

          // Free up the pipeline!
          const preFetch = wsPreFetchQueue.shift();
          if (preFetch && result.profiler) {
            const roundTripMs = performance.now() - preFetch;
            result.profiler.network = Math.max(0, roundTripMs - result.profiler.total);
            adaptQuality(roundTripMs);
          }

          if ($id('ui-fps'))  $id('ui-fps').textContent  = uiFps;
          state.lastFrameReceived = performance.now();

          processScene(result);
          if (state.screen === 'running') updateRunningUI(result);
          if (state.screen === 'dev') {
            updateDevUI(result, result.depth_stats, {
              fps: result.server_fps || 0.0,
              uiFps: uiFps,
              data_source_mode: 'LIVE'
            });
          }
          pendingResult = null;
        }
      } catch (e) { console.warn('[WebSocket JSON Parse]', e); }
      return;
    }

    // 2. Handle Binary Image Blob
    if (event.data instanceof Blob) {
      if (!pendingResult) return; 

      const result = pendingResult;
      const blob = event.data;
      
      const preFetch = wsPreFetchQueue.shift();
      if (preFetch && result.profiler) {
        const roundTripMs = performance.now() - preFetch;
        result.profiler.network = Math.max(0, roundTripMs - result.profiler.total);
        adaptQuality(roundTripMs);
      }

      // Convert Binary Blob to URL for the <img> tag
      result.render = URL.createObjectURL(blob);

      // Track UI FPS since the Binary blob completes the cycle
      const now = performance.now();
      state.uiFpsHistory.push(now);
      while (state.uiFpsHistory.length > 0 && now - state.uiFpsHistory[0] > 1000) {
        state.uiFpsHistory.shift();
      }
      const uiFps = state.uiFpsHistory.length;

      if ($id('ui-fps'))  $id('ui-fps').textContent  = uiFps;
      state.lastFrameReceived = performance.now();

      processScene(result);
      if (state.screen === 'running') updateRunningUI(result);
      if (state.screen === 'dev') {
        updateDevUI(result, result.depth_stats, {
          fps: result.server_fps || 0.0,
          uiFps: uiFps,
          data_source_mode: 'LIVE'
        });
      }
      
      pendingResult = null;
    }
  };

  ws.onclose = (evt) => { 
    console.log('[WebSocket] Disconnected, code:', evt.code); 
    pendingResult = null;
    wsPreFetchQueue = [];

    // Only auto-reconnect if the user did NOT press Stop
    if (state.running && !state.userStopped) {
      const delay = Math.min(1000 * Math.pow(2, state.reconnectAttempts), 30000);
      state.reconnectAttempts++;
      console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts})...`);
      toast(`Connection lost. Reconnecting in ${Math.round(delay/1000)}s...`);
      setTimeout(() => {
        if (state.running && !state.userStopped) {
          ws = null;
          initWebSocket();
        }
      }, delay);
    }
  };

  ws.onerror = (e) => {
    console.warn('[WebSocket] Error:', e);
  };
}

function startNavCapture() {
  if (state.running) return;
  state.running = true;
  state.userStopped = false;
  state.reconnectAttempts = 0;
  state.lastFrameReceived = performance.now(); // initialise so watchdog doesn't fire immediately
  initWebSocket();
  initCamera().then(() => {
    if (state.running) navCaptureLoop();
  });
  startWatchdog();
}

function stopNavCapture() {
  state.running = false;
  state.userStopped = true;  // Prevent auto-reconnect
  stopWatchdog();
  if (ws) { ws.close(); ws = null; }
  const video = $id('nav-video');
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
}

// --- CONNECTION WATCHDOG ---
// Independently detects frozen UI, dead sockets, and unreachable backend.
let _watchdogTimer = null;
const WATCHDOG_INTERVAL_MS = 5000;  // Check every 5s
const FROZEN_FRAME_TIMEOUT = 8000; // No frames for 8s = frozen

function startWatchdog() {
  stopWatchdog(); // Clear any previous timer
  _watchdogTimer = setInterval(async () => {
    if (!state.running || state.userStopped) return;

    const now = performance.now();
    const timeSinceFrame = now - state.lastFrameReceived;

    // 1. Frozen UI: socket looks open but no frames are arriving
    const socketDead = !ws || ws.readyState === WebSocket.CLOSING || ws.readyState === WebSocket.CLOSED;
    const uiFrozen   = timeSinceFrame > FROZEN_FRAME_TIMEOUT;

    if (socketDead || uiFrozen) {
      console.warn(`[Watchdog] Triggering reconnect — socketDead:${socketDead}, uiFrozen:${uiFrozen} (${Math.round(timeSinceFrame)}ms since last frame)`);
      reconnectNow();
      return;
    }

    // 2. Backend health check — catches FastAPI crashes even if WebSocket is still in OPEN state
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 3000);
      const resp = await fetch(`${API_BASE}/api/health`, { signal: ctrl.signal });
      clearTimeout(t);
      if (!resp.ok) throw new Error('unhealthy');
      // Health OK — reset reconnect backoff since server is clearly alive
      if (state.reconnectAttempts > 0) state.reconnectAttempts = 0;
    } catch (e) {
      console.warn('[Watchdog] Health check failed:', e.message);
      reconnectNow();
    }
  }, WATCHDOG_INTERVAL_MS);
}

function stopWatchdog() {
  if (_watchdogTimer) { clearInterval(_watchdogTimer); _watchdogTimer = null; }
}

function reconnectNow() {
  if (!state.running || state.userStopped) return;
  const delay = Math.min(1000 * Math.pow(2, state.reconnectAttempts), 30000);
  state.reconnectAttempts++;
  console.log(`[Watchdog] Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts})`);
  toast(`Reconnecting... (${state.reconnectAttempts})`);
  // Force-close and re-open
  if (ws) { try { ws.close(); } catch(_) {} ws = null; }
  state.lastFrameReceived = performance.now(); // Reset so watchdog doesn't immediately re-fire
  setTimeout(() => {
    if (state.running && !state.userStopped) initWebSocket();
  }, delay);
}

async function initCamera(deviceId = null) {
  const video = $id('nav-video');
  if (!video) return;
  try {
    // Stop any existing tracks first
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    const constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 } } };
    if (deviceId) {
      constraints.video.deviceId = { exact: deviceId };
    } else {
      // Prefer rear camera; works on both Android Chrome and iOS Safari
      constraints.video.facingMode = { ideal: 'environment' };
    }
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    // play() may throw if the element isn't in the DOM yet, use a guard
    try { await video.play(); } catch(_) {}
    if (video.readyState < 2) {
      await new Promise(resolve => { video.onloadeddata = resolve; });
    }
    console.log('[Camera] Ready:', video.videoWidth, 'x', video.videoHeight);
  } catch (err) {
    console.error('[initCamera] Failed:', err);
    toast('Camera access denied or unavailable');
  }
}

async function switchCamera() {
  try {
    // On Android/Chrome, device labels are empty until permission is granted.
    // Always re-enumerate after the first getUserMedia call to get populated labels.
    const devices = await navigator.mediaDevices.enumerateDevices();
    state.videoDevices = devices.filter(d => d.kind === 'videoinput');

    if (state.videoDevices.length < 2) {
      toast('No other cameras found');
      return;
    }

    state.cameraIndex = (state.cameraIndex + 1) % state.videoDevices.length;
    const device = state.videoDevices[state.cameraIndex];
    const label = device.label || `Camera ${state.cameraIndex + 1}`;
    toast(`Switching to ${label}...`);
    await initCamera(device.deviceId);
  } catch (err) {
    console.warn('[switchCamera] Error:', err);
    toast('Camera switch failed');
  }
}

// ── Frame capture loop ────────────────────────────────────────────
// Rate-limited to MAX_FPS to avoid flooding the backend.
// MAX_IN_FLIGHT=1 guarantees frames are always applied in order.
const FRAME_INTERVAL_MS = 33;  // Reduced to 33ms (30 fps cap) to allow full system speed

function navCaptureLoop() {
  if (!state.running) return;
  const now = performance.now();
  if (now - state.lastFrameTs >= FRAME_INTERVAL_MS) {
    state.lastFrameTs = now;
    captureAndSend();  // fire-and-forget
  }
  requestAnimationFrame(navCaptureLoop);
}

async function captureAndSend() {
  const video  = $id('nav-video');
  const canvas = $id('nav-canvas');
  if (!video || !canvas || !video.videoWidth) return;

  // Prevent sending frames if the websocket connection drops or buffers too much
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (wsPreFetchQueue.length >= MAX_IN_FLIGHT) return;

  // Adaptive: if the send buffer is backing up, the network is struggling
  if (ws.bufferedAmount > 200000) {
    adaptQuality('poor');
    console.warn('[WebSocket] Network bottleneck, dropping frame');
    return;
  }

  // Capture: center-crop to square and scale to 640×640
  const ctx  = canvas.getContext('2d');
  const size = Math.min(video.videoWidth, video.videoHeight);
  const sx   = (video.videoWidth  - size) / 2;
  const sy   = (video.videoHeight - size) / 2;
  ctx.drawImage(video, sx, sy, size, size, 0, 0, 640, 640);

  // Compress to JPEG blob using the current adaptive quality
  const blob = await new Promise(resolve =>
    canvas.toBlob(resolve, 'image/jpeg', state.captureQuality)
  );
  if (!blob) return;

  // Fire directly into the Web Socket
  wsPreFetchQueue.push(performance.now());
  ws.send(blob);
}

// Adaptive quality controller — called after each measured round-trip
function adaptQuality(hint) {
  let target = state.captureQuality;

  if (hint === 'poor') {
    target = QUALITY_LOW;
  } else if (typeof hint === 'number') {
    // hint is the measured RTT in ms
    const rtt = hint;
    state.rttHistory.push(rtt);
    if (state.rttHistory.length > 10) state.rttHistory.shift();
    const avgRtt = state.rttHistory.reduce((a, b) => a + b, 0) / state.rttHistory.length;

    if (avgRtt < RTT_GOOD_MS) {
      target = QUALITY_HIGH;
    } else if (avgRtt < RTT_POOR_MS) {
      target = QUALITY_MED;
    } else {
      target = QUALITY_LOW;
    }

    // Show quality in dev UI if available
    const qEl = $id('prof-quality');
    if (qEl) {
      const pct = Math.round(target * 100);
      const color = target === QUALITY_HIGH ? '#4ade80' : target === QUALITY_MED ? '#fbbf24' : '#f87171';
      qEl.textContent = `${pct}%`;
      qEl.style.color = color;
    }
  }

  // Only update if there is a meaningful change (avoid thrashing)
  if (Math.abs(target - state.captureQuality) > 0.05) {
    console.log(`[AdaptiveQuality] ${Math.round(state.captureQuality*100)}% -> ${Math.round(target*100)}%`);
    state.captureQuality = target;
    // Reset reconnect counter on quality recovery — connection is alive
    if (target >= QUALITY_MED) state.reconnectAttempts = 0;
  }
}

// ── Scene processor (danger + TTS) ───────────────────────────────
function getNormalizedTokens(text) {
  if (!text) return [];
  // Lowercase, remove numbers, units, and punctuation
  let clean = text.toLowerCase()
    .replace(/[0-9.]/g, '')
    .replace(/\b(metres|meters|metris)\b/g, '')
    .replace(/[.,!?;:]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  
  // SYNONYM CONSOLIDATION: Map similar words to base forms to prevent re-triggering
  clean = clean
    .replace(/\b(unrecognised|unidentified|unknown)\b/g, 'unknown')
    .replace(/\b(object|obstacle|hazard)\b/g, 'obstacle');

  // Return set of significant words (length > 2, ignore pure numbers)
  return clean.split(' ').filter(word => {
    return word.length > 2 && !/^\d+(\.\d+)?$/.test(word);
  });
}

function isSimilar(msg1, msg2) {
  const tokens1 = getNormalizedTokens(msg1);
  const tokens2 = getNormalizedTokens(msg2);
  if (tokens1.length === 0 || tokens2.length === 0) return msg1 === msg2;

  // Check overlap percentage (Jaccard-ish)
  const set1 = new Set(tokens1);
  const set2 = new Set(tokens2);
  let matches = 0;
  set1.forEach(t => { if (set2.has(t)) matches++; });

  const maxLen = Math.max(set1.size, set2.size);
  const similarity = matches / maxLen;
  return similarity > 0.75; // 75% word overlap threshold
}

function processScene(scene) {
  // --- SYNC DYNAMIC UI CONFIG ---
  if (scene.config && scene.config.ui_font_size) {
    document.documentElement.style.setProperty('--dynamic-font-size', scene.config.ui_font_size + 'rem');
  }

  const danger    = !!scene.immediate_danger;
  const risk      = scene.risk_level || 'clear';
  const displayMsg = scene.final_instruction || scene.llm_response || '';
  const ttsAction = scene.tts_action || "SILENCE";

  // --- 1. HANDLE DANGER OVERLAYS (Visual Only) ---
  if (danger && !state.inDanger) {
    state.inDanger = true;
    triggerDangerOverlay();
  } else if (!danger && state.inDanger) {
    state.inDanger = false;
    clearDangerOverlay();
  }

  // --- 2. HANDLE TTS DECISIONS FROM BACKEND ---
  if (ttsAction !== "SILENCE" && displayMsg && displayMsg !== "...") {
    const now = Date.now();
    
    // Priority: INTERRUPT forces immediate speech
    const isInterrupt = (ttsAction === "INTERRUPT");
    
    if (isInterrupt) {
      window.speechSynthesis?.cancel();
      state.speaking = false;
      // Small delay to let the browser clear the audio buffer before starting the next priority message
      setTimeout(() => {
        speak(displayMsg, { rate: (risk === 'clear' ? 0.9 : 1.0), priority: true });
      }, 50);
      
      state.lastLLM = displayMsg;
      state.lastSpokenTime = now;
      return; // Handled in timeout
    }

    // --- STANDARD SPEAK ---
    // Cooldown and Beeps
    if (risk === 'warning' || risk === 'danger') {
      const timeSinceBeep = now - (state.lastBeepTime || 0);
      if (timeSinceBeep > 3000) {
        state.lastBeepTime = now;
        if (risk === 'danger') playDangerBeep();
        else playWarningBeep();
        vibrate([80, 60, 80]);
      }
      speak(displayMsg, { rate: 1.0 });
    } else {
      speak(displayMsg, { rate: 0.9 });
    }
    
    state.lastLLM = displayMsg;
    state.lastSpokenTime = now;
  }
}

function triggerDangerOverlay() {
  $id('danger-overlay').classList.add('visible');
  vibrate([200, 100, 200, 100, 400]);
  const stopBtn = $id('stop-btn');
  if (stopBtn) stopBtn.classList.add('danger-pulse');
}

function clearDangerOverlay() {
  $id('danger-overlay').classList.remove('visible');
  const stopBtn = $id('stop-btn');
  if (stopBtn) stopBtn.classList.remove('danger-pulse');
}

function triggerDangerAlert(llmMsg) {
  // 1. Full screen overlay
  $id('danger-overlay').classList.add('visible');
  // 2. Vibrate SOS pattern
  vibrate([200, 100, 200, 100, 400]);
  // 4. Beep
  playDangerBeep();
  // 5. Speak
  speakDanger(llmMsg || 'Stop! Immediate danger ahead.');
  // 6. Eye btn danger pulse
  const stopBtn = $id('stop-btn');
  if (stopBtn) stopBtn.classList.add('danger-pulse');
}

function clearDangerAlert() {
  $id('danger-overlay').classList.remove('visible');
  const stopBtn = $id('stop-btn');
  if (stopBtn) stopBtn.classList.remove('danger-pulse');
}

// ── Running screen UI ─────────────────────────────────────────────
const RISK_CONFIG = {
  clear:   { icon: '🟢', text: 'ALL CLEAR',    cls: 'clear',   dotCls: '',       cardCls: '' },
  warning: { icon: '🟡', text: 'CAUTION',       cls: 'warning', dotCls: 'warn',   cardCls: 'warn-card' },
  danger:  { icon: '🔴', text: 'DANGER',         cls: 'danger',  dotCls: 'danger', cardCls: 'danger-card' },
};

function updateRunningUI(scene) {
  const risk = scene.risk_level || 'clear';
  const cfg  = RISK_CONFIG[risk] || RISK_CONFIG.clear;
  const msg  = scene.final_instruction || scene.llm_response || '—';

  // Risk badge
  const badge = $id('risk-badge');
  badge.className = `risk-badge ${cfg.cls}`;
  $id('risk-icon').textContent = cfg.icon;
  $id('risk-text').textContent = cfg.text;

  // Status dot
  const dot = $id('run-dot');
  dot.className = `status-dot ${cfg.dotCls}`;

  // LLM card (Main Instruction Area)
  const card = $id('llm-card');
  card.className = `llm-card ${cfg.cardCls}`;
  $id('llm-icon').textContent = risk === 'danger' ? '🔴' : risk === 'warning' ? '🟡' : '🔵';
  $id('llm-text').textContent = msg;
}

// ── Dev screen UI ─────────────────────────────────────────────────
function updateDevUI(scene, depth, status) {
  const risk = scene.risk_level || 'clear';
  const cfg  = RISK_CONFIG[risk] || RISK_CONFIG.clear;
  const fpsNum = parseFloat(status?.fps);
  const uiFpsNum = parseInt(status?.uiFps) || 0;

  // 0. Update Server Feed Render (only when Video Feed is toggled on)
  if (state.videoFeedOn) {
    const imgData = scene.render;
    const imgEl   = $id('dev-render-img');
    const phEl    = $id('dev-feed-placeholder');
    if (imgData && imgEl) {
      // The image is already a blob URL or base64 from processScene
      imgEl.src = imgData;
      if (phEl) phEl.style.display = 'none';
    }


  }

  // 1. Risk badge
  const dbadge = $id('dev-risk-badge');
  if (dbadge) {
    dbadge.className = `risk-badge ${cfg.cls}`;
    $id('dev-risk-icon').textContent = cfg.icon;
    $id('dev-risk-text').textContent = cfg.text;
  }

  // 2. FPS
  $id('dev-fps').textContent = !isNaN(fpsNum) ? fpsNum.toFixed(1) : '—';
  if ($id('ui-fps')) $id('ui-fps').textContent = uiFpsNum || '—';

  // 2.5 Profiler
  if (scene.profiler) {
    if ($id('prof-total')) $id('prof-total').textContent = scene.profiler.total.toFixed(0) + 'ms';
    if ($id('prof-bottleneck')) $id('prof-bottleneck').textContent = scene.profiler.bottleneck;
    if ($id('prof-network')) $id('prof-network').textContent = (scene.profiler.network || 0).toFixed(0) + 'ms';
    if ($id('prof-yolo')) $id('prof-yolo').textContent = scene.profiler.yolo.toFixed(0) + 'ms';
    if ($id('prof-depth')) $id('prof-depth').textContent = (scene.profiler.depth || 0).toFixed(0) + 'ms';
    if ($id('prof-path')) $id('prof-path').textContent = scene.profiler.path.toFixed(0) + 'ms';
    if ($id('prof-encode')) $id('prof-encode').textContent = scene.profiler.encode.toFixed(0) + 'ms';
  }

  // 3. LLM (Store previous so it doesn't flicker when loading)
  if (scene.llm_response && scene.llm_response !== '...') {
    state.lastLLM = scene.llm_response;
  }
  const displayLLM = scene.llm_response && scene.llm_response !== '...' ? scene.llm_response : (state.lastLLM || '—');
  $id('dev-llm-text').textContent = displayLLM;
  
  const devCard = $id('dev-llm-box');
  if (devCard) devCard.className = `llm-card ${cfg.cardCls}`;

  // 4. Object count
  $id('dev-count').textContent = scene.count ?? 0;

  // 5. Source
  const ds = $id('dev-source');
  if (ds && status?.data_source_mode) ds.textContent = status.data_source_mode.toUpperCase();

  // Detection list
  renderDetections(scene.objects || []);

  // Depth
  if (depth) {
    const min  = depth.min_distance  ?? 0;
    const mean = depth.mean_distance ?? 10;
    const max  = depth.max_distance  ?? 20;
    const pct  = Math.min(95, Math.max(5, ((20 - mean) / 20) * 100));
    $id('depth-bar').style.width = pct + '%';
    $id('d-min').textContent  = min.toFixed(1)  + 'm';
    $id('d-mean').textContent = mean.toFixed(1) + 'm';
    $id('d-max').textContent  = max.toFixed(1)  + 'm';
  }

  // Dev screen dot
  $id('dev-dot').className = `status-dot ${cfg.dotCls}`;
}

function renderDetections(objects) {
  const list = $id('detection-list');
  if (!objects || !objects.length) {
    list.innerHTML = '<p style="font-size:0.8rem;color:var(--text-3);padding:6px 0;">No objects detected.</p>';
    return;
  }

  // Build a Set of labels currently being reported to the user (in the active SOS message)
  const activeMsg = (state.lastLLM || '').toLowerCase();
  
  list.innerHTML = objects.map(obj => {
    const dist   = parseFloat(obj.distance);
    const pos    = Array.isArray(obj.position) ? obj.position[0] : (obj.position || '');
    const distStr = !isNaN(dist) ? dist.toFixed(1) + 'm' : '?m';
    
    // Determine if this object is one the AI is actively reporting
    const labelLower = (obj.label || '').toLowerCase();
    const isReported = activeMsg.includes(labelLower);

    // Color: red if currently being reported aloud, green if detected but suppressed
    const labelColor  = isReported ? '#f87171' : '#4ade80';
    const badgeText   = isReported ? '● ACTIVE' : '○ SILENT';
    const badgeColor  = isReported ? '#f87171' : '#4ade80';

    // Severity color for distance
    const distColor = !isNaN(dist) ? (dist < 2 ? '#f87171' : dist < 5 ? '#fbbf24' : '#94a3b8') : '#94a3b8';

    return `<div style="display:flex;align-items:center;justify-content:space-between;padding:5px 8px;border-bottom:1px solid var(--border,#2a3a50);gap:6px;">
      <span style="font-weight:700;font-size:0.78rem;color:${labelColor};flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(obj.label)}</span>
      <span style="font-size:0.65rem;color:#64748b;white-space:nowrap;">${esc(pos)}</span>
      <span style="font-weight:700;font-size:0.8rem;color:${distColor};white-space:nowrap;min-width:38px;text-align:right;">${distStr}</span>
      <span style="font-size:0.6rem;color:${badgeColor};white-space:nowrap;font-weight:600;">${badgeText}</span>
    </div>`;
  }).join('');
}

const esc = s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ── Settings ──────────────────────────────────────────────────────
const S_MAP = {
  's-companion':  'companion_mode',
  's-distance':   'distance_estimation',
  's-descriptive':'descriptive_mode',
  's-dev':        'developer_mode',
};

async function fetchSettings() {
  const d = await apiGet('/api/settings');
  if (d) {
    state.settings = d;
    state.devMode  = !!d.developer_mode;
    updateDevBtn();
  }
}

function loadSettingsUI() {
  for (const [id, key] of Object.entries(S_MAP)) {
    const el = $id(id);
    if (el) el.checked = !!state.settings[key];
  }
}

async function saveSettings() {
  const draft = {};
  for (const [id, key] of Object.entries(S_MAP)) {
    const el = $id(id);
    if (el) draft[key] = el.checked;
  }
  const r = await apiPost('/api/settings', draft);
  if (r?.status === 'saved') {
    state.settings = r.settings;
    state.devMode  = !!r.settings.developer_mode;
    updateDevBtn();
    toast('Settings saved ✓');
    goTo('home');
  } else {
    toast('Could not save — backend running?');
  }
}

function discardSettings() {
  loadSettingsUI();
  toast('Changes discarded');
  goTo('home');
}

function updateDevBtn() {
  const btn = $id('btn-dev-open');
  if (btn) btn.style.display = state.devMode ? 'block' : 'none';
}

// ── Nav: start / stop ─────────────────────────────────────────────
async function startNav() {
  ensureAudioCtx();
  if (state.audioCtx?.state === 'suspended') await state.audioCtx.resume();
  primeTTS();

  const r = await apiPost('/api/command', { action: 'start' });
  if (r !== null) {
    state.inDanger = false;
    state.lastLLM  = '';
    goTo('running');         // goTo calls startNavCapture internally
    toast('Eye-NAV started');
  } else {
    toast('Backend not reachable – is server running?');
  }
}

async function stopNav() {
  await apiPost('/api/command', { action: 'stop' });
  stopNavCapture();
  clearDangerAlert();
  window.speechSynthesis?.cancel();
  goTo('home');
  toast('Navigation stopped');
}

// ── Toggle helpers ────────────────────────────────────────────────
function toggleAudio() {
  state.audioOn = !state.audioOn;
  const btn = $id('audio-toggle');
  const ico = $id('audio-icon');
  btn.classList.toggle('on', state.audioOn);
  btn.setAttribute('aria-pressed', state.audioOn);
  ico.textContent = state.audioOn ? '🔊' : '🔇';
  if (state.audioOn) speak('Audio on');
  else toast('Audio muted');
}

function toggleHaptic() {
  state.hapticOn = !state.hapticOn;
  const btn = $id('haptic-toggle');
  const ico = $id('haptic-icon');
  btn.classList.toggle('on', state.hapticOn);
  btn.setAttribute('aria-pressed', state.hapticOn);
  ico.textContent = state.hapticOn ? '📳' : '📵';
  toast(state.hapticOn ? 'Haptic on' : 'Haptic off');
  if (state.hapticOn) vibrate([60, 40, 60]);
}

function toggleVideoFeed() {
  state.videoFeedOn = !state.videoFeedOn;
  updateVideoFeedUI();
}

function updateVideoFeedUI() {
  const btn    = $id('dev-feed-toggle');
  const lbl    = $id('dev-feed-lbl');
  const panel  = $id('dev-feed-panel');
  const fsBtn  = $id('dev-fullscreen-btn');

  if (btn)   btn.style.borderColor = state.videoFeedOn ? 'var(--blue-light,#4fc3f7)' : 'var(--border,#2a3a50)';
  if (lbl)   lbl.textContent = state.videoFeedOn ? 'Video On' : 'Video Off';
  if (panel) panel.style.display = state.videoFeedOn ? 'block' : 'none';
  if (fsBtn) fsBtn.style.display = state.videoFeedOn ? 'flex'  : 'none';

  if (!state.videoFeedOn && state.devFullScreen) {
    exitDevFullScreen();
  }

  if (state.running && state.videoFeedOn) {
    toast('Video Feed enabled');
  } else if (state.running && !state.videoFeedOn) {
    toast('Video Feed disabled (faster)');
  }
}

function toggleDevFullScreen() {
  if (!state.videoFeedOn) return;
  state.devFullScreen = !state.devFullScreen;
  
  const screen = $id('screen-dev');
  const panel = $id('dev-feed-panel');
  
  if (screen) screen.classList.toggle('dev-full-screen-mode', state.devFullScreen);
  if (panel) panel.classList.toggle('full-screen-feed', state.devFullScreen);
  
  if (state.devFullScreen) {
    document.body.style.overflow = 'hidden';
  } else {
    document.body.style.overflow = '';
  }
}

function exitDevFullScreen() {
  state.devFullScreen = false;
  const screen = $id('screen-dev');
  const panel = $id('dev-feed-panel');
  if (screen) screen.classList.remove('dev-full-screen-mode');
  if (panel) panel.classList.remove('full-screen-feed');
  document.body.style.overflow = '';
}

// ── Event binding ─────────────────────────────────────────────────
function bindEvents() {
  $id('start-btn').addEventListener('click', startNav);
  $id('btn-settings-home').addEventListener('click', () => goTo('settings'));
  $id('stop-btn').addEventListener('click', stopNav);
  $id('audio-toggle').addEventListener('click', toggleAudio);
  $id('haptic-toggle').addEventListener('click', toggleHaptic);
  $id('dev-feed-toggle').addEventListener('click', toggleVideoFeed);
  $id('dev-fullscreen-btn').addEventListener('click', toggleDevFullScreen);
  $id('dev-minimize-btn').addEventListener('click', exitDevFullScreen);
  $id('btn-dev-open').addEventListener('click', () => goTo('dev'));
  $id('btn-cam-switch')?.addEventListener('click', switchCamera);
  $id('dev-cam-switch')?.addEventListener('click', switchCamera);
  $id('btn-dev-exit').addEventListener('click', () => goTo('running'));
  $id('btn-settings-back').addEventListener('click', discardSettings);
  $id('btn-discard').addEventListener('click', discardSettings);
  $id('btn-save').addEventListener('click', saveSettings);

  // Collapsible panel toggles
  function makeToggle(btnId, panelId) {
    const btn = $id(btnId);
    const panel = $id(panelId);
    if (!btn || !panel) return;
    btn.addEventListener('click', () => {
      const isOpen = panel.style.display !== 'none';
      panel.style.display = isOpen ? 'none' : 'block';
      btn.setAttribute('aria-expanded', String(!isOpen));
      btn.textContent = isOpen ? '\u25b6 Show' : '\u25bc Hide';
    });
  }
  makeToggle('speech-log-toggle', 'dev-speech-log');
  makeToggle('det-list-toggle', 'detection-list');
}

// ── Voices warmup (some browsers need a trigger) ──────────────────
function warmupVoices() {
  if (!window.speechSynthesis) return;
  if (window.speechSynthesis.getVoices().length === 0) {
    window.speechSynthesis.addEventListener('voiceschanged', () => {}, { once: true });
  }
}

// ── Init ──────────────────────────────────────────────────────────
// ── Init ──────────────────────────────────────────────────────────
async function init() {
  $s.screens = {
    splash:   $id('screen-splash'),
    home:     $id('screen-home'),
    running:  $id('screen-running'),
    dev:      $id('screen-dev'),
    settings: $id('screen-settings'),
  };

  bindEvents();
  warmupVoices();
  updateVideoFeedUI(); // Sync default state with UI

  // ALIGNMENT: Synchronize UI transition with backend readiness
  const minTimer = new Promise(resolve => setTimeout(resolve, 1000));
  
  const pollBackend = async () => {
    while (true) {
      try {
        const resp = await fetch(`${API_BASE}/api/settings`);
        if (resp.ok) {
          const d = await resp.json();
          state.settings = d;
          state.devMode  = !!d.developer_mode;
          updateDevBtn();
          return true;
        }
      } catch (e) {
        // Backend not ready yet (loading models...)
      }
      await new Promise(r => setTimeout(r, 500));
    }
  };

  // Wait for at least 1s AND the backend to respond
  await Promise.all([minTimer, pollBackend()]);
  
  goTo('home');
}

document.addEventListener('DOMContentLoaded', init);
