(() => {
  let __chatbotInitialized = false;

  function initChatbot() {
    if (__chatbotInitialized) return;
    __chatbotInitialized = true;
  // =====================
  // CONFIG
  // =====================
  const GLOBAL_CONFIG = window.BILChatbotConfig || {};
  const ROOT = GLOBAL_CONFIG.root || document;
  const byId = (id) =>
    ROOT.getElementById ? ROOT.getElementById(id) : ROOT.querySelector(`#${id}`);
  const API_BASE = GLOBAL_CONFIG.apiBase || "";
  const ASSET_BASE = (GLOBAL_CONFIG.assetBase || "").replace(/\/$/, "");
  const API_URL = GLOBAL_CONFIG.apiUrl || (API_BASE ? `${API_BASE}/chat` : "/chat");
  const STT_URL = GLOBAL_CONFIG.sttUrl || (API_BASE ? `${API_BASE}/stt` : "/stt");
  const BOT_NAME = GLOBAL_CONFIG.botName || "Norbu";
  const BOT_LOGO =
    GLOBAL_CONFIG.botLogo ||
    (ASSET_BASE ? `${ASSET_BASE}/assets/logo.svg` : "./assets/logo.svg");

  // Typewriter speed
  const TYPE_MIN_DELAY = 6;
  const TYPE_MAX_DELAY = 10;

  // =====================
  // DOM
  // =====================
  const fab = byId("chat-fab");
  const panel = byId("chat-panel");
  const closeBtn = byId("chat-close");
  const teaser = byId("chat-teaser");
  const teaserClose = byId("teaser-close");
  const fabIcon = byId("fab-icon");

  const messagesEl = byId("chat-messages");
  const inputEl = byId("chat-input");
  const sendBtn = byId("send-btn");
  const statusEl = byId("bot-status");

  const micBtn = byId("mic-btn");
  const hintEl = byId("hint");
  const micDefaultHtml = micBtn ? micBtn.innerHTML : "";

  const composerNormal = byId("composer-normal");
  const composerRecording = byId("composer-recording");

  const recTimerEl = byId("rec-timer");
  const recCancelBtn = byId("rec-cancel");
  const recSendBtn = byId("rec-send");

  // Validate required nodes
  const required = {
    fab,
    panel,
    closeBtn,
    messagesEl,
    inputEl,
    sendBtn,
    micBtn,
    hintEl,
    composerNormal,
    composerRecording,
    recTimerEl,
    recCancelBtn,
    recSendBtn,
  };
  for (const [k, v] of Object.entries(required)) {
    if (!v) console.error(`Missing element: ${k}`);
  }

  // =====================
  // SESSION + HISTORY
  // =====================
  let history = []; // [{role:"user"|"assistant", content:"..."}]

  let sessionId = localStorage.getItem("bil_session_id");
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem("bil_session_id", sessionId);
  }

  // =====================
  // UI helpers
  // =====================
  const TEASER_DELAY_MS = 2500;
  const FAB_OPEN_ICON =
    GLOBAL_CONFIG.fabOpenIcon ||
    (ASSET_BASE ? `${ASSET_BASE}/assets/popup.svg` : "./assets/popup.svg");
  const FAB_CLOSE_ICON =
    GLOBAL_CONFIG.fabCloseIcon ||
    (ASSET_BASE ? `${ASSET_BASE}/assets/down.svg` : "./assets/down.svg");
  let teaserTimer = null;

  function scheduleTeaser() {
    if (!teaser) return;
    if (teaserTimer) clearTimeout(teaserTimer);
    teaserTimer = setTimeout(() => {
      teaser.classList.remove("hidden");
    }, TEASER_DELAY_MS);
  }

  function togglePanel(show) {
    panel.classList.toggle("hidden", !show);
    if (show) {
      if (teaser) teaser.classList.add("hidden");
      if (teaserTimer) clearTimeout(teaserTimer);
      if (fabIcon) fabIcon.src = FAB_CLOSE_ICON;
      if (fab) fab.setAttribute("aria-label", "Close chat");
      if (fab) {
        fab.classList.add("fab-close");
        fab.classList.remove("fab-open");
      }
      ensureIntroMessage();
      setTimeout(() => inputEl.focus(), 50);
    } else {
      if (fabIcon) fabIcon.src = FAB_OPEN_ICON;
      if (fab) fab.setAttribute("aria-label", "Open chat");
      if (fab) {
        fab.classList.add("fab-open");
        fab.classList.remove("fab-close");
      }
      scheduleTeaser();
    }
  }

  // Initialize FAB state
  if (fab) {
    fab.classList.add("fab-open");
    fab.classList.remove("fab-close");
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function showHint(msg) {
    const text = (msg || "").trim();
    const isTranscribing = /transcrib/i.test(text);
    const isCancelled = /cancel/i.test(text);

    if (!text) {
      if (composerNormal) composerNormal.classList.remove("hinting");
      if (inputEl) {
        inputEl.disabled = false;
        inputEl.style.display = "";
        inputEl.placeholder = "Ask a question...";
      }
      const inline = composerNormal
        ? composerNormal.querySelector(".hint-inline")
        : null;
      if (inline) inline.remove();
      if (micBtn) {
        micBtn.innerHTML = micDefaultHtml;
        micBtn.classList.remove("hint-cancel");
      }
      if (sendBtn) sendBtn.disabled = false;
      return;
    }

    if (isCancelled) {
      if (composerNormal) composerNormal.classList.remove("hinting");
      if (inputEl) {
        inputEl.disabled = false;
        inputEl.style.display = "";
        inputEl.placeholder = text;
      }
      const inline = composerNormal
        ? composerNormal.querySelector(".hint-inline")
        : null;
      if (inline) inline.remove();
      if (micBtn) {
        micBtn.innerHTML = micDefaultHtml;
        micBtn.classList.remove("hint-cancel");
      }
      if (sendBtn) sendBtn.disabled = false;
      return;
    }

    if (composerNormal) composerNormal.classList.add("hinting");
    if (inputEl) {
      inputEl.disabled = true;
      inputEl.style.display = "none";
    }

    let inline = composerNormal.querySelector(".hint-inline");
    if (!inline) {
      inline = document.createElement("div");
      inline.className = "hint-inline";
      composerNormal.appendChild(inline);
    }
    inline.innerHTML = `
      <span>${text}</span>
      ${isTranscribing ? '<span class="hint-spinner" aria-hidden="true"></span>' : ""}
    `;

    if (micBtn) {
      micBtn.innerHTML = "✕";
      micBtn.classList.add("hint-cancel");
    }
    if (sendBtn) sendBtn.disabled = true;
  }

  function getGreeting() {
    const h = new Date().getHours();
    if (h < 12) return "Good morning";
    if (h < 17) return "Good afternoon";
    return "Good evening";
  }

  function getRandomGreeting() {
    const base = getGreeting();
    const options = [
      `${base}! I’m Norbu, here to help with BIL insurance, claims, loans, and forms.`,
      `${base}! I’m your BIL assistant. I can help with insurance, claims, loans, and forms.`,
      `${base}! I’m here to help with BIL questions and support.`,
    ];
    return options[Math.floor(Math.random() * options.length)];
  }

  function addMessageRow(type = "incoming") {
    const row = document.createElement("div");
    row.className = `msg-row ${type}`;

    let bubble = null;

    if (type === "incoming") {
      const avatar = document.createElement("div");
      avatar.className = "msg-avatar";

      const img = document.createElement("img");
      img.src = BOT_LOGO;
      img.alt = "BIL logo";
      avatar.appendChild(img);

      const content = document.createElement("div");
      content.className = "msg-content";

      const title = document.createElement("div");
      title.className = "msg-title";
      title.textContent = BOT_NAME;

      bubble = document.createElement("div");
      bubble.className = "bubble";

      content.appendChild(title);
      content.appendChild(bubble);

      row.appendChild(avatar);
      row.appendChild(content);
    } else {
      const wrap = document.createElement("div");
      wrap.className = "msg-outgoing";

      bubble = document.createElement("div");
      bubble.className = "bubble";

      wrap.appendChild(bubble);
      row.appendChild(wrap);
    }

    messagesEl.appendChild(row);
    scrollToBottom();

    return { row, bubble };
  }

  function addTextMessage(text, type = "incoming") {
    const { bubble } = addMessageRow(type);
    bubble.textContent = text;
    scrollToBottom();
  }

  function ensureIntroMessage() {
    if (messagesEl.childElementCount > 0) return;
    if (statusEl) statusEl.textContent = "Online";
    addTextMessage(getRandomGreeting(), "incoming");
  }

  function addTypingBubble() {
    const { row, bubble } = addMessageRow("incoming");
    const dots = document.createElement("span");
    dots.className = "typing";
    dots.innerHTML = "<span></span><span></span><span></span>";
    bubble.appendChild(dots);
    bubble.classList.add("is-typing");
    return { row, bubble };
  }

  function isFileRequest(message) {
    const q = String(message || "").toLowerCase();
    return /form|forms|pdf|download|document|documents|file|files/.test(q);
  }

  // little dots animation via JS (no extra CSS needed)
  function randomDelay() {
    return (
      Math.floor(Math.random() * (TYPE_MAX_DELAY - TYPE_MIN_DELAY + 1)) +
      TYPE_MIN_DELAY
    );
  }

  // =====================
  // Markdown: progressive preview without showing markdown
  // =====================
  function stripMarkdown(md) {
    if (!md) return "";

    let s = String(md);

    // Normalize newlines
    s = s.replace(/\r\n/g, "\n");

    // Remove fenced code blocks completely (or keep their text)
    s = s.replace(/```[\s\S]*?```/g, (m) => {
      return m
        .replace(/```[\w-]*\n?/g, "")
        .replace(/```/g, "")
        .trim();
    });

    // Headings: "# Title" -> "Title"
    s = s.replace(/^\s{0,3}#{1,6}\s+/gm, "");

    // Blockquotes: "> text" -> "text"
    s = s.replace(/^\s{0,3}>\s?/gm, "");

    // Horizontal rules
    s = s.replace(/^\s{0,3}(-{3,}|_{3,}|\*{3,})\s*$/gm, "");

    // Links: [text](url) -> text
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "$1");

    // Images: ![alt](url) -> alt
    s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, "$1");

    // Inline code: `x` -> x
    s = s.replace(/`([^`]+)`/g, "$1");

    // Bold/italic: **x** __x__ *x* _x_ -> x
    s = s.replace(/\*\*([\s\S]*?)\*\*/g, "$1");
    s = s.replace(/__([\s\S]*?)__/g, "$1");
    s = s.replace(/\*([\s\S]*?)\*/g, "$1");
    s = s.replace(/_([\s\S]*?)_/g, "$1");
    s = s.replace(/:\s*-\s+/g, ":\n- ");

    // Lists:
    // "- item" "* item" "+ item"  -> "• item"
    s = s.replace(/^\s*[-*+]\s+/gm, "• ");
    // "1. item" -> "• item"
    s = s.replace(/^\s*\d+\.\s+/gm, "• ");

    // Tables: remove pipes but keep text
    s = s.replace(/^\s*\|?/gm, "");
    s = s.replace(/\|/g, "  ");

    // Cleanup excessive whitespace
    s = s.replace(/[ \t]+\n/g, "\n");
    s = s.replace(/\n{3,}/g, "\n\n");
    s = s.trim();

    return s;
  }

  async function typeMarkdown(bubble, md) {
    bubble.classList.add("is-typing");

    if (!window.marked) {
      const plain = stripMarkdown(md);
      bubble.classList.add("is-typing-plain");
      bubble.textContent = "";
      for (let i = 0; i < plain.length; i++) {
        bubble.textContent += plain[i];
        if (i % 10 === 0) scrollToBottom();
        await new Promise((r) => setTimeout(r, randomDelay()));
      }
      bubble.classList.remove("is-typing-plain");
      bubble.classList.remove("is-typing");
      return;
    }

    // Render markdown first, then type into text nodes so formatting is visible
    bubble.innerHTML = marked.parse(md);

    // Hide list markers until their text starts typing
    const listItems = bubble.querySelectorAll("li");
    listItems.forEach((li) => {
      if (li.textContent && li.textContent.trim()) {
        li.classList.add("li-typing");
      }
    });

    const walker = document.createTreeWalker(
      bubble,
      NodeFilter.SHOW_TEXT,
      null,
    );
    const nodes = [];
    let node;
    while ((node = walker.nextNode())) {
      const text = node.textContent || "";
      if (!text.trim()) continue;
      nodes.push({ node, text });
      node.textContent = "";
    }

    for (const { node: n, text } of nodes) {
      const li = n.parentElement ? n.parentElement.closest("li") : null;
      for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        n.textContent += ch;
        if (li && li.classList.contains("li-typing") && ch.trim()) {
          li.classList.remove("li-typing");
        }
        if (i % 8 === 0) scrollToBottom();
        await new Promise((r) => setTimeout(r, randomDelay()));
      }
    }

    bubble.querySelectorAll("li.li-typing").forEach((li) => {
      li.classList.remove("li-typing");
    });

    bubble.classList.remove("is-typing");
    scrollToBottom();
  }

  // =====================
  // Downloads UI
  // =====================
  function addDownloadsUI(downloads) {
    if (!Array.isArray(downloads) || downloads.length === 0) return;

    const container = document.createElement("div");
    container.style.marginTop = "8px";
    container.style.display = "flex";
    container.style.flexWrap = "wrap";
    container.style.gap = "8px";

    downloads.forEach((d) => {
      const card = document.createElement("div");
      card.style.background = "rgba(0,0,0,0.05)";
      card.style.border = "1px solid rgba(0,0,0,0.08)";
      card.style.borderRadius = "14px";
      card.style.padding = "10px";
      card.style.minWidth = "180px";

      const title = document.createElement("div");
      title.style.fontWeight = "700";
      title.style.fontSize = "13px";
      title.textContent = d.title || "Download";

      const a = document.createElement("a");
      a.href = d.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "⬇ Download";
      a.style.display = "inline-flex";
      a.style.marginTop = "8px";
      a.style.padding = "9px 12px";
      a.style.borderRadius = "12px";
      a.style.textDecoration = "none";
      a.style.background = "#1877f2";
      a.style.color = "#fff";
      a.style.fontWeight = "800";
      a.style.fontSize = "12.5px";

      card.appendChild(title);
      card.appendChild(a);
      container.appendChild(card);
    });

    // attach to last incoming bubble
    const lastIncomingBubble = [
      ...messagesEl.querySelectorAll(".msg-row.incoming .bubble"),
    ].pop();
    if (lastIncomingBubble) lastIncomingBubble.appendChild(container);
    scrollToBottom();
  }

  // =====================
  // API calls
  // =====================
  async function sendMessageToAPI(message) {
    const payload = {
      message,
      history: history.slice(-8),
      session_id: sessionId,
    };

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return await res.json();
  }

  async function transcribeAudio(blob) {
    const fd = new FormData();
    fd.append("file", blob, "voice.webm");

    const res = await fetch(STT_URL, { method: "POST", body: fd });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `HTTP ${res.status}`);
    }
    return await res.json(); // {text:"..."}
  }

  // =====================
  // Chat send
  // =====================
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    addTextMessage(text, "outgoing");
    history.push({ role: "user", content: text });

    inputEl.value = "";

    // typing indicator
    const fileish = isFileRequest(text);
    const { row: typingRow, bubble: typingBubble } = addTypingBubble();
    const typingStart = Date.now();
    const minTypingMs = fileish ? 1500 : 1500;
    try {
      const data = await sendMessageToAPI(text);
      const serverDelay =
        data && typeof data.client_delay_ms === "number"
          ? data.client_delay_ms
          : 0;
      const elapsed = Date.now() - typingStart;
      const targetDelay = Math.max(minTypingMs, serverDelay);
      if (elapsed < targetDelay) {
        await new Promise((r) => setTimeout(r, targetDelay - elapsed));
      }

      // remove typing bubble
      typingRow.remove();

      // pick markdown if available
      const md =
        data && typeof data.answer_md === "string" && data.answer_md.trim()
          ? data.answer_md
          : data && data.answer
            ? String(data.answer)
            : "Sorry, I couldn't process that.";

      // create bot bubble and progressive render
      const { bubble } = addMessageRow("incoming");
      await typeMarkdown(bubble, md);

      // store plain answer in history (not HTML)
      const plainAnswer =
        data && data.answer ? String(data.answer) : stripMarkdown(md);
      history.push({ role: "assistant", content: plainAnswer });

      // downloads
      if (data && Array.isArray(data.downloads) && data.downloads.length > 0) {
        addDownloadsUI(data.downloads);
      }
    } catch (err) {
      typingRow.remove();
      addTextMessage(
        "Sorry, I’m having trouble connecting right now. Please try again.",
        "incoming",
      );
      console.error(err);
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  // =====================
  // Recording UI + Audio Meter (same as your design code)
  // =====================
  let isRecording = false;
  let timerInterval = null;
  let startTime = 0;
  let audioContext = null;
  let analyser = null;
  let dataArray = null;
  let rafId = null;
  let mediaStream = null;
  const bars = Array.from(ROOT.querySelectorAll(".sound-bars span"));

  // real recording (MediaRecorder) -> STT backend
  let recorder = null;
  let chunks = [];
  let recordingCanceled = false;

  function formatTime(ms) {
    const total = Math.floor(ms / 1000);
    const m = String(Math.floor(total / 60)).padStart(2, "0");
    const s = String(total % 60).padStart(2, "0");
    return `${m}:${s}`;
  }

  function startTimer() {
    startTime = Date.now();
    recTimerEl.textContent = "00:00";
    timerInterval = setInterval(() => {
      recTimerEl.textContent = formatTime(Date.now() - startTime);
    }, 250);
  }

  function stopTimer() {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  function setComposerMode(mode) {
    const recording = mode === "recording";
    composerNormal.classList.toggle("hidden", recording);
    composerRecording.classList.toggle("hidden", !recording);
    if (!recording) setTimeout(() => inputEl.focus(), 30);
  }

  async function startAudioMeter() {
    if (!navigator.mediaDevices?.getUserMedia || !bars.length) return;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;

      const source = audioContext.createMediaStreamSource(mediaStream);
      source.connect(analyser);

      dataArray = new Uint8Array(analyser.frequencyBinCount);

      const update = () => {
        analyser.getByteFrequencyData(dataArray);
        const step = Math.floor(dataArray.length / bars.length) || 1;

        for (let i = 0; i < bars.length; i++) {
          const idx = i * step;
          const v = dataArray[idx] || 0;
          const scale = 0.5 + (v / 255) * 2.0;
          bars[i].style.transform = `scaleY(${scale})`;
          bars[i].style.opacity = `${0.55 + (v / 255) * 0.45}`;
        }

        rafId = requestAnimationFrame(update);
      };
      update();
    } catch (err) {
      showHint("Mic access blocked. Audio bars will stay idle.");
    }
  }

  function stopAudioMeter() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;

    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }

    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }

    analyser = null;
    dataArray = null;

    for (const bar of bars) {
      bar.style.transform = "";
      bar.style.opacity = "";
    }
  }

  async function beginRecording() {
    if (isRecording) return;

    isRecording = true;
    recordingCanceled = false;
    setComposerMode("recording");
    micBtn.classList.add("recording");
    showHint("");
    startTimer();

    // start meter + recorder
    await startAudioMeter();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        // stop tracks
        stream.getTracks().forEach((t) => t.stop());

        const blob = new Blob(chunks, { type: "audio/webm" });

        if (recordingCanceled) {
          chunks = [];
          showHint("");
          return;
        }

        // transcribing hint
        showHint("Transcribing…");

        try {
          const result = await transcribeAudio(blob);
          const text = (result.text || "").trim();
          showHint("");

          if (text) {
            inputEl.value = text;
            // auto-send like your design? (optional)
            // sendMessage();
          } else {
            showHint("Couldn’t hear clearly.");
            setTimeout(() => showHint(""), 1200);
          }
        } catch (e) {
          showHint("Transcription failed.");
          setTimeout(() => showHint(""), 1400);
        }
      };

      recorder.start();
    } catch (err) {
      showHint("Mic permission denied.");
      endRecording(true);
    }
  }

  function endRecording(cancel = false) {
    if (!isRecording) return;

    isRecording = false;
    recordingCanceled = cancel;
    micBtn.classList.remove("recording");
    stopTimer();
    stopAudioMeter();
    setComposerMode("normal");

    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {}
    }

    if (cancel) {
      inputEl.value = "";
      showHint("Recording cancelled.");
      setTimeout(() => showHint(""), 900);
    }
  }

  micBtn.addEventListener("click", () => {
    if (composerNormal && composerNormal.classList.contains("hinting")) {
      showHint("");
      return;
    }
    beginRecording();
  });
  recCancelBtn.addEventListener("click", () => endRecording(true));
  recSendBtn.addEventListener("click", () => {
    // stop recording -> transcription fills input -> then send
    endRecording(false);
    setTimeout(() => {
      if (inputEl.value.trim()) sendMessage();
    }, 50);
  });

  // =====================
  // Open/close events
  // =====================
  fab.addEventListener("click", () => {
    const isOpen = !panel.classList.contains("hidden");
    togglePanel(!isOpen);
  });
  closeBtn.addEventListener("click", () => togglePanel(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") togglePanel(false);
  });

  if (teaserClose && teaser) {
    teaserClose.addEventListener("click", () => {
      teaser.classList.add("hidden");
    });
  }
  }

  window.BILChatbotInit = initChatbot;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initChatbot());
  } else {
    initChatbot();
  }
})();
