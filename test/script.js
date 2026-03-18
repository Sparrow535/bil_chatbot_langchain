document.addEventListener("DOMContentLoaded", () => {
  const fab = document.getElementById("chat-fab");
  const panel = document.getElementById("chat-panel");
  const closeBtn = document.getElementById("chat-close");

  const messagesEl = document.getElementById("chat-messages");
  const inputEl = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");

  const micBtn = document.getElementById("mic-btn");
  const hintEl = document.getElementById("hint");

  const composerNormal = document.getElementById("composer-normal");
  const composerRecording = document.getElementById("composer-recording");

  const recTimerEl = document.getElementById("rec-timer");
  const recCancelBtn = document.getElementById("rec-cancel");
  const recSendBtn = document.getElementById("rec-send");

  // ✅ quick validation (THIS will tell you immediately if something is null)
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

  let recognition = null;
  let isRecording = false;
  let timerInterval = null;
  let startTime = 0;
  let audioContext = null;
  let analyser = null;
  let dataArray = null;
  let rafId = null;
  let mediaStream = null;
  const bars = Array.from(document.querySelectorAll(".sound-bars span"));

  // ---------- UI helpers ----------
  function togglePanel(show) {
    panel.classList.toggle("hidden", !show);
    if (show) setTimeout(() => inputEl.focus(), 50);
  }

  function addMessage(text, type = "incoming") {
    const row = document.createElement("div");
    row.className = `msg-row ${type}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function botReply(userText) {
    setTimeout(() => {
      addMessage(
        `You said: "${userText}"\n(Connect to your AI API for real replies.)`,
        "incoming",
      );
    }, 500);
  }

  function showHint(msg) {
    hintEl.textContent = msg || "";
  }

  // ---------- Chat ----------
  function normalizeOutgoingText(value) {
    if (typeof value === "string") return value.trim();
    if (
      value &&
      typeof value === "object" &&
      (typeof value.preventDefault === "function" || typeof value.type === "string")
    ) {
      try {
        value.preventDefault();
      } catch {}
      return inputEl.value.trim();
    }
    return inputEl.value.trim();
  }

  function sendMessage(messageOverride = "") {
    const text = normalizeOutgoingText(messageOverride);
    if (!text) return;
    addMessage(text, "outgoing");
    inputEl.value = "";
    botReply(text);
  }

  sendBtn.addEventListener("click", (e) => sendMessage(e));
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  });

  // ---------- Open/close ----------
  fab.addEventListener("click", () => togglePanel(true));
  closeBtn.addEventListener("click", () => togglePanel(false));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") togglePanel(false);
  });

  // ---------- Recording UI helpers ----------
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

  // ---------- Speech Recognition (optional) ----------
  function setupRecognition() {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return null;

    const r = new SpeechRecognition();
    r.lang = "en-US";
    r.interimResults = true;
    r.continuous = true;
    return r;
  }

  function beginRecording() {
    if (isRecording) return;

    // ✅ DEBUG: if this logs, your click is working
    console.log("beginRecording() fired ✅");

    recognition = recognition || setupRecognition();
    if (!recognition) {
      showHint(
        "Voice transcription not supported here, but recording UI works.",
      );
    } else {
      showHint("");
    }

    isRecording = true;

    // ✅ THIS is what makes the animation visible
    micBtn.classList.add("recording");
    setComposerMode("recording");
    startTimer();
    startAudioMeter();

    if (!recognition) return;

    let finalTranscript = "";
    let interim = "";

    recognition.onresult = (event) => {
      interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTranscript += t + " ";
        else interim += t;
      }
      inputEl.value = (finalTranscript + interim).trim();
    };

    recognition.onerror = (e) => {
      showHint(`Mic error: ${e.error}`);
      // ✅ keep UI; user can cancel manually
    };

    recognition.onend = () => {
      // ✅ Don't auto-cancel the UI. Keep recording UI visible until user presses Cancel/Send.
      // SpeechRecognition often stops immediately on some browsers, causing UI to revert instantly.
      if (isRecording) {
        showHint("Mic stopped by browser. Use Cancel or Send.");
      }
    };

    try {
      recognition.start();
    } catch {}
  }

  function endRecording(cancel = false) {
    if (!isRecording) return;

    isRecording = false;
    micBtn.classList.remove("recording");
    stopTimer();
    stopAudioMeter();
    setComposerMode("normal");

    if (recognition) {
      try {
        recognition.stop();
      } catch {}
    }

    if (cancel) {
      inputEl.value = "";
      showHint("Recording cancelled.");
    } else {
      showHint("Recording finished. Tap send to send the transcript.");
    }
  }

  // ---------- Recording buttons ----------
  micBtn.addEventListener("click", beginRecording);
  recCancelBtn.addEventListener("click", () => endRecording(true));
  recSendBtn.addEventListener("click", () => {
    endRecording(false);
    if (inputEl.value.trim()) sendMessage();
  });

  addMessage("Hi! Tap the mic to see the recording animation 🙂", "incoming");

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
          const scale = 0.4 + (v / 255) * 2.2;
          bars[i].style.transform = `scaleY(${scale})`;
          bars[i].style.opacity = `${0.5 + (v / 255) * 0.5}`;
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
});
