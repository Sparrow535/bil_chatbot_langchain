(() => {
  let __chatbotInitialized = false;

  function initChatbot() {
    if (__chatbotInitialized) return;
    // =====================
    // CONFIG
    // =====================
    const GLOBAL_CONFIG = window.BILChatbotConfig || {};
    const ROOT = GLOBAL_CONFIG.root || document;
    const byId = (id) =>
      ROOT.getElementById
        ? ROOT.getElementById(id)
        : ROOT.querySelector(`#${id}`);
    const API_BASE = GLOBAL_CONFIG.apiBase || "";
    const ASSET_BASE = (GLOBAL_CONFIG.assetBase || "").replace(/\/$/, "");
    const API_URL =
      GLOBAL_CONFIG.apiUrl || (API_BASE ? `${API_BASE}/chat` : "/chat");
    const STT_URL =
      GLOBAL_CONFIG.sttUrl || (API_BASE ? `${API_BASE}/stt` : "/stt");
    const GREETING_URL =
      GLOBAL_CONFIG.greetingUrl ||
      (API_BASE ? `${API_BASE}/greeting` : "/greeting");
    const BOT_LOGO =
      GLOBAL_CONFIG.botLogo ||
      (ASSET_BASE ? `${ASSET_BASE}/assets/logo.svg` : "./assets/logo.svg");
    const USER_AVATAR =
      GLOBAL_CONFIG.userAvatar ||
      (ASSET_BASE
        ? `${ASSET_BASE}/assets/user-avatar.svg`
        : "./assets/user-avatar.svg");

    // Typewriter speed
    const TYPE_MIN_DELAY = 6;
    const TYPE_MAX_DELAY = 10;

    // =====================
    // DOM
    // =====================
    const widget = byId("chat-widget");
    const fab = byId("chat-fab");
    const panel = byId("chat-panel");
    const closeBtn = byId("chat-close");
    const teaser = byId("chat-teaser");
    const teaserClose = byId("teaser-close");
    const teaserTitle = byId("teaser-title");
    const teaserSubtitle = byId("teaser-subtitle");
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

    // Validate required nodes (retry if not ready yet)
    const required = {
      widget,
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
    const missing = Object.entries(required).filter(([, v]) => !v);
    if (missing.length) {
      missing.forEach(([k]) => console.warn(`Chatbot: missing element ${k}`));
      setTimeout(initChatbot, 200);
      return;
    }
    __chatbotInitialized = true;

    // =====================
    // SESSION + HISTORY
    // =====================
    let history = []; // [{role:"user"|"assistant", content:"...", followup_query?:"..."}]

    const SESSION_TIMEOUT_MS = 30 * 60 * 1000;
    const SESSION_ID_KEY = "bil_session_id";
    const SESSION_LAST_SEEN_KEY = "bil_session_last_seen";

    function createSessionId() {
      if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return window.crypto.randomUUID();
      }
      return `bil-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    }

    function persistSessionId(nextId) {
      sessionId = nextId;
      localStorage.setItem(SESSION_ID_KEY, nextId);
    }

    function touchSessionActivity() {
      localStorage.setItem(SESSION_LAST_SEEN_KEY, String(Date.now()));
    }

    function hasSessionExpired() {
      const raw = localStorage.getItem(SESSION_LAST_SEEN_KEY);
      if (!raw) return false;
      const lastSeen = Number(raw);
      if (!Number.isFinite(lastSeen) || lastSeen <= 0) return false;
      return Date.now() - lastSeen > SESSION_TIMEOUT_MS;
    }

    function resetSessionState({ clearThread = false } = {}) {
      history = [];
      introGreetingInFlight = false;
      persistSessionId(createSessionId());
      touchSessionActivity();
      if (clearThread && messagesEl) {
        messagesEl.innerHTML = "";
      }
      if (statusEl) statusEl.textContent = "Online";
      showHint("");
      setAssistantBusy(false);
    }

    function ensureActiveSession({ clearThread = false } = {}) {
      if (hasSessionExpired()) {
        resetSessionState({ clearThread });
        return true;
      }
      touchSessionActivity();
      return false;
    }

    let sessionId = localStorage.getItem(SESSION_ID_KEY);
    if (!sessionId) {
      persistSessionId(createSessionId());
    }
    if (!localStorage.getItem(SESSION_LAST_SEEN_KEY)) {
      touchSessionActivity();
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
    let lastTeaserLine = "";
    let introGreetingInFlight = false;
    let isAssistantBusy = false;

    function syncComposerControls() {
      const isHinting = Boolean(
        composerNormal && composerNormal.classList.contains("hinting"),
      );
      const isTranscribing = Boolean(
        composerNormal && composerNormal.classList.contains("transcribing"),
      );

      if (sendBtn) {
        sendBtn.disabled = isAssistantBusy || isHinting;
        sendBtn.setAttribute("aria-hidden", isTranscribing ? "true" : "false");
      }

      if (micBtn) {
        micBtn.disabled = isAssistantBusy && !isHinting;
      }
    }

    function setAssistantBusy(next) {
      isAssistantBusy = Boolean(next);
      syncComposerControls();
    }

    function scheduleTeaser() {
      if (!teaser) return;
      if (teaserTimer) clearTimeout(teaserTimer);
      refreshTeaserCopy();
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
        updateKeyboardOffset();
        setTimeout(() => inputEl.focus(), 50);
      } else {
        if (fabIcon) fabIcon.src = FAB_OPEN_ICON;
        if (fab) fab.setAttribute("aria-label", "Open chat");
        if (fab) {
          fab.classList.add("fab-open");
          fab.classList.remove("fab-close");
        }
        forceKeyboardClosed();
        scheduleTeaser();
      }
    }

    // Initialize FAB state
    if (fab) {
      fab.classList.add("fab-open");
      fab.classList.remove("fab-close");
    }
    refreshTeaserCopy();

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function showHint(msg) {
      const text = (msg || "").trim();
      const isTranscribing = /transcrib/i.test(text);
      const isCancelled = /cancel/i.test(text);
      const isErrorLike =
        /no voice|no clear voice|couldn['’]?t|failed|blocked|denied|not supported/i.test(
          text,
        );

      if (!text) {
        if (composerNormal) {
          composerNormal.classList.remove("hinting");
          composerNormal.classList.remove("transcribing");
        }
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
        syncComposerControls();
        return;
      }

      if (isCancelled) {
        if (composerNormal) {
          composerNormal.classList.remove("hinting");
          composerNormal.classList.remove("transcribing");
        }
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
        syncComposerControls();
        return;
      }

      if (composerNormal) {
        composerNormal.classList.add("hinting");
        composerNormal.classList.toggle("transcribing", isTranscribing);
      }
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
      inline.classList.remove(
        "hint-inline--progress",
        "hint-inline--error",
        "hint-inline--info",
      );
      if (isTranscribing) inline.classList.add("hint-inline--progress");
      else if (isErrorLike) inline.classList.add("hint-inline--error");
      else inline.classList.add("hint-inline--info");

      inline.innerHTML = `
      <span>${text}</span>
      ${isTranscribing ? '<span class="hint-spinner" aria-hidden="true"></span>' : ""}
    `;

      if (micBtn) {
        micBtn.innerHTML = "✕";
        micBtn.classList.add("hint-cancel");
      }
      syncComposerControls();
    }

    function getGreeting() {
      const h = new Date().getHours();
      if (h < 12) return "Good morning";
      if (h < 17) return "Good afternoon";
      return "Good evening";
    }

    function getRandomGreeting() {
      return "Hello! I am the official chatbot for Bhutan Insurance Limited (BIL), here to assist you with information about the company and its services, including insurance, credit management (loans), and fund management (provident fund).";
    }

    function pickTeaserLine(pool) {
      const options = Array.isArray(pool) ? pool.filter(Boolean) : [];
      if (!options.length) return "Need help?";
      const filtered = options.filter((line) => line !== lastTeaserLine);
      const nextPool = filtered.length ? filtered : options;
      const next = nextPool[Math.floor(Math.random() * nextPool.length)];
      lastTeaserLine = next;
      return next;
    }

    function getRecentTopicTeaserLines() {
      const lastUser = [...history]
        .reverse()
        .find((m) => m && m.role === "user" && String(m.content || "").trim());
      const q = String((lastUser && lastUser.content) || "").toLowerCase();
      const lines = [];

      if (/loan|loans|housing|personal loan|transport loan/.test(q)) {
        lines.push("Need loan details?", "Need rates or forms?");
      }
      if (/claim|claims|motor claim|fire claim|travel claim/.test(q)) {
        lines.push("Need claim help?", "Need claim forms?");
      }
      if (/travel insurance|motor insurance|fire insurance|insurance/.test(q)) {
        lines.push("Need coverage details?", "Need the right form?");
      }
      if (/annual report|profit|income|earnings|report|20\d{2}/.test(q)) {
        lines.push("Need a report figure?", "Need report highlights?");
      }

      lines.push(
        "Need the next detail?",
        "Need a quick answer?",
        "Need help with this?",
      );
      return lines;
    }

    function refreshTeaserCopy() {
      if (!teaserTitle) return;

      teaserTitle.textContent = "Kuzuzangpo la! How may I assist you today?";

      if (teaserSubtitle) {
        teaserSubtitle.textContent = "";
      }
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

        bubble = document.createElement("div");
        bubble.className = "bubble";

        content.appendChild(bubble);

        row.appendChild(avatar);
        row.appendChild(content);
      } else {
        const wrap = document.createElement("div");
        wrap.className = "msg-outgoing";

        bubble = document.createElement("div");
        bubble.className = "bubble";

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar user-avatar";

        const img = document.createElement("img");
        img.src = USER_AVATAR;
        img.alt = "User";
        avatar.appendChild(img);

        wrap.appendChild(bubble);
        row.appendChild(wrap);
        row.appendChild(avatar);
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

    async function fetchIntroGreeting() {
      const res = await fetch(GREETING_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          history: history.slice(-8),
          session_id: sessionId,
        }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      const data = await res.json();
      touchSessionActivity();
      return data;
    }

    async function ensureIntroMessage() {
      ensureActiveSession({ clearThread: true });
      if (messagesEl.childElementCount > 0 || introGreetingInFlight) return;
      introGreetingInFlight = true;
      setAssistantBusy(true);
      if (statusEl) statusEl.textContent = "Online";

      const { row: typingRow } = addTypingBubble();
      const typingStart = Date.now();
      const fallbackGreeting = getRandomGreeting();
      try {
        const data = await fetchIntroGreeting();
        const serverDelay =
          data && typeof data.client_delay_ms === "number"
            ? data.client_delay_ms
            : 0;
        const elapsed = Date.now() - typingStart;
        const targetDelay = Math.max(700, serverDelay);
        if (elapsed < targetDelay) {
          await new Promise((r) => setTimeout(r, targetDelay - elapsed));
        }

        if (typingRow.isConnected) typingRow.remove();

        const md =
          data && typeof data.answer_md === "string" && data.answer_md.trim()
            ? data.answer_md
            : data && data.answer
              ? String(data.answer)
              : fallbackGreeting;

        const { bubble } = addMessageRow("incoming");
        await typeMarkdown(bubble, md);
      } catch (err) {
        if (typingRow.isConnected) typingRow.remove();
        addTextMessage(fallbackGreeting, "incoming");
        console.error(err);
      } finally {
        introGreetingInFlight = false;
        setAssistantBusy(false);
      }
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

    function readComposerText() {
      return String((inputEl && inputEl.value) || "").trim();
    }

    function isEventLike(value) {
      return !!(
        value &&
        typeof value === "object" &&
        (typeof value.preventDefault === "function" ||
          typeof value.stopPropagation === "function" ||
          (typeof value.type === "string" && "target" in value))
      );
    }

    function normalizeOutgoingText(value) {
      const fallback = readComposerText();

      if (typeof value === "string") {
        const text = value.trim();
        if (/^\[object\s+\w*Event\]$/i.test(text)) return fallback;
        return text || fallback;
      }

      if (isEventLike(value)) {
        try {
          value.preventDefault();
        } catch {}
        try {
          value.stopPropagation();
        } catch {}

        const currentTarget = value.currentTarget;
        if (currentTarget && currentTarget.form) {
          const field = currentTarget.form.querySelector(
            "#chat-input, input[type='text'], textarea",
          );
          if (field && typeof field.value === "string" && field.value.trim()) {
            return field.value.trim();
          }
        }
        return fallback;
      }

      if (value == null) return fallback;

      const text = String(value).trim();
      if (/^\[object\s+\w*Event\]$/i.test(text)) return fallback;
      return text || fallback;
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
      enhanceMarkdownLinkUI(bubble);

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
        const linkChip = n.parentElement
          ? n.parentElement.closest("a.msg-link-chip")
          : null;
        const pendingHelpfulTitle = n.parentElement
          ? n.parentElement.closest(".helpful-links-title-pending")
          : null;
        const pendingHelpfulList = n.parentElement
          ? n.parentElement.closest(".helpful-links-list-pending")
          : null;
        for (let i = 0; i < text.length; i++) {
          const ch = text[i];
          n.textContent += ch;
          if (pendingHelpfulTitle && ch.trim()) {
            pendingHelpfulTitle.classList.remove("helpful-links-title-pending");
          }
          if (pendingHelpfulList && ch.trim()) {
            pendingHelpfulList.classList.remove("helpful-links-list-pending");
          }
          if (
            linkChip &&
            linkChip.classList.contains("link-pending") &&
            (n.textContent || "").trim()
          ) {
            linkChip.classList.remove("link-pending");
          }
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

    function enhanceMarkdownLinkUI(bubble) {
      if (!bubble) return;

      const cleanup = (s) =>
        (s || "")
          .toLowerCase()
          .replace(/\s+/g, " ")
          .replace(/[:\s]+$/, "")
          .trim();

      const markHelpfulLinksList = (titleEl) => {
        if (!titleEl) return;
        const p = titleEl.closest("p") || titleEl;
        p.classList.add("helpful-links-title", "helpful-links-title-pending");
        const next = p.nextElementSibling;
        if (next && (next.tagName === "UL" || next.tagName === "OL")) {
          next.classList.add(
            "helpful-links-list",
            "helpful-links-list-pending",
          );
        }
      };

      bubble
        .querySelectorAll("p, h1, h2, h3, h4, h5, h6, strong, b")
        .forEach((el) => {
          if (cleanup(el.textContent) === "helpful links") {
            markHelpfulLinksList(el);
          }
        });

      bubble.querySelectorAll("a[href]").forEach((a) => {
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.classList.add("msg-link");

        const li = a.closest("li");
        if (!li) return;
        const inHelpfulList = !!li.closest(".helpful-links-list");
        if (inHelpfulList) {
          li.classList.add("msg-link-item");
          a.classList.add("msg-link-chip", "link-pending");
        }
      });
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
      const normalizedMessage = normalizeOutgoingText(message);
      const payload = {
        message: normalizedMessage,
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
      const data = await res.json();
      touchSessionActivity();
      return data;
    }

    async function transcribeAudio(blob) {
      const fd = new FormData();
      const mime = String(blob?.type || "").toLowerCase();
      let ext = "webm";
      if (mime.includes("mp4") || mime.includes("m4a")) ext = "m4a";
      else if (mime.includes("ogg")) ext = "ogg";
      else if (mime.includes("wav")) ext = "wav";
      else if (mime.includes("mpeg") || mime.includes("mp3")) ext = "mp3";
      fd.append("file", blob, `voice.${ext}`);

      const res = await fetch(STT_URL, { method: "POST", body: fd });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      const data = await res.json(); // {text:"..."}
      touchSessionActivity();
      return data;
    }

    // =====================
    // Chat send
    // =====================
    async function sendMessage(messageOverride = "") {
      if (isAssistantBusy || sttInFlight) return;

      const text = normalizeOutgoingText(messageOverride);
      if (!text) return;
      ensureActiveSession({ clearThread: true });
      const usedOverrideText =
        typeof messageOverride === "string" && messageOverride.trim() === text;

      if (composerNormal && composerNormal.classList.contains("hinting")) {
        showHint("");
      }

      setAssistantBusy(true);
      addTextMessage(text, "outgoing");
      history.push({ role: "user", content: text });

      if (!usedOverrideText) {
        inputEl.value = "";
      }

      // typing indicator
      const fileish = isFileRequest(text);
      const { row: typingRow } = addTypingBubble();
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

        if (typingRow.isConnected) typingRow.remove();

        const md =
          data && typeof data.answer_md === "string" && data.answer_md.trim()
            ? data.answer_md
            : data && data.answer
              ? String(data.answer)
              : "Sorry, I couldn't process that.";

        const { bubble } = addMessageRow("incoming");
        await typeMarkdown(bubble, md);

        const plainAnswer =
          data && data.answer ? String(data.answer) : stripMarkdown(md);
        const followupQuery =
          data && typeof data.followup_query === "string"
            ? data.followup_query.trim()
            : "";
        history.push({
          role: "assistant",
          content: plainAnswer,
          followup_query: followupQuery || undefined,
        });

        if (
          data &&
          Array.isArray(data.downloads) &&
          data.downloads.length > 0
        ) {
          addDownloadsUI(data.downloads);
        }
      } catch (err) {
        if (typingRow.isConnected) typingRow.remove();
        addTextMessage(
          "Sorry, I’m having trouble connecting right now. Please try again.",
          "incoming",
        );
        console.error(err);
      } finally {
        setAssistantBusy(false);
      }
    }

    sendBtn.addEventListener("click", (e) => sendMessage(e));
    sendBtn.addEventListener("pointerup", (e) => {
      e.preventDefault();
    });
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();
      }
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
    let timeDataArray = null;
    let meterSink = null;
    let meterMaxRms = 0;
    let meterRmsSum = 0;
    let meterRmsCount = 0;
    let meterSignalFrames = 0;
    let rafId = null;
    let mediaStream = null;
    let bars = Array.from(ROOT.querySelectorAll(".sound-bars span"));

    // real recording (MediaRecorder) -> STT backend
    let recorder = null;
    let recorderStream = null;
    let chunks = [];
    let recordingCanceled = false;
    let autoSendAfterTranscribe = false;
    let sttInFlight = null;
    let recordingMimeType = "audio/webm";
    let holdToTalkRecording = false;
    let micPressTimer = null;
    let micPressPointerId = null;
    let micPressActive = false;
    let micLongPressStarted = false;
    const MIC_LONG_PRESS_MS = 280;
    const VAD_MIN_MAX_RMS = 0.0085;
    const VAD_MIN_AVG_RMS = 0.0025;
    const VAD_MIN_SIGNAL_FRAMES = 5;
    const VAD_MIN_SIGNAL_RATIO = 0.045;

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

    function setComposerMode(mode, opts = {}) {
      const recording = mode === "recording";
      const holdMode = recording && Boolean(opts.holdToTalk);
      composerNormal.classList.toggle("hidden", recording);
      composerRecording.classList.toggle("hidden", !recording);
      composerRecording.classList.toggle("hold-mode", holdMode);
      recCancelBtn.classList.toggle("hidden", holdMode);
      recSendBtn.classList.toggle("hidden", holdMode);
      if (!holdMode) {
        recSendBtn.textContent = "■";
        recSendBtn.setAttribute("aria-label", "Stop recording");
        recSendBtn.setAttribute("title", "Stop recording");
      }
      if (!recording) setTimeout(() => inputEl.focus(), 30);
    }

    function pickRecordingMimeType() {
      if (
        !window.MediaRecorder ||
        typeof MediaRecorder.isTypeSupported !== "function"
      ) {
        return "";
      }
      const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
      ];
      for (const c of candidates) {
        if (MediaRecorder.isTypeSupported(c)) return c;
      }
      return "";
    }

    function computeRms(arr) {
      if (!arr || !arr.length) return 0;
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        const n = (arr[i] - 128) / 128;
        sum += n * n;
      }
      return Math.sqrt(sum / arr.length);
    }

    async function startAudioMeter(stream) {
      if (!stream) return;
      try {
        mediaStream = stream;
        bars = Array.from(ROOT.querySelectorAll(".sound-bars span"));
        if (!bars.length) return;

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === "suspended") {
          try {
            await audioContext.resume();
          } catch {}
        }
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.72;
        analyser.minDecibels = -92;
        analyser.maxDecibels = -16;

        const source = audioContext.createMediaStreamSource(mediaStream);
        source.connect(analyser);
        // Keep the WebAudio graph active on browsers that pause unconnected analyzers.
        meterSink = audioContext.createGain();
        meterSink.gain.value = 0;
        analyser.connect(meterSink);
        meterSink.connect(audioContext.destination);

        dataArray = new Uint8Array(analyser.frequencyBinCount);
        timeDataArray = new Uint8Array(analyser.fftSize);
        meterMaxRms = 0;
        meterRmsSum = 0;
        meterRmsCount = 0;
        meterSignalFrames = 0;

        const update = () => {
          if (!analyser || !dataArray || !timeDataArray) return;
          analyser.getByteFrequencyData(dataArray);
          analyser.getByteTimeDomainData(timeDataArray);
          const rms = computeRms(timeDataArray);
          meterMaxRms = Math.max(meterMaxRms, rms);
          meterRmsSum += rms;
          meterRmsCount += 1;
          if (rms > 0.0075) meterSignalFrames += 1;
          const rmsBoost = Math.min(1, rms * 30);
          const step = Math.floor(dataArray.length / bars.length) || 1;
          const tStep = Math.floor(timeDataArray.length / bars.length) || 1;

          for (let i = 0; i < bars.length; i++) {
            const from = i * step;
            const to = Math.min(dataArray.length, from + step);
            let bucket = 0;
            for (let j = from; j < to; j++) bucket += dataArray[j] || 0;
            const bucketAvg = to > from ? bucket / (to - from) : 0;
            const vFreq = bucketAvg / 255;
            const tFrom = i * tStep;
            const tTo = Math.min(timeDataArray.length, tFrom + tStep);
            let absSum = 0;
            for (let j = tFrom; j < tTo; j++) {
              absSum += Math.abs((timeDataArray[j] - 128) / 128);
            }
            const vTime =
              tTo > tFrom ? Math.min(1, (absSum / (tTo - tFrom)) * 5.8) : 0;
            const v = Math.max(
              vFreq,
              vTime,
              rmsBoost * (0.72 + (i % 3) * 0.08),
            );
            const scale = 0.5 + v * 2.25;
            bars[i].style.transform = `scaleY(${scale})`;
            bars[i].style.opacity = `${0.5 + v * 0.5}`;
          }

          rafId = requestAnimationFrame(update);
        };
        update();
      } catch (err) {
        console.error("Audio meter error:", err);
        showHint("Mic access blocked. Audio bars will stay idle.");
      }
    }

    function stopAudioMeter() {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;

      try {
        if (analyser) analyser.disconnect();
      } catch {}
      try {
        if (meterSink) meterSink.disconnect();
      } catch {}
      meterSink = null;

      if (audioContext) {
        audioContext.close();
        audioContext = null;
      }

      analyser = null;
      dataArray = null;
      timeDataArray = null;
      meterRmsSum = 0;
      meterRmsCount = 0;
      meterSignalFrames = 0;

      for (const bar of bars) {
        bar.style.transform = "";
        bar.style.opacity = "";
      }
    }

    function stopRecorderStreamTracks() {
      if (recorderStream) {
        recorderStream.getTracks().forEach((t) => t.stop());
        recorderStream = null;
      }
      mediaStream = null;
    }

    async function beginRecording(opts = {}) {
      if (isRecording) return;
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        showHint("Voice recording is not supported on this device.");
        setTimeout(() => showHint(""), 1400);
        return;
      }

      isRecording = true;
      recordingCanceled = false;
      autoSendAfterTranscribe = false;
      recordingMimeType = "audio/webm";
      meterMaxRms = 0;
      meterRmsSum = 0;
      meterRmsCount = 0;
      meterSignalFrames = 0;
      holdToTalkRecording = Boolean(opts.holdToTalk);
      setComposerMode("recording", { holdToTalk: holdToTalkRecording });
      micBtn.classList.add("recording");
      showHint("");
      startTimer();

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
          },
        });
        recorderStream = stream;
        chunks = [];
        recordingMimeType = pickRecordingMimeType() || "audio/webm";

        try {
          if (recordingMimeType) {
            recorder = new MediaRecorder(stream, {
              mimeType: recordingMimeType,
              audioBitsPerSecond: 128000,
            });
          } else {
            recorder = new MediaRecorder(stream);
          }
        } catch {
          recorder = new MediaRecorder(stream);
          recordingMimeType = recorder.mimeType || "audio/webm";
        }

        await startAudioMeter(stream);

        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) chunks.push(e.data);
        };

        recorder.onstop = async () => {
          stopRecorderStreamTracks();
          const blob = new Blob(chunks, {
            type: recordingMimeType || "audio/webm",
          });
          const durationMs = Math.max(0, Date.now() - startTime);

          if (recordingCanceled) {
            chunks = [];
            showHint("");
            return;
          }
          if (!blob.size || blob.size < 300) {
            showHint("No voice detected.");
            setTimeout(() => showHint(""), 1200);
            return;
          }
          const avgRms = meterRmsCount ? meterRmsSum / meterRmsCount : 0;
          const signalRatio = meterRmsCount
            ? meterSignalFrames / meterRmsCount
            : 0;
          const hasSpeechEnergy =
            meterMaxRms >= VAD_MIN_MAX_RMS ||
            avgRms >= VAD_MIN_AVG_RMS ||
            (meterSignalFrames >= VAD_MIN_SIGNAL_FRAMES &&
              signalRatio >= VAD_MIN_SIGNAL_RATIO);
          if (!hasSpeechEnergy || (durationMs < 350 && meterMaxRms < 0.012)) {
            showHint(
              "No clear voice detected. Please check mic and try again.",
            );
            setTimeout(() => showHint(""), 1500);
            return;
          }

          // transcribing hint
          showHint("Transcribing…");

          try {
            sttInFlight = transcribeAudio(blob);
            const result = await sttInFlight;
            const text = String(result.text || "").trim();
            showHint("");
            sttInFlight = null;

            if (!text) {
              showHint("Couldn’t hear clearly.");
              setTimeout(() => showHint(""), 1200);
              return;
            }

            // Guard against silence hallucinations like one-word outputs from near-silent audio.
            const wordCount = text.split(/\s+/).filter(Boolean).length;
            const probablySilenceHallucination =
              wordCount <= 2 && signalRatio < 0.03 && meterMaxRms < 0.0115;
            if (probablySilenceHallucination) {
              showHint("No clear voice detected. Please try again.");
              setTimeout(() => showHint(""), 1400);
              return;
            }

            if (autoSendAfterTranscribe) {
              await new Promise((r) => setTimeout(r, 80));
              await sendMessage(text);
            } else {
              inputEl.value = text;
            }
          } catch (e) {
            sttInFlight = null;
            console.error("Transcription failed:", e);
            showHint("Transcription failed.");
            setTimeout(() => showHint(""), 1400);
          }
        };

        recorder.start(250);
      } catch (err) {
        console.error("Recording setup failed:", err);
        showHint("Mic permission denied.");
        endRecording(true);
      }
    }

    function endRecording(cancel = false, opts = {}) {
      if (!isRecording) return;
      const autoSend = Boolean(opts && opts.autoSend);

      isRecording = false;
      recordingCanceled = cancel;
      autoSendAfterTranscribe = !cancel && autoSend;
      micBtn.classList.remove("recording");
      stopTimer();
      stopAudioMeter();
      setComposerMode("normal");
      holdToTalkRecording = false;

      if (recorder && recorder.state !== "inactive") {
        try {
          recorder.stop();
        } catch {}
      } else {
        stopRecorderStreamTracks();
      }

      if (cancel) {
        inputEl.value = "";
        showHint("Recording cancelled.");
        setTimeout(() => showHint(""), 900);
      }
    }

    function clearMicPressTimer() {
      if (micPressTimer) {
        clearTimeout(micPressTimer);
        micPressTimer = null;
      }
    }

    function clearMicPressState() {
      clearMicPressTimer();
      micPressActive = false;
      micPressPointerId = null;
      micLongPressStarted = false;
    }

    function finishMicPress(pointerId) {
      if (isAssistantBusy) return;
      if (!micPressActive) return;
      if (
        micPressPointerId !== null &&
        typeof pointerId === "number" &&
        pointerId !== micPressPointerId
      ) {
        return;
      }

      const wasLongPress = micLongPressStarted;
      clearMicPressState();

      if (wasLongPress) {
        if (isRecording) endRecording(false, { autoSend: true });
        return;
      }

      if (composerNormal && composerNormal.classList.contains("hinting")) {
        showHint("");
        return;
      }

      if (!isRecording && !sttInFlight) {
        beginRecording({ holdToTalk: false });
      }
    }

    function startMicPress(pointerId = null) {
      if (isAssistantBusy || isRecording || sttInFlight || micPressActive)
        return;

      micPressActive = true;
      micPressPointerId = typeof pointerId === "number" ? pointerId : null;
      micLongPressStarted = false;
      clearMicPressTimer();
      micPressTimer = setTimeout(() => {
        if (!micPressActive || isRecording || sttInFlight) return;
        micLongPressStarted = true;
        beginRecording({ holdToTalk: true });
      }, MIC_LONG_PRESS_MS);
    }

    if (window.PointerEvent) {
      micBtn.addEventListener("pointerdown", (e) => {
        if (e.pointerType === "mouse" && e.button !== 0) return;
        startMicPress(typeof e.pointerId === "number" ? e.pointerId : null);
      });
      window.addEventListener("pointerup", (e) => {
        finishMicPress(typeof e.pointerId === "number" ? e.pointerId : null);
      });
      window.addEventListener("pointercancel", (e) => {
        if (micLongPressStarted && isRecording) endRecording(true);
        finishMicPress(typeof e.pointerId === "number" ? e.pointerId : null);
      });
    } else {
      // Fallback for older browsers.
      micBtn.addEventListener("touchstart", () => startMicPress(null), {
        passive: true,
      });
      window.addEventListener("touchend", () => finishMicPress(null), {
        passive: true,
      });
      window.addEventListener("touchcancel", () => {
        if (micLongPressStarted && isRecording) endRecording(true);
        finishMicPress(null);
      });
      micBtn.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        startMicPress(null);
      });
      window.addEventListener("mouseup", () => finishMicPress(null));
    }

    // Keyboard accessibility for the mic button.
    micBtn.addEventListener("keydown", (e) => {
      if (isAssistantBusy) return;
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      if (composerNormal && composerNormal.classList.contains("hinting")) {
        showHint("");
        return;
      }
      if (!isRecording && !sttInFlight) beginRecording({ holdToTalk: false });
    });

    syncComposerControls();

    recCancelBtn.addEventListener("click", () => endRecording(true));
    recSendBtn.addEventListener("click", () => {
      // stop recording, then auto-send after transcription finishes
      endRecording(false, { autoSend: true });
    });

    // =====================
    // Mobile keyboard handling (attach panel to keyboard)
    // =====================
    const isIOS = /iP(ad|hone|od)/.test(navigator.userAgent);
    let baseViewportHeight = 0;
    let baseInnerHeight = 0;
    let baseDocHeight = 0;
    let kbCloseTimer = null;
    const KB_THRESHOLD = 60;

    function updateBaseHeights(force = false) {
      const vv = window.visualViewport;
      const currentVVH = vv ? vv.height : 0;
      const currentInner = window.innerHeight || 0;
      const currentDoc = document.documentElement
        ? document.documentElement.clientHeight || 0
        : 0;

      if (force || !baseViewportHeight)
        baseViewportHeight = currentVVH || currentInner;
      if (force || !baseInnerHeight) baseInnerHeight = currentInner;
      if (force || !baseDocHeight) baseDocHeight = currentDoc;

      // If keyboard is closed, refresh baselines to handle orientation changes
      if (!widget || !widget.classList.contains("kb-open")) {
        if (currentVVH) baseViewportHeight = currentVVH;
        if (currentInner) baseInnerHeight = currentInner;
        if (currentDoc) baseDocHeight = currentDoc;
      }
    }

    function updateKeyboardOffset() {
      if (!widget) return;
      updateBaseHeights();

      let offset = 0;
      if (window.visualViewport) {
        const vv = window.visualViewport;
        const currentVVH = vv.height || window.innerHeight;
        const base = baseViewportHeight || currentVVH || window.innerHeight;
        const delta = Math.max(0, base - currentVVH);
        const safeDelta = Math.max(
          0,
          window.innerHeight - (currentVVH || 0) - (vv.offsetTop || 0),
        );
        const currentDoc = document.documentElement
          ? document.documentElement.clientHeight || 0
          : 0;
        const deltaDoc = Math.max(
          0,
          (baseDocHeight || currentDoc) - currentDoc,
        );
        offset = Math.max(delta, safeDelta, deltaDoc);

        // iOS sometimes reports small deltas; apply threshold
        if (isIOS && offset < KB_THRESHOLD && delta > KB_THRESHOLD) {
          offset = delta;
        }
      } else {
        const base = baseInnerHeight || window.innerHeight;
        const currentDoc = document.documentElement
          ? document.documentElement.clientHeight || 0
          : 0;
        const deltaDoc = Math.max(
          0,
          (baseDocHeight || currentDoc) - currentDoc,
        );
        offset = Math.max(0, base - window.innerHeight, deltaDoc);
      }

      widget.style.setProperty("--kb-offset", `${offset}px`);
      if (offset > KB_THRESHOLD) {
        if (kbCloseTimer) {
          clearTimeout(kbCloseTimer);
          kbCloseTimer = null;
        }
        widget.classList.add("kb-open");
      } else {
        if (kbCloseTimer) clearTimeout(kbCloseTimer);
        kbCloseTimer = setTimeout(() => {
          widget.classList.remove("kb-open");
        }, 180);
      }
    }

    function forceKeyboardClosed() {
      if (!widget) return;
      widget.style.setProperty("--kb-offset", "0px");
      if (kbCloseTimer) clearTimeout(kbCloseTimer);
      kbCloseTimer = setTimeout(() => {
        widget.classList.remove("kb-open");
      }, 180);
    }

    updateBaseHeights(true);
    if (window.visualViewport) {
      window.visualViewport.addEventListener("resize", updateKeyboardOffset);
      window.visualViewport.addEventListener("scroll", updateKeyboardOffset);
    }
    window.addEventListener("resize", updateKeyboardOffset);
    document.addEventListener("focusin", updateKeyboardOffset);
    document.addEventListener("focusout", () => {
      updateKeyboardOffset();
      setTimeout(() => {
        if (!widget) return;
        if (!widget.contains(document.activeElement)) {
          forceKeyboardClosed();
        }
      }, 120);
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
