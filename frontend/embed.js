(() => {
  const scriptTag = document.currentScript;
  const existing = document.getElementById("chat-widget");
  if (existing) return;

  const fromAttr = (name, fallback = "") =>
    (scriptTag && scriptTag.getAttribute(name)) || fallback;

  const src = scriptTag && scriptTag.src ? new URL(scriptTag.src) : null;
  const baseFromScript = src ? `${src.origin}${src.pathname.replace(/\/[^/]+$/, "")}` : "";

  const assetBase = fromAttr("data-asset-base", baseFromScript);
  const apiBase = fromAttr("data-api-base", baseFromScript);

  window.BILChatbotConfig = {
    ...(window.BILChatbotConfig || {}),
    apiBase,
    assetBase,
    botName: fromAttr("data-bot-name", "Norbu"),
    botLogo: fromAttr("data-bot-logo", assetBase ? `${assetBase}/assets/logo.svg` : "./assets/logo.svg"),
  };

  const ensureCss = () => {
    const href = assetBase ? `${assetBase}/styles.css` : "./styles.css";
    const exists = Array.from(document.styleSheets || []).some((s) => s.href === href);
    if (exists) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  };

  const ensureMarked = () =>
    new Promise((resolve) => {
      if (window.marked) return resolve();
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
      s.onload = () => resolve();
      document.head.appendChild(s);
    });

  const ensureScripts = () =>
    new Promise((resolve) => {
      if (window.BILChatbotInit) return resolve();
      const s = document.createElement("script");
      s.src = assetBase ? `${assetBase}/scripts.js` : "./scripts.js";
      s.defer = true;
      s.onload = () => resolve();
      document.body.appendChild(s);
    });

  const injectHtml = () => {
    const widget = document.createElement("div");
    widget.id = "chat-widget";
    widget.innerHTML = `
      <div id="chat-teaser" class="chat-teaser" role="status" aria-live="polite">
        <button id="teaser-close" class="teaser-close" aria-label="Dismiss">×</button>
        <div class="teaser-row">
          <div class="teaser-avatar">
            <img src="${window.BILChatbotConfig.botLogo}" alt="BIL logo">
          </div>
          <div class="teaser-content">
            <div class="teaser-title">Hi! I’m Norbu. How can I help you today?</div>
            <div class="teaser-subtitle">Norbu · AI Agent · Just now</div>
          </div>
        </div>
      </div>

      <button id="chat-fab" aria-label="Open chat">
        <img id="fab-icon" class="fab-image" src="${assetBase ? `${assetBase}/assets/popup.svg` : "./assets/popup.svg"}" alt="Open chat">
      </button>

      <section id="chat-panel" class="hidden" role="dialog" aria-label="AI Chatbot">
        <header class="chat-header">
          <div class="bot-id">
            <div class="bot-logo"><img src="${window.BILChatbotConfig.botLogo}" alt="logo"></div>
            <div class="bot-meta">
              <div class="bot-name">NORBU <span>AI Assistant</span></div>
              <div class="bot-status" id="bot-status">Online</div>
            </div>
          </div>
          <button id="chat-close" class="icon-btn" aria-label="Close chat">✕</button>
        </header>

        <main id="chat-messages" class="chat-messages" aria-live="polite"></main>

        <footer class="chat-composer">
          <div id="composer-normal" class="composer-pill">
            <input id="chat-input" type="text" placeholder="Ask a question..." autocomplete="off" />

            <button id="mic-btn" class="pill-icon pill-mic" type="button" aria-label="Voice message">
              <img src="${assetBase ? `${assetBase}/assets/mic.svg` : "./assets/mic.svg"}" alt="microphone icon" class="pill-svg" />
            </button>

            <button id="send-btn" class="pill-icon pill-send" type="button" aria-label="Send">
              <svg class="pill-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M3 11.5 21 3l-8.5 18-2.7-6.3L3 11.5Zm8 2.2 1.6 3.7L17.8 6.2 11 13.7Z" />
              </svg>
            </button>
          </div>

          <div id="composer-recording" class="recording-row hidden" aria-live="polite">
            <button id="rec-cancel" class="rec-action ghost" type="button" aria-label="Cancel recording">✕</button>

            <div class="rec-visual">
              <div class="rec-dot"></div>
              <div class="sound-bars" aria-hidden="true">
                <span></span><span></span><span></span><span></span><span></span>
                <span></span><span></span><span></span><span></span><span></span>
                <span></span><span></span><span></span><span></span><span></span>
                <span></span><span></span><span></span><span></span><span></span>
              </div>
              <div class="rec-time" id="rec-timer">00:00</div>
            </div>

            <button id="rec-send" class="rec-action primary" type="button" aria-label="Send recording">➤</button>
          </div>

          <div class="hint" id="hint"></div>
        </footer>
      </section>
    `;
    document.body.appendChild(widget);
  };

  (async () => {
    ensureCss();
    injectHtml();
    await ensureMarked();
    await ensureScripts();
    if (window.BILChatbotInit) window.BILChatbotInit();
  })();
})();
