const SESSION_KEY = "academic_assistant_session_id";

const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const queryInput = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const chatListEl = document.getElementById("chat-list");
const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebar = document.getElementById("sidebar");
const toastEl = document.getElementById("toast");

let sessionId = localStorage.getItem(SESSION_KEY);
let isLoading = false;

const WELCOME_HTML = `
  <div class="welcome">
    <div class="welcome-icon">AA</div>
    <h2>How can I help you today?</h2>
    <p>Ask about transfer requirements, application deadlines, and admissions.</p>
    <div class="suggestions">
      <button class="suggestion" type="button" data-query="What is the Fall transfer application deadline?">Fall transfer deadline</button>
      <button class="suggestion" type="button" data-query="What courses do I need to transfer to CSU?">Transfer course requirements</button>
      <button class="suggestion" type="button" data-query="When does the Spring application period open?">Spring application dates</button>
    </div>
  </div>
`;

function showToast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.hidden = false;
  toastEl.classList.toggle("error", isError);
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    toastEl.hidden = true;
  }, 3500);
}

async function parseError(response) {
  const data = await response.json().catch(() => ({}));
  const detail = data.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  return "Something went wrong. Please try again.";
}

function showWelcome() {
  messagesEl.innerHTML = WELCOME_HTML;
  bindSuggestions();
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setActiveSession(activeId) {
  chatListEl.querySelectorAll(".chat-list-row").forEach((row) => {
    row.classList.toggle("active", row.dataset.sessionId === activeId);
  });
}

async function deleteSession(targetSessionId, event) {
  event?.stopPropagation();
  if (isLoading) return;

  const confirmed = window.confirm("Delete this chat? This cannot be undone.");
  if (!confirmed) return;

  try {
    const response = await fetch(`/session/${targetSessionId}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await parseError(response));

    if (targetSessionId === sessionId) {
      await createSession();
      showWelcome();
      setActiveSession(sessionId);
    }

    await refreshChatList();
    showToast("Chat deleted");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function refreshChatList() {
  try {
    const response = await fetch("/sessions");
    if (!response.ok) return;

    const data = await response.json();
    const sessions = data.sessions || [];

    if (sessions.length === 0) {
      chatListEl.innerHTML = '<p class="chat-list-empty">No previous chats yet</p>';
      return;
    }

    chatListEl.innerHTML = sessions
      .map(
        (session) => `
          <div
            class="chat-list-row${session.session_id === sessionId ? " active" : ""}"
            data-session-id="${session.session_id}"
          >
            <button
              type="button"
              class="chat-list-item"
              data-session-id="${session.session_id}"
              title="${escapeHtml(session.title)}"
            >
              ${escapeHtml(session.title)}
            </button>
            <button
              type="button"
              class="chat-delete-btn"
              data-session-id="${session.session_id}"
              aria-label="Delete chat"
              title="Delete chat"
            >×</button>
          </div>
        `
      )
      .join("");

    chatListEl.querySelectorAll(".chat-list-item").forEach((button) => {
      button.addEventListener("click", () => {
        switchToSession(button.dataset.sessionId);
      });
    });

    chatListEl.querySelectorAll(".chat-delete-btn").forEach((button) => {
      button.addEventListener("click", (event) => {
        deleteSession(button.dataset.sessionId, event);
      });
    });
  } catch {
    chatListEl.innerHTML = '<p class="chat-list-empty">Unable to load chats</p>';
  }
}

async function createSession() {
  const response = await fetch("/session/new", { method: "POST" });
  if (!response.ok) throw new Error(await parseError(response));
  const data = await response.json();
  sessionId = data.session_id;
  localStorage.setItem(SESSION_KEY, sessionId);
  return sessionId;
}

async function ensureSession() {
  if (sessionId) return sessionId;
  return createSession();
}

function renderHistory(history) {
  messagesEl.innerHTML = "";

  if (!history.length) {
    showWelcome();
    return;
  }

  history.forEach((turn) => {
    const bubble = createMessageRow(turn.role);
    bubble.textContent = turn.content;
    if (turn.role === "assistant" && turn.content.startsWith("Query failed")) {
      bubble.closest(".message-row").classList.add("error");
    }
  });

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function loadSession(targetSessionId) {
  const response = await fetch(`/session/${targetSessionId}/history`);
  if (!response.ok) throw new Error(await parseError(response));

  const data = await response.json();
  sessionId = targetSessionId;
  localStorage.setItem(SESSION_KEY, sessionId);
  renderHistory(data.history || []);
  setActiveSession(sessionId);
  sidebar?.classList.remove("open");
}

async function switchToSession(targetSessionId) {
  if (isLoading || targetSessionId === sessionId) return;

  try {
    await loadSession(targetSessionId);
  } catch (error) {
    showToast(error.message, true);
  }
}

function clearWelcome() {
  messagesEl.querySelector(".welcome")?.remove();
}

function createMessageRow(role) {
  clearWelcome();

  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AA";

  const content = document.createElement("div");
  content.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  content.appendChild(bubble);

  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return bubble;
}

function showTypingIndicator() {
  clearWelcome();
  const bubble = createMessageRow("assistant");
  bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  bubble.closest(".message-row").id = "typing-indicator";
}

function hideTypingIndicator() {
  document.getElementById("typing-indicator")?.remove();
}

function setLoading(loading) {
  isLoading = loading;
  sendBtn.disabled = loading;
  queryInput.disabled = loading;
  newChatBtn.disabled = loading;
}

async function sendQuery(query) {
  const text = query.trim();
  if (!text || isLoading) return;

  setLoading(true);
  createMessageRow("user").textContent = text;
  queryInput.value = "";
  queryInput.style.height = "auto";
  showTypingIndicator();

  try {
    await ensureSession();
    setActiveSession(sessionId);

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("query", text);

    const response = await fetch("/ask", { method: "POST", body: formData });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    hideTypingIndicator();
    const bubble = createMessageRow("assistant");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      fullText += decoder.decode(value, { stream: true });
      bubble.textContent = fullText;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    await refreshChatList();
  } catch (error) {
    hideTypingIndicator();
    const bubble = createMessageRow("assistant");
    bubble.textContent = error.message;
    bubble.closest(".message-row").classList.add("error");
    showToast(error.message, true);
  } finally {
    setLoading(false);
    queryInput.focus();
  }
}

async function startNewChat() {
  if (isLoading) return;

  await createSession();
  showWelcome();
  setActiveSession(sessionId);
  await refreshChatList();
  sidebar?.classList.remove("open");
  queryInput.focus();
}

function bindSuggestions() {
  document.querySelectorAll(".suggestion").forEach((btn) => {
    btn.addEventListener("click", () => sendQuery(btn.dataset.query));
  });
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuery(queryInput.value);
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendQuery(queryInput.value);
  }
});

queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 160)}px`;
});

newChatBtn.addEventListener("click", startNewChat);

sidebarToggle?.addEventListener("click", () => {
  sidebar?.classList.toggle("open");
});

async function initApp() {
  await refreshChatList();

  if (sessionId) {
    try {
      const response = await fetch(`/session/${sessionId}/history`);
      if (response.ok) {
        const data = await response.json();
        if (data.history?.length) {
          renderHistory(data.history);
          setActiveSession(sessionId);
          return;
        }
      }
    } catch {
      // fall through to new session
    }
  }

  await createSession();
  showWelcome();
  setActiveSession(sessionId);
}

initApp().catch((error) => showToast(error.message, true));
