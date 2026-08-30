/**
 * chat.js — FAQ Chatbot frontend logic
 *
 * Responsibilities:
 *  - Render the welcome card on load
 *  - Fetch and render FAQ suggestion buttons in the sidebar
 *  - Send user messages to /chat and render bot replies
 *  - Show / hide the typing indicator
 *  - Auto-resize the textarea
 *  - Track and display message count
 *  - Handle "Clear Chat" with a confirmation prompt
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl   = document.getElementById("chat-messages");
const form         = document.getElementById("chat-form");
const inputEl      = document.getElementById("user-input");
const sendBtn      = document.getElementById("send-btn");
const typingEl     = document.getElementById("typing-indicator");
const clearBtn     = document.getElementById("clear-btn");
const suggestionEl = document.getElementById("faq-suggestions");
const msgCountEl   = document.getElementById("msg-count");
const charCountEl  = document.getElementById("char-count");

let messageCount = 0;

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  renderWelcome();
  loadSuggestions();
  inputEl.focus();
});

// ── Welcome card ──────────────────────────────────────────────────────────────
function renderWelcome() {
  const card = document.createElement("div");
  card.className = "welcome-card";
  card.innerHTML = `
    <div style="font-size:2.5rem;margin-bottom:12px;">🤖</div>
    <h2>Hello! I'm your FAQ Assistant</h2>
    <p>
      Ask me anything about <strong>orders</strong>, <strong>returns</strong>,
      <strong>shipping</strong>, <strong>payments</strong>, or your
      <strong>account</strong>. I'll find the best answer for you!
    </p>
    <p style="margin-top:12px;font-size:0.78rem;color:var(--text-muted);">
      Powered by NLP · TF-IDF Cosine Similarity
    </p>
  `;
  messagesEl.appendChild(card);
}

// ── Load FAQ suggestions ───────────────────────────────────────────────────────
async function loadSuggestions() {
  try {
    const res  = await fetch("/faqs");
    const list = await res.json();

    // Show only first 6 suggestions in sidebar
    const items = list.slice(0, 6);
    suggestionEl.innerHTML = "";

    items.forEach(q => {
      const li  = document.createElement("li");
      const btn = document.createElement("button");
      btn.type        = "button";
      btn.textContent = q;
      btn.addEventListener("click", () => sendMessage(q));
      li.appendChild(btn);
      suggestionEl.appendChild(li);
    });
  } catch {
    suggestionEl.innerHTML =
      "<li style='font-size:0.75rem;color:var(--text-muted);padding:4px 0;'>Could not load suggestions</li>";
  }
}

// ── Form submit ───────────────────────────────────────────────────────────────
form.addEventListener("submit", e => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  sendMessage(text);
});

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage(text) {
  if (!text.trim()) return;

  // Render user bubble
  appendMessage("user", text);
  inputEl.value = "";
  updateCharCount();
  resizeTextarea();
  setLoading(true);

  try {
    const res  = await fetch("/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();

    // Small artificial delay so the typing indicator feels natural
    await sleep(420);

    setLoading(false);
    appendBotMessage(data);

  } catch (err) {
    setLoading(false);
    appendMessage("bot",
      "⚠️ Something went wrong connecting to the server. Please try again."
    );
    console.error(err);
  }
}

// ── Render user bubble ────────────────────────────────────────────────────────
function appendMessage(role, text, timestamp) {
  // Remove welcome card on first real message
  const welcomeCard = messagesEl.querySelector(".welcome-card");
  if (welcomeCard) welcomeCard.remove();

  const ts  = timestamp || currentTime();
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const avatarHTML = role === "bot"
    ? `<div class="bot-avatar tiny">🤖</div>`
    : "";

  div.innerHTML = `
    ${avatarHTML}
    <div>
      <div class="bubble">${escapeHtml(text)}</div>
    </div>
    <span class="msg-time">${ts}</span>
  `;

  messagesEl.appendChild(div);
  scrollToBottom();

  messageCount++;
  msgCountEl.textContent = `${messageCount} message${messageCount !== 1 ? "s" : ""}`;
}

// ── Render bot bubble (with optional match badge) ─────────────────────────────
function appendBotMessage(data) {
  const welcomeCard = messagesEl.querySelector(".welcome-card");
  if (welcomeCard) welcomeCard.remove();

  const div = document.createElement("div");
  div.className = "message bot";

  // Format multi-line answers (bullet points etc.)
  const formattedAnswer = escapeHtml(data.answer).replace(/\n/g, "<br>");

  // Match badge — only when FAQ match is confident
  let badgeHTML = "";
  if (data.confident && data.matched_q) {
    const pct = Math.round(data.score * 100);
    badgeHTML = `
      <div class="match-badge">
        Matched: "${escapeHtml(data.matched_q)}"
        <span class="score">${pct}%</span>
      </div>
    `;
  }

  div.innerHTML = `
    <div class="bot-avatar tiny">🤖</div>
    <div>
      <div class="bubble">${formattedAnswer}${badgeHTML}</div>
    </div>
    <span class="msg-time">${data.timestamp || currentTime()}</span>
  `;

  messagesEl.appendChild(div);
  scrollToBottom();

  messageCount++;
  msgCountEl.textContent = `${messageCount} message${messageCount !== 1 ? "s" : ""}`;
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function setLoading(on) {
  sendBtn.disabled = on;
  typingEl.classList.toggle("hidden", !on);
  if (on) scrollToBottom();
}

// ── Clear chat ────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", async () => {
  if (!confirm("Clear the conversation?")) return;

  await fetch("/clear", { method: "POST" });

  messagesEl.innerHTML = "";
  messageCount = 0;
  msgCountEl.textContent = "0 messages";
  renderWelcome();
  inputEl.focus();
});

// ── Textarea auto-resize ──────────────────────────────────────────────────────
inputEl.addEventListener("input", () => {
  resizeTextarea();
  updateCharCount();
});

inputEl.addEventListener("keydown", e => {
  // Submit on Enter, new line on Shift+Enter
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event("submit"));
  }
});

function resizeTextarea() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
}

function updateCharCount() {
  const len     = inputEl.value.length;
  const counter = charCountEl.parentElement;
  charCountEl.textContent = len;
  counter.className = "char-counter";
  if (len > 250) counter.classList.add("warn");
  if (len >= 300) counter.classList.add("over");
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

function currentTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
