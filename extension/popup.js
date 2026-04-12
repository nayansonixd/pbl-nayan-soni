/**
 * popup.js
 * ========
 * Controls all UI logic in the extension popup.
 *
 * Flow:
 *   1. On open → check backend health (/health)
 *   2. "Detect from Gmail" → message content.js → get email text → classify
 *   3. "Classify Email"    → read textarea      → classify
 *   4. classifyEmail()     → message background.js → /predict API
 *   5. showResult()        → render category, confidence pill, bar, scores
 */

"use strict";

// ── Category metadata ─────────────────────────────────────────────────────────
const CATEGORY_META = {
  Promotions:  { emoji: "📢", color: "#f59e0b" },
  Social:      { emoji: "👥", color: "#3b82f6" },
  Updates:     { emoji: "🔔", color: "#22d3ee" },
  Forum:       { emoji: "💬", color: "#a78bfa" },
  Spam:        { emoji: "🚨", color: "#ef4444" },
  Verify_Code: { emoji: "🔑", color: "#4ade80" },
};

const API_BASE = "http://localhost:8000";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const statusDot   = $("statusDot");
const statusText  = $("statusText");
const btnRetry    = $("btnRetry");
const btnDetect   = $("btnDetect");
const btnClassify = $("btnClassify");
const emailInput  = $("emailInput");

const loadingEl   = $("loading");
const errCard     = $("errCard");
const errText     = $("errText");
const resultCard  = $("resultCard");

const resultIcon     = $("resultIcon");
const resultIconWrap = $("resultIconWrap");
const resultCategory = $("resultCategory");
const confPill       = $("confPill");
const confFill       = $("confFill");
const confPct        = $("confPct");
const scoresList     = $("scoresList");


// ══════════════════════════════════════════════════════════════════════════════
// 1. Backend health check
// ══════════════════════════════════════════════════════════════════════════════
async function checkHealth() {
  statusDot.className = "status-dot";           // amber/blinking
  statusText.textContent = "Checking backend…";

  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(4000),
    });
    if (res.ok) {
      statusDot.className   = "status-dot online";
      statusText.textContent = "Backend online — ready to classify";
    } else {
      throw new Error();
    }
  } catch {
    statusDot.className   = "status-dot offline";
    statusText.textContent = "Backend offline — start the server first";
  }
}

btnRetry.addEventListener("click", checkHealth);


// ══════════════════════════════════════════════════════════════════════════════
// 2. UI state helpers
// ══════════════════════════════════════════════════════════════════════════════
function setLoading(on) {
  loadingEl.style.display = on ? "flex" : "none";
  btnDetect.disabled   = on;
  btnClassify.disabled = on;
}

function showError(msg) {
  setLoading(false);
  resultCard.style.display = "none";
  errCard.style.display    = "flex";
  errText.textContent      = msg;
}

function clearError() {
  errCard.style.display = "none";
}


// ══════════════════════════════════════════════════════════════════════════════
// 3. Render result
// ══════════════════════════════════════════════════════════════════════════════
/**
 * @param {{ category: string, confidence: number, all_scores: Array }} data
 */
function showResult(data) {
  setLoading(false);
  clearError();

  const { category, confidence, all_scores } = data;
  const pct  = Math.round(confidence * 100);
  const meta = CATEGORY_META[category] || { emoji: "📂", color: "#7c6ef4" };

  // Top row
  resultIcon.textContent      = meta.emoji;
  resultCategory.textContent  = category;
  confPill.textContent        = `${pct}%`;

  // Confidence bar (animate after paint)
  confFill.style.width = "0%";
  requestAnimationFrame(() => {
    setTimeout(() => { confFill.style.width = `${pct}%`; }, 40);
  });
  confPct.textContent = `${pct}%`;

  // Find max score to identify the top category
  const maxScore = Math.max(...all_scores.map(s => s.score));

  // Scores list
  scoresList.innerHTML = "";
  [...all_scores]
    .sort((a, b) => b.score - a.score)
    .forEach(({ label, score }) => {
      const rowPct  = Math.round(score * 100);
      const isTop   = score === maxScore;
      const m       = CATEGORY_META[label] || { emoji: "📂" };

      const row = document.createElement("div");
      row.className = "score-row";
      row.innerHTML = `
        <span class="score-emoji">${m.emoji}</span>
        <span class="score-name ${isTop ? "top" : ""}">${label}</span>
        <div class="score-bar-track">
          <div class="score-bar-fill ${isTop ? "top" : "rest"}"
               style="width: ${rowPct}%"></div>
        </div>
        <span class="score-pct ${isTop ? "top" : ""}">${rowPct}%</span>
      `;
      scoresList.appendChild(row);
    });

  resultCard.style.display = "flex";
}


// ══════════════════════════════════════════════════════════════════════════════
// 4. Core classify function (routes through background.js)
// ══════════════════════════════════════════════════════════════════════════════
function classifyText(text) {
  const trimmed = (text || "").trim();

  if (trimmed.length < 10) {
    showError("Please provide at least 10 characters of email text.");
    return;
  }

  clearError();
  resultCard.style.display = "none";
  setLoading(true);

  chrome.runtime.sendMessage(
    { action: "CLASSIFY_EMAIL", text: trimmed },
    response => {
      setLoading(false);

      if (chrome.runtime.lastError) {
        showError("Extension error: " + chrome.runtime.lastError.message);
        return;
      }
      if (!response) {
        showError("No response from background worker. Try reloading the extension.");
        return;
      }
      if (response.success) {
        showResult(response.data);
      } else {
        showError(response.error || "Classification failed. Is the backend running?");
      }
    }
  );
}


// ══════════════════════════════════════════════════════════════════════════════
// 5. Button handlers
// ══════════════════════════════════════════════════════════════════════════════

// ── Detect from Gmail ─────────────────────────────────────────────────────────
btnDetect.addEventListener("click", async () => {
  clearError();

  // Verify we're on Gmail
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url?.includes("mail.google.com")) {
    showError(
      "Please navigate to Gmail and open an email, then click Detect again."
    );
    return;
  }

  // Ask content.js to scrape the email
  chrome.tabs.sendMessage(
    tab.id,
    { action: "GET_EMAIL_TEXT" },
    response => {
      if (chrome.runtime.lastError || !response) {
        showError(
          "Could not connect to the Gmail page. " +
          "Try refreshing Gmail and then the popup."
        );
        return;
      }
      if (!response.success) {
        showError(
          response.error || "No open email found. Open an email in Gmail first."
        );
        return;
      }
      // Pre-fill textarea so the user can see what was detected
      emailInput.value = response.text;
      classifyText(response.text);
    }
  );
});

// ── Manual classify ───────────────────────────────────────────────────────────
btnClassify.addEventListener("click", () => {
  classifyText(emailInput.value);
});

// ── Classify on Ctrl/Cmd + Enter ─────────────────────────────────────────────
emailInput.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    classifyText(emailInput.value);
  }
});


// ══════════════════════════════════════════════════════════════════════════════
// 6. Init
// ══════════════════════════════════════════════════════════════════════════════
checkHealth();
