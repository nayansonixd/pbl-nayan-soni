/**
 * background.js  (Service Worker — Manifest v3)
 * ==============================================
 * Acts as a relay between popup.js and the FastAPI backend.
 *
 * Why route through the background worker?
 *   Chrome Extensions running on https://mail.google.com cannot make
 *   direct fetch() calls to http://localhost due to mixed-content
 *   restrictions.  The service worker has no such restriction.
 *
 * Messages handled:
 *   { action: "CLASSIFY_EMAIL", text: string }
 *     → POSTs to /predict
 *     → replies { success: true,  data: { category, confidence, all_scores } }
 *            or { success: false, error: string }
 */

"use strict";

const API_BASE = "http://localhost:8000";

// ── Message listener ─────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action !== "CLASSIFY_EMAIL") return false;

  classifyEmail(message.text)
    .then(data  => sendResponse({ success: true,  data }))
    .catch(err  => sendResponse({ success: false, error: err.message }));

  // Return true to use sendResponse asynchronously
  return true;
});

// ── API call ──────────────────────────────────────────────────────────────────
/**
 * POST /predict with the email text.
 * @param {string} text
 * @returns {Promise<{category: string, confidence: number, all_scores: Array}>}
 */
async function classifyEmail(text) {
  let response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
  } catch (networkErr) {
    throw new Error(
      "Cannot reach the backend server. " +
      "Make sure it is running:  uvicorn backend.app:app --reload --port 8000"
    );
  }

  if (!response.ok) {
    let detail = `Server error (HTTP ${response.status})`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_) { /* ignore parse error */ }
    throw new Error(detail);
  }

  return response.json();
}
