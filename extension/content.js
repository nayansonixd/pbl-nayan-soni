/**
 * content.js
 * ==========
 * Injected into every Gmail page (https://mail.google.com/*).
 *
 * Responsibilities:
 *   • Listen for a GET_EMAIL_TEXT message from popup.js
 *   • Scrape the currently-open email body from the Gmail DOM
 *   • Return the plain text to the popup via sendResponse
 *
 * Gmail DOM notes (as of 2024):
 *   The readable email body lives inside  div.a3s.aiL  (single email)
 *   or  div.a3s  (thread messages).  We try multiple selectors so that
 *   the extension is resilient to minor Gmail layout changes.
 */

"use strict";

/**
 * Attempts to extract the email body text from Gmail's DOM.
 * Tries several CSS selectors in order of specificity.
 *
 * @returns {string|null} Plain text of the email, or null if not found.
 */
function scrapeGmailBody() {
  // ── Strategy 1: single-message open view (most common) ──────────────────
  const primary = document.querySelector("div.a3s.aiL");
  if (primary && primary.innerText.trim().length > 0) {
    return primary.innerText.trim();
  }

  // ── Strategy 2: most recent message in a thread ──────────────────────────
  const allBodies = document.querySelectorAll("div.a3s");
  if (allBodies.length > 0) {
    // The last expanded message is the most recent one
    for (let i = allBodies.length - 1; i >= 0; i--) {
      const text = allBodies[i].innerText.trim();
      if (text.length > 20) return text;
    }
  }

  // ── Strategy 3: subject + body fallback ─────────────────────────────────
  const subject = document.querySelector("h2.hP");
  const body    = document.querySelector("[data-message-id] div.ii.gt");
  if (body) {
    const subjectText = subject ? subject.innerText.trim() : "";
    const bodyText    = body.innerText.trim();
    return `${subjectText}\n\n${bodyText}`.trim();
  }

  return null;   // nothing found
}

// ── Message listener ─────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action !== "GET_EMAIL_TEXT") return false;

  const text = scrapeGmailBody();

  if (text) {
    sendResponse({ success: true,  text });
  } else {
    sendResponse({
      success: false,
      text:    "",
      error:   "No open email detected. Please open an email in Gmail first.",
    });
  }

  // Return true to keep the message channel open for async sendResponse
  return true;
});
