/**
 * api.js — Backend API service layer.
 * All backend calls go through this module so the base URL stays in one place.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Session / User Identity ───────────────────────────────────────────────────

/**
 * Get or create a persistent user UUID stored in localStorage.
 * This is used as an anonymous user identifier across sessions.
 */
export function getUserId() {
  let userId = localStorage.getItem("rag_user_id");
  if (!userId) {
    userId = crypto.randomUUID();
    localStorage.setItem("rag_user_id", userId);
  }
  return userId;
}

/**
 * Get current session ID from localStorage.
 */
export function getSessionId() {
  return localStorage.getItem("rag_session_id") || "";
}

/**
 * Start a new chat session — generates a fresh session UUID.
 */
export function newSession() {
  const sessionId = crypto.randomUUID();
  localStorage.setItem("rag_session_id", sessionId);
  return sessionId;
}

// ── SSE Streaming ─────────────────────────────────────────────────────────────

/**
 * Open an SSE stream for a question.
 * Returns an EventSource-compatible fetch stream wrapped as an async generator.
 *
 * @param {string} question - The user's question
 * @param {boolean} decompose - Whether to enable query decomposition
 * @param {string} sessionId - Current chat session UUID
 * @param {string} userId - Persistent user UUID
 *
 * Usage:
 *   for await (const event of streamAsk(question, true, sessionId, userId)) {
 *     if (event.type === "decomposition") ... // sub-queries
 *     if (event.type === "sources") ...
 *     if (event.type === "token")   ...
 *     if (event.type === "done")    ...  // done.data.message_id is now available
 *     if (event.type === "error")   ...
 *   }
 */
export async function* streamAsk(question, decompose = false, sessionId = "", userId = "") {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      decompose,
      session_id: sessionId,
      user_id: userId,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop(); // keep incomplete last chunk

    for (const part of parts) {
      if (!part.trim()) continue;

      // Parse SSE: "event: xxx\ndata: {...}"
      const lines = part.split("\n");
      let eventType = "message";
      let dataStr = "";

      for (const line of lines) {
        if (line.startsWith("event: ")) eventType = line.slice(7).trim();
        if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
      }

      if (!dataStr) continue;

      try {
        const data = JSON.parse(dataStr);
        yield { type: eventType, data };
      } catch {
        // ignore malformed SSE
      }
    }
  }
}

// ── Feedback API ──────────────────────────────────────────────────────────────

/**
 * Submit user feedback (thumbs up/down) for a specific message.
 *
 * @param {string} messageId - UUID of the chat message to rate
 * @param {string} userId - Persistent user UUID
 * @param {"up"|"down"} vote - User's vote
 * @param {string} [comment=""] - Optional free-text comment
 */
export async function submitFeedback(messageId, userId, vote, comment = "") {
  const response = await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message_id: messageId,
      user_id: userId,
      vote,
      comment,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  return response.json();
}

// ── Sessions History API ──────────────────────────────────────────────────────

/**
 * Fetch recent sessions for a given user.
 *
 * @param {string} userId
 * @param {number} [limit=20]
 */
export async function fetchSessions(userId, limit = 20) {
  const url = `${API_BASE}/api/sessions?user_id=${encodeURIComponent(userId)}&limit=${limit}`;
  const resp = await fetch(url, { signal: AbortSignal.timeout(8000) });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

/**
 * Fetch messages for a given session.
 *
 * @param {string} sessionId
 */
export async function fetchSessionMessages(sessionId) {
  const url = `${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}/messages`;
  const resp = await fetch(url, { signal: AbortSignal.timeout(8000) });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

// ── Health Check ──────────────────────────────────────────────────────────────

/**
 * Health check — returns true if backend is reachable.
 */
export async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    return resp.ok;
  } catch {
    return false;
  }
}
