/**
 * api.js — Backend API service layer.
 * All backend calls go through this module so the base URL stays in one place.
 */

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * Open an SSE stream for a question.
 * Returns an EventSource-compatible fetch stream wrapped as an async generator.
 *
 * Usage:
 *   for await (const event of streamAsk(question)) {
 *     if (event.type === "sources") ...
 *     if (event.type === "token")   ...
 *     if (event.type === "done")    ...
 *     if (event.type === "error")   ...
 *   }
 */
export async function* streamAsk(question) {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
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
