import { useState, useCallback, useRef } from "react";
import { streamAsk, getUserId, getSessionId, newSession } from "../services/api";

/**
 * useChat — manages chat state and SSE streaming lifecycle.
 *
 * Returns:
 *   answer        - current streamed answer text
 *   sources       - list of source dicts from backend
 *   isLoading     - true while streaming
 *   status        - "idle" | "retrieving" | "streaming" | "done" | "error"
 *   activeModel   - name of the LLM that answered
 *   error         - error message if status === "error"
 *   progressStep  - { step, label, eta_s } | null — current pipeline step
 *   etaRemaining  - countdown seconds remaining for current step
 *   subQueries    - string[] — decomposed sub-queries (empty if decompose=false)
 *   messageId     - UUID of the last completed message (for feedback)
 *   sessionId     - current chat session UUID
 *   userId        - persistent anonymous user UUID
 *   submit(q, decompose)  - async function to start a new query
 *   reset()       - clear answer state (keeps session)
 *   startNewSession() - generate a new session UUID
 */
export function useChat() {
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("idle");
  const [activeModel, setActiveModel] = useState("");
  const [error, setError] = useState("");
  const [progressStep, setProgressStep] = useState(null);
  const [etaRemaining, setEtaRemaining] = useState(0);
  const [subQueries, setSubQueries] = useState([]);
  const [messageId, setMessageId] = useState(null);

  // Session & user identity (stable across renders)
  const [sessionId, setSessionId] = useState(() => {
    // Restore existing session or create new one
    return getSessionId() || newSession();
  });
  const [userId] = useState(() => getUserId());

  const abortRef = useRef(false);
  const etaIntervalRef = useRef(null);

  // ── ETA countdown helpers ─────────────────────────────────────────────────

  const stopCountdown = () => {
    if (etaIntervalRef.current) {
      clearInterval(etaIntervalRef.current);
      etaIntervalRef.current = null;
    }
    setEtaRemaining(0);
  };

  const startCountdown = (seconds) => {
    stopCountdown();
    if (!seconds || seconds <= 0) return;
    setEtaRemaining(seconds);
    etaIntervalRef.current = setInterval(() => {
      setEtaRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(etaIntervalRef.current);
          etaIntervalRef.current = null;
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  // ── Actions ───────────────────────────────────────────────────────────────

  const startNewSession = useCallback(() => {
    const sid = newSession();
    setSessionId(sid);
    setAnswer("");
    setSources([]);
    setError("");
    setActiveModel("");
    setProgressStep(null);
    setEtaRemaining(0);
    setSubQueries([]);
    setMessageId(null);
    setStatus("idle");
    return sid;
  }, []);

  const reset = useCallback(() => {
    abortRef.current = true;
    stopCountdown();
    setAnswer("");
    setSources([]);
    setIsLoading(false);
    setStatus("idle");
    setActiveModel("");
    setError("");
    setProgressStep(null);
    setEtaRemaining(0);
    setSubQueries([]);
    setMessageId(null);
  }, []);

  const submit = useCallback(
    async (question, decompose = false) => {
      if (!question.trim() || isLoading) return;

      // Reset state for new query
      abortRef.current = false;
      setAnswer("");
      setSources([]);
      setError("");
      setActiveModel("");
      setProgressStep(null);
      setEtaRemaining(0);
      setSubQueries([]);
      setMessageId(null);
      setIsLoading(true);
      setStatus("retrieving");

      try {
        for await (const event of streamAsk(question, decompose, sessionId, userId)) {
          if (abortRef.current) break;

          switch (event.type) {
            case "progress": {
              // Update current pipeline step and restart ETA countdown
              setProgressStep(event.data);
              startCountdown(event.data.eta_s || 0);
              break;
            }

            case "decomposition":
              // Deep analysis sub-queries received
              setSubQueries(event.data.sub_queries || []);
              break;

            case "sources":
              stopCountdown();
              setSources(event.data.sources || []);
              setProgressStep(null);
              setStatus("streaming");
              break;

            case "token":
              if (event.data.model) setActiveModel(event.data.model);
              setAnswer((prev) => prev + (event.data.token || ""));
              break;

            case "done":
              stopCountdown();
              if (event.data.model) setActiveModel(event.data.model);
              if (event.data.message_id) setMessageId(event.data.message_id);
              setProgressStep(null);
              setStatus("done");
              break;

            case "error":
              stopCountdown();
              setError(event.data.message || "An unknown error occurred. Please try again.");
              setProgressStep(null);
              setStatus("error");
              break;

            default:
              break;
          }
        }
      } catch (err) {
        stopCountdown();
        if (!abortRef.current) {
          setError(classifyNetworkError(err));
          setStatus("error");
        }
      } finally {
        stopCountdown();
        setIsLoading(false);
      }
    },
    [isLoading, sessionId, userId]
  );

  return {
    answer,
    sources,
    isLoading,
    status,
    activeModel,
    error,
    progressStep,
    etaRemaining,
    subQueries,
    messageId,
    sessionId,
    userId,
    submit,
    reset,
    startNewSession,
  };
}

// ── Network error classifier ──────────────────────────────────────────────────

/**
 * Maps raw JS network/fetch exceptions to user-friendly English error messages.
 */
function classifyNetworkError(err) {
  const msg = (err.message || "").toLowerCase();

  if (
    msg.includes("failed to fetch") ||
    msg.includes("networkerror") ||
    msg.includes("load failed") ||
    msg.includes("network request failed")
  ) {
    return "⏳ Cannot reach backend. It may be starting up (cold start ~30s). Please try again.";
  }
  if (msg.includes("http 5")) {
    return `⛔ Server error (${err.message}). Please try again later.`;
  }
  if (msg.includes("http 4")) {
    return `❌ Request error (${err.message}).`;
  }
  if (
    msg.includes("timeout") ||
    msg.includes("aborterror") ||
    msg.includes("aborted")
  ) {
    return "⏱️ Request timed out. Please try again.";
  }
  return err.message || "❌ Unknown error. Please try again.";
}
