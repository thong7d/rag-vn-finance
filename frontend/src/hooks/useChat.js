import { useState, useCallback, useRef } from "react";
import { streamAsk } from "../services/api";

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
 *   submit(q, decompose)  - async function to start a new query
 *   reset()       - clear all state
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
      setIsLoading(true);
      setStatus("retrieving");

      try {
        for await (const event of streamAsk(question, decompose)) {
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
              setProgressStep(null);
              setStatus("done");
              break;

            case "error":
              stopCountdown();
              setError(event.data.message || "Đã xảy ra lỗi không xác định.");
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
    [isLoading]
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
    submit,
    reset,
  };
}

// ── Network error classifier ──────────────────────────────────────────────────

/**
 * Maps raw JS network/fetch exceptions to user-friendly Vietnamese messages.
 */
function classifyNetworkError(err) {
  const msg = (err.message || "").toLowerCase();

  if (
    msg.includes("failed to fetch") ||
    msg.includes("networkerror") ||
    msg.includes("load failed") ||
    msg.includes("network request failed")
  ) {
    return "⏳ Không thể kết nối đến backend. Backend có thể đang khởi động lại (cold start ~30 giây). Vui lòng thử lại sau.";
  }
  if (msg.includes("http 5")) {
    return `⛔ Lỗi máy chủ (${err.message}). Vui lòng thử lại sau.`;
  }
  if (msg.includes("http 4")) {
    return `❌ Lỗi yêu cầu (${err.message}).`;
  }
  if (
    msg.includes("timeout") ||
    msg.includes("aborterror") ||
    msg.includes("aborted")
  ) {
    return "⏱️ Yêu cầu bị hủy do quá thời gian chờ. Vui lòng thử lại.";
  }
  return err.message || "❌ Lỗi không xác định. Vui lòng thử lại.";
}
