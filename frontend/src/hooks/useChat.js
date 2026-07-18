import { useState, useCallback, useRef } from "react";
import { streamAsk } from "../services/api";

/**
 * useChat — manages the full chat state and SSE streaming lifecycle.
 *
 * Returns:
 *   answer        - current streamed answer text
 *   sources       - list of source dicts from backend
 *   isLoading     - true while streaming
 *   status        - "idle" | "retrieving" | "streaming" | "done" | "error"
 *   activeModel   - name of the LLM that answered
 *   error         - error message if status === "error"
 *   submit(q)     - async function to start a new query
 *   reset()       - clear all state
 */
export function useChat() {
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("idle");
  const [activeModel, setActiveModel] = useState("");
  const [error, setError] = useState("");

  // Ref to allow cancellation
  const abortRef = useRef(false);

  const reset = useCallback(() => {
    abortRef.current = true; // signal any in-flight stream to stop
    setAnswer("");
    setSources([]);
    setIsLoading(false);
    setStatus("idle");
    setActiveModel("");
    setError("");
  }, []);

  const submit = useCallback(async (question) => {
    if (!question.trim() || isLoading) return;

    // Reset state for new query
    abortRef.current = false;
    setAnswer("");
    setSources([]);
    setError("");
    setActiveModel("");
    setIsLoading(true);
    setStatus("retrieving");

    try {
      for await (const event of streamAsk(question)) {
        if (abortRef.current) break;

        switch (event.type) {
          case "sources":
            setSources(event.data.sources || []);
            setStatus("streaming");
            break;

          case "token":
            if (event.data.model) setActiveModel(event.data.model);
            setAnswer((prev) => prev + (event.data.token || ""));
            break;

          case "done":
            if (event.data.model) setActiveModel(event.data.model);
            setStatus("done");
            break;

          case "error":
            setError(event.data.message || "Unknown error");
            setStatus("error");
            break;
        }
      }
    } catch (err) {
      if (!abortRef.current) {
        setError(err.message || "Network error — backend may be starting up (cold start ~30s). Please retry.");
        setStatus("error");
      }
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  return { answer, sources, isLoading, status, activeModel, error, submit, reset };
}
