/**
 * StatusBar — rich pipeline progress + status display.
 *
 * Props:
 *   status:       "idle" | "retrieving" | "streaming" | "done" | "error"
 *   error:        string
 *   progressStep: { step: string, label: string, eta_s: number } | null
 *   etaRemaining: number — live countdown seconds
 *   activeModel:  string — LLM name during streaming
 */
import { useState } from "react";

// Pipeline step definitions (order matters — used to determine done/active/pending)
const PIPELINE = [
  { id: "embedding",     icon: "◈", label: "Embedding"  },
  { id: "sparse_search", icon: "⊙", label: "BM25"       },
  { id: "rerank",        icon: "⊛", label: "Rerank"     },
];

export default function StatusBar({ status, error, progressStep, etaRemaining, activeModel }) {
  const [errorExpanded, setErrorExpanded] = useState(false);

  if (status === "idle") return null;

  // ── Retrieval phase — show pipeline step progress ─────────────────────────
  if (status === "retrieving" && progressStep) {
    const currentIdx = PIPELINE.findIndex((s) => s.id === progressStep.step);

    return (
      <div className="status-bar loading" role="status" aria-live="polite">
        <div className="pipeline-row">
          <span className="spinner" style={{ flexShrink: 0 }} />
          <div className="pipeline-chips">
            {PIPELINE.map((step, idx) => {
              const chipState =
                idx < currentIdx
                  ? "done"
                  : idx === currentIdx
                  ? "active"
                  : "pending";
              return (
                <span
                  key={step.id}
                  className={`pipeline-chip pipeline-chip--${chipState}`}
                >
                  {chipState === "done" ? "✓" : step.icon} {step.label}
                </span>
              );
            })}
          </div>
          {etaRemaining > 0 && (
            <span className="eta-badge" aria-label={`Ước tính ${etaRemaining} giây`}>
              {etaRemaining}s
            </span>
          )}
        </div>
        <div className="pipeline-label">{progressStep.label}</div>
      </div>
    );
  }

  // ── Retrieving without step (initial state) ───────────────────────────────
  if (status === "retrieving") {
    return (
      <div className="status-bar loading" role="status" aria-live="polite">
        <span className="spinner" />
        <span>Đang xử lý câu hỏi...</span>
      </div>
    );
  }

  // ── Streaming phase — LLM generating ─────────────────────────────────────
  if (status === "streaming") {
    return (
      <div className="status-bar loading" role="status" aria-live="polite">
        <div className="pipeline-row">
          <span className="spinner" style={{ flexShrink: 0 }} />
          <div className="pipeline-chips">
            {PIPELINE.map((step) => (
              <span key={step.id} className="pipeline-chip pipeline-chip--done">
                ✓ {step.label}
              </span>
            ))}
          </div>
        </div>
        <div className="pipeline-label">
          Đang sinh câu trả lời
          {activeModel && (
            <span style={{ color: "var(--accent-green)", marginLeft: "0.3rem" }}>
              · {activeModel}
            </span>
          )}
          ...
        </div>
      </div>
    );
  }

  // ── Done ──────────────────────────────────────────────────────────────────
  if (status === "done") {
    return (
      <div className="status-bar success" role="status">
        ✓ <span>Hoàn thành</span>
      </div>
    );
  }

  // ── Error ─────────────────────────────────────────────────────────────────
  if (status === "error") {
    const displayError = error || "Đã xảy ra lỗi. Vui lòng thử lại.";
    const isLong = displayError.length > 120;
    return (
      <div className="status-bar error" role="alert" aria-live="assertive">
        <span className="error-icon">⚠</span>
        <span className="error-text">
          {isLong && !errorExpanded
            ? displayError.slice(0, 120) + "…"
            : displayError}
        </span>
        {isLong && (
          <button
            className="error-toggle"
            onClick={() => setErrorExpanded((v) => !v)}
            aria-expanded={errorExpanded}
          >
            {errorExpanded ? "Thu gọn ▲" : "Chi tiết ▼"}
          </button>
        )}
      </div>
    );
  }

  return null;
}
