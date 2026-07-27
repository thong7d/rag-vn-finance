import { useState } from "react";

/**
 * DecompositionPanel — displays sub-queries from query decomposition.
 *
 * Props:
 *   subQueries: string[] — list of decomposed sub-questions
 *   isStreaming: bool — true when answer is streaming (panel collapses)
 */
export default function DecompositionPanel({ subQueries, isStreaming }) {
  const [collapsed, setCollapsed] = useState(false);

  // Auto-collapse once streaming begins
  const isCollapsed = collapsed || isStreaming;

  if (!subQueries || subQueries.length === 0) return null;

  return (
    <div className="decomposition-panel" aria-label="Query decomposition results">
      <button
        className="decomposition-header"
        onClick={() => setCollapsed((v) => !v)}
        aria-expanded={!isCollapsed}
      >
        <span className="decomposition-icon">🔍</span>
        <span className="decomposition-title">
          Deep Analysis — {subQueries.length} sub-queries
        </span>
        <span className="decomposition-chevron">{isCollapsed ? "▼" : "▲"}</span>
      </button>

      {!isCollapsed && (
        <div className="decomposition-body">
          {subQueries.map((q, i) => (
            <div key={i} className="sub-query-item">
              <span className="sub-query-index">{i + 1}</span>
              <span className="sub-query-text">{q}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
