import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * AdminDashboard — Enterprise Observability & Monitoring Modal.
 *
 * Displays metrics: total chats, sessions, positive feedback %, average latency,
 * LLM model usage breakdown, recent query audit log, and automated evaluation logs.
 *
 * Props:
 *   isOpen: bool
 *   onClose: fn
 */
export default function AdminDashboard({ isOpen, onClose }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen) {
      loadMetrics();
    }
  }, [isOpen]);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/api/admin/metrics`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setMetrics(data);
    } catch (err) {
      console.error("Failed to load admin metrics:", err);
      setError("Failed to fetch observability metrics. Make sure backend is running with DATABASE_URL.");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="admin-backdrop" onClick={onClose}>
      <div className="admin-modal" onClick={(e) => e.stopPropagation()}>
        <div className="admin-header">
          <div className="admin-title">
            <span>📊</span> Enterprise Observability & Audit Log
          </div>
          <button className="admin-close-btn" onClick={onClose} title="Close">
            ✕
          </button>
        </div>

        <div className="admin-body">
          {loading ? (
            <div className="admin-loading">Loading metrics from PostgreSQL...</div>
          ) : error ? (
            <div className="admin-error">{error}</div>
          ) : metrics && metrics.enabled === false ? (
            <div className="admin-warning">
              ⚠️ Database Observability disabled. Add DATABASE_URL to backend/.env to activate Enterprise Logging.
            </div>
          ) : metrics ? (
            <>
              {/* ── KPI Cards ── */}
              <div className="kpi-grid">
                <div className="kpi-card">
                  <div className="kpi-label">Total Answers</div>
                  <div className="kpi-value">{metrics.total_chats || 0}</div>
                  <div className="kpi-sub">{metrics.total_sessions || 0} sessions</div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-label">User Satisfaction</div>
                  <div className="kpi-value">{metrics.positive_rate}%</div>
                  <div className="kpi-sub">
                    👍 {metrics.up_votes} | 👎 {metrics.down_votes}
                  </div>
                </div>

                <div className="kpi-card">
                  <div className="kpi-label">Avg Pipeline Latency</div>
                  <div className="kpi-value">{metrics.avg_latency_ms} <span className="kpi-unit">ms</span></div>
                  <div className="kpi-sub">Retrieval + Rerank + LLM</div>
                </div>
              </div>

              {/* ── LLM Model Usage Distribution ── */}
              <div className="admin-section">
                <div className="admin-section-title">🤖 LLM Model Usage Breakdown</div>
                <div className="model-bar-container">
                  {Object.entries(metrics.model_breakdown || {}).map(([modelName, count]) => {
                    const total = metrics.total_chats || 1;
                    const pct = Math.round((count / total) * 100);
                    return (
                      <div key={modelName} className="model-chip-row">
                        <span className="model-chip-label">{modelName}</span>
                        <div className="model-progress-bg">
                          <div className="model-progress-fill" style={{ width: `${pct}%` }} />
                        </div>
                        <span className="model-chip-count">{count} ({pct}%)</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* ── Recent Chat Logs ── */}
              <div className="admin-section">
                <div className="admin-section-title">📝 Recent Audit Logs (Last 10 Q&A)</div>
                <div className="admin-table-wrapper">
                  <table className="admin-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Question</th>
                        <th>Model</th>
                        <th>Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(metrics.recent_messages || []).map((m) => (
                        <tr key={m.message_id}>
                          <td>
                            {m.created_at
                              ? new Date(m.created_at).toLocaleTimeString("vi-VN", {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                  second: "2-digit",
                                })
                              : "-"}
                          </td>
                          <td className="table-question">{m.question}</td>
                          <td>
                            <span className="table-tag">{m.model_used || "Unknown"}</span>
                          </td>
                          <td>{m.latency_ms ? `${m.latency_ms}ms` : "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* ── Automated Evaluation Logs ── */}
              {metrics.recent_evaluations && metrics.recent_evaluations.length > 0 && (
                <div className="admin-section">
                  <div className="admin-section-title">🧪 Regression & Quality Evaluation Logs</div>
                  <div className="admin-table-wrapper">
                    <table className="admin-table">
                      <thead>
                        <tr>
                          <th>Run Type</th>
                          <th>Status</th>
                          <th>Faithfulness</th>
                          <th>Answer Relevancy</th>
                          <th>Question</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.recent_evaluations.map((ev) => (
                          <tr key={ev.id}>
                            <td>{ev.run_type}</td>
                            <td>
                              <span className={`status-badge ${ev.status.toLowerCase()}`}>
                                {ev.status}
                              </span>
                            </td>
                            <td>{ev.faithfulness != null ? ev.faithfulness.toFixed(2) : "N/A"}</td>
                            <td>{ev.answer_relevancy != null ? ev.answer_relevancy.toFixed(2) : "N/A"}</td>
                            <td className="table-question">{ev.question}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}
