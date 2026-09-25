import { useEffect, useState } from "react";
import { fetchSessions } from "../services/api";

/**
 * ChatHistorySidebar — drawer displaying past chat sessions for the user.
 *
 * Props:
 *   isOpen: bool
 *   onClose: fn
 *   userId: string
 *   activeSessionId: string
 *   onSelectSession: fn(sessionId)
 *   onNewChat: fn()
 */
export default function ChatHistorySidebar({
  isOpen,
  onClose,
  userId,
  activeSessionId,
  onSelectSession,
  onNewChat,
}) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen && userId) {
      loadSessions();
    }
  }, [isOpen, userId]);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const res = await fetchSessions(userId);
      setSessions(res.sessions || []);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="sidebar-backdrop" onClick={onClose}>
      <div className="sidebar-container" onClick={(e) => e.stopPropagation()}>
        <div className="sidebar-header">
          <div className="sidebar-title">
            <span>📜</span> Chat History
          </div>
          <button className="sidebar-close-btn" onClick={onClose} title="Close sidebar">
            ✕
          </button>
        </div>

        <button
          className="new-chat-btn"
          onClick={() => {
            onNewChat();
            onClose();
          }}
        >
          ➕ New Chat Session
        </button>

        <div className="sidebar-body">
          {loading ? (
            <div className="sidebar-loading">Loading sessions...</div>
          ) : sessions.length === 0 ? (
            <div className="sidebar-empty">No past sessions found. Start asking questions!</div>
          ) : (
            <div className="session-list">
              {sessions.map((s) => {
                const isActive = s.session_id === activeSessionId;
                const dateStr = s.updated_at
                  ? new Date(s.updated_at).toLocaleDateString("vi-VN", {
                      day: "2-digit",
                      month: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                    })
                  : "";

                return (
                  <div
                    key={s.session_id}
                    className={`session-item ${isActive ? "active" : ""}`}
                    onClick={() => {
                      onSelectSession(s.session_id);
                      onClose();
                    }}
                  >
                    <div className="session-question">{s.first_question || "New Chat"}</div>
                    <div className="session-date">{dateStr}</div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
