import { useState } from "react";
import "./styles/index.css";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import DecompositionPanel from "./components/DecompositionPanel";
import SourceCard from "./components/SourceCard";
import StatusBar from "./components/StatusBar";
import ChatHistorySidebar from "./components/ChatHistorySidebar";
import AdminDashboard from "./components/AdminDashboard";
import { useChat } from "./hooks/useChat";
import { fetchSessionMessages } from "./services/api";

export default function App() {
  const {
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
    startNewSession,
  } = useChat();

  const [decompose, setDecompose] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isAdminOpen, setIsAdminOpen] = useState(false);
  const [historyMessages, setHistoryMessages] = useState([]);

  // Sample chips only visible when idle (no answer, no loading)
  const showSamples = status === "idle" && !answer && !isLoading && historyMessages.length === 0;

  const handleSubmit = (question, decomp) => {
    setHistoryMessages([]);
    submit(question, decomp);
  };

  const handleSelectSession = async (sId) => {
    try {
      const res = await fetchSessionMessages(sId);
      setHistoryMessages(res.messages || []);
    } catch (e) {
      console.error("Failed to load session messages:", e);
    }
  };

  const handleNewSessionClick = () => {
    setHistoryMessages([]);
    startNewSession();
  };

  return (
    <>
      {/* ── Header ── */}
      <header className="app-header">
        <div className="app-container">
          <div className="header-inner">
            <div className="header-brand">
              <span className="header-icon">📈</span>
              <div>
                <div className="header-title">RAG Finance VN</div>
                <div className="header-subtitle">Vietnamese Financial News Q&amp;A</div>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <button
                type="button"
                className="nav-action-btn"
                onClick={() => setIsHistoryOpen(true)}
                title="View past sessions"
              >
                📜 History
              </button>

              <button
                type="button"
                className="nav-action-btn"
                onClick={() => setIsAdminOpen(true)}
                title="View Enterprise Observability & Audit Log"
              >
                📊 Admin &amp; Metrics
              </button>

              <a
                href="https://huggingface.co/spaces/thong7d/financial-news-rag"
                target="_blank"
                rel="noopener noreferrer"
                style={{ fontSize: "0.75rem", color: "var(--text-muted)", textDecoration: "none", marginLeft: "0.25rem" }}
              >
                🤗 Demo ↗
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* ── Chat History Sidebar ── */}
      <ChatHistorySidebar
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        userId={userId}
        activeSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewSessionClick}
      />

      {/* ── Enterprise Admin Dashboard Modal ── */}
      <AdminDashboard isOpen={isAdminOpen} onClose={() => setIsAdminOpen(false)} />

      {/* ── Hero ── */}
      <section className="hero-section">
        <div className="app-container">
          <h1 className="hero-title">
            Intelligent Q&amp;A on<br />
            Vietnamese Finance
          </h1>
          <p className="hero-sub">
            Specialized RAG system — 10,000+ Vietnamese financial news articles
            from 2015–2024. Answers are cited and verifiable.
          </p>
        </div>
      </section>

      {/* ── Main ── */}
      <main className="main-content">
        <div className="app-container">
          {/* Input + Sample Chips + Decompose Toggle */}
          <ChatInput
            onSubmit={handleSubmit}
            isLoading={isLoading}
            showSamples={showSamples}
            decompose={decompose}
            onToggleDecompose={() => setDecompose((v) => !v)}
          />

          {/* Status */}
          <StatusBar
            status={status}
            error={error}
            progressStep={progressStep}
            etaRemaining={etaRemaining}
            activeModel={activeModel}
          />

          {/* History Messages View (when loading session from sidebar) */}
          {historyMessages.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", marginBottom: "1.5rem" }}>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                📜 Past Messages in Session
              </div>
              {historyMessages.map((msg) => (
                <div key={msg.message_id} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  <div style={{ fontWeight: 600, color: "var(--accent-blue)", fontSize: "0.9rem" }}>
                    ❓ {msg.question}
                  </div>
                  <ChatMessage
                    answer={msg.answer}
                    isStreaming={false}
                    model={msg.model_used}
                    messageId={msg.message_id}
                    userId={userId}
                  />
                </div>
              ))}
            </div>
          )}

          {/* Decomposition Panel (shown when deep analysis is active) */}
          {subQueries && subQueries.length > 0 && (
            <DecompositionPanel subQueries={subQueries} isStreaming={status === "streaming" || status === "done"} />
          )}

          {/* Current Answer */}
          {(answer || isLoading) && historyMessages.length === 0 && (
            <ChatMessage
              answer={answer}
              isStreaming={status === "streaming" || status === "retrieving"}
              model={activeModel}
              messageId={messageId}
              userId={userId}
            />
          )}

          {/* Sources */}
          {sources.length > 0 && historyMessages.length === 0 && (
            <section className="sources-section" aria-label="Cited sources">
              <div className="sources-title">
                📄 Sources ({sources.length} retrieved passages)
              </div>
              <div className="sources-grid">
                {sources.map((s, i) => (
                  <SourceCard key={s.chunk_id || i} source={s} index={i + 1} />
                ))}
              </div>
            </section>
          )}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <div className="app-container">
          Powered by Gemini · Mistral · Qdrant Cloud · Neon PostgreSQL · Cohere Reranker ·&nbsp;
          <a
            href="https://github.com/thong7d/rag-vn-finance"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent-blue)", textDecoration: "none" }}
          >
            GitHub ↗
          </a>
        </div>
      </footer>
    </>
  );
}
