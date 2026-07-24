import "./styles/index.css";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import SourceCard from "./components/SourceCard";
import StatusBar from "./components/StatusBar";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { answer, sources, isLoading, status, activeModel, error, progressStep, etaRemaining, submit } = useChat();

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
            <a
              href="https://huggingface.co/spaces/thong7d/financial-news-rag"
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: "0.75rem", color: "var(--text-muted)", textDecoration: "none" }}
            >
              🤗 Gradio Demo ↗
            </a>
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="hero-section">
        <div className="app-container">
          <h1 className="hero-title">
            Hỏi đáp <span>thông minh</span> về<br />
            Tài chính Việt Nam
          </h1>
          <p className="hero-sub">
            Hệ thống RAG chuyên biệt — Dữ liệu 10,000+ bài báo tài chính
            từ 2015–2024. Câu trả lời được trích dẫn nguồn và có thể kiểm chứng.
          </p>
        </div>
      </section>

      {/* ── Main ── */}
      <main className="main-content">
        <div className="app-container">
          {/* Input */}
          <ChatInput onSubmit={submit} isLoading={isLoading} />

          {/* Status */}
          <StatusBar status={status} error={error} progressStep={progressStep} etaRemaining={etaRemaining} activeModel={activeModel} />

          {/* Answer */}
          {(answer || isLoading) && (
            <ChatMessage
              answer={answer}
              isStreaming={status === "streaming" || status === "retrieving"}
              model={activeModel}
            />
          )}

          {/* Sources */}
          {sources.length > 0 && (
            <section className="sources-section" aria-label="Nguồn trích dẫn">
              <div className="sources-title">
                📄 Nguồn trích dẫn ({sources.length} đoạn ngữ cảnh)
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
          Powered by Gemini · Mistral · Qdrant Cloud · Cohere Reranker ·&nbsp;
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
