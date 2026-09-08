/**
 * ChatMessage — displays the streamed answer with a blinking cursor while loading.
 *
 * Props:
 *   answer: string
 *   isStreaming: bool
 *   model: string
 */
export default function ChatMessage({ answer, isStreaming, model }) {
  const isEmpty = !answer;

  return (
    <div className="answer-panel">
      <div className="panel-header">
        {isStreaming && <span className="panel-dot" />}
        🤖 Answer
        {model && (
          <span style={{ marginLeft: "auto", fontWeight: 400, color: "var(--accent-green)", textTransform: "none", letterSpacing: 0 }}>
            via {model}
          </span>
        )}
      </div>
      <div className={`answer-body ${isEmpty ? "empty" : ""}`}>
        {isEmpty && !isStreaming ? (
          "Your answer will appear here once you ask a question..."
        ) : (
          <>
            {answer}
            {isStreaming && <span className="cursor" aria-hidden="true" />}
          </>
        )}
      </div>
    </div>
  );
}
