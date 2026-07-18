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
        🤖 Câu trả lời
        {model && (
          <span style={{ marginLeft: "auto", fontWeight: 400, color: "var(--accent-green)", textTransform: "none", letterSpacing: 0 }}>
            via {model}
          </span>
        )}
      </div>
      <div className={`answer-body ${isEmpty ? "empty" : ""}`}>
        {isEmpty && !isStreaming ? (
          "Câu trả lời sẽ xuất hiện ở đây sau khi bạn đặt câu hỏi..."
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
