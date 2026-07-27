import { useState, useRef, useEffect } from "react";

/**
 * ChatInput — question text area + submit button + sample question chips.
 *
 * Props:
 *   onSubmit(question: string, decompose: boolean) - called when user submits
 *   isLoading: bool
 *   showSamples: bool - whether to show sample question chips
 *   decompose: bool - current decompose toggle state (passed from parent)
 *   onToggleDecompose: () => void - callback to toggle decompose mode
 */
export default function ChatInput({ onSubmit, isLoading, showSamples = true, decompose = false, onToggleDecompose }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [value]);

  const handleSubmit = () => {
    const q = value.trim();
    if (!q || isLoading) return;
    onSubmit(q, decompose);
    setValue("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSampleClick = (question) => {
    if (isLoading) return;
    onSubmit(question, decompose);
  };

  const PLACEHOLDER_EXAMPLES = [
    "Lợi nhuận của Vietcombank quý 1/2023 là bao nhiêu?",
    "Tình hình thị trường bất động sản 2024 như thế nào?",
    "VIC có kế hoạch phát hành cổ phiếu không?",
  ];

  return (
    <div className={`chat-input-wrapper${decompose ? " chat-input-wrapper--decompose" : ""}`}>
      <label className="chat-input-label" htmlFor="chat-input">
        📊 Câu hỏi tài chính
      </label>
      <textarea
        id="chat-input"
        ref={textareaRef}
        className="chat-textarea"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={PLACEHOLDER_EXAMPLES[0]}
        rows={2}
        disabled={isLoading}
        aria-label="Nhập câu hỏi về tài chính Việt Nam"
      />
      <div className="chat-input-footer">
        {/* Deep Analysis toggle */}
        <div className="decompose-toggle-area">
          <button
            type="button"
            className={`decompose-toggle${decompose ? " decompose-toggle--active" : ""}`}
            onClick={onToggleDecompose}
            aria-pressed={decompose}
            aria-label="Toggle deep analysis mode"
          >
            <span className="toggle-track">
              <span className="toggle-thumb" />
            </span>
            <span className="toggle-label">🔍 Deep Analysis</span>
          </button>
        </div>
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={isLoading || !value.trim()}
          aria-label="Gửi câu hỏi"
        >
          {isLoading ? (
            <>
              <span className="spinner" /> Đang xử lý...
            </>
          ) : (
            <>
              ↗ Gửi câu hỏi
            </>
          )}
        </button>
      </div>

      {/* Sample question chips — only shown in idle state */}
      {showSamples && !isLoading && (
        <div className="sample-questions" aria-label="Sample questions">
          {PLACEHOLDER_EXAMPLES.map((q, i) => (
            <button
              key={i}
              className="sample-chip"
              onClick={() => handleSampleClick(q)}
              type="button"
            >
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
