import { useState, useRef, useEffect } from "react";

/**
 * ChatInput — question text area + submit button.
 *
 * Props:
 *   onSubmit(question: string) - called when user submits
 *   isLoading: bool
 */
export default function ChatInput({ onSubmit, isLoading }) {
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
    onSubmit(q);
    setValue("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const PLACEHOLDER_EXAMPLES = [
    "Lợi nhuận của Vietcombank quý 1/2023 là bao nhiêu?",
    "Tình hình thị trường bất động sản 2024 như thế nào?",
    "VIC có kế hoạch phát hành cổ phiếu không?",
  ];

  return (
    <div className="chat-input-wrapper">
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
        <span className="char-count">
          {value.length > 0 ? `${value.length} ký tự · Enter để gửi` : "Shift+Enter để xuống dòng"}
        </span>
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
    </div>
  );
}
