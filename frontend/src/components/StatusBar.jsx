/**
 * StatusBar — shows current system status to the user.
 *
 * Props:
 *   status: "idle" | "retrieving" | "streaming" | "done" | "error"
 *   error: string
 */
export default function StatusBar({ status, error }) {
  if (status === "idle") return null;

  const configs = {
    retrieving: {
      cls: "loading",
      icon: <span className="spinner" />,
      text: "Đang truy xuất ngữ cảnh từ Qdrant Cloud...",
    },
    streaming: {
      cls: "loading",
      icon: <span className="spinner" />,
      text: "Đang sinh câu trả lời...",
    },
    done: {
      cls: "success",
      icon: "✓",
      text: "Hoàn thành.",
    },
    error: {
      cls: "error",
      icon: "✕",
      text: error || "Đã xảy ra lỗi. Vui lòng thử lại.",
    },
  };

  const config = configs[status];
  if (!config) return null;

  return (
    <div className={`status-bar ${config.cls}`} role="status" aria-live="polite">
      {config.icon}
      <span>{config.text}</span>
    </div>
  );
}
