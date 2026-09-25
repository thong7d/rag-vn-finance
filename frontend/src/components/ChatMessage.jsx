import { useState } from "react";
import { submitFeedback } from "../services/api";

/**
 * ChatMessage — displays the streamed answer with a blinking cursor while loading,
 * and thumbs up/down feedback controls once complete.
 *
 * Props:
 *   answer: string
 *   isStreaming: bool
 *   model: string
 *   messageId: string | null
 *   userId: string
 */
export default function ChatMessage({ answer, isStreaming, model, messageId, userId }) {
  const isEmpty = !answer;
  const [voted, setVoted] = useState(null); // "up" | "down" | null
  const [showCommentBox, setShowCommentBox] = useState(false);
  const [commentText, setCommentText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState(false);

  const handleVote = async (voteType) => {
    if (!messageId || isSubmitting) return;
    setVoted(voteType);
    if (voteType === "down") {
      setShowCommentBox(true);
    } else {
      setShowCommentBox(false);
      await sendFeedback(voteType, "");
    }
  };

  const sendFeedback = async (voteType, comment) => {
    if (!messageId) return;
    setIsSubmitting(true);
    try {
      await submitFeedback(messageId, userId, voteType, comment);
      setFeedbackSent(true);
    } catch (e) {
      console.error("Failed to submit feedback:", e);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCommentSubmit = (e) => {
    e.preventDefault();
    if (voted) {
      sendFeedback(voted, commentText);
      setShowCommentBox(false);
    }
  };

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

      {!isEmpty && !isStreaming && messageId && (
        <div className="feedback-bar">
          <span className="feedback-label">Was this answer helpful?</span>
          <button
            type="button"
            className={`feedback-btn ${voted === "up" ? "active-up" : ""}`}
            onClick={() => handleVote("up")}
            title="Helpful"
            disabled={isSubmitting}
          >
            👍
          </button>
          <button
            type="button"
            className={`feedback-btn ${voted === "down" ? "active-down" : ""}`}
            onClick={() => handleVote("down")}
            title="Needs improvement"
            disabled={isSubmitting}
          >
            👎
          </button>

          {feedbackSent && (
            <span className="feedback-thanks">
              {voted === "up" ? "Thanks for your feedback! ✨" : "Thank you, we will work to improve this answer! 🙏"}
            </span>
          )}

          {showCommentBox && (
            <form onSubmit={handleCommentSubmit} className="feedback-comment-form">
              <input
                type="text"
                className="feedback-comment-input"
                placeholder="What could be improved? (Optional)"
                value={commentText}
                onChange={(e) => setCommentText(e.target.value)}
              />
              <button type="submit" className="feedback-submit-btn" disabled={isSubmitting}>
                Submit
              </button>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
