/**
 * SourceCard — displays a single retrieved passage with relevance score bar,
 * collapsible excerpt, and link to original article.
 *
 * Props:
 *   source: { chunk_id, text, title, url, score }
 *   index: number (1-based display index)
 */
export default function SourceCard({ source, index }) {
  const { title, text, url, score } = source;

  // Score is Cohere relevance score (0–1). Normalize for the bar.
  const barWidth = Math.min(Math.round(score * 100), 100);
  const scoreLabel = score.toFixed(4);

  const shortTitle = title?.trim() || `Đoạn ngữ cảnh ${index}`;
  const excerpt = text?.trim().slice(0, 240) + (text?.length > 240 ? "..." : "");

  return (
    <div className="source-card" role="article" aria-label={`Nguồn ${index}: ${shortTitle}`}>
      <div className="source-card-header">
        <span className="source-index">{index}</span>
        <span className="source-title">{shortTitle}</span>
        <span className="source-score">{scoreLabel}</span>
      </div>

      <div className="score-bar-track" title={`Relevance: ${scoreLabel}`}>
        <div
          className="score-bar-fill"
          style={{ width: `${barWidth}%` }}
          role="meter"
          aria-valuenow={barWidth}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>

      <p className="source-excerpt">{excerpt}</p>

      {url && url !== "Không có liên kết" ? (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="source-link"
          aria-label={`Xem bài gốc: ${shortTitle}`}
        >
          🔗 Xem bài gốc ↗
        </a>
      ) : (
        <span className="source-link" style={{ cursor: "default", opacity: 0.4 }}>
          Không có liên kết
        </span>
      )}
    </div>
  );
}
