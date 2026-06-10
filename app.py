"""
app.py — Gradio UI for Vietnamese Financial News RAG System (Phase 9)
Configured with default_concurrency_limit=1 to protect Auto-Fallback limits.
Phase 9 enhancements:
  - URL trích dẫn trong Retrieved Context để người dùng kiểm chứng.
  - Cohere Reranker (rerank-v3.0) để lọc top-5 ngữ cảnh chất lượng cao cho LLM.
"""

import os
import json
import faiss
import pandas as pd
import gradio as gr
import requests
import numpy as np
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from requests.exceptions import ConnectionError, Timeout
from huggingface_hub import InferenceClient
import cohere

from src.utils import setup_logger
from src.retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from src.indexing import load_bm25_index
from src.generation import generate_answer

logger = setup_logger("GradioApp")

# ---------------------------------------------------------------------------
# Global initialization
# ---------------------------------------------------------------------------

# Hardcoded best config for deployment to save RAM (Max ~6GB)
STRATEGY = "fixed_size"
INDEXES_DIR = os.environ.get("INDEXES_DIR", "indexes")
BM25_DIR = os.environ.get("BM25_DIR", "bm25")

logger.info("Khởi tạo hệ thống RAG...")

# ---------------------------------------------------------------------------
# HF Embedding API integration (OpenRouter Cloud Provider)
# ---------------------------------------------------------------------------

import time  # Thêm import thư viện time ở đầu tệp nếu chưa có

class OpenRouterEmbeddingAPI:
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.model_name = "intfloat/multilingual-e5-large"
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY chưa được thiết lập trong Settings!")

    def encode(self, texts, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
            
        prefixed_texts = [t if t.startswith("query: ") else f"query: {t}" for t in texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": prefixed_texts
        }
        
        # Bắt đầu đo hiệu năng thời gian thực
        start_time = time.time()
        logger.info(f"[OPENROUTER] Đang gửi yêu cầu nhúng {len(prefixed_texts)} văn bản sang model '{self.model_name}'...")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=15)
            latency = time.time() - start_time # Tính toán tổng thời gian phản hồi mạng
            
            if response.status_code == 200:
                data = response.json()["data"]
                embeddings = [item["embedding"] for item in data]
                emb_array = np.array(embeddings, dtype=np.float32)
                
                # IN LOG CHI TIẾT ĐỂ XÁC THỰC THÀNH CÔNG
                logger.info(
                    f"[SUCCESS] Gọi thành công OpenRouter API | Model: {self.model_name} | "
                    f"Thời gian phản hồi: {latency:.2f}s | Kích thước Vector trả về: {emb_array.shape}"
                )
                return emb_array
            else:
                logger.error(f"[API ERROR] OpenRouter trả về lỗi: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"[NETWORK ERROR] Không thể kết nối đến OpenRouter Endpoint: {e}")
            return None

# Kích hoạt thực thể kết nối API trực tiếp
embedding_model = OpenRouterEmbeddingAPI()

# Load FAISS
faiss_dir = os.path.join(INDEXES_DIR, STRATEGY)
logger.info(f"Loading FAISS từ {faiss_dir}...")
if not os.path.exists(faiss_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục FAISS: {faiss_dir}")

faiss_index = faiss.read_index(os.path.join(faiss_dir, 'index.faiss'))
with open(os.path.join(faiss_dir, 'chunk_ids.json'), 'r', encoding='utf-8') as f:
    dense_chunk_ids = json.load(f)

# Load metadata for Context display
df_meta = pd.read_parquet(os.path.join(faiss_dir, 'metadata.parquet'))
chunk_text_map = dict(zip(df_meta['chunk_id'], df_meta['text']))
chunk_title_map = dict(zip(df_meta['chunk_id'], df_meta.get('title', pd.Series(dtype=str))))
# URL map — khai báo an toàn để đề phòng lệch tên cột (url / link / source)
chunk_url_map = dict(zip(
    df_meta['chunk_id'],
    df_meta.get('url', df_meta.get('link', pd.Series("Không có liên kết", index=df_meta.index)))
))

dense_retriever = DenseRetriever(faiss_index, dense_chunk_ids, embedding_model)

# Load BM25
logger.info(f"Loading BM25 từ {BM25_DIR}...")
bm25_index, sparse_chunk_ids = load_bm25_index(BM25_DIR, STRATEGY)
sparse_retriever = SparseRetriever(bm25_index, sparse_chunk_ids)

# Hybrid Retriever
retriever = HybridRetriever(dense_retriever, sparse_retriever, rrf_k=60)
logger.info("Hệ thống RAG đã khởi tạo thành công!")

# ---------------------------------------------------------------------------
# Cohere Reranker — lọc top-5 từ tập ứng viên thô 30 kết quả
# ---------------------------------------------------------------------------

def rerank_with_cohere(query: str, candidates: list) -> list:
    """
    Nhận danh sách (chunk_id, score) thô từ Hybrid Retriever,
    gọi Cohere Rerank API để tái xếp hạng ngữ nghĩa sâu,
    và trả về top-5 kết quả dạng [(chunk_id, rerank_score), ...].

    Nếu COHERE_API_KEY chưa được thiết lập hoặc API lỗi,
    tự động fallback về top-5 kết quả thô ban đầu.
    """
    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if not cohere_key:
        logger.warning("[Reranker] COHERE_API_KEY chưa được thiết lập, bỏ qua bước Rerank.")
        return candidates[:5]

    # Chuẩn bị danh sách văn bản ứng viên để gửi cho Cohere
    docs = [chunk_text_map.get(cid, "") for cid, _ in candidates]
    # Lọc bỏ chuỗi rỗng để tránh lỗi API
    valid_pairs = [(cid_score, doc) for cid_score, doc in zip(candidates, docs) if doc.strip()]
    if not valid_pairs:
        return candidates[:5]

    valid_candidates, valid_docs = zip(*valid_pairs)

    try:
        co = cohere.Client(api_key=cohere_key)
        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=list(valid_docs),
            top_n=5,
        )
        # Ánh xạ ngược kết quả Cohere về cấu trúc [(chunk_id, score), ...]
        reranked = [
            (valid_candidates[result.index][0], result.relevance_score)
            for result in response.results
        ]
        logger.info(f"[Reranker] Cohere Rerank thành công — Top-5 được chọn từ {len(candidates)} ứng viên.")
        return reranked
    except Exception as e:
        logger.warning(f"[Reranker] Cohere Rerank thất bại ({e}), fallback về top-5 thô.")
        return candidates[:5]


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def process_query(question: str):
    if not question.strip():
        return "Vui lòng nhập câu hỏi.", ""

    try:
        logger.info(f"Truy vấn: {question}")
        
        try:
            # Bước 1: Truy xuất 30 ứng viên thô từ Hybrid Retriever (FAISS + BM25)
            # top_k=30 để Cohere Reranker có đủ không gian ứng viên tái xếp hạng
            retrieved_results = retriever.retrieve(question, top_k=30)
            mode_status = ""
        except Exception as net_err:
            # Chế độ dự phòng: Hạ cấp sang BM25 Offline khi OpenRouter sập
            logger.warning(f"[FALLBACK ACTIVATED] Kênh Dense gặp sự cố ({net_err}). Hạ cấp sang BM25 Offline.")
            retrieved_results = sparse_retriever.retrieve(question, top_k=30)
            mode_status = "⚠️ *Hệ thống đang tự động vận hành ở Chế độ Dự phòng Khẩn cấp (BM25 Từ khóa) do lỗi nghẽn mạch API của đám mây OpenRouter. Kết quả trả ra vẫn đảm bảo trích dẫn nguồn chính xác.*\n\n"

        # Bước 2: Rerank với Cohere để lọc top-5 ngữ cảnh chất lượng cao nhất
        top_results = rerank_with_cohere(question, retrieved_results)

        # Bước 3: Xây dựng contexts (text sạch cho LLM) và source_texts (có URL cho UI)
        contexts = []      # Chỉ chứa text sạch — KHÔNG chứa URL để không làm nhiễu LLM
        source_texts = []
        for i, (cid, score) in enumerate(top_results):
            text = chunk_text_map.get(cid, "")
            title = chunk_title_map.get(cid, "Không xác định")
            url = chunk_url_map.get(cid, "Không có liên kết")
            if text.strip():
                contexts.append(text)   # Text sạch cho LLM
                # Định dạng link Markdown cho Gradio — chỉ hiển thị ở UI, không vào LLM
                if url and url != "Không có liên kết":
                    link_md = f"[Xem bài gốc]({url})"
                else:
                    link_md = "*Không có liên kết nguồn*"
                source_texts.append(
                    f"**[Nguồn {i+1}] {title}** (Score: {score:.4f}) — {link_md}\n\n{text}\n"
                )

        # Bước 4: Sinh câu trả lời qua tầng Auto-Fallback 3 lớp
        answer = generate_answer(question, contexts)
        
        final_answer = f"{mode_status}{answer}"
        sources_formatted = "\n---\n".join(source_texts) if source_texts else "Không tìm thấy ngữ cảnh phù hợp."
        return final_answer, sources_formatted

    except Exception as e:
        logger.error(f"Lỗi hệ thống nghiêm trọng tại tầng UI: {e}")
        return f"Đã xảy ra lỗi cục bộ: {str(e)}", ""

# ---------------------------------------------------------------------------
# UI Definition
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📈 Vietnamese Financial News RAG System")
    gr.Markdown("Hệ thống hỏi đáp tài chính được trang bị Auto-Fallback 3 lớp (Groq ⚡ -> OpenRouter -> Gemini).")
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Câu hỏi của bạn", 
                placeholder="Ví dụ: Lợi nhuận của Vietcombank quý 1 năm 2022 là bao nhiêu?",
                lines=3
            )
            submit_btn = gr.Button("Gửi câu hỏi", variant="primary")
            
            answer_output = gr.Markdown(label="Câu trả lời")
            
        with gr.Column(scale=1):
            with gr.Accordion("Nguồn trích dẫn (Retrieved Context)", open=False):
                sources_output = gr.Markdown("Các đoạn thông tin được hệ thống truy xuất sẽ hiển thị ở đây.")
                
    submit_btn.click(
        fn=process_query,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )
    question_input.submit(
        fn=process_query,
        inputs=[question_input],
        outputs=[answer_output, sources_output]
    )

# ---------------------------------------------------------------------------
# App Launch (Concurrency Limit is MANDATORY to protect rate limits)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Giữ nguyên cấu hình queue bảo vệ rate limit
    demo.queue(default_concurrency_limit=1)
    
    # Ép buộc bind đúng cổng hạ tầng của HF Spaces mong muốn
    demo.launch(server_name="0.0.0.0", server_port=7860)