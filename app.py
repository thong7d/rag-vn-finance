"""
app.py — Gradio UI for Vietnamese Financial News RAG System (Phase 9)
Configured with default_concurrency_limit=1 to protect Auto-Fallback limits.
"""

import os
import json
import faiss
import pandas as pd
import gradio as gr
import requests
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from requests.exceptions import ConnectionError, Timeout
from huggingface_hub import InferenceClient

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
# HF Embedding API integration
# ---------------------------------------------------------------------------

class HFAPIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(f"HF API Error {status_code}: {message}")

def is_retryable_exception(exception):
    # Tự động bắt lỗi NameResolutionError/DNS drop (nằm trong ConnectionError) và Timeout để retry
    if isinstance(exception, (ConnectionError, Timeout)):
        logger.warning(f"Phát hiện lỗi mạng hoặc DNS tạm thời, đang kích hoạt hàm thử lại tự động: {exception}")
        return True
    # Bắt các lỗi overload phía server Hugging Face
    if isinstance(exception, HFAPIError):
        return exception.status_code in [429, 503]
    return False

class HFEmbeddingAPI:
    def __init__(self, model_id: str = "intfloat/multilingual-e5-large", token: str = None):
        self.model_id = model_id
        self.token = token or os.environ.get("HF_TOKEN", "")
        # Khởi tạo client chính thức của Hugging Face
        self.client = InferenceClient(token=self.token)
        
        if not self.token:
            logger.warning("HF_TOKEN chưa được thiết lập!")

    def _call_api(self, payload):
        # Tách mảng văn bản đầu vào từ payload gốc
        texts = payload["inputs"]
        
        # InferenceClient tự động tối ưu định tuyến và retry nội bộ trong Cluster Space
        response = self.client.feature_extraction(texts, model=self.model_id)
        
        # Chuyển đổi kết quả trả về thành list chuẩn để đồng bộ dữ liệu với hàm encode
        if isinstance(response, np.ndarray):
            return response.tolist()
        return response

    @retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_exponential(multiplier=2, min=4, max=16),
        stop=stop_after_attempt(6),
        reraise=True
    )
    def _call_api_with_retry(self, payload):
        return self._call_api(payload)

    def encode(self, texts, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
            
        prefixed_texts = [t if t.startswith("query: ") else f"query: {t}" for t in texts]
        payload = {"inputs": prefixed_texts}
        
        try:
            logger.info(f"Sending {len(prefixed_texts)} texts to HF Inference API via Hub Client...")
            response_json = self._call_api_with_retry(payload)
        except Exception as e:
            logger.error(f"Kênh Embedding sập hoàn toàn sau tất cả các lượt thử lại: {e}")
            return None

        if isinstance(response_json, dict) and "error" in response_json:
            logger.error(f"HF Inference API error: {response_json['error']}")
            return None

        emb_arr = np.array(response_json, dtype=np.float32)
        if emb_arr.ndim == 1:
            emb_arr = emb_arr.reshape(1, -1)
        elif emb_arr.ndim == 3:
            emb_arr = np.mean(emb_arr, axis=1)
            
        return emb_arr

# Load embedding model via HF Inference API
logger.info("Initializing Hugging Face Inference API for multilingual-e5-large...")
embedding_model = HFEmbeddingAPI()

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

dense_retriever = DenseRetriever(faiss_index, dense_chunk_ids, embedding_model)

# Load BM25
logger.info(f"Loading BM25 từ {BM25_DIR}...")
bm25_index, sparse_chunk_ids = load_bm25_index(BM25_DIR, STRATEGY)
sparse_retriever = SparseRetriever(bm25_index, sparse_chunk_ids)

# Hybrid Retriever
retriever = HybridRetriever(dense_retriever, sparse_retriever, rrf_k=60)
logger.info("Hệ thống RAG đã khởi tạo thành công!")

# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def process_query(question: str):
    if not question.strip():
        return "Vui lòng nhập câu hỏi.", ""

    try:
        logger.info(f"Truy vấn: {question}")
        
        # Kiểm tra tính khả dụng của mô hình Embedding nhúng
        embedding_available = True
        try:
            # Thử nghiệm trích xuất vector cho câu hỏi của người dùng
            query_vector = embedding_model.encode(question)
            if query_vector is None:
                embedding_available = False
        except Exception:
            embedding_available = False

        if embedding_available:
            # CHẾ ĐỘ TIÊU CHUẨN: Thực hiện truy xuất kết hợp Hybrid (FAISS + BM25)
            retrieved_results = retriever.retrieve(question, top_k=10)
            mode_status = ""
        else:
            # CHẾ ĐỘ DỰ PHÒNG KHẨN CẤP: Ép hệ thống chạy độc lập bằng bộ từ khóa BM25 Offline
            logger.warning("[FALLBACK ACTIVATED] Chuyển đổi hệ thống sang trạng thái chạy BM25 thuần túy.")
            retrieved_results = sparse_retriever.retrieve(question, top_k=10)
            mode_status = "⚠️ *Hệ thống đang tự động vận hành ở Chế độ Dự phòng Khẩn cấp (BM25 Từ khóa) do lỗi nghẽn mạch DNS của đám mây Hugging Face. Kết quả trả ra vẫn đảm bảo trích dẫn nguồn chính xác.*\n\n"

        contexts = []
        source_texts = []
        for i, (cid, score) in enumerate(retrieved_results):
            text = chunk_text_map.get(cid, "")
            title = chunk_title_map.get(cid, "Không xác định")
            if text.strip():
                contexts.append(text)
                source_texts.append(f"**[Nguồn {i+1}] {title}** (Score: {score:.4f})\n{text}\n")

        # 2. Generation (Gọi mô hình ngôn ngữ lớn qua tầng Auto-Fallback 3 lớp sẵn có)
        answer = generate_answer(question, contexts)
        
        # Đính kèm cảnh báo chế độ vận hành vào đầu văn bản phản hồi
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
                placeholder="Ví dụ: Lợi nhuận của Vietcombank quý 3 năm 2023 là bao nhiêu?",
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