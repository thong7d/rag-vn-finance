"""
app.py — Gradio UI for Vietnamese Financial News RAG System (Phase 9)
Configured with default_concurrency_limit=1 to protect Auto-Fallback limits.
"""

import os
import json
import pandas as pd
import gradio as gr
import requests
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
# Nhập đầy đủ cây ngoại lệ của thư viện requests
from requests.exceptions import RequestException, SSLError, ConnectionError, Timeout

from src.utils import setup_logger
logger = setup_logger("GradioApp")

class HFAPIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(f"HF API Error {status_code}: {message}")

def is_retryable_exception(exception):
    """
    Bộ lọc phân loại lỗi triệt để cho tầng Tenacity.
    Trả về True nếu lỗi có khả năng phục hồi khi thử lại.
    Trả về False nếu lỗi nghiêm trọng cần ngắt tiến trình để qua tầng Fallback.
    """
    # 1. Trường hợp lỗi bảo mật SSL (Hostname mismatch, Cert expired...)
    if isinstance(exception, SSLError):
        logger.error(f"[CRITICAL SSL] Lỗi chứng chỉ SSL hoặc lệch cổng định tuyến: {exception}. Ngắt truy cập HF.")
        return False
        
    # 2. Trường hợp lỗi Overload từ phía máy chủ API Hugging Face
    if isinstance(exception, HFAPIError):
        return exception.status_code in [429, 503]
        
    # 3. Trường hợp rớt gói tin DNS (ConnectionError) hoặc nghẽn hàng đợi (Timeout)
    if isinstance(exception, (ConnectionError, Timeout)):
        logger.warning(f"[NETWORK WARNING] Lỗi kết nối hoặc phân giải DNS tạm thời: {type(exception).__name__}. Đang thử lại...")
        return True
        
    return False

class HFEmbeddingAPI:
    def __init__(self, model_id: str = "intfloat/multilingual-e5-large", token: str = None):
        self.model_id = model_id
        self.token = token or os.environ.get("HF_TOKEN", "")
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
        
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=0 
        )
        self.session.mount("https://", adapter)

    def _call_api(self, payload):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        response = self.session.post(self.api_url, headers=headers, json=payload, timeout=10)
        
        if response.status_code != 200:
            raise HFAPIError(response.status_code, response.text)
            
        return response.json()

    @retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_exponential(multiplier=2, min=4, max=16),
        stop=stop_after_attempt(4),
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
            # Tiến trình gọi API có quản lý cấu trúc retry
            response_json = self._call_api_with_retry(payload)
        except RequestException as network_error:
            # BẮT TRIỆT ĐỂ: Mọi lỗi mạng phát sinh không thể cứu vãn sau cấu trúc retry
            logger.error(f"[RAG FALLBACK ACTIVATED] Kênh Embedding HF sập hoàn toàn do: {network_error}")
            # Ép buộc ném lỗi kiểm soát ra ngoài để hàm process_query kích hoạt tầng Fallback sang LLM độc lập
            raise RuntimeError(f"Hệ thống Embedding tạm thời gián đoạn. {str(network_error)}")

        if isinstance(response_json, dict) and "error" in response_json:
            raise Exception(f"HF Inference API error: {response_json['error']}")

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
        # 1. Retrieval
        logger.info(f"Truy vấn: {question}")
        retrieved_results = retriever.retrieve(question, top_k=10)
        
        contexts = []
        source_texts = []
        for i, (cid, score) in enumerate(retrieved_results):
            text = chunk_text_map.get(cid, "")
            title = chunk_title_map.get(cid, "Không xác định")
            if text.strip():
                contexts.append(text)
                source_texts.append(f"**[Nguồn {i+1}] {title}** (Score: {score:.4f})\n{text}\n")

        # 2. Generation (3-layer Auto-Fallback)
        answer = generate_answer(question, contexts)
        
        sources_formatted = "\n---\n".join(source_texts) if source_texts else "Không tìm thấy ngữ cảnh phù hợp."
        return answer, sources_formatted

    except Exception as e:
        logger.error(f"Lỗi hệ thống: {e}")
        return f"Đã xảy ra lỗi: {str(e)}", ""

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