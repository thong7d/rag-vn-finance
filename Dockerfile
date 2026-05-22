FROM python:3.10-slim

# Ngăn Python ghi cache pyc và đẩy log ra terminal ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài đặt gcc cho các thư viện C/C++ dependencies (nếu có)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sao chép và cài đặt requirements
COPY requirements.txt .

# Cài đặt các thư viện
RUN pip install --default-timeout=1000 -r requirements.txt gradio fastapi uvicorn

# Sao chép mã nguồn cốt lõi và file khởi chạy
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY app.py /app/app.py

# Expose port mặc định của Gradio
EXPOSE 7860

# Lệnh khởi chạy
CMD ["python", "app.py"]
