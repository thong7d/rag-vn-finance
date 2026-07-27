FROM python:3.11-slim

# Tạo non-root user để tăng bảo mật
RUN useradd -m -u 1000 appuser
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

WORKDIR /app

# Copy thư mục hiện tại (backend) vào /app
COPY --chown=appuser ./requirements.txt /app/requirements.txt

# Cài đặt thư viện
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy toàn bộ code
COPY --chown=appuser . /app/

EXPOSE 8000

# Lệnh khởi chạy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
