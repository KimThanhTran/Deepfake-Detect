FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small.
# For GPU: change the index URL to https://download.pytorch.org/whl/cu121
# and add a `deploy` section with GPU reservation in docker-compose.yml.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "gradio>=4" numpy Pillow

COPY networks/ networks/
COPY gradio_ui.py util.py ./

EXPOSE 7860

CMD ["python", "gradio_ui.py"]
