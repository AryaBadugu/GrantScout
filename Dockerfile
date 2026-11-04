FROM python:3.11-slim

WORKDIR /app

# System deps for Docling (pdf + image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 poppler-utils libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Streamlit config
COPY .streamlit/config.toml .streamlit/config.toml

# Port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
