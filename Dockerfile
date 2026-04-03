# ------------------------------------------------------------------ #
# Stage 1: builder                                                     #
# Install all Python dependencies into a virtual environment          #
# ------------------------------------------------------------------ #
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for scipy / torch compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------ #
# Stage 2: runtime                                                     #
# ------------------------------------------------------------------ #
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Make venv the default Python
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY semiyield/ ./semiyield/
COPY dashboard/ ./dashboard/
COPY pyproject.toml .

# Install the package itself (editable not needed in container)
RUN pip install --no-cache-dir -e . --no-deps

# Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run the dashboard
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
