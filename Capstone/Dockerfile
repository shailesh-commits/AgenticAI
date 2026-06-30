FROM python:3.11-slim

WORKDIR /app

# System deps for Playwright / lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libxml2-dev libxslt-dev wget curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (only needed when mock_mode=false)
RUN playwright install chromium --with-deps || true

COPY . .

# Ensure data directory exists for SQLite
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "main.py"]
