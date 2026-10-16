# CPU-only news keyword collector and offline test runner.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

COPY news_agent/requirements.txt /tmp/news-agent-requirements.txt

# Explicit CPU torch avoids unexpectedly downloading CUDA dependencies.
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      'torch>=2.5,<3' \
 && python -m pip install --no-cache-dir \
      -r /tmp/news-agent-requirements.txt 'pytest>=8,<10' \
 && python -m spacy download en_core_web_sm \
 && mkdir -p /hf-cache \
 && chmod 1777 /hf-cache

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "news_agent.cli"]
