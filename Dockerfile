FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY webapp ./webapp

# Copy everything else and install dependencies

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir ".[webapp]"

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["webapp/sado/webapp_full.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.baseUrlPath=sado"]
