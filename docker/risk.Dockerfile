FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY adas_stack /app/adas_stack
RUN pip install --no-cache-dir -e .

EXPOSE 8003
CMD ["uvicorn", "adas_stack.services.risk_service.app:app", "--host", "0.0.0.0", "--port", "8003"]
