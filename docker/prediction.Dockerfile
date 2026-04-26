FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY adas_stack /app/adas_stack
RUN pip install --no-cache-dir -e .

EXPOSE 8002
CMD ["uvicorn", "adas_stack.services.prediction_service.app:app", "--host", "0.0.0.0", "--port", "8002"]
