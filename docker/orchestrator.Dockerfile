FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY adas_stack /app/adas_stack
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "adas_stack.pipeline.orchestrator", "--frames", "30", "--target-fps", "12"]
