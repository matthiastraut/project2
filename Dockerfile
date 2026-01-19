FROM python:3.12-slim

# Install uv by copying the binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Set environment variables for uv
# 1. We force the CPU-only version of torch and torchvision
# 2. We enable system python so uv installs into the container's python
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --locked

# Copy the model files
COPY model.onnx ./
COPY model.onnx.data ./

# Copy the application code
COPY model_deploy.py ./

# Expose the port FastAPI runs on
EXPOSE 9696

# Run the application
CMD ["uv", "run", "model_deploy.py"]
