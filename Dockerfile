# Imagen base reproducible y ligera
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias nativas para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar con pyproject
COPY pyproject.toml README.md ./
COPY app/ app/
COPY src/ src/
RUN pip install --no-cache-dir -e .

# (si quieres dev tools dentro de la imagen, añade)
# COPY requirements-dev.txt .
# RUN pip install --no-cache-dir -r requirements-dev.txt

# Copiar código
COPY src/ src/
COPY app/ app/

# Directorios de trabajo y variable del modelo
RUN mkdir -p model outputs
ENV MODEL_PATH=/app/model/conv_MLP_84.h5

# Comando por defecto: ayuda del CLI
CMD ["python", "-m", "app.cli", "--help"]

