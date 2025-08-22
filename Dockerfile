# Imagen base ligera y reproducible
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencias nativas mínimas (OpenCV / Pillow runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no root
RUN useradd -m -u 10001 appuser
WORKDIR /app

# (1) Instalar dependencias primero para maximizar caché
#    Copiamos solo archivos que definen deps
COPY pyproject.toml ./
# Si tienes 'requirements.txt' o 'uv.lock', cópialos aquí también

# Actualizar pip y herramientas de build
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar el paquete (a partir de pyproject)
# Si tu paquete se llama 'app' o está editable, lo instalamos luego con el código.
# Aquí sólo resolvemos deps base si tu pyproject lo permite.
# Si necesitas el código para resolver deps, puedes saltar este paso.

# (2) Copiar código fuente
COPY src/ src/
COPY app/ app/

# Instalar el proyecto en modo editable (útil para depurar dentro del contenedor)
RUN pip install --no-cache-dir -e .

# Directorios de trabajo y variable del modelo
RUN mkdir -p model outputs && chown -R appuser:appuser /app
ENV MODEL_PATH=/app/model/conv_MLP_84.h5

# Cambiar a usuario no root
USER appuser

# Comando por defecto: ayuda del CLI
CMD ["python", "-m", "app.cli", "--help"]


