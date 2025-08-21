# Herramienta para la detección rápida de neumonía

Deep Learning aplicado a imágenes radiográficas de tórax (DICOM/JPG/PNG) para clasificarlas en:

1. **Neumonía bacteriana**
2. **Neumonía viral**
3. **Sin neumonía**

Incluye explicación visual **Grad-CAM**, que superpone un mapa de calor sobre la radiografía para resaltar las regiones que más influyen en la predicción.

> ⚠️ **Aviso**: Proyecto con fines **académicos**. **No** usar para diagnóstico médico real.

---

## Instalación

Este repo soporta **dos** formas de instalar dependencias:

### ✅ Opción A (recomendada): `pyproject.toml` como fuente única de *runtime*
Usa `setuptools` + *editable install*. Puedes usar `uv` (recomendado) o `pip`.

#### **Con `uv`:**

uv venv
uv pip install -e .
uv pip install -r requirements-dev.txt (opcional: herramientas de desarrollo)

#### Con `pip` clásico

  python -m venv .venv

  pip install -e .
  pip install -r requirements-dev.txt    (opcional)

### Opcion B (respaldo): requirements.txt

  Mantenemos requirements.txt como respaldo. Úsalo solo si no quieres instalar el paquete con -e .

  #### con uv:

  uv venv
  uv pip install -r requirements.txt
  uv pip install -r requirements-dev.txt     (opcional)

  #### Con pip: (mismo patrón que arriba, sustituyendo pip install -e . por -r requirements.txt)
  nota: Evite duplicar dependencias en ambos sitios. La fuente canónica de runtime es pyproject.toml; requirements.txt queda como plan B

## Ejecución (GUI)

### tras instalar (Opción A o B)
  ##### forma canónica como módulo:
  python -m app.gui

  ##### si instalaste con -e . (pyproject), también tienes script:
  uao-neumonia-gui

  ##### si dejaste wrapper:
  python main.py

## CLI (sin GUI, útil para Docker/automatizar)

  ### si instalaste con -e .
  uao-neumonia-cli --img samples/bacteria.jpg --out outputs

  ### como módulo (vale en cualquier caso):
  python -m app.cli --img samples/bacteria.jpg --out outputs


## Uso de la interfaz
  
  1. Ingrese la cédula del paciente.
  2. Clic en Cargar Imagen y seleccione un archivo (.dcm, .jpg/.jpeg o .png).
  3. Clic en Predecir → verá el Resultado, la Probabilidad y el Heatmap (Grad-CAM).
  4. Guardar → añade una fila a historial.csv (cédula, etiqueta, probabilidad).
  5. PDF → exporta una captura de la ventana como ReporteN.pdf.
  6. Borrar → limpia todos los campos y deshabilita Predecir hasta cargar una nueva imagen.

## Estructura del proyecto (desacoplado)

  UAO-Neumonia/
  ├─ app/
  │  ├─ __init__.py
  │  ├─ gui.py      # Interfaz Tkinter (orquestación)
  │  └─ cli.py      # Modo línea de comandos (inferencia + guardado de heatmap)
  ├─ src/
  │  ├─ __init__.py
  │  ├─ io_imgs.py      # Lectura DICOM/JPG/PNG → RGB + PIL para GUI
  │  ├─ preprocess.py   # 512x512, gris, CLAHE, normalización, batch
  │  ├─ model.py        # Carga del modelo (.h5) → model_fun()
  │  ├─ explain.py      # Grad-CAM (tf.GradientTape) → heatmap superpuesto
  │  └─ inference.py    # predict() → (label, proba, heatmap)
  ├─ model/             # (vacío en git; montar .h5 en runtime)
  ├─ outputs/           # (resultados de CLI/GUI)
  ├─ samples/           # (imágenes locales de prueba; no versionadas)
  ├─ tests/             # Pruebas unitarias (pytest)
  ├─ pyproject.toml
  ├─ requirements.txt         # respaldo (no canónico)
  ├─ requirements-dev.txt     # herramientas dev (pytest, black, ruff…)
  ├─ pytest.ini
  ├─ Dockerfile
  ├─ .dockerignore
  ├─ .gitignore
  └─ README.md

## Pruebas (pytest)
  Ejecuta desde la raíz:

  uv run -m pytest -q
  o
  pytest -q

  ### Incluye (mínimo):

    - tests/test_preprocess.py: valida shape y dtype de preprocess().
    - tests/test_inference_smoke.py: “smoke test” de predict() con monkeypatch (sin depender del .h5 real).

## Docker (CLI)

Construir:

docker build -t uao-neumonia-cli .

Ejecutar (montando modelo e imágenes):

### Windows PowerShell
docker run --rm `
  -v "$PWD\model:/app/model" `
  -v "$PWD\samples:/app/samples" `
  -v "$PWD\outputs:/app/outputs" `
  -e MODEL_PATH=/app/model/conv_MLP_84.h5 `
  uao-neumonia-cli python -m app.cli --img /app/samples/bacteria.jpg --out /app/outputs

(macOS/Linux (equivalente con \ y $PWD))


## Configuración del modelo

Por defecto se busca model/conv_MLP_84.h5. Para cambiar la ruta:
  
  # Windows (PowerShell)

    $env:MODEL_PATH = "C:\ruta\mi_modelo.h5"
    uv run -m app.gui

  # macOS/Linux

  export MODEL_PATH=/ruta/mi_modelo.h5
  uv run -m app.gui

## Solución de problemas

## Dependencias principales

- Runtime: definidas en pyproject.toml (dependencies).
requirements.txt se mantiene como respaldo (instalación alternativa).

- Dev: requirements-dev.txt (pytest, pytest-cov, black, ruff)

## Acerca del modelo (resumen)
  CNN eficiente para rayos X de tórax con skip connections y capas Dense finales.
  Regularización con Dropout y explicación por Grad-CAM (pesos por gradiente promedio y suma ponderada de mapas de activación).

## Licencia

### MIT.
::contentReference[oaicite:0]{index=0}

## Proyecto realizado por:

Camilo Eduardo Rosada Caicedo - https://github.com/CamiloRosadaC

