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

### Opción A (recomendada): `pyproject.toml` como fuente única de *runtime*
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
  
  1. Cargar imagen y seleccionar archivo (.dcm, .jpg/.jpeg o .png).
  2. Ingrese la cédula del paciente.
  3. Clic en Predecir → verá el Resultado, la Probabilidad y el Heatmap (Grad-CAM).
  4. Con cédula + predicción disponibles, se habilitan:
    - Guardar → agrega una fila a resultados/historial/historial.csv (cédula, etiqueta, probabilidad).
    - PDF → exporta una captura como resultados/reportes/ReporteN.pdf.

  5. Borrar → limpia todos los campos y deshabilita Predecir hasta cargar una nueva imagen.

    ### notas: 
    - Los campos Resultado y Probabilidad son solo lectura.

    - La app evita acciones inválidas con mensajes claros:

      * “Falta la cédula.”

      * “No se ha cargado una imagen.”

      * “No se ha realizado una predicción.”

    - Al cargar una nueva imagen se limpian resultado/heatmap previos.

    - Tras Borrar, Predecir queda deshabilitado hasta cargar otra imagen.

## Estructura de resultados
Al ejecutar la app se crean automáticamente estas carpetas: 
<img width="502" height="83" alt="image" src="https://github.com/user-attachments/assets/76cc0b03-044d-4d35-9116-4081cdee00a0" />

- Los reportes se guardan como resultados/reportes/ReporteN.pdf, donde N es el primer índice libre (no se sobreescriben aunque cierres/abras la app).

- La imagen temporal utilizada para generar el PDF se elimina automáticamente.

## Estructura del proyecto (desacoplado)
<img width="609" height="642" alt="image" src="https://github.com/user-attachments/assets/07ba9e8e-924a-40ba-874c-b5688f88bea6" />

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

## Solución de problemas comunes

- Asegúrate de que MODEL_PATH apunta a un .h5 válido.

- Si no se habilita Predecir, verifica que has cargado una imagen.

- Si Guardar/PDF están deshabilitados, revisa que haya cédula + predicción.

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

