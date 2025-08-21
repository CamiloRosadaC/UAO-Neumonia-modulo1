# CHANGELOG – Proyecto Neumonía  

Este documento registra los cambios realizados en el proyecto desde su inicio hasta su estado actual.  

---

## 1 Desarrollo inicial – Código monolítico integrado  

Durante esta fase se construyó y ajustó el primer script único que contenía todo el flujo de la aplicación (GUI, modelo y lógica).  

### Cambios principales:
- **Cambio 1 – Importar TensorFlow**  
  Se añadió `import tensorflow as tf` para poder usar Keras y cargar el modelo preentrenado.  

- **Cambio 2 – Ajustes en Keras Backend**  
  Se incorporaron y luego comentaron utilidades de `keras.backend` para el manejo de gradientes y compatibilidad con Grad-CAM.  

- **Cambio 3 – Creación de `model_fun()`**  
  Se centralizó la carga del modelo en una función `model_fun()` para evitar duplicación de código y facilitar modificaciones posteriores.  

- **Cambio 4 – Actualización de `read_dicom_file()`**  
  Se mejoró la lectura de imágenes DICOM usando `dicom.dcmread`, normalizando intensidades y convirtiendo a RGB.  

- **Cambio 5 – Soporte JPG/PNG en el diálogo de apertura**  
  La función `load_img_file()` pasó de aceptar solo DICOM a soportar también **JPEG/PNG**, con detección automática de la extensión.  

- **Cambio 6 – Preprocesamiento robusto**  
  Se actualizó `preprocess()` para manejar imágenes en **RGB, BGR o escala de grises**, aplicando **CLAHE** y normalización estandarizada.  

- **Cambio 7 – Grad-CAM sin parámetros hardcodeados**  
  Se rediseñó Grad-CAM para no depender de capas o filtros fijos, haciéndolo dinámico y compatible con cualquier última capa convolucional.  

- **Cambio 8.1 y 8.2 – Manejo seguro de limpieza de widgets**  
  Se corrigieron los métodos `delete()` y `run_model()` para **evitar errores de Tkinter al borrar widgets** o limpiar imágenes antes de cargar nuevas.  

- **Cambio 9 – Compatibilidad con modo Eager/Graph**  
  Se ajustaron flags de TensorFlow (`disable_eager_execution`) para pruebas, y luego se migró a un enfoque estable con Eager.  

- **Cambio 10 – Grad-CAM con Eager Execution**  
  Se implementó una versión de Grad-CAM usando **tf.GradientTape**, eliminando dependencias de Keras backend.  

- **Cambio 11 – Grad-CAM refinado**  
  Se consolidó la versión final de Grad-CAM:  
  - Detección automática de la última capa convolucional.  
  - Uso de `GradientTape` y ponderación dinámica de canales.  
  - Normalización y superposición con OpenCV.  

- **Cambio 12 – Ajustes de layout en GUI (cédula paciente)**  
  Se reubicaron etiquetas y cajas de texto para mostrar correctamente la identificación del paciente.  

- **Cambio 13 – Visualización de textos y paneles de imagen**  
  Se reorganizó la interfaz (labels, cajas de texto, paneles de imágenes) para lograr una disposición clara, con paneles fijos de **250x250 px** para las imágenes de entrada y Grad-CAM.  

Con estos cambios, el código monolítico pasó de ser un prototipo inicial a una aplicación funcional que:  
- Carga imágenes (DICOM/JPG/PNG).  
- Preprocesa y clasifica con el modelo CNN.  
- Genera mapas Grad-CAM interpretables.  
- Muestra resultados en una **interfaz Tkinter** con opciones de exportar a PDF o CSV.  

---

## 2 Refactorización – División en módulos y directorios  

- Se reorganizó el código en una estructura modular dentro de `src/` (`utils.py`, `model.py`, `gradcam.py`).  
- Se crearon carpetas para **ejemplos** (`examples/`) y **resultados** (`results/`).  

---

## 3 Documentación y configuración del entorno  

- Se creó y amplió el `README.md`.  
- Se añadieron **`requirements.txt`** y **`pyproject.toml`** para instalación vía `pip` o `uv`.  
- Se estableció un `main.py` como punto de entrada.  

---

## 4 Contenerización – Dockerfile y .dockerignore  

- Se configuró un **Dockerfile** optimizado para ejecutar el proyecto en contenedores.  
- Se añadió `.dockerignore` para excluir archivos innecesarios en la build.  

---

## 5 Revisión de gitignore y publicación en GitHub  

- Se ajustó `.gitignore` para excluir entornos virtuales, resultados y cachés de Python.  
- Se realizó el **primer commit oficial** con la estructura final y el proyecto fue publicado en **GitHub**.  

---

## 6 Correcciones y mejoras de UX 

> Objetivo: impedir acciones inválidas, evitar choques de archivos y organizar salidas para una experiencia más clara.  
> **Archivo afectado**: `app/gui.py`  
> **Nuevos directorios**: `resultados/`, `resultados/reportes/`, `resultados/historial/`

- **Bloqueos y mensajes preventivos**
  - Si se intenta predecir sin imagen: aviso **“Carga un nuevo archivo antes de analizar.”**
  - Tras **Borrar**, el botón **Predecir** queda **deshabilitado** hasta cargar una imagen nueva.
  - **Exportaciones protegidas** (Guardar/ PDF): se validan 3 casos con mensajes específicos:
    1. **Falta la cédula.**
    2. **No se ha cargado una imagen.**
    3. **No se ha realizado una predicción.**

- **Campos no editables**
  - **Resultado** y **Probabilidad** ahora son `ttk.Entry(state="readonly")` con `StringVar` para evitar edición manual y entradas inconsistentes.

- **Limpieza de estado al cargar imagen**
  - Al seleccionar nueva imagen se **limpian** heatmap, resultado y probabilidad; así se evitan confusiones con resultados anteriores.
  - En esa misma acción, se **deshabilitan** Guardar/PDF hasta que haya **cédula + nueva predicción**.

- **Habilitado inteligente de exportaciones**
  - Guardar y PDF están **deshabilitados por defecto** y se habilitan solo cuando:
    - Hay **cédula** (entrada `ID`) y
    - Existe **predicción** (resultado presente).
  - Se actualizan dinámicamente con `trace` del `StringVar` de cédula.

- **Rutas de salida organizadas**
  - CSV se guarda en: `resultados/historial/historial.csv`.
  - PDFs se guardan en: `resultados/reportes/ReporteN.pdf`.
  - La estructura de carpetas se **crea automáticamente** si no existe.

- **Evitar conflictos de nombres de reporte**
  - Al generar PDF se busca el **siguiente índice libre** (`Reporte0.pdf`, `Reporte1.pdf`, …) en `resultados/reportes/`.
  - Se elimina el `.jpg` temporal utilizado para capturar la ventana.

- **Textos consistentes**
  - Mensajería de diálogos estandarizada (Predicción/Guardar/PDF/Borrar) para feedback claro y accionable.

---

# Estado final  

- **Código modular** con pipeline estable y Grad-CAM robusto.  
- **GUI pulida** con validaciones y estados coherentes (botones y campos).  
- **Resultados organizados** en carpetas dedicadas (`resultados/reportes`, `resultados/historial`).  
- **Exportaciones seguras** (sin archivos en blanco, sin sobrescrituras accidentales).
