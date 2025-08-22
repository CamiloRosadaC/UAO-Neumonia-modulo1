"""Paquete src: núcleo de la herramienta de detección de neumonía.

Incluye:
- io_imgs: lectura de imágenes DICOM/JPG/PNG
- preprocess: funciones de preprocesamiento
- model: construcción del modelo de clasificación
- explain: Grad-CAM para interpretabilidad
- inference: predicción con etiquetas y probabilidades
"""

from .io_imgs import read_dicom_file, read_jpg_file
from .preprocess import preprocess
from .model import model_fun
from .explain import grad_cam
from .inference import LABELS, predict

__all__ = [
    "read_dicom_file",
    "read_jpg_file",
    "preprocess",
    "model_fun",
    "grad_cam",
    "predict",
    "LABELS",
]

