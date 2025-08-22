#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrada/salida de imágenes para el pipeline.

- DICOM → RGB `np.ndarray` + `PIL.Image` para mostrar.
- JPG/PNG → RGB `np.ndarray` + `PIL.Image`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pydicom
from PIL import Image


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Escala un array float/integro a rango [0, 255] y lo castea a uint8."""
    img = np.asarray(img)
    if img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    img = img.astype("float32", copy=False)
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax <= vmin:  # imagen plana o valores inválidos
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - vmin) / (vmax - vmin)
    img = (img * 255.0).clip(0, 255)
    return img.astype(np.uint8)


def read_dicom_file(path: str | Path) -> Tuple[np.ndarray, Image.Image]:
    """Lee un DICOM y lo devuelve como RGB + imagen PIL para visualización.

    Aplica RescaleSlope/Intercept si existen y corrige MONOCHROME1
    (invertido) según `PhotometricInterpretation`.

    Args:
        path: Ruta al archivo DICOM.

    Returns:
        rgb: Imagen RGB `np.ndarray` (H, W, 3) en uint8.
        img2show: `PIL.Image` para mostrar/guardar.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el DICOM es ilegible o no tiene `pixel_array`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo DICOM: {p}")

    try:
        ds = pydicom.dcmread(str(p))
    except Exception as exc:  # pydicom lanza distintos tipos
        raise ValueError(f"No se pudo leer el DICOM: {p}") from exc

    if not hasattr(ds, "pixel_array"):
        raise ValueError("El DICOM no contiene datos de imagen (pixel_array).")

    img = ds.pixel_array.astype("float32", copy=False)

    # Rescale (slope/intercept) si están presentes
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        img = img * slope + intercept

    # Invertir MONOCHROME1 (blanco = valores bajos)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        img = np.max(img) - img

    # A 8 bits y a RGB
    arr_u8 = _to_uint8(img)
    rgb = cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)

    img2show = Image.fromarray(arr_u8)  # en escala de grises para mostrar
    return rgb, img2show


def read_jpg_file(path: str | Path) -> Tuple[np.ndarray, Image.Image]:
    """Lee JPG/PNG y devuelve RGB `np.ndarray` + `PIL.Image`.

    Args:
        path: Ruta a la imagen (.jpg/.jpeg/.png).

    Returns:
        rgb: Imagen RGB `np.ndarray` (H, W, 3) en uint8.
        img2show: `PIL.Image` para mostrar/guardar.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si OpenCV no puede leer la imagen.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo: {p}")

    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {p}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img2show = Image.fromarray(rgb)
    return rgb, img2show


