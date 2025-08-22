#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocesamiento de imágenes para inferencia."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def preprocess(
    array: np.ndarray,
    size: Tuple[int, int] = (512, 512),
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Convierte una imagen a gris, normaliza y añade dimensiones de batch.

    Args:
        array: Imagen RGB (H, W, 3) o en gris (H, W). Se acepta uint8 o float.
        size: Tamaño objetivo (ancho, alto) para `resize`.
        use_clahe: Si `True`, aplica CLAHE para mejorar contraste local.
        clip_limit: Parámetro de CLAHE.
        tile_grid_size: Parámetro de CLAHE (tamaño de teselas).

    Returns:
        Batch `float32` con shape (1, H, W, 1) y valores en [0.0, 1.0].

    Raises:
        ValueError: Si la imagen no es RGB (H, W, 3) ni GRIS (H, W).
    """
    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError("La imagen debe ser RGB (H, W, 3) o GRIS (H, W).")

    # Sanitizar valores numéricos
    gray = np.nan_to_num(gray, copy=False)

    # Asegurar uint8 para CLAHE y operaciones posteriores
    if gray.dtype != np.uint8:
        gmin, gmax = float(np.min(gray)), float(np.max(gray))
        if gmax > 1.0:  # ya parece estar en 0..255
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        else:  # 0..1 → escalar a 0..255
            gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)

    # Redimensionar primero (mejor que después de normalizar)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    # Contraste local opcional
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        gray = clahe.apply(gray)

    # Normalizar a [0, 1] y añadir canales/batch
    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=-1)  # (H, W, 1)
    batch = np.expand_dims(gray, axis=0)  # (1, H, W, 1)
    return batch

