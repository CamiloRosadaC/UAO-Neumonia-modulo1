#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inferencia: predicción de clase y generación de Grad‑CAM."""

from __future__ import annotations

import numpy as np

from .explain import grad_cam
from .model import model_fun
from .preprocess import preprocess

LABELS: tuple[str, ...] = ("bacteriana", "normal", "viral")


def predict(array: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Predice la clase de una imagen y genera el heatmap Grad‑CAM.

    Args:
        array: Imagen de entrada (H, W, C) en RGB o escala de grises.

    Returns:
        label: Etiqueta predicha (en español).
        proba: Probabilidad de la clase predicha en porcentaje [0.0, 100.0].
        heatmap: Imagen RGB (H, W, 3 aprox. tras resize) con Grad‑CAM.

    Notes:
        - `proba` se calcula como el máximo de `softmax` * 100.
        - El tamaño/shape del `heatmap` depende de la implementación de
          `grad_cam` (por defecto, 512 x 512 en nuestra versión).
    """
    # Preprocesar para el modelo
    batch = preprocess(array)  # (1, H, W, C)

    # Cargar/crear el modelo y predecir
    model = model_fun()
    probs = model.predict(batch, verbose=0)[0]

    # Índice y etiqueta
    class_idx = int(np.argmax(probs))
    label = (
        LABELS[class_idx] if 0 <= class_idx < len(LABELS) else f"class_{class_idx}"
    )

    # Probabilidad como porcentaje
    proba = float(np.max(probs) * 100.0)
    proba = float(np.clip(proba, 0.0, 100.0))

    # Grad‑CAM (sobre la imagen original)
    heatmap = grad_cam(array)

    return label, proba, heatmap

