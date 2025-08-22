#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Carga del modelo de clasificación para neumonía."""

from __future__ import annotations

import os
from pathlib import Path

from tensorflow.keras.models import Model, load_model


def model_fun() -> Model:
    """Carga el modelo entrenado desde la ruta definida en `MODEL_PATH`.

    Busca primero en la variable de entorno `MODEL_PATH`; si no está
    definida, usa por defecto `model/conv_MLP_84.h5`.

    Returns:
        Un objeto `tf.keras.Model` listo para inferencia (no compilado).

    Raises:
        FileNotFoundError: Si no existe el archivo del modelo.
    """
    path = Path(os.getenv("MODEL_PATH", "model/conv_MLP_84.h5"))

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en: {path.resolve()}"
        )

    return load_model(str(path), compile=False)

