"""Generación de explicaciones visuales con Grad-CAM."""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from .model import model_fun
from .preprocess import preprocess


def grad_cam(array: np.ndarray, target_size: int = 512) -> np.ndarray:
    """Genera un mapa Grad-CAM y lo superpone sobre la imagen original.

    Args:
        array: Imagen de entrada (H, W, C) en formato RGB o escala de grises.
        target_size: Tamaño al que redimensionar la salida (por defecto 512).

    Returns:
        Imagen RGB (target_size, target_size, 3) con el heatmap superpuesto.

    Raises:
        ValueError: Si el modelo no contiene capas convolucionales.
    """
    # Preprocesar y ejecutar predicción
    img = preprocess(array)  # shape (1, H, W, C)
    model = model_fun()
    preds = model.predict(img, verbose=0)
    class_idx = int(np.argmax(preds[0]))

    # Localizar la última capa convolucional
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.SeparableConv2D,
                tf.keras.layers.DepthwiseConv2D,
            ),
        ):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("El modelo no contiene capas convolucionales.")

    # Modelo para obtener (activaciones_conv, salida)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output],
    )

    # Calcular gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img, training=False)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Combinar activaciones y gradientes
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    # Redimensionar y colorear
    heat_u8 = np.uint8(255 * cv2.resize(heatmap, (target_size, target_size)))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    # Superponer sobre la imagen base
    base = cv2.resize(array, (target_size, target_size))
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay_rgb

