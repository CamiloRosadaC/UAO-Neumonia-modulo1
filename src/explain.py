import numpy as np
import tensorflow as tf
import cv2
from .preprocess import preprocess
from .model import model_fun

def grad_cam(array):
    img = preprocess(array)  # (1,512,512,1)
    model = model_fun()

    preds = model.predict(img)
    class_idx = int(np.argmax(preds[0]))

    # última conv automáticamente
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.SeparableConv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No se encontró una capa convolucional en el modelo.")

    # modelo para (activaciones_conv, salida)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img, training=False)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H,W,C)
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # colorear y superponer
    heat_u8 = np.uint8(255 * cv2.resize(heatmap, (512, 512)))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    base = cv2.resize(array, (512, 512))
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb
