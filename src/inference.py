import numpy as np
from .preprocess import preprocess
from .model import model_fun
from .explain import grad_cam

LABELS = ["bacteriana", "normal", "viral"]

def predict(array):
    batch = preprocess(array)
    model = model_fun()

    probs = model.predict(batch)[0]
    class_idx = int(np.argmax(probs))
    proba = float(np.max(probs) * 100.0)
    label = LABELS[class_idx] if class_idx < len(LABELS) else f"class_{class_idx}"

    heatmap = grad_cam(array)
    return label, proba, heatmap
