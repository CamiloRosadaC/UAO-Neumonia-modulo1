import numpy as np
import cv2

def preprocess(array):
    # array: RGB (H,W,3) o GRIS (H,W)
    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError("La imagen debe ser RGB (H,W,3) o GRIS (H,W).")

    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=-1)    # (H,W,1)
    batch = np.expand_dims(gray, axis=0)    # (1,H,W,1)
    return batch

