import numpy as np
import pydicom as dicom
import cv2
from PIL import Image

def read_dicom_file(path):
    ds = dicom.dcmread(path)
    arr = ds.pixel_array.astype(float)
    arr = (np.maximum(arr, 0) / (arr.max() if arr.max() > 0 else 1)) * 255.0
    arr_u8 = np.uint8(arr)
    rgb = cv2.cvtColor(arr_u8, cv2.COLOR_GRAY2RGB)
    img2show = Image.fromarray(arr_u8)
    return rgb, img2show

def read_jpg_file(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img2show = Image.fromarray(rgb)
    return rgb, img2show

