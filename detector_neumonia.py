#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aplicación monolítica (GUI Tkinter + modelo + Grad-CAM).

Uso académico. No apto para diagnóstico médico real.
Mantiene todo en un solo archivo por requerimiento del proyecto.
"""

from __future__ import annotations

import csv
from tkinter import END, StringVar, Text, Tk
from tkinter import filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

import cv2
import numpy as np
import pydicom as dicom
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.models import Model, load_model

# ---------------------------- Configuración -----------------------------

MODEL_PATH = "model/conv_MLP_84.h5"
TARGET_SIZE = 512
LABELS: tuple[str, ...] = ("bacteriana", "normal", "viral")


# --------------------------- Modelo y Grad-CAM --------------------------

def model_fun() -> Model:
    """Carga el modelo entrenado para inferencia (no compilado)."""
    return load_model(MODEL_PATH, compile=False)


def preprocess(array: np.ndarray) -> np.ndarray:
    """Convierte imagen a gris, normaliza [0,1], y añade dims (1,H,W,1)."""
    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError("La imagen debe ser RGB (H,W,3) o GRIS (H,W).")

    gray = cv2.resize(gray, (TARGET_SIZE, TARGET_SIZE), cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=-1)   # (H, W, 1)
    batch = np.expand_dims(gray, axis=0)   # (1, H, W, 1)
    return batch


def grad_cam(array: np.ndarray, model: Model | None = None) -> np.ndarray:
    """Genera Grad‑CAM superpuesto a la imagen base."""
    img = preprocess(array)
    model = model or model_fun()

    preds = model.predict(img, verbose=0)
    class_idx = int(np.argmax(preds[0]))

    # Buscar la última capa convolucional utilizable
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
        raise ValueError("No se encontró una capa convolucional en el modelo.")

    # Modelo que devuelve (activaciones, salida)
    grad_model = tf.keras.models.Model(
        inputs=model.input, outputs=[last_conv_layer.output, model.output]
    )

    # Gradientes de la clase objetivo vs. activaciones
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img, training=False)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Colorear y superponer
    heat_u8 = np.uint8(255 * cv2.resize(heatmap, (TARGET_SIZE, TARGET_SIZE)))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    base = cv2.resize(array, (TARGET_SIZE, TARGET_SIZE))
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def predict(array: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Devuelve (label, probabilidad_en_% , heatmap_RGB)."""
    batch = preprocess(array)
    model = model_fun()

    probs = model.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    label = LABELS[class_idx] if 0 <= class_idx < len(LABELS) else "desconocida"
    proba = float(np.max(probs) * 100.0)

    # Reusar el mismo modelo para Grad‑CAM (evita recarga)
    heatmap = grad_cam(array, model=model)
    return label, proba, heatmap


# ------------------------------ Lectura I/O -----------------------------

def read_dicom_file(path: str) -> tuple[np.ndarray, Image.Image]:
    """Lee DICOM, normaliza a 8 bits y devuelve (RGB_uint8, PIL_para_mostrar)."""
    ds = dicom.dcmread(path)
    img_array = ds.pixel_array.astype(float)

    # Escalado simple a 0..255 (monolito: mantenemos lo básico)
    denom = img_array.max() if img_array.max() > 0 else 1.0
    img_u8 = np.uint8((np.maximum(img_array, 0) / denom) * 255.0)

    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    img2show = Image.fromarray(img_u8)
    return rgb, img2show


def read_jpg_file(path: str) -> tuple[np.ndarray, Image.Image]:
    """Lee JPG/PNG y devuelve (RGB_uint8, PIL_para_mostrar)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img2show = Image.fromarray(rgb)
    return rgb, img2show


# --------------------------------- GUI ---------------------------------

class App:
    """Aplicación Tkinter mínima para cargar imagen, predecir y exportar."""

    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        bold = font.Font(weight="bold")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # Labels
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=bold)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=bold)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=bold)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=bold)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=bold)

        # Variables y campos
        self.patient_id = StringVar()
        self.text1 = ttk.Entry(self.root, textvariable=self.patient_id, width=10)

        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        # Botones
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(self.root, text="Guardar", command=self.save_csv)

        # Layout absoluto
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab5.place(x=122, y=25)

        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)

        self.text_img1.place(x=65, y=90, width=250, height=250)
        self.text_img2.place(x=500, y=90, width=250, height=250)

        self.lab4.place(x=65, y=360)
        self.text1.place(x=230, y=356, width=140, height=24)

        self.lab3.place(x=500, y=350)
        self.text2.place(x=640, y=346, width=140, height=24)

        self.lab6.place(x=500, y=400)
        self.text3.place(x=640, y=396, width=140, height=24)

        # Estado
        self.text1.focus_set()
        self.array: np.ndarray | None = None
        self.label: str = ""
        self.proba: float = 0.0
        self.heatmap: np.ndarray | None = None
        self.report_id = 0

        self.root.mainloop()

    # -------------------------- Callbacks UI --------------------------

    def load_img_file(self) -> None:
        """Abre un diálogo, carga la imagen y la muestra en la UI."""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
        )
        if not filepath:
            return

        ext = filepath.lower().split(".")[-1]
        if ext == "dcm":
            self.array, img2show = read_dicom_file(filepath)
        else:
            self.array, img2show = read_jpg_file(filepath)

        try:
            img2show = img2show.resize(
                (250, 250), Image.Resampling.LANCZOS
            )
        except AttributeError:
            img2show = img2show.resize((250, 250), Image.ANTIALIAS)

        self.img1 = ImageTk.PhotoImage(img2show)
        self.text_img1.delete("1.0", "end")
        self.text_img1.image_create(END, image=self.img1)
        self.button1["state"] = "enabled"

    def run_model(self) -> None:
        """Ejecuta inferencia + Grad‑CAM y presenta resultados."""
        if self.array is None:
            showinfo(title="Predicción", message="Cargue una imagen primero.")
            return

        self.text_img2.delete("1.0", "end")
        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")

        self.label, self.proba, self.heatmap = predict(self.array)

        img_heat = Image.fromarray(self.heatmap)
        try:
            img_heat = img_heat.resize(
                (250, 250), Image.Resampling.LANCZOS
            )
        except AttributeError:
            img_heat = img_heat.resize((250, 250), Image.ANTIALIAS)
        self.img2 = ImageTk.PhotoImage(img_heat)
        self.text_img2.image_create(END, image=self.img2)

        self.text2.insert(END, self.label)
        self.text3.insert(END, f"{self.proba:.2f}%")

    def save_csv(self) -> None:
        """Guarda cédula, resultado y probabilidad en un CSV local."""
        with open("historial.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow(
                [self.text1.get(), self.label, f"{self.proba:.2f}%"]
            )
        showinfo(title="Guardar", message="Datos guardados con éxito.")

    def create_pdf(self) -> None:
        """Exporta la ventana completa a PDF (técnica de captura + PIL)."""
        # Nota: en esta versión monolítica mantenemos la técnica simple.
        import tkcap  # import local para evitar dependencia si no se usa

        cap = tkcap.CAP(self.root)
        jpg_name = f"Reporte{self.report_id}.jpg"
        cap.capture(jpg_name)

        img = Image.open(jpg_name).convert("RGB")
        pdf_path = f"Reporte{self.report_id}.pdf"
        img.save(pdf_path)
        self.report_id += 1
        showinfo(title="PDF", message=f"PDF generado: {pdf_path}")

    def delete(self) -> None:
        """Limpia campos, imágenes y resultados."""
        answer = askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=WARNING,
        )
        if not answer:
            return

        self.text1.delete(0, "end")
        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")
        self.text_img1.delete("1.0", "end")
        self.text_img2.delete("1.0", "end")
        self.array = None
        self.label = ""
        self.proba = 0.0
        self.heatmap = None
        showinfo(title="Borrar", message="Los datos se borraron con éxito")


# --------------------------------- Main --------------------------------

def main() -> int:
    """Punto de entrada de la app monolítica."""
    App()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
