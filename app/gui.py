#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import csv
import tkcap

from src.io_imgs import read_dicom_file, read_jpg_file
from src.inference import predict

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        fonti = font.Font(weight="bold")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(self.root, text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", font=fonti)
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.ID = StringVar()
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        self.button1 = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button2 = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(self.root, text="Guardar", command=self.save_results_csv)

        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab5.place(x=122, y=25)

        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)

        self.text_img1.place(x=65,  y=90,  width=250, height=250)
        self.text_img2.place(x=500, y=90,  width=250, height=250)

        self.lab4.place(x=65,  y=360)
        self.text1.place(x=230, y=356, width=140, height=24)

        self.lab3.place(x=500, y=350)
        self.text2.place(x=640, y=346, width=140, height=24)

        self.lab6.place(x=500, y=400)
        self.text3.place(x=640, y=396, width=140, height=24)

        self.text1.focus_set()
        self.array = None
        self.reportID = 0
        self.root.mainloop()

    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(("DICOM", "*.dcm"), ("JPEG", "*.jpeg"), ("JPG", "*.jpg"), ("PNG", "*.png")),
        )
        if not filepath:
            return
        ext = filepath.lower().split(".")[-1]
        if ext == "dcm":
            self.array, img2show = read_dicom_file(filepath)
        else:
            self.array, img2show = read_jpg_file(filepath)

        try:
            img2show = img2show.resize((250, 250), Image.Resampling.LANCZOS)
        except AttributeError:
            img2show = img2show.resize((250, 250), Image.ANTIALIAS)
        self.img1 = ImageTk.PhotoImage(img2show)

        self.text_img1.delete("1.0", "end")
        self.text_img1.image_create(END, image=self.img1)
        self.button1["state"] = "enabled"

    def run_model(self):
        if self.array is None:
            showinfo(title="Predicción", message="Primero carga una imagen.")
            return
        self.text_img2.delete("1.0", "end")
        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")

        label, proba, heatmap = predict(self.array)

        img_heat = Image.fromarray(heatmap)
        try:
            img_heat = img_heat.resize((250, 250), Image.Resampling.LANCZOS)
        except AttributeError:
            img_heat = img_heat.resize((250, 250), Image.ANTIALIAS)
        self.img2 = ImageTk.PhotoImage(img_heat)
        self.text_img2.image_create(END, image=self.img2)

        self.text2.insert(END, label)
        self.text3.insert(END, f"{proba:.2f}%")

    def save_results_csv(self):
        if self.array is None or not self.text2.get("1.0", "end").strip():
            showinfo(title="Guardar", message="No hay predicción para guardar.")
            return
        with open("historial.csv", "a", newline="") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow([self.text1.get(), self.text2.get("1.0", "end").strip(),
                        self.text3.get("1.0", "end").strip()])
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        cap = tkcap.CAP(self.root)
        img_name = f"Reporte{self.reportID}.jpg"
        cap.capture(img_name)
        img = Image.open(img_name).convert("RGB")
        pdf_path = f"Reporte{self.reportID}.pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        answer = askokcancel(title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING)
        if not answer:
            return
        self.text1.delete(0, "end")
        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")
        self.text_img1.delete("1.0", "end")
        self.text_img2.delete("1.0", "end")
        self.array = None
        self.img1 = None
        self.img2 = None
        self.button1["state"] = "disabled"
        showinfo(title="Borrar", message="Los datos se borraron con éxito")

def main():
    App()
    return 0

if __name__ == "__main__":
    main()
