from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import csv
import tkcap
import os  # manejo de directorios/archivos

from src.io_imgs import read_dicom_file, read_jpg_file
from src.inference import predict

# Directorios de salida
RESULTS_DIR = "resultados"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reportes")
HIST_DIR = os.path.join(RESULTS_DIR, "historial")

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        # Asegurar estructura de resultados
        for d in (RESULTS_DIR, REPORTS_DIR, HIST_DIR):
            os.makedirs(d, exist_ok=True)

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

        # Campos NO editables para resultado y probabilidad
        self.result_var = StringVar()
        self.proba_var = StringVar()
        self.text2 = ttk.Entry(self.root, textvariable=self.result_var, state="readonly")
        self.text3 = ttk.Entry(self.root, textvariable=self.proba_var, state="readonly")

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

        # Botones de exportación deshabilitados al inicio
        self.button4["state"] = "disabled"  # PDF
        self.button6["state"] = "disabled"  # Guardar

        # Cuando cambia la cédula, re-evaluar habilitado de exportación
        self.ID.trace_add("write", self._on_id_change)

        self.root.mainloop()

    # ---------- Helpers de estado ----------
    def _has_prediction(self) -> bool:
        return bool(self.result_var.get().strip())

    def _refresh_export_buttons(self):
        """Habilita PDF/Guardar solo si hay cédula + predicción."""
        has_id = bool(self.ID.get().strip())
        can_export = has_id and self._has_prediction()
        self.button4["state"] = "enabled" if can_export else "disabled"
        self.button6["state"] = "enabled" if can_export else "disabled"

    def _on_id_change(self, *args):
        self._refresh_export_buttons()

    # ---------- Flujo UI ----------
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(("DICOM", "*.dcm"), ("JPEG", "*.jpeg"), ("JPG", "*.jpg"), ("PNG", "*.png")),
        )
        if not filepath:
            return

        # Limpiar resultados previos al cargar una nueva imagen
        self.text_img2.delete("1.0", "end")
        self.result_var.set("")
        self.proba_var.set("")

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

        # Al cargar nueva imagen se limpian predicciones → deshabilitar exportación
        self._refresh_export_buttons()

    def run_model(self):
        if self.array is None:
            showinfo(title="Predicción", message="Carga un nuevo archivo antes de analizar.")
            return

        self.text_img2.delete("1.0", "end")
        self.result_var.set("")
        self.proba_var.set("")

        label, proba, heatmap = predict(self.array)

        img_heat = Image.fromarray(heatmap)
        try:
            img_heat = img_heat.resize((250, 250), Image.Resampling.LANCZOS)
        except AttributeError:
            img_heat = img_heat.resize((250, 250), Image.ANTIALIAS)
        self.img2 = ImageTk.PhotoImage(img_heat)
        self.text_img2.image_create(END, image=self.img2)

        self.result_var.set(label)
        self.proba_var.set(f"{proba:.2f}%")

        # Si hay cédula + predicción, habilitar exportación
        self._refresh_export_buttons()

    def save_results_csv(self):
        # Validaciones con mensajes específicos
        if not self.ID.get().strip():
            showinfo(title="Guardar", message="Falta la cédula.")
            return
        if self.array is None:
            showinfo(title="Guardar", message="No se ha cargado una imagen.")
            return
        if not self._has_prediction():
            showinfo(title="Guardar", message="No se ha realizado una predicción.")
            return

        csv_path = os.path.join(HIST_DIR, "historial.csv")
        os.makedirs(HIST_DIR, exist_ok=True)

        with open(csv_path, "a", newline="") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow([
                self.text1.get(),
                self.result_var.get().strip(),
                self.proba_var.get().strip()
            ])
        showinfo(title="Guardar", message=f"Los datos se guardaron con éxito en\n{csv_path}")

    def create_pdf(self):
        # Validaciones con mensajes específicos
        if not self.ID.get().strip():
            showinfo(title="PDF", message="Falta la cédula.")
            return
        if self.array is None:
            showinfo(title="PDF", message="No se ha cargado una imagen.")
            return
        if not self._has_prediction():
            showinfo(title="PDF", message="No se ha realizado una predicción.")
            return

        os.makedirs(REPORTS_DIR, exist_ok=True)

        # Buscar el siguiente índice libre para evitar conflictos
        idx = 0
        while True:
            jpg_candidate = os.path.join(REPORTS_DIR, f"Reporte{idx}.jpg")
            pdf_candidate = os.path.join(REPORTS_DIR, f"Reporte{idx}.pdf")
            if not (os.path.exists(jpg_candidate) or os.path.exists(pdf_candidate)):
                break
            idx += 1

        cap = tkcap.CAP(self.root)
        img_name = os.path.join(REPORTS_DIR, f"Reporte{idx}.jpg")
        cap.capture(img_name)

        pdf_path = os.path.join(REPORTS_DIR, f"Reporte{idx}.pdf")
        Image.open(img_name).convert("RGB").save(pdf_path)
        try:
            os.remove(img_name)
        except OSError:
            pass

        self.reportID = idx + 1
        showinfo(title="PDF", message=f"El PDF fue generado con éxito:\n{pdf_path}")

    def delete(self):
        answer = askokcancel(title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING)
        if not answer:
            return
        self.text1.delete(0, "end")
        self.result_var.set("")
        self.proba_var.set("")
        self.text_img1.delete("1.0", "end")
        self.text_img2.delete("1.0", "end")
        self.array = None
        self.img1 = None
        self.img2 = None
        self.button1["state"] = "disabled"
        # Al borrar, deshabilitar exportación
        self._refresh_export_buttons()
        showinfo(title="Borrar", message="Los datos se borraron con éxito")

def main():
    App()
    return 0

if __name__ == "__main__":
    main()





