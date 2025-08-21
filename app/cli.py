#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from PIL import Image

from src.io_imgs import read_dicom_file, read_jpg_file
from src.inference import predict

SUPPORTED_EXTS = [".dcm", ".jpg", ".jpeg", ".png"]


def resolve_image_path(raw: str) -> Path:
    """Resuelve la ruta de la imagen. Si no tiene extensión,
    intenta SUPPORTED_EXTS en ese orden."""
    p = Path(raw)
    if p.exists():
        return p

    # Si no existe y no tiene extensión, probamos con las soportadas
    if p.suffix == "":
        for ext in SUPPORTED_EXTS:
            cand = p.with_suffix(ext)
            if cand.exists():
                return cand

    # También probamos dentro de 'samples/' si el usuario solo pasó el nombre
    samples = Path("samples")
    if samples.exists():
        # p.ej. "bacteria" -> "samples/bacteria{ext}"
        base = samples / p.name
        if base.exists():
            return base
        if base.suffix == "":
            for ext in SUPPORTED_EXTS:
                cand = base.with_suffix(ext)
                if cand.exists():
                    return cand

    # Nada funcionó
    raise SystemExit(f"No se encontró la imagen: {p.resolve()}\n"
                     f"Prueba con ruta absoluta o coloca el archivo en 'samples/' "
                     f"con una de estas extensiones: {', '.join(SUPPORTED_EXTS)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia por CLI (sin GUI) con Grad-CAM."
    )
    parser.add_argument(
        "--img",
        default="samples/bacteria",
        help="Ruta a la imagen (.dcm/.jpg/.jpeg/.png). "
             "Por defecto intenta 'samples/bacteria{ext}'."
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help="Carpeta de salida para guardar el heatmap (PNG)."
    )
    args = parser.parse_args()

    img_path = resolve_image_path(args.img)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = img_path.suffix.lower()
    if ext == ".dcm":
        array, _ = read_dicom_file(str(img_path))
    elif ext in {".jpg", ".jpeg", ".png"}:
        array, _ = read_jpg_file(str(img_path))
    else:
        raise SystemExit(f"Extensión no soportada: {ext}")

    label, proba, heatmap = predict(array)

    out_file = out_dir / f"heatmap_{img_path.stem}.png"
    Image.fromarray(heatmap).save(out_file)

    print(f"Imagen:       {img_path.resolve()}")
    print(f"Resultado:    {label}")
    print(f"Probabilidad: {proba:.2f}%")
    print(f"Heatmap:      {out_file.resolve()}")


if __name__ == "__main__":
    main()

