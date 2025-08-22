#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI para inferencia y generación de Grad-CAM (uso académico).

Ejecuta predicción sobre una imagen (DICOM/JPG/PNG), guarda el heatmap
como PNG y muestra etiqueta + probabilidad por consola.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.io_imgs import read_dicom_file, read_jpg_file
from src.inference import predict

# Extensiones soportadas (en orden de prueba cuando no se especifica).
SUPPORTED_EXTS: tuple[str, ...] = (".dcm", ".jpg", ".jpeg", ".png")

# Valores por defecto de la CLI.
DEFAULT_IMG_HINT = "samples/bacteria"
DEFAULT_OUTPUT_DIR = "outputs"


def _iter_candidates(base: Path, exts: Iterable[str]) -> Iterable[Path]:
    """Genera rutas candidatas probando sufijos de `exts`."""
    for ext in exts:
        yield base.with_suffix(ext)


def resolve_image_path(raw: str) -> Path:
    """Resuelve la ruta de la imagen a partir de un argumento de usuario.

    Reglas:
      1) Si `raw` existe tal cual, se usa.
      2) Si no tiene extensión, se prueban SUPPORTED_EXTS en ese orden.
      3) Si no existe, se intenta dentro de `samples/` con el mismo nombre
         (con y sin extensión, probando SUPPORTED_EXTS).

    Args:
        raw: Ruta provista por el usuario (con o sin extensión).

    Returns:
        Ruta existente hacia el archivo de imagen.

    Raises:
        SystemExit: Si no se encuentra ninguna ruta válida.
    """
    p = Path(raw)

    # Caso 1: existe tal cual.
    if p.exists():
        return p

    # Caso 2: sin extensión → probar sufijos.
    if p.suffix == "":
        for cand in _iter_candidates(p, SUPPORTED_EXTS):
            if cand.exists():
                return cand

    # Caso 3: buscar dentro de 'samples/'.
    samples = Path("samples")
    if samples.exists():
        base = samples / p.name
        if base.exists():
            return base
        if base.suffix == "":
            for cand in _iter_candidates(base, SUPPORTED_EXTS):
                if cand.exists():
                    return cand

    supported = ", ".join(SUPPORTED_EXTS)
    msg = (
        f"No se encontró la imagen: {p.resolve()}\n"
        "Prueba con ruta absoluta o coloca el archivo en 'samples/' "
        f"con una de estas extensiones: {supported}"
    )
    raise SystemExit(msg)


def main() -> None:
    """Punto de entrada de la CLI."""
    parser = argparse.ArgumentParser(
        description="Inferencia por CLI (sin GUI) con Grad-CAM.",
    )
    parser.add_argument(
        "--img",
        default=DEFAULT_IMG_HINT,
        help=(
            "Ruta a la imagen (.dcm/.jpg/.jpeg/.png). "
            "Por defecto intenta 'samples/bacteria{ext}'."
        ),
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta de salida para guardar el heatmap (PNG).",
    )
    args = parser.parse_args()

    # Resolver ruta de entrada y carpeta de salida.
    img_path = resolve_image_path(args.img)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar imagen según la extensión (E/S centralizada en src.io_imgs).
    ext = img_path.suffix.lower()
    if ext == ".dcm":
        array, _meta = read_dicom_file(str(img_path))
    elif ext in {".jpg", ".jpeg", ".png"}:
        array, _meta = read_jpg_file(str(img_path))
    else:
        raise SystemExit(f"Extensión no soportada: {ext}")

    # Ejecutar inferencia (devuelve etiqueta, probabilidad y heatmap RGB).
    label, proba, heatmap = predict(array)

    # Guardar heatmap como PNG.
    out_file = out_dir / f"heatmap_{img_path.stem}.png"
    Image.fromarray(heatmap).save(out_file)

    # Reporte breve por consola.
    print(f"Imagen:       {img_path.resolve()}")
    print(f"Resultado:    {label}")
    print(f"Probabilidad: {proba:.2f}%")
    print(f"Heatmap:      {out_file.resolve()}")


if __name__ == "__main__":
    main()


