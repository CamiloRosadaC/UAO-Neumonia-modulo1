#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Punto de entrada principal del proyecto.

Por defecto ejecuta la interfaz gráfica (GUI).
Para la CLI, usar directamente: `python -m app.cli`.
"""

from __future__ import annotations

from app.gui import main as gui_main


if __name__ == "__main__":
    raise SystemExit(gui_main())

