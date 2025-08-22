import numpy as np
from src.preprocess import preprocess


def test_preprocess_rgb_and_gray():
    """Preprocess convierte RGB o gris a batch (1, H, W, 1) float32 en [0, 1]."""
    # Imagen dummy RGB
    fake_rgb = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    out_rgb = preprocess(fake_rgb)
    assert out_rgb.shape == (1, 512, 512, 1)
    assert out_rgb.dtype == np.float32
    assert 0.0 <= float(out_rgb.min()) <= 1.0
    assert 0.0 <= float(out_rgb.max()) <= 1.0

    # Imagen dummy en escala de grises
    fake_gray = (np.random.rand(512, 512) * 255).astype(np.uint8)
    out_gray = preprocess(fake_gray)
    assert out_gray.shape == (1, 512, 512, 1)
    assert out_gray.dtype == np.float32
    assert 0.0 <= float(out_gray.min()) <= 1.0
    assert 0.0 <= float(out_gray.max()) <= 1.0

