import numpy as np
from src.preprocess import preprocess

def test_preprocess_shape_dtype():
    # imagen dummy RGB
    fake = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    out = preprocess(fake)
    assert out.shape == (1, 512, 512, 1)
    assert out.dtype == np.float32
