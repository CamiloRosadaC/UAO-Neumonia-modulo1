import numpy as np
from src import inference

def test_predict_smoke(monkeypatch):
    # Imagen gris en RGB (512x512x3)
    img = np.dstack([np.full((512, 512), 128, np.uint8)] * 3)

    # Stub del modelo para no depender del .h5 real
    class FakeModel:
        def predict(self, batch):
            import numpy as np
            # Simula 3 clases con una distribución válida
            return np.array([[0.2, 0.3, 0.5]], dtype=float)

    # Reemplaza model_fun y grad_cam dentro de src.inference
    monkeypatch.setattr(inference, "model_fun", lambda: FakeModel())
    monkeypatch.setattr(
        inference, "grad_cam",
        lambda x: np.dstack([np.full((512, 512), 10, np.uint8)] * 3)
    )

    label, proba, heat = inference.predict(img)
    assert label in {"bacteriana", "normal", "viral"}
    assert 0 <= proba <= 100
    assert heat.shape == (512, 512, 3)

