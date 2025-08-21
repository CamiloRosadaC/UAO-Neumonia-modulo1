import os
from tensorflow.keras.models import load_model

def model_fun():
    path = os.getenv("MODEL_PATH", "model/conv_MLP_84.h5")
    return load_model(path, compile=False)
