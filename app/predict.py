import os
import numpy as np
import pickle
from typing import Tuple, Optional

_model = None

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "disease prediction model.pkl")

def load_model(path: Optional[str] = None):
    global _model
    if _model is None:
        p = path or MODEL_PATH
        p = os.path.join(os.path.dirname(__file__), p) if p.startswith("..") else p
        # safe load
        with open(p, "rb") as f:
            _model = pickle.load(f)
    return _model

def predict(features: list) -> Tuple[str, Optional[float]]:
    """
    features: list of 24 floats
    returns: (prediction_string, confidence_float_or_None)
    """
    model = load_model()
    arr = np.array(features, dtype=float).reshape(1, -1)
    try:
        pred = model.predict(arr)
        pred_label = pred[0]
    except Exception as e:
        raise RuntimeError(f"Model predict error: {e}")

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(arr)
            if probs.shape[1] > 1:
                idx = list(model.classes_).index(pred_label)
                confidence = float(probs[0, idx])
            else:
                confidence = float(probs[0, 0])
        except Exception:
            confidence = None

    return str(pred_label), confidence
