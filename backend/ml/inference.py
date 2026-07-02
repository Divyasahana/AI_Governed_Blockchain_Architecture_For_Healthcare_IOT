from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict

import joblib
import numpy as np

from .features import LABELS, SEQUENCE_LENGTH, feature_row, normalize_payload, sequence_matrix
from .fusion import entropy, fuse
from .models import AdaptiveFusionGate

try:
    import torch
except Exception:
    torch = None


class ModelNotReadyError(RuntimeError):
    pass


class ModelService:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=120))
        self.xgb = self._load("xgboost_model.pkl")
        self.iforest = self._load("isolation_forest.pkl")
        self.scaler_if = self._load("iforest_scaler.pkl") or self._load("scaler_if.pkl")
        self.feature_scaler = self._load("feature_scaler.pkl")
        self.lstm_scaler = self._load("lstm_scaler.pkl")
        self.fusion_scaler = self._load("fusion_scaler.pkl")
        self.metadata = self._load_json("metadata.json")
        self.feature_columns = self.metadata.get("feature_columns", [])
        self.sequence_columns = self.metadata.get("sequence_columns", [])
        self.class_labels = self.metadata.get("class_labels") or sorted(LABELS)
        self.fusion_gate = self._load_fusion_gate()
        self.status = {
            "xgboost": self.xgb is not None,
            "lstm": (self.artifacts_dir / "lstm_model.pt").exists(),
            "isolation_forest": self.iforest is not None,
            "fusion_gate": self.fusion_gate is not None,
            "labels": LABELS,
            "sequence_length": SEQUENCE_LENGTH,
            "fallback_enabled": False,
        }

    def _load(self, name: str):
        path = self.artifacts_dir / name
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                return None
        return None

    def _load_json(self, name: str) -> dict:
        path = self.artifacts_dir / name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _load_fusion_gate(self):
        path = self.artifacts_dir / "adaptive_fusion.pt"
        if torch is None or not path.exists() or self.fusion_scaler is None:
            return None
        try:
            input_dim = int(getattr(self.fusion_scaler, "n_features_in_", 16))
            model = AdaptiveFusionGate(input_dim=input_dim)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model
        except Exception:
            return None

    def _vector(self, row: dict) -> np.ndarray:
        if not self.feature_columns:
            raise ModelNotReadyError("metadata.json is missing feature_columns. Run model training first.")
        columns = self.feature_columns
        return np.array([[float(row.get(col, 0.0)) for col in columns]], dtype=float)

    def _xgb_probs(self, row: dict) -> dict:
        if self.xgb is None:
            raise ModelNotReadyError("XGBoost artifact is missing. Run backend.ml.train_models first.")
        if self.feature_scaler is None:
            raise ModelNotReadyError("feature_scaler.pkl is missing. Run backend.ml.train_models first.")
        try:
            x = self._vector(row)
            x = self.feature_scaler.transform(x)
            probs = self.xgb.predict_proba(x)[0]
            classes = getattr(self.xgb, "classes_", LABELS)
            if all(isinstance(cls, (int, np.integer)) for cls in classes):
                mapped = {self.class_labels[int(cls)]: float(prob) for cls, prob in zip(classes, probs)}
            else:
                mapped = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
            return {label: mapped.get(label, 0.0) for label in LABELS}
        except ModelNotReadyError:
            raise
        except Exception as exc:
            raise ModelNotReadyError(f"XGBoost prediction failed: {exc}") from exc

    def _lstm_probs(self, row: dict, history: Deque[dict]) -> tuple[dict, bool]:
        seq = sequence_matrix(history, row)
        if len(seq) < SEQUENCE_LENGTH:
            return self._xgb_probs(row), False
        if torch is None:
            raise ModelNotReadyError("PyTorch is not available, so LSTM cannot run.")
        path = self.artifacts_dir / "lstm_model.pt"
        if not path.exists():
            raise ModelNotReadyError("LSTM artifact is missing. Run backend.ml.train_models first.")
        if self.lstm_scaler is None:
            raise ModelNotReadyError("lstm_scaler.pkl is missing. Run backend.ml.train_models first.")
        try:
            arr = np.array(seq, dtype=float)
            flat = arr.reshape(-1, arr.shape[-1])
            arr = self.lstm_scaler.transform(flat).reshape(arr.shape)
            model = torch.jit.load(str(path)) if self.metadata.get("lstm_torchscript") else None
            if model is None:
                raise ModelNotReadyError("LSTM metadata does not indicate a TorchScript model.")
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(arr[None, :, :], dtype=torch.float32))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            mapped = {label: float(prob) for label, prob in zip(self.class_labels, probs)}
            return {label: mapped.get(label, 0.0) for label in LABELS}, True
        except ModelNotReadyError:
            raise
        except Exception as exc:
            raise ModelNotReadyError(f"LSTM prediction failed: {exc}") from exc

    def _anomaly_score(self, row: dict) -> float:
        if self.iforest is None:
            raise ModelNotReadyError("Isolation Forest artifact is missing. Run backend.ml.train_models first.")
        if self.scaler_if is None:
            raise ModelNotReadyError("iforest_scaler.pkl is missing. Run backend.ml.train_models first.")
        try:
            x = self._vector(row)
            x = self.scaler_if.transform(x)
            raw = float(self.iforest.decision_function(x)[0])
            return round(max(0.0, min(1.0, 0.5 - raw)), 4)
        except ModelNotReadyError:
            raise
        except Exception as exc:
            raise ModelNotReadyError(f"Isolation Forest prediction failed: {exc}") from exc

    def _fusion_alpha(self, row: dict, xgb_probs: dict, lstm_probs: dict, anomaly_score: float, sequence_available: bool) -> float:
        if self.fusion_gate is None or self.fusion_scaler is None or torch is None:
            raise ModelNotReadyError("Adaptive fusion gate artifacts are missing. Run backend.ml.train_fusion_gate first.")
        try:
            xgb_values = np.array([xgb_probs[label] for label in LABELS], dtype=float)
            lstm_values = np.array([lstm_probs[label] for label in LABELS], dtype=float)
            confidence = np.array([xgb_values.max(), lstm_values.max(), entropy(xgb_probs), entropy(lstm_probs)], dtype=float)
            features = np.concatenate([
                np.array([row[key] for key in ["temperature", "heart_rate", "spo2", "respiratory_rate"]], dtype=float),
                xgb_values,
                lstm_values,
                np.array([anomaly_score], dtype=float),
                confidence,
                np.array([1.0 if sequence_available else 0.0], dtype=float),
            ]).reshape(1, -1)
            features = self.fusion_scaler.transform(features)
            with torch.no_grad():
                return float(self.fusion_gate(torch.tensor(features, dtype=torch.float32)).item())
        except Exception as exc:
            raise ModelNotReadyError(f"Adaptive fusion gate prediction failed: {exc}") from exc

    def predict(self, payload: dict) -> dict:
        vitals = normalize_payload(payload)
        history = self.history[vitals["device_id"]]
        row = feature_row(vitals, history)
        xgb_probs = self._xgb_probs(row)
        lstm_probs, sequence_available = self._lstm_probs(row, history)
        anomaly_score = self._anomaly_score(row)
        alpha = self._fusion_alpha(row, xgb_probs, lstm_probs, anomaly_score, sequence_available)
        fusion = fuse(xgb_probs, lstm_probs, anomaly_score, sequence_available, alpha)
        history.append(vitals)
        trust_score = round(float(fusion["final_probability"]), 4)
        return {
            "input": vitals,
            "features": row,
            "isolation_forest": {"anomaly_score": anomaly_score},
            "xgboost": {"probabilities": {k: round(v, 4) for k, v in xgb_probs.items()}},
            "lstm": {"probabilities": {k: round(v, 4) for k, v in lstm_probs.items()}, "sequence_available": sequence_available},
            "fusion": fusion,
            "final_label": fusion["final_label"],
            "confidence": fusion["final_probability"],
            "trust_score": trust_score,
        }
