from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Iterable, List

VITALS = ["temperature", "heart_rate", "spo2", "respiratory_rate"]
LABELS = ["normal_vitals", "critical_vitals", "device_error"]
SEQUENCE_LENGTH = 12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_payload(payload: dict) -> dict:
    aliases = {
        "temperature": ["temperature", "temp", "Temp"],
        "heart_rate": ["heart_rate", "heartbeat", "hr", "HR", "bpm"],
        "spo2": ["spo2", "SpO2", "oxygen"],
        "respiratory_rate": ["respiratory_rate", "resp_rate", "RespRate", "rr"],
        "timestamp": ["timestamp", "time", "createdAt"],
        "device_id": ["device_id", "deviceId", "patient_device_id"],
        "patient_id": ["patient_id", "patientId"],
    }
    out = {}
    for target, keys in aliases.items():
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                out[target] = payload[key]
                break
    out.setdefault("timestamp", utc_now())
    out.setdefault("device_id", os.getenv("DEFAULT_DEVICE_ID", "device-unknown"))
    out.setdefault("patient_id", os.getenv("DEFAULT_PATIENT_ID", "patient-unknown"))
    for key in VITALS:
        out[key] = float(out.get(key, 0))
    return out


def _delta(current: dict, previous: dict | None, key: str) -> float:
    if not previous:
        return 0.0
    return float(current[key]) - float(previous[key])


def feature_row(current: dict, history: Iterable[dict]) -> dict:
    hist = list(history)
    previous = hist[-1] if hist else None
    row = {key: float(current[key]) for key in VITALS}
    for key in VITALS:
        row[f"{key}_delta"] = _delta(current, previous, key)
    window = hist[-11:] + [current]
    for key in VITALS:
        values = [float(item[key]) for item in window if key in item]
        row[f"{key}_mean_12"] = sum(values) / len(values) if values else row[key]
        row[f"{key}_min_12"] = min(values) if values else row[key]
        row[f"{key}_max_12"] = max(values) if values else row[key]
    row["sequence_available"] = 1.0 if len(window) >= SEQUENCE_LENGTH else 0.0
    return row


def sequence_matrix(history: Deque[dict], current: dict) -> List[List[float]]:
    rows = list(history)[-(SEQUENCE_LENGTH - 1):] + [current]
    if len(rows) < SEQUENCE_LENGTH:
        return []
    features = []
    prev = None
    for item in rows:
        vals = [float(item[k]) for k in VITALS]
        vals += [_delta(item, prev, k) for k in VITALS]
        features.append(vals)
        prev = item
    return features


def probs_to_label(probs: Dict[str, float]) -> tuple[str, float]:
    label = max(probs, key=probs.get)
    return label, float(probs[label])
