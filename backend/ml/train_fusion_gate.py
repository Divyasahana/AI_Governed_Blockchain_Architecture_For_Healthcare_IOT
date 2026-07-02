from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.ml.models import AdaptiveFusionGate, LABELS

VITALS = ["temperature", "heart_rate", "spo2", "respiratory_rate"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def entropy(p):
    p = np.clip(p, 1e-7, 1.0)
    return -np.sum(p * np.log(p), axis=1, keepdims=True)


def build_lstm_probabilities(df: pd.DataFrame, artifacts: Path, metadata: dict, fallback_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sequence_columns = metadata["sequence_columns"]
    sequence_length = int(metadata["sequence_length"])
    lstm_scaler = joblib.load(artifacts / "lstm_scaler.pkl")
    model = torch.jit.load(str(artifacts / "lstm_model.pt"))
    model.eval()
    lstm_probs = fallback_probs.copy()
    sequence_available = np.zeros((len(df), 1), dtype=float)
    ordered = df.sort_values(["patient_id", "timestamp"])
    with torch.no_grad():
        for _, group in ordered.groupby("patient_id", sort=False):
            values = group[sequence_columns].to_numpy(dtype=np.float32)
            row_indices = group.index.to_numpy()
            if len(values) < sequence_length:
                continue
            sequences, target_indices = [], []
            for i in range(sequence_length - 1, len(values)):
                sequences.append(values[i - sequence_length + 1:i + 1])
                target_indices.append(row_indices[i])
            seq = np.asarray(sequences, dtype=np.float32)
            flat = seq.reshape(-1, seq.shape[-1])
            seq = lstm_scaler.transform(flat).reshape(seq.shape)
            probs = torch.softmax(model(torch.tensor(seq, dtype=torch.float32)), dim=-1).cpu().numpy()
            lstm_probs[target_indices] = probs
            sequence_available[target_indices, 0] = 1.0
    return lstm_probs, sequence_available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned/cleaned_vitals.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    df = pd.read_csv(args.input)
    metadata = json.loads((artifacts / "metadata.json").read_text())
    feature_cols = metadata["feature_columns"]
    scaler = joblib.load(artifacts / "feature_scaler.pkl")
    xgb = joblib.load(artifacts / "xgboost_model.pkl")
    iforest = joblib.load(artifacts / "isolation_forest.pkl")
    if_scaler = joblib.load(artifacts / "iforest_scaler.pkl")
    xgb_probs = xgb.predict_proba(scaler.transform(df[feature_cols]))
    lstm_probs, seq_flag = build_lstm_probabilities(df, artifacts, metadata, xgb_probs)
    anomaly = np.clip(0.5 - iforest.decision_function(if_scaler.transform(df[feature_cols])), 0, 1).reshape(-1, 1)
    confidence = np.hstack([xgb_probs.max(axis=1, keepdims=True), lstm_probs.max(axis=1, keepdims=True), entropy(xgb_probs), entropy(lstm_probs)])
    fusion_x = np.hstack([df[VITALS].to_numpy(float), xgb_probs, lstm_probs, anomaly, confidence, seq_flag])
    labels = df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    fusion_scaler = StandardScaler().fit(fusion_x[train_idx])
    x_train = fusion_scaler.transform(fusion_x[train_idx])
    x_val = fusion_scaler.transform(fusion_x[val_idx])
    y_train, y_val = labels[train_idx], labels[val_idx]
    model = AdaptiveFusionGate(input_dim=x_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    for _ in range(60):
        alpha = model(xt)
        px = torch.tensor(xgb_probs[train_idx], dtype=torch.float32)
        pl = torch.tensor(lstm_probs[train_idx], dtype=torch.float32)
        fused = alpha * px + (1 - alpha) * pl
        entropy_reg = -0.01 * torch.mean(alpha * torch.log(alpha + 1e-7) + (1 - alpha) * torch.log(1 - alpha + 1e-7))
        bounded_reg = 0.02 * torch.mean(torch.relu(0.08 - alpha) + torch.relu(alpha - 0.92))
        loss = ce(torch.log(fused + 1e-7), yt) + entropy_reg + bounded_reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        av = model(torch.tensor(x_val, dtype=torch.float32)).numpy()
        pv = av * xgb_probs[val_idx] + (1 - av) * lstm_probs[val_idx]
        print("fusion validation accuracy", accuracy_score(y_val, pv.argmax(axis=1)))
        print("fusion training rows with real LSTM sequence", int(seq_flag.sum()), "of", len(df))
    torch.save(model.state_dict(), artifacts / "adaptive_fusion.pt")
    joblib.dump(fusion_scaler, artifacts / "fusion_scaler.pkl")
    fusion_metadata = {
        "training_inputs": ["original_vitals", "xgboost_probabilities", "lstm_probabilities", "isolation_forest_anomaly_score", "model_confidence_entropy_features", "sequence_available_flag"],
        "original_vitals": VITALS,
        "probability_labels": LABELS,
        "rows": int(len(df)),
        "rows_with_real_lstm_sequence": int(seq_flag.sum()),
        "rows_without_lstm_sequence_use_xgboost_fallback": int(len(df) - seq_flag.sum()),
    }
    (artifacts / "adaptive_fusion_metadata.json").write_text(json.dumps(fusion_metadata, indent=2))


if __name__ == "__main__":
    main()
