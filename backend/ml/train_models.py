from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.models import LABELS, LSTMClassifier

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
VITALS = ["temperature", "heart_rate", "spo2", "respiratory_rate"]
FEATURE_COLUMNS = VITALS + [f"{v}_delta" for v in VITALS] + [f"{v}_mean_12" for v in VITALS] + [f"{v}_min_12" for v in VITALS] + [f"{v}_max_12" for v in VITALS]
SEQUENCE_COLUMNS = VITALS + [f"{v}_delta" for v in VITALS]
SEQ_LEN = 12


def make_sequences(df: pd.DataFrame):
    xs, ys = [], []
    for _, group in df.sort_values(["patient_id", "timestamp"]).groupby("patient_id"):
        values = group[SEQUENCE_COLUMNS].to_numpy(dtype=np.float32)
        labels = group["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        for i in range(SEQ_LEN - 1, len(group)):
            xs.append(values[i - SEQ_LEN + 1:i + 1])
            ys.append(labels[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def train_lstm(x_train, y_train, x_val, y_val, artifact_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=x_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_x = torch.tensor(x_train, dtype=torch.float32).to(device)
    train_y = torch.tensor(y_train, dtype=torch.long).to(device)
    val_x = torch.tensor(x_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.long).to(device)
    best = 0
    best_state = None
    for _ in range(25):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(val_x).argmax(1) == val_y).float().mean().item()
        if acc > best:
            best = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    scripted = torch.jit.trace(model.cpu(), torch.randn(1, SEQ_LEN, x_train.shape[-1]))
    scripted.save(str(artifact_dir / "lstm_model.pt"))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned/cleaned_vitals.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    y = df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    xgb = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="mlogloss")
    xgb.fit(x_train_s, y_train)
    print(classification_report(y_val, xgb.predict(x_val_s), target_names=LABELS))
    joblib.dump(xgb, artifact_dir / "xgboost_model.pkl")
    joblib.dump(scaler, artifact_dir / "feature_scaler.pkl")
    normal_critical = df[df["label"].isin(["normal_vitals", "critical_vitals"])]
    if_scaler = StandardScaler().fit(normal_critical[FEATURE_COLUMNS])
    iforest = IsolationForest(n_estimators=150, contamination=0.12, random_state=42).fit(if_scaler.transform(normal_critical[FEATURE_COLUMNS]))
    joblib.dump(iforest, artifact_dir / "isolation_forest.pkl")
    joblib.dump(if_scaler, artifact_dir / "iforest_scaler.pkl")
    seq_x, seq_y = make_sequences(df)
    flat_scaler = StandardScaler().fit(seq_x.reshape(-1, seq_x.shape[-1]))
    seq_x_s = flat_scaler.transform(seq_x.reshape(-1, seq_x.shape[-1])).reshape(seq_x.shape)
    sx_train, sx_val, sy_train, sy_val = train_test_split(seq_x_s, seq_y, test_size=0.2, random_state=42, stratify=seq_y)
    lstm_acc = train_lstm(sx_train, sy_train, sx_val, sy_val, artifact_dir)
    joblib.dump(flat_scaler, artifact_dir / "lstm_scaler.pkl")
    metadata = {
        "labels": LABELS,
        "class_labels": LABELS,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_columns": SEQUENCE_COLUMNS,
        "sequence_length": SEQ_LEN,
        "lstm_torchscript": True,
        "lstm_validation_accuracy": lstm_acc,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
