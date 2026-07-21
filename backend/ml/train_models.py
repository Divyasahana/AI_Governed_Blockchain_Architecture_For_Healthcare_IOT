from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.ml.models import LABELS, LSTMClassifier

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
VITALS = ["temperature", "heart_rate", "spo2", "respiratory_rate"]
FEATURE_COLUMNS = VITALS + [f"{v}_delta" for v in VITALS] + [f"{v}_mean_12" for v in VITALS] + [f"{v}_min_12" for v in VITALS] + [f"{v}_max_12" for v in VITALS]
SEQUENCE_COLUMNS = VITALS + [f"{v}_delta" for v in VITALS]
SEQ_LEN = 12
SMOTE_RANDOM_STATE = 42
LSTM_MAX_EPOCHS = 100
LSTM_PATIENCE = 12
LSTM_LEARNING_RATE = 5e-4
LSTM_HIDDEN_DIM = 96
LSTM_GRAD_CLIP = 1.0
LSTM_BATCH_SIZE = 512
LSTM_USE_CLASS_WEIGHTS = False


def log(message: str):
    print(message, flush=True)


def make_sequences(df: pd.DataFrame):
    xs, ys = [], []
    for _, group in df.sort_values(["patient_id", "timestamp"]).groupby("patient_id"):
        values = group[SEQUENCE_COLUMNS].to_numpy(dtype=np.float32)
        labels = group["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
        for i in range(SEQ_LEN - 1, len(group)):
            xs.append(values[i - SEQ_LEN + 1:i + 1])
            ys.append(labels[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def class_weights_from_labels(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(LABELS) * counts)
    return weights / weights.mean()


def train_lstm(x_train, y_train, x_val, y_val, artifact_dir: Path, class_weights: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=x_train.shape[-1], hidden_dim=LSTM_HIDDEN_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LSTM_LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    loss_weights = torch.tensor(class_weights, dtype=torch.float32).to(device) if LSTM_USE_CLASS_WEIGHTS else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights)
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)
    best_acc = 0.0
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(1, LSTM_MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad()
            train_logits = model(batch_x)
            loss = loss_fn(train_logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), LSTM_GRAD_CLIP)
            opt.step()
            train_loss_sum += float(loss.item()) * len(batch_y)
            train_count += len(batch_y)
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_logits = model(batch_x)
                val_loss = loss_fn(val_logits, batch_y)
                val_loss_sum += float(val_loss.item()) * len(batch_y)
                val_count += len(batch_y)
                correct += int((val_logits.argmax(1) == batch_y).sum().item())
        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        acc = correct / max(1, val_count)
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(acc),
            "learning_rate": float(opt.param_groups[0]["lr"]),
        })
        percent_done = epoch / LSTM_MAX_EPOCHS * 100
        remaining = LSTM_MAX_EPOCHS - epoch
        log(
            "LSTM training "
            f"{percent_done:6.2f}% done | "
            f"epoch {epoch}/{LSTM_MAX_EPOCHS} | "
            f"remaining max {remaining} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={acc * 100:.2f}% | "
            f"lr={opt.param_groups[0]['lr']:.6f}"
        )
        improved = acc > best_acc or (acc == best_acc and val_loss < best_loss)
        if improved:
            best_acc = acc
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= LSTM_PATIENCE:
            log(f"LSTM early stopping triggered after {epoch} epochs.")
            break
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    pd.DataFrame(history).to_csv(artifact_dir / "lstm_training_history.csv", index=False)
    model.load_state_dict(best_state)
    lstm_pred = predict_lstm_batches(model, x_val)
    log("\nLSTM validation performance:")
    log(classification_report(y_val, lstm_pred, target_names=LABELS))
    scripted = torch.jit.trace(model.cpu(), torch.randn(1, SEQ_LEN, x_train.shape[-1]))
    scripted.save(str(artifact_dir / "lstm_model.pt"))
    return best_acc, history


def predict_lstm_batches(model, x_val: np.ndarray, batch_size: int = LSTM_BATCH_SIZE) -> np.ndarray:
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_val), batch_size):
            batch = torch.tensor(x_val[start:start + batch_size], dtype=torch.float32).to(device)
            preds.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def print_fault_detection_report(iforest, if_scaler, x_val: np.ndarray, y_val: np.ndarray):
    actual_fault = (y_val == LABEL_TO_ID["device_error"]).astype(int)
    detected_fault = (iforest.predict(if_scaler.transform(x_val)) == -1).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual_fault, detected_fault, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alarm = fp / (fp + tn) if fp + tn else 0.0
    log("\nIsolation Forest fault-detection performance:")
    log(f"  true positives:  {tp}")
    log(f"  true negatives:  {tn}")
    log(f"  false positives: {fp}")
    log(f"  false negatives: {fn}")
    log(f"  precision: {precision * 100:.2f}%")
    log(f"  recall:    {recall * 100:.2f}%")
    log(f"  f1-score:  {f1 * 100:.2f}%")
    log(f"  false alarm rate: {false_alarm * 100:.2f}%")


def smote_tabular(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return SMOTE(random_state=SMOTE_RANDOM_STATE).fit_resample(x_train, y_train)


def smote_sequences(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    original_shape = x_train.shape
    flat = x_train.reshape(original_shape[0], -1)
    flat_resampled, y_resampled = SMOTE(random_state=SMOTE_RANDOM_STATE).fit_resample(flat, y_train)
    return flat_resampled.reshape(-1, original_shape[1], original_shape[2]).astype(np.float32), y_resampled.astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned/cleaned_vitals.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log("Loading cleaned dataset...")
    df = pd.read_csv(args.input)
    y = df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    log(f"Dataset loaded: {len(df)} rows, {len(FEATURE_COLUMNS)} tabular features, {len(LABELS)} classes.")
    log("Splitting train/validation data...")
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    log(f"Train rows: {len(y_train)} | Validation rows: {len(y_val)}")

    log("\n[1/3] Training XGBoost...")
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    log("Applying SMOTE to XGBoost training rows...")
    x_train_smote, y_train_smote = smote_tabular(x_train_s, y_train)
    log(f"XGBoost rows after SMOTE: {len(y_train_smote)}")
    xgb = XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="mlogloss")
    xgb.fit(x_train_smote, y_train_smote, eval_set=[(x_val_s, y_val)], verbose=25)
    log("\nXGBoost validation performance:")
    log(classification_report(y_val, xgb.predict(x_val_s), target_names=LABELS))
    joblib.dump(xgb, artifact_dir / "xgboost_model.pkl")
    joblib.dump(scaler, artifact_dir / "feature_scaler.pkl")

    log("\n[2/3] Training Isolation Forest...")
    normal_critical = df[df["label"].isin(["normal_vitals", "critical_vitals"])]
    if_scaler = StandardScaler().fit(normal_critical[FEATURE_COLUMNS])
    iforest = IsolationForest(n_estimators=150, contamination=0.12, random_state=42).fit(if_scaler.transform(normal_critical[FEATURE_COLUMNS]))
    print_fault_detection_report(iforest, if_scaler, x_val, y_val)
    joblib.dump(iforest, artifact_dir / "isolation_forest.pkl")
    joblib.dump(if_scaler, artifact_dir / "iforest_scaler.pkl")

    log("\n[3/3] Training LSTM...")
    log("Building 12-reading sequences...")
    seq_x, seq_y = make_sequences(df)
    log(f"LSTM sequences built: {len(seq_y)} rows, shape={seq_x.shape}")
    flat_scaler = StandardScaler().fit(seq_x.reshape(-1, seq_x.shape[-1]))
    seq_x_s = flat_scaler.transform(seq_x.reshape(-1, seq_x.shape[-1])).reshape(seq_x.shape)
    sx_train, sx_val, sy_train, sy_val = train_test_split(seq_x_s, seq_y, test_size=0.2, random_state=42, stratify=seq_y)
    log(f"LSTM train sequences: {len(sy_train)} | Validation sequences: {len(sy_val)}")
    log("Applying SMOTE to flattened LSTM training sequences...")
    sx_train_smote, sy_train_smote = smote_sequences(sx_train, sy_train)
    log(f"LSTM training sequences after SMOTE: {len(sy_train_smote)}")
    lstm_class_weights = class_weights_from_labels(sy_train)
    lstm_acc, lstm_history = train_lstm(sx_train_smote, sy_train_smote, sx_val, sy_val, artifact_dir, lstm_class_weights)
    joblib.dump(flat_scaler, artifact_dir / "lstm_scaler.pkl")
    metadata = {
        "labels": LABELS,
        "class_labels": LABELS,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_columns": SEQUENCE_COLUMNS,
        "sequence_length": SEQ_LEN,
        "lstm_torchscript": True,
        "lstm_validation_accuracy": lstm_acc,
        "lstm_training": {
            "max_epochs": LSTM_MAX_EPOCHS,
            "epochs_ran": len(lstm_history),
            "early_stopping_patience": LSTM_PATIENCE,
            "learning_rate": LSTM_LEARNING_RATE,
            "hidden_dim": LSTM_HIDDEN_DIM,
            "batch_size": LSTM_BATCH_SIZE,
            "gradient_clip": LSTM_GRAD_CLIP,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "class_weights_enabled": LSTM_USE_CLASS_WEIGHTS,
            "class_weights": {label: float(weight) for label, weight in zip(LABELS, lstm_class_weights)},
            "history_file": "lstm_training_history.csv",
        },
        "smote": {
            "enabled": True,
            "applied_after_split": True,
            "validation_untouched": True,
            "xgboost_training_rows_before": int(len(y_train)),
            "xgboost_training_rows_after": int(len(y_train_smote)),
            "lstm_training_sequences_before": int(len(sy_train)),
            "lstm_training_sequences_after": int(len(sy_train_smote)),
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
