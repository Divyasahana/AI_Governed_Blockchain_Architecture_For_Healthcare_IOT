from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

try:
    import psutil
except Exception:
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from backend.blockchain.crypto import decrypt_record, encrypt_record, encrypted_data_hash
from backend.blockchain.service import BlockchainService
from backend.database.influx_client import InfluxStore
from backend.ml.models import AdaptiveFusionGate, LABELS


DATA = ROOT / "data" / "cleaned" / "cleaned_vitals.csv"
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "experimental results.md"
VITALS = ["temperature", "heart_rate", "spo2", "respiratory_rate"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0)
    return -np.sum(p * np.log(p), axis=1, keepdims=True)


def build_lstm_probabilities(df: pd.DataFrame, metadata: dict, fallback_probs: np.ndarray):
    sequence_columns = metadata["sequence_columns"]
    sequence_length = int(metadata["sequence_length"])
    lstm_scaler = joblib.load(ARTIFACTS / "lstm_scaler.pkl")
    model = torch.jit.load(str(ARTIFACTS / "lstm_model.pt"))
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


def specificities(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    values = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        values.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return values


def metric_row(name, y_true, probs):
    pred = probs.argmax(axis=1)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "specificity_macro": float(np.mean(specificities(y_true, pred))),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc,
        "mcc": matthews_corrcoef(y_true, pred),
        "kappa": cohen_kappa_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }


def per_class_table(y_true, probs):
    pred = probs.argmax(axis=1)
    specs = specificities(y_true, pred)
    rows = []
    for i, label in enumerate(LABELS):
        binary_true = (y_true == i).astype(int)
        binary_pred = (pred == i).astype(int)
        rows.append({
            "class": label,
            "precision": precision_score(binary_true, binary_pred, zero_division=0),
            "recall": recall_score(binary_true, binary_pred, zero_division=0),
            "specificity": specs[i],
            "f1": f1_score(binary_true, binary_pred, zero_division=0),
            "roc_auc": roc_auc_score(binary_true, probs[:, i]),
        })
    return rows


def timed_ms(fn, repeats=100):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return float(np.mean(values)), float(np.std(values))


def fmt(value, digits=4):
    if isinstance(value, str):
        return value
    if value != value:
        return "N/A"
    return f"{value:.{digits}f}"


def pct(value):
    return f"{value * 100:.2f}%"


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def main():
    df = pd.read_csv(DATA)
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
    feature_cols = metadata["feature_columns"]
    y = df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )[1]
    y_val = y[val_idx]

    xgb = joblib.load(ARTIFACTS / "xgboost_model.pkl")
    feature_scaler = joblib.load(ARTIFACTS / "feature_scaler.pkl")
    iforest = joblib.load(ARTIFACTS / "isolation_forest.pkl")
    if_scaler = joblib.load(ARTIFACTS / "iforest_scaler.pkl")
    fusion_scaler = joblib.load(ARTIFACTS / "fusion_scaler.pkl")

    x_frame = df[feature_cols]
    x_all = x_frame.to_numpy(float)
    x_scaled = feature_scaler.transform(x_all)
    xgb_probs = xgb.predict_proba(x_scaled)
    lstm_probs, seq_flag = build_lstm_probabilities(df, metadata, xgb_probs)

    anomaly = np.clip(0.5 - iforest.decision_function(if_scaler.transform(x_frame)), 0, 1).reshape(-1, 1)
    confidence = np.hstack([
        xgb_probs.max(axis=1, keepdims=True),
        lstm_probs.max(axis=1, keepdims=True),
        entropy(xgb_probs),
        entropy(lstm_probs),
    ])
    fusion_x = np.hstack([df[VITALS].to_numpy(float), xgb_probs, lstm_probs, anomaly, confidence, seq_flag])
    fusion_model = AdaptiveFusionGate(input_dim=fusion_x.shape[1])
    fusion_model.load_state_dict(torch.load(ARTIFACTS / "adaptive_fusion.pt", map_location="cpu"))
    fusion_model.eval()
    with torch.no_grad():
        alpha = fusion_model(torch.tensor(fusion_scaler.transform(fusion_x), dtype=torch.float32)).numpy()
    fusion_probs = alpha * xgb_probs + (1 - alpha) * lstm_probs
    fusion_probs = fusion_probs / fusion_probs.sum(axis=1, keepdims=True)

    model_metrics = [
        metric_row("XGBoost", y_val, xgb_probs[val_idx]),
        metric_row("LSTM with XGBoost fallback", y_val, lstm_probs[val_idx]),
        metric_row("Adaptive Fusion", y_val, fusion_probs[val_idx]),
    ]

    sequence_val = val_idx[seq_flag[val_idx, 0] == 1.0]
    sequence_only_metrics = metric_row("LSTM sequence-only rows", y[sequence_val], lstm_probs[sequence_val])
    fusion_per_class = per_class_table(y_val, fusion_probs[val_idx])
    cm = confusion_matrix(y_val, fusion_probs[val_idx].argmax(axis=1), labels=[0, 1, 2])

    actual_fault = (y_val == LABEL_TO_ID["device_error"]).astype(int)
    detected_fault = (iforest.predict(if_scaler.transform(x_frame.iloc[val_idx])) == -1).astype(int)
    fault_tp = int(((actual_fault == 1) & (detected_fault == 1)).sum())
    fault_tn = int(((actual_fault == 0) & (detected_fault == 0)).sum())
    fault_fp = int(((actual_fault == 0) & (detected_fault == 1)).sum())
    fault_fn = int(((actual_fault == 1) & (detected_fault == 0)).sum())

    sample_idx = val_idx[: min(300, len(val_idx))]
    sample_x = x_frame.iloc[sample_idx]
    sample_feature_scaled = x_scaled[sample_idx]
    sample_if_scaled = if_scaler.transform(sample_x)
    sample_fusion = fusion_x[sample_idx]

    xgb_time = timed_ms(lambda: xgb.predict_proba(sample_feature_scaled[:1]), repeats=300)
    if_time = timed_ms(lambda: iforest.decision_function(sample_if_scaled[:1]), repeats=300)
    lstm_model = torch.jit.load(str(ARTIFACTS / "lstm_model.pt"))
    lstm_model.eval()
    seq_cols = metadata["sequence_columns"]
    seq_len = int(metadata["sequence_length"])
    seq_group = df.sort_values(["patient_id", "timestamp"]).groupby("patient_id").filter(lambda x: len(x) >= seq_len)
    seq_values = seq_group[seq_cols].head(seq_len).to_numpy(dtype=np.float32)
    seq_values = joblib.load(ARTIFACTS / "lstm_scaler.pkl").transform(seq_values).reshape(1, seq_len, -1)
    seq_tensor = torch.tensor(seq_values, dtype=torch.float32)
    lstm_time = timed_ms(lambda: torch.softmax(lstm_model(seq_tensor), dim=-1), repeats=300)
    fusion_time = timed_ms(lambda: fusion_model(torch.tensor(fusion_scaler.transform(sample_fusion[:1]), dtype=torch.float32)), repeats=300)
    total_ai_time = (
        if_time[0] + xgb_time[0] + lstm_time[0] + fusion_time[0],
        (if_time[1] ** 2 + xgb_time[1] ** 2 + lstm_time[1] ** 2 + fusion_time[1] ** 2) ** 0.5,
    )

    chain = BlockchainService()
    chain_records = chain.list_records()
    web3, contract = chain._contract_connection()
    chain_count_time = timed_ms(lambda: contract.functions.recordCount().call(), repeats=20) if contract else (float("nan"), float("nan"))
    if contract and chain_records:
        retrieve_time = timed_ms(lambda: contract.functions.getRecord(0).call(), repeats=20)
    else:
        retrieve_time = (float("nan"), float("nan"))

    payload = {
        "heart_rate": 82,
        "temperature": 37.1,
        "respiratory_rate": 18,
        "spo2": 98,
    }
    encrypted = encrypt_record(payload, "keys/public.pem")
    aes_rsa_encrypt_time = timed_ms(lambda: encrypt_record(payload, "keys/public.pem"), repeats=100)
    rsa_aes_decrypt_time = timed_ms(lambda: decrypt_record(encrypted, "keys/private.pem"), repeats=100)
    hash_time = timed_ms(lambda: encrypted_data_hash(encrypted), repeats=300)

    gas_estimate = float("nan")
    gas_price_gwei = float("nan")
    estimated_cost_eth = float("nan")
    try:
        account = web3.eth.account.from_key(__import__("os").getenv("ETH_PRIVATE_KEY") or __import__("os").getenv("PRIVATE_KEY"))
        tx_args = (
            "patient-demo",
            bytes.fromhex(encrypted_data_hash(encrypted).removeprefix("0x")),
            "influxdb://vitals/medical_vitals/evaluation",
            account.address,
            int(time.time()),
            "normal_vitals",
            9900,
        )
        gas_estimate = contract.functions.storeRecord(*tx_args).estimate_gas({"from": account.address})
        gas_price = web3.eth.gas_price
        gas_price_gwei = float(web3.from_wei(gas_price, "gwei"))
        estimated_cost_eth = float(web3.from_wei(gas_estimate * gas_price, "ether"))
    except Exception:
        pass

    db = InfluxStore()
    db_status = db.status()
    db_read_time = timed_ms(lambda: db.list_records(100, require_influx=True), repeats=20) if db.connected else (float("nan"), float("nan"))
    try:
        api_time = timed_ms(lambda: urllib.request.urlopen("http://127.0.0.1:5000/api/blockchain-records", timeout=10).read(), repeats=10)
    except Exception:
        api_time = (float("nan"), float("nan"))
    if psutil is not None:
        resource_sample = {
            "cpu_percent": f"{psutil.cpu_percent(interval=1):.2f}%",
            "ram_mb": f"{psutil.virtual_memory().used / (1024 ** 2):.2f} MB",
            "disk_gb": f"{psutil.disk_usage(str(ROOT)).used / (1024 ** 3):.2f} GB",
        }
    else:
        resource_sample = {
            "cpu_percent": "N/A - psutil is not installed",
            "ram_mb": "N/A - psutil is not installed",
            "disk_gb": "N/A - psutil is not installed",
        }

    fusion_pred = fusion_probs[val_idx].argmax(axis=1)
    xgb_correct = (xgb_probs[val_idx].argmax(axis=1) == y_val).astype(float)
    lstm_correct = (lstm_probs[val_idx].argmax(axis=1) == y_val).astype(float)
    fusion_correct = (fusion_pred == y_val).astype(float)
    try:
        t_xgb = ttest_rel(fusion_correct, xgb_correct).pvalue
        w_xgb = wilcoxon(fusion_correct, xgb_correct).pvalue
    except Exception:
        t_xgb = float("nan")
        w_xgb = float("nan")
    try:
        t_lstm = ttest_rel(fusion_correct, lstm_correct).pvalue
        w_lstm = wilcoxon(fusion_correct, lstm_correct).pvalue
    except Exception:
        t_lstm = float("nan")
        w_lstm = float("nan")

    lines = [
        "# Experimental Results",
        "",
        "This file contains measured experimental values computed from the current project artifacts and dataset. Values were generated from `data/cleaned/cleaned_vitals.csv` and the trained models in `artifacts/`.",
        "",
        "## Dataset and Split",
        "",
        markdown_table(
            ["Item", "Value"],
            [
                ["Total records", len(df)],
                ["Validation records", len(val_idx)],
                ["Training records", len(df) - len(val_idx)],
                ["Class labels", ", ".join(LABELS)],
                ["Sequence length", metadata["sequence_length"]],
                ["Rows with real LSTM sequence", int(seq_flag.sum())],
                ["Rows using XGBoost fallback for LSTM", int(len(df) - seq_flag.sum())],
            ],
        ),
        "",
        "## Model Performance",
        "",
        markdown_table(
            ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "MCC", "Kappa", "Balanced Acc."],
            [
                [
                    m["model"],
                    pct(m["accuracy"]),
                    pct(m["precision_macro"]),
                    pct(m["recall_macro"]),
                    pct(m["specificity_macro"]),
                    pct(m["f1_macro"]),
                    fmt(m["roc_auc_macro"]),
                    fmt(m["mcc"]),
                    fmt(m["kappa"]),
                    pct(m["balanced_accuracy"]),
                ]
                for m in model_metrics
            ],
        ),
        "",
        "## LSTM Sequence-Only Performance",
        "",
        "The deployed inference code falls back to XGBoost probabilities when a 12-reading temporal sequence is unavailable. The table below reports LSTM performance only on validation rows where a real LSTM sequence exists.",
        "",
        markdown_table(
            ["Model", "Rows", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [[
                sequence_only_metrics["model"],
                len(sequence_val),
                pct(sequence_only_metrics["accuracy"]),
                pct(sequence_only_metrics["precision_macro"]),
                pct(sequence_only_metrics["recall_macro"]),
                pct(sequence_only_metrics["f1_macro"]),
                fmt(sequence_only_metrics["roc_auc_macro"]),
            ]],
        ),
        "",
        "## Adaptive Fusion Per-Class Results",
        "",
        markdown_table(
            ["Class", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"],
            [
                [row["class"], pct(row["precision"]), pct(row["recall"]), pct(row["specificity"]), pct(row["f1"]), fmt(row["roc_auc"])]
                for row in fusion_per_class
            ],
        ),
        "",
        "## Adaptive Fusion Confusion Matrix",
        "",
        markdown_table(
            ["Actual / Predicted", *LABELS],
            [[LABELS[i], *[int(cm[i, j]) for j in range(3)]] for i in range(3)],
        ),
        "",
        "## Sensor Fault Detection: Isolation Forest",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["True positives", fault_tp],
                ["True negatives", fault_tn],
                ["False positives", fault_fp],
                ["False negatives", fault_fn],
                ["Detection rate / recall", pct(fault_tp / (fault_tp + fault_fn))],
                ["False alarm rate", pct(fault_fp / (fault_fp + fault_tn))],
                ["False negative rate", pct(fault_fn / (fault_fn + fault_tp))],
                ["Precision", pct(fault_tp / (fault_tp + fault_fp)) if fault_tp + fault_fp else "0.00%"],
                ["F1-score", pct(2 * fault_tp / (2 * fault_tp + fault_fp + fault_fn)) if (2 * fault_tp + fault_fp + fault_fn) else "0.00%"],
            ],
        ),
        "",
        "## Trust and Confidence Analysis",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Mean trust score", fmt(float(np.mean(fusion_probs[val_idx].max(axis=1))))],
                ["Std. dev. trust score", fmt(float(np.std(fusion_probs[val_idx].max(axis=1))))],
                ["Minimum trust score", fmt(float(np.min(fusion_probs[val_idx].max(axis=1))))],
                ["Maximum trust score", fmt(float(np.max(fusion_probs[val_idx].max(axis=1))))],
                ["Mean adaptive alpha", fmt(float(np.mean(alpha[val_idx])))],
                ["Std. dev. adaptive alpha", fmt(float(np.std(alpha[val_idx])))],
            ],
        ),
        "",
        "## Stage Runtime Performance",
        "",
        markdown_table(
            ["Stage", "Mean Time", "Std. Dev."],
            [
                ["Isolation Forest inference", f"{if_time[0]:.4f} ms", f"{if_time[1]:.4f} ms"],
                ["XGBoost inference", f"{xgb_time[0]:.4f} ms", f"{xgb_time[1]:.4f} ms"],
                ["LSTM inference", f"{lstm_time[0]:.4f} ms", f"{lstm_time[1]:.4f} ms"],
                ["Adaptive fusion gate", f"{fusion_time[0]:.4f} ms", f"{fusion_time[1]:.4f} ms"],
                ["Total AI stage", f"{total_ai_time[0]:.4f} ms", f"{total_ai_time[1]:.4f} ms"],
                ["Flask /api/blockchain-records response", f"{api_time[0]:.4f} ms", f"{api_time[1]:.4f} ms"],
            ],
        ),
        "",
        "## Cryptography Performance",
        "",
        markdown_table(
            ["Operation", "Mean Time", "Std. Dev."],
            [
                ["Hybrid AES-256-GCM + RSA-OAEP-256 encryption", f"{aes_rsa_encrypt_time[0]:.4f} ms", f"{aes_rsa_encrypt_time[1]:.4f} ms"],
                ["Hybrid AES-256-GCM + RSA-OAEP-256 decryption", f"{rsa_aes_decrypt_time[0]:.4f} ms", f"{rsa_aes_decrypt_time[1]:.4f} ms"],
                ["SHA-256 hash generation", f"{hash_time[0]:.4f} ms", f"{hash_time[1]:.4f} ms"],
                ["Total encryption/hash overhead", f"{aes_rsa_encrypt_time[0] + hash_time[0]:.4f} ms", f"{(aes_rsa_encrypt_time[1] ** 2 + hash_time[1] ** 2) ** 0.5:.4f} ms"],
            ],
        ),
        "",
        "## Blockchain Read Performance",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Sepolia contract records returned", len(chain_records)],
                ["recordCount() mean latency", f"{chain_count_time[0]:.4f} ms"],
                ["recordCount() std. dev.", f"{chain_count_time[1]:.4f} ms"],
                ["getRecord(0) mean latency", f"{retrieve_time[0]:.4f} ms"],
                ["getRecord(0) std. dev.", f"{retrieve_time[1]:.4f} ms"],
                ["Estimated storeRecord gas", fmt(float(gas_estimate), 0)],
                ["Current gas price", f"{gas_price_gwei:.4f} gwei" if gas_price_gwei == gas_price_gwei else "N/A"],
                ["Estimated storeRecord cost", f"{estimated_cost_eth:.10f} ETH" if estimated_cost_eth == estimated_cost_eth else "N/A"],
            ],
        ),
        "",
        "## Database Performance",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["InfluxDB connected", db_status["connected"]],
                ["InfluxDB bucket", db_status["bucket"]],
                ["InfluxDB read latency mean", f"{db_read_time[0]:.4f} ms" if db_read_time[0] == db_read_time[0] else "N/A"],
                ["InfluxDB read latency std. dev.", f"{db_read_time[1]:.4f} ms" if db_read_time[1] == db_read_time[1] else "N/A"],
                ["InfluxDB error", db_status.get("error") or ""],
            ],
        ),
        "",
        "## Current Resource Utilization",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["CPU usage", resource_sample["cpu_percent"]],
                ["RAM used", resource_sample["ram_mb"]],
                ["Disk used on project drive", resource_sample["disk_gb"]],
            ],
        ),
        "",
        "## Statistical Validation",
        "",
        markdown_table(
            ["Comparison", "Fusion Mean Acc.", "Baseline Mean Acc.", "Paired t-test p", "Wilcoxon p"],
            [
                ["Fusion vs XGBoost", fmt(float(np.mean(fusion_correct))), fmt(float(np.mean(xgb_correct))), fmt(float(t_xgb)), fmt(float(w_xgb))],
                ["Fusion vs LSTM fallback", fmt(float(np.mean(fusion_correct))), fmt(float(np.mean(lstm_correct))), fmt(float(t_lstm)), fmt(float(w_lstm))],
            ],
        ),
        "",
        "## Notes",
        "",
        "- Training and validation loss curves are not included because the current training scripts do not persist per-epoch loss history in `artifacts/`.",
        "- The blockchain gas value is an estimate from `estimate_gas`; the script did not send a new Sepolia transaction.",
        "- Blockchain transaction confirmation timing requires submitting new transactions and is not measured by this non-destructive evaluator.",
        "- InfluxDB write latency and scalability values require a live load test that writes new records; this evaluator only measures non-destructive reads.",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
