from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample

VITAL_MAP = {"Temp": "temperature", "HR": "heart_rate", "O2Sat": "spo2", "Resp": "respiratory_rate"}
VITALS = list(VITAL_MAP.values())
LABELS = ["normal_vitals", "critical_vitals", "device_error"]


def read_psv_files(raw_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    frames = []
    for idx, path in enumerate(sorted(raw_dir.glob("*.psv"))):
        if max_files and idx >= max_files:
            break
        df = pd.read_csv(path, sep="|")
        if any(col not in df.columns for col in VITAL_MAP):
            continue
        df = df[list(VITAL_MAP)].rename(columns=VITAL_MAP)
        df["patient_id"] = path.stem
        df["timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="h")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No Challenge 2019 .psv files with required vitals found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in VITALS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby("patient_id")[col].transform(lambda s: s.interpolate(limit_direction="both"))
        out[col] = out[col].fillna(out[col].median())
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["patient_id", "timestamp"]).copy()
    for col in VITALS:
        out[f"{col}_delta"] = out.groupby("patient_id")[col].diff().fillna(0)
        out[f"{col}_mean_12"] = out.groupby("patient_id")[col].transform(lambda s: s.rolling(12, min_periods=1).mean())
        out[f"{col}_min_12"] = out.groupby("patient_id")[col].transform(lambda s: s.rolling(12, min_periods=1).min())
        out[f"{col}_max_12"] = out.groupby("patient_id")[col].transform(lambda s: s.rolling(12, min_periods=1).max())
    return out


def label_clinical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    signals = (
        ((out["temperature"] >= 38.3) | (out["temperature"] <= 35.5)).astype(int) +
        ((out["heart_rate"] >= 120) | (out["heart_rate"] <= 45)).astype(int) +
        (out["spo2"] < 92).astype(int) +
        ((out["respiratory_rate"] >= 26) | (out["respiratory_rate"] <= 8)).astype(int)
    )
    trend = (
        (out["heart_rate_delta"].abs() > 35) |
        (out["spo2_delta"].abs() > 8) |
        (out["respiratory_rate_delta"].abs() > 10)
    )
    out["label"] = np.where((signals >= 2) | ((signals == 1) & trend), "critical_vitals", "normal_vitals")
    return out


def simulate_device_errors(df: pd.DataFrame, fraction: float = 0.20) -> pd.DataFrame:
    n = max(1, int(len(df) * fraction))
    faults = df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)
    fault_type = np.random.default_rng(42).choice(["stuck", "impossible", "jump", "flat_missing"], size=n)
    for i, kind in enumerate(fault_type):
        if kind == "stuck":
            value = round(float(faults.at[i, "temperature"]), 2)
            faults.loc[i, VITALS] = [value, value, value, value]
        elif kind == "impossible":
            faults.loc[i, VITALS] = [0, 0, 130, 0]
        elif kind == "jump":
            faults.loc[i, VITALS] = [44.5, 230, 55, 58]
        else:
            faults.loc[i, VITALS] = [np.nan, np.nan, np.nan, np.nan]
    faults["label"] = "device_error"
    return add_features(clean(faults))


def balance(df: pd.DataFrame) -> pd.DataFrame:
    groups = [df[df["label"] == label] for label in LABELS]
    target = min(len(group) for group in groups if len(group) > 0)
    balanced = [resample(group, replace=False, n_samples=target, random_state=42) for group in groups]
    return pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)


def visualize(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df["label"].value_counts().reindex(LABELS).plot(kind="bar", color=["#16a34a", "#dc2626", "#f59e0b"])
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png")
    plt.close()
    sample = df.groupby("label").head(80)
    for col in VITALS:
        for label, group in sample.groupby("label"):
            plt.plot(range(len(group)), group[col].values, label=label, alpha=0.8)
        plt.title(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_trends.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/cleaned")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    base = add_features(clean(read_psv_files(raw_dir, args.max_files)))
    clinical = label_clinical(base)
    faults = simulate_device_errors(base)
    combined = balance(pd.concat([clinical, faults], ignore_index=True))
    combined.to_csv(output_dir / "cleaned_vitals.csv", index=False)
    visualize(combined, processed_dir)
    print(f"Saved {len(combined)} balanced rows to {output_dir / 'cleaned_vitals.csv'}")


if __name__ == "__main__":
    main()
