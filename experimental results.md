# Experimental Results

This file contains measured experimental values computed from the current project artifacts and dataset. Values were generated from `data/cleaned/cleaned_vitals.csv` and the trained models in `artifacts/`.

## Dataset and Split

| Item | Value |
| --- | --- |
| Total records | 150824 |
| Validation records | 30165 |
| Training records | 120659 |
| Class labels | normal_vitals, critical_vitals, device_error |
| Sequence length | 12 |
| Rows with real LSTM sequence | 115308 |
| Rows using XGBoost fallback for LSTM | 35516 |

## Model Performance

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | MCC | Kappa | Balanced Acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 98.04% | 88.15% | 98.78% | 99.26% | 92.26% | 0.9997 | 0.9455 | 0.9438 | 98.78% |
| LSTM with XGBoost fallback | 97.77% | 87.31% | 96.83% | 98.90% | 91.05% | 0.9964 | 0.9372 | 0.9357 | 96.83% |
| Adaptive Fusion | 98.84% | 91.86% | 99.17% | 99.55% | 95.00% | 0.9998 | 0.9667 | 0.9661 | 99.17% |

## LSTM Sequence-Only Performance

The deployed inference code falls back to XGBoost probabilities when a 12-reading temporal sequence is unavailable. The table below reports LSTM performance only on validation rows where a real LSTM sequence exists.

| Model | Rows | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM sequence-only rows | 23062 | 97.58% | 86.83% | 96.39% | 90.56% | 0.9963 |

## Adaptive Fusion Per-Class Results

| Class | Precision | Recall | Specificity | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| normal_vitals | 99.95% | 98.60% | 99.80% | 99.27% | 0.9999 |
| critical_vitals | 75.63% | 98.96% | 98.84% | 85.74% | 0.9995 |
| device_error | 100.00% | 99.96% | 100.00% | 99.98% | 1.0000 |

## Adaptive Fusion Confusion Matrix

| Actual / Predicted | normal_vitals | critical_vitals | device_error |
| --- | --- | --- | --- |
| normal_vitals | 23741 | 337 | 0 |
| critical_vitals | 11 | 1049 | 0 |
| device_error | 1 | 1 | 5025 |

## Sensor Fault Detection: Isolation Forest

| Metric | Value |
| --- | --- |
| True positives | 5027 |
| True negatives | 22102 |
| False positives | 3036 |
| False negatives | 0 |
| Detection rate / recall | 100.00% |
| False alarm rate | 12.08% |
| False negative rate | 0.00% |
| Precision | 62.35% |
| F1-score | 76.81% |

## Trust and Confidence Analysis

| Metric | Value |
| --- | --- |
| Mean trust score | 0.9695 |
| Std. dev. trust score | 0.0746 |
| Minimum trust score | 0.4964 |
| Maximum trust score | 0.9997 |
| Mean adaptive alpha | 0.5469 |
| Std. dev. adaptive alpha | 0.1644 |

## Stage Runtime Performance

| Stage | Mean Time | Std. Dev. |
| --- | --- | --- |
| Isolation Forest inference | 1.7529 ms | 0.1842 ms |
| XGBoost inference | 0.1917 ms | 0.0341 ms |
| LSTM inference | 0.3168 ms | 1.0887 ms |
| Adaptive fusion gate | 0.2162 ms | 0.0191 ms |
| Total AI stage | 2.4775 ms | 1.1048 ms |
| Flask /api/blockchain-records response | nan ms | nan ms |

## Cryptography Performance

| Operation | Mean Time | Std. Dev. |
| --- | --- | --- |
| Hybrid AES-256-GCM + RSA-OAEP-256 encryption | 0.2167 ms | 0.0376 ms |
| Hybrid AES-256-GCM + RSA-OAEP-256 decryption | 32.0565 ms | 0.8303 ms |
| SHA-256 hash generation | 0.0013 ms | 0.0014 ms |
| Total encryption/hash overhead | 0.2180 ms | 0.0376 ms |

## Blockchain Read Performance

| Metric | Value |
| --- | --- |
| Sepolia contract records returned | 61 |
| recordCount() mean latency | 42.3545 ms |
| recordCount() std. dev. | 2.2262 ms |
| getRecord(0) mean latency | 41.7320 ms |
| getRecord(0) std. dev. | 1.8738 ms |
| Estimated storeRecord gas | 242850 |
| Current gas price | 1.0997 gwei |
| Estimated storeRecord cost | 0.0002670560 ETH |

## Database Performance

| Metric | Value |
| --- | --- |
| InfluxDB connected | True |
| InfluxDB bucket | vitals |
| InfluxDB read latency mean | 263.9098 ms |
| InfluxDB read latency std. dev. | 35.8237 ms |
| InfluxDB error |  |

## Current Resource Utilization

| Metric | Value |
| --- | --- |
| CPU usage | N/A - psutil is not installed |
| RAM used | N/A - psutil is not installed |
| Disk used on project drive | N/A - psutil is not installed |

## Statistical Validation

| Comparison | Fusion Mean Acc. | Baseline Mean Acc. | Paired t-test p | Wilcoxon p |
| --- | --- | --- | --- | --- |
| Fusion vs XGBoost | 0.9884 | 0.9804 | 0.0000 | 0.0000 |
| Fusion vs LSTM fallback | 0.9884 | 0.9777 | 0.0000 | 0.0000 |

## Notes

- Training and validation loss curves are not included because the current training scripts do not persist per-epoch loss history in `artifacts/`.
- The blockchain gas value is an estimate from `estimate_gas`; the script did not send a new Sepolia transaction.
- Blockchain transaction confirmation timing requires submitting new transactions and is not measured by this non-destructive evaluator.
- InfluxDB write latency and scalability values require a live load test that writes new records; this evaluator only measures non-destructive reads.