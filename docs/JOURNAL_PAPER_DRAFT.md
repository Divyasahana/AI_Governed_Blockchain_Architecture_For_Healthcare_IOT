# A Secure AI-Driven IoT Healthcare Monitoring Framework Using Adaptive Model Fusion and Blockchain-Based Encrypted Storage

## Abstract

The increasing adoption of Internet of Things (IoT) medical sensors enables continuous monitoring of patient vital signs such as body temperature, heart rate, oxygen saturation, and respiratory rate. However, real-time IoT healthcare systems face three major challenges: reliable classification of physiological status, detection of faulty sensor readings, and secure tamper-resistant storage of sensitive medical data. This work presents an integrated IoT healthcare monitoring framework that combines machine learning, adaptive model fusion, time-series database storage, asymmetric encryption, and Ethereum blockchain-based record anchoring.

The proposed system preprocesses the PhysioNet/Computing in Cardiology Challenge 2019 medical dataset by selecting temperature, heart rate, SpO2, and respiratory rate. The data are cleaned, feature-engineered using temporal trends, labelled into `normal_vitals`, `critical_vitals`, and `device_error`, and balanced for model training. The AI layer consists of an Isolation Forest for anomaly and device-fault detection, XGBoost for current-reading classification, and an LSTM network for temporal sequence classification. A pretrained adaptive fusion gate learns the reliability of XGBoost and LSTM predictions using original vitals, XGBoost probabilities, LSTM probabilities, and Isolation Forest anomaly score. The final labelled record is stored in InfluxDB and encrypted using hybrid asymmetric encryption before blockchain storage.

The novelty of this work lies in its complete layered architecture that unifies sensor-level vital acquisition, multi-model health-state classification, adaptive fusion, device-error identification, time-series medical record storage, and encrypted blockchain anchoring in a single deployable system. The framework is implemented with a Flask backend, React dashboard, IoT simulator, ESP32 sensor interface, InfluxDB, and Ethereum smart contract support.

## Keywords

IoT healthcare, medical sensors, vital sign monitoring, machine learning, Isolation Forest, XGBoost, LSTM, adaptive fusion, blockchain, Ethereum, asymmetric encryption, InfluxDB, secure medical data storage

## 1. Introduction

Remote patient monitoring has become an important component of modern healthcare systems. IoT-enabled medical devices can continuously measure vital signs and transmit them to cloud or edge platforms for clinical interpretation. Commonly monitored physiological parameters include body temperature, heart rate, oxygen saturation, and respiratory rate. These parameters provide useful indicators of patient condition and can assist in early detection of critical health states.

Despite these advantages, IoT healthcare monitoring systems face several practical limitations. First, sensor readings may be noisy, incomplete, or faulty due to device malfunction, poor contact, environmental interference, or communication errors. Second, patient criticality cannot always be inferred from a single vital sign threshold because combinations of multiple vitals and their temporal behavior may indicate deterioration. Third, medical data are sensitive and require secure storage, integrity protection, and controlled access. Conventional databases can support fast retrieval, but they do not inherently provide tamper-evident record integrity. Blockchain can provide immutable record anchoring, but raw medical data should not be stored openly on-chain.

To address these issues, this project proposes a secure AI-driven IoT health monitoring system. The system classifies incoming sensor readings into three categories: `normal_vitals`, `critical_vitals`, and `device_error`. It uses three models: Isolation Forest, XGBoost, and LSTM. Isolation Forest detects abnormal or faulty sensor behavior, XGBoost classifies individual feature-rich readings, and LSTM captures temporal patterns over sequences of readings. Their outputs are combined using an adaptive feed-forward fusion gate that learns how much trust should be assigned to XGBoost and LSTM for each input condition. The final prediction, confidence score, trust score, and original vitals are stored in InfluxDB and encrypted before being anchored in Ethereum blockchain.

## 2. Problem Statement

The objective of this work is to design and implement an IoT medical monitoring framework that:

- Acquires or simulates medical sensor readings for temperature, heart rate, SpO2, and respiratory rate.
- Preprocesses and labels Challenge 2019 medical data into normal, critical, and device-error classes.
- Detects faulty sensor readings using an unsupervised anomaly detection model.
- Classifies patient vital status using supervised machine learning and deep learning models.
- Uses temporal information and combinations of multiple vitals instead of relying only on single-value thresholds.
- Applies adaptive fusion to combine model predictions.
- Stores input vitals and model outputs in a time-series database.
- Encrypts final medical records before blockchain storage.
- Provides a dashboard for live monitoring, manual JSON testing, database inspection, and blockchain record viewing.

## 3. Novelty and Contributions

The main novelty of this work is the integration of AI-based vital classification, adaptive model fusion, sensor-fault detection, secure time-series storage, and blockchain-based integrity protection into a single working IoT healthcare architecture.

The key contributions are:

1. **Three-class IoT medical vital classification**

   The system classifies incoming readings into `normal_vitals`, `critical_vitals`, and `device_error`, allowing the framework to distinguish between patient deterioration and possible sensor/device faults.

2. **Hybrid AI model design**

   The AI layer combines:

   - Isolation Forest for unsupervised anomaly and device-error detection.
   - XGBoost for current-reading classification using engineered vital features.
   - LSTM for sequence-based classification when at least 12 readings are available.

3. **Adaptive fusion gate instead of fixed voting**

   Instead of manually assigning fixed weights to XGBoost and LSTM, the system trains a feed-forward neural network that outputs an adaptive alpha value. This alpha controls how much trust is given to XGBoost versus LSTM for each input.

4. **Stage 3 fusion trained on model-output-level evidence**

   The adaptive fusion gate is trained using:

   - Original vitals.
   - XGBoost probabilities.
   - LSTM probabilities.
   - Isolation Forest anomaly score.
   - Confidence and entropy-based reliability features.
   - Sequence availability flag.

5. **Fault-aware medical prediction pipeline**

   The system does not directly assume every abnormal reading indicates a critical patient state. It first evaluates whether the reading may be a device error or sensor fault.

6. **Time-series and blockchain dual-storage strategy**

   InfluxDB is used for queryable time-series medical monitoring, while Ethereum is used for tamper-evident record anchoring.

7. **Hybrid encryption for medical record protection**

   Full medical records are encrypted using AES-GCM, while the AES key is encrypted using RSA-OAEP. This avoids the payload-size limitation of direct RSA encryption while preserving asymmetric-key security.

8. **Complete demonstration stack**

   The project includes a Flask API, React dashboard, simulator, ESP32 firmware scaffold, InfluxDB integration, Ethereum smart contract, and blockchain interaction scripts.

## 4. Proposed System Architecture

The proposed system is organized into the following layers:

1. IoT Data Acquisition Layer
2. Data Preprocessing and Visualization Layer
3. AI Model Training Layer
4. Real-Time Inference Layer
5. Adaptive Fusion Layer
6. Database Layer
7. Blockchain and Encryption Layer
8. Flask Server Layer
9. Dashboard Layer

The overall data flow is:

```text
IoT Sensor / Simulator
        |
        v
Flask API
        |
        v
Feature Engineering
        |
        v
Isolation Forest -> XGBoost -> LSTM
        |
        v
Adaptive Fusion Gate
        |
        v
Final Label + Confidence + Trust Score
        |
        +--> InfluxDB Time-Series Storage
        |
        +--> Encryption + Ethereum Blockchain Record Anchoring
        |
        v
React Dashboard Visualization
```

## 5. IoT Data Acquisition Layer

The IoT data acquisition layer is responsible for collecting patient vital signs. The target hardware setup includes:

- ESP32 microcontroller.
- MAX30102 sensor for heart rate and SpO2.
- DS18B20 temperature sensor.
- Respiratory rate estimation from sensor signal trends or external respiratory sensor input.

For demonstration and testing, a simulator is included. The simulator generates three types of readings:

- Normal vitals.
- Critical vitals.
- Device-error readings.

Example JSON payload:

```json
{
  "temperature": 39.4,
  "heart_rate": 135,
  "spo2": 88,
  "respiratory_rate": 31,
  "device_id": "ESP32-SIM-01",
  "patient_id": "patient-demo",
  "timestamp": "2026-07-01T10:00:00Z"
}
```

## 6. Data Preprocessing and Visualization Layer

The preprocessing layer uses the Challenge 2019 medical dataset. Only the required vitals are selected:

| Original Column | Project Feature |
|---|---|
| `Temp` | `temperature` |
| `HR` | `heart_rate` |
| `O2Sat` | `spo2` |
| `Resp` | `respiratory_rate` |

The preprocessing steps are:

1. Load `.psv` files from `data/raw/`.
2. Select temperature, heart rate, SpO2, and respiratory rate.
3. Convert values to numeric format.
4. Interpolate missing values per patient.
5. Fill remaining missing values using median values.
6. Generate temporal features:
   - Vital deltas.
   - Rolling 12-reading mean.
   - Rolling 12-reading minimum.
   - Rolling 12-reading maximum.
7. Generate clinical labels:
   - `normal_vitals`
   - `critical_vitals`
   - `device_error`
8. Simulate device-error rows if real device-error examples are unavailable.
9. Balance all three classes.
10. Save the final dataset as `data/cleaned/cleaned_vitals.csv`.

Visualization outputs include:

- Class distribution plot.
- Temperature trend plot.
- Heart-rate trend plot.
- SpO2 trend plot.
- Respiratory-rate trend plot.

## 7. AI Model Training Layer

### 7.1 Isolation Forest

Isolation Forest is trained using only `normal_vitals` and `critical_vitals`. Device-error data are excluded during training so that future sensor faults can appear as anomalies.

Input features:

- Original vitals.
- Delta features.
- Rolling-window features.

Output:

- Anomaly score.
- Device-error flag when the anomaly score is high.

### 7.2 XGBoost Classifier

XGBoost is trained on all three classes:

- `normal_vitals`
- `critical_vitals`
- `device_error`

XGBoost is useful when only the current reading or a short history is available.

Output:

```json
{
  "normal_vitals": 0.12,
  "critical_vitals": 0.84,
  "device_error": 0.04
}
```

### 7.3 LSTM Classifier

The LSTM model is trained on sequences of 12 readings. It captures temporal patterns that cannot be represented by a single input row.

Input:

- 12-reading sequence.
- Original vitals.
- Delta features.

Output:

```json
{
  "normal_vitals": 0.20,
  "critical_vitals": 0.75,
  "device_error": 0.05
}
```

If fewer than 12 readings are available during runtime, the system uses XGBoost probabilities as fallback for the LSTM path and marks sequence availability as false.

## 8. Real-Time Inference Pipeline

When a new IoT reading arrives:

1. The payload is normalized.
2. Feature engineering is applied.
3. Isolation Forest calculates anomaly score.
4. XGBoost predicts class probabilities.
5. LSTM predicts class probabilities if 12 readings are available.
6. Adaptive fusion gate computes alpha.
7. Final class probabilities are generated.
8. Final label, confidence score, and trust score are produced.

The three final labels are:

- `normal_vitals`
- `critical_vitals`
- `device_error`

## 9. Adaptive Fusion Layer

The adaptive fusion layer is a feed-forward neural network that learns how much trust should be assigned to XGBoost and LSTM.

### 9.1 Fusion Inputs

The adaptive fusion gate is trained using:

- Original vitals:
  - Temperature.
  - Heart rate.
  - SpO2.
  - Respiratory rate.
- XGBoost probability vector.
- LSTM probability vector.
- Isolation Forest anomaly score.
- XGBoost confidence.
- LSTM confidence.
- XGBoost entropy.
- LSTM entropy.
- Sequence availability flag.

### 9.2 Fusion Output

The fusion gate outputs alpha:

```text
alpha = value between 0 and 1
```

The final probability is calculated as:

```text
Final Probability = alpha * XGBoost Probability + (1 - alpha) * LSTM Probability
```

Interpretation:

- Higher alpha means XGBoost is trusted more.
- Lower alpha means LSTM is trusted more.
- XGBoost is more useful for stable current readings.
- LSTM is more useful when temporal sequence patterns are available.

## 10. Database Layer

InfluxDB is used as the time-series database. Each reading is stored as a `medical_vitals` measurement.

Stored fields include:

- `temperature`
- `heart_rate`
- `spo2`
- `respiratory_rate`
- `anomaly_score`
- `alpha`
- `confidence`
- `trust_score`
- `xgb_normal_vitals`
- `xgb_critical_vitals`
- `xgb_device_error`
- `lstm_normal_vitals`
- `lstm_critical_vitals`
- `lstm_device_error`
- `fusion_normal_vitals`
- `fusion_critical_vitals`
- `fusion_device_error`

Tags include:

- `device_id`
- `patient_id`
- `final_label`

InfluxDB supports fast retrieval and visualization of time-series medical data.

## 11. Blockchain and Encryption Layer

The blockchain layer protects the integrity of final medical records.

The full medical prediction record contains:

- Input vitals.
- Timestamp.
- Device ID.
- Patient ID.
- Model probabilities.
- Isolation Forest anomaly score.
- Adaptive fusion alpha.
- Final label.
- Confidence score.
- Trust score.

### 11.1 Encryption

The system uses hybrid encryption:

1. AES-GCM encrypts the full medical record.
2. RSA-OAEP encrypts the AES key.
3. The encrypted payload is stored or referenced securely.

This design is used because direct RSA encryption cannot encrypt large medical JSON payloads.

### 11.2 Blockchain Storage

Ethereum stores metadata rather than raw medical data:

- SHA-256 record hash.
- Encrypted data reference.
- Timestamp.
- Device ID.
- Final label.
- Trust score.

The smart contract is `MedicalRecordStore.sol`.

This provides tamper-evident record anchoring without exposing sensitive patient data directly on-chain.

## 12. Flask Server Layer

The Flask backend provides REST APIs:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check backend and InfluxDB status |
| `POST /api/vitals` | Submit vitals and receive prediction |
| `GET /api/latest` | Get latest prediction |
| `GET /api/db-records` | Read records from InfluxDB |
| `GET /api/db-status` | Check InfluxDB connection status |
| `POST /api/blockchain/store` | Encrypt and store latest or supplied record |
| `GET /api/blockchain-records` | View blockchain records |
| `GET /api/models/status` | Check model artifact status |
| `POST /api/simulator/start` | Start simulator from dashboard |
| `POST /api/simulator/stop` | Stop simulator |

## 13. Dashboard Layer

The React dashboard provides:

1. **Live Monitoring Tab**
   - Latest vitals.
   - Label.
   - Trust score.
   - Alpha.
   - Anomaly score.
   - XGBoost, LSTM, and fusion probabilities.

2. **Test Prediction Tab**
   - User can paste sample JSON.
   - Dashboard displays full model output.
   - User can encrypt and store the prediction in blockchain.

3. **Database Records Tab**
   - Shows InfluxDB records.
   - Displays each vital in a separate column.
   - Shows label, confidence, anomaly score, alpha, trust score, and blockchain transaction hash.

4. **Blockchain Records Tab**
   - Shows encrypted blockchain storage metadata.
   - Supports optional local decryption for demonstration.

## 14. Experimental Setup

The implementation uses:

| Component | Technology |
|---|---|
| Backend | Flask |
| Frontend | React + Vite |
| Machine learning | scikit-learn, XGBoost, PyTorch |
| Time-series database | InfluxDB |
| Blockchain | Ethereum, Solidity, Hardhat |
| Encryption | RSA-OAEP, AES-GCM |
| Simulator | Python |
| IoT hardware target | ESP32 |

Model artifacts:

- `xgboost_model.pkl`
- `isolation_forest.pkl`
- `lstm_model.pt`
- `adaptive_fusion.pt`
- Scalers and metadata files

## 15. Evaluation Metrics

The following metrics can be used to evaluate the framework:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Class-wise performance for normal, critical, and device-error labels.
- Anomaly detection behavior for sensor-fault examples.
- Fusion validation accuracy.
- Inference latency.
- Blockchain storage latency.
- Database write/read latency.

The system should be evaluated separately for:

1. XGBoost performance.
2. LSTM performance.
3. Isolation Forest anomaly detection.
4. Adaptive fusion performance.
5. End-to-end runtime behavior.

## 16. Expected Results and Observations

The expected behavior is:

- Normal vitals should be classified as `normal_vitals`.
- Abnormal combinations of vitals should be classified as `critical_vitals`.
- Impossible or faulty readings should be classified as `device_error`.
- LSTM should become active after at least 12 readings from the same device.
- Adaptive fusion alpha should change depending on model confidence and sequence availability.
- InfluxDB should store each vital and model output as separate fields.
- Blockchain should store encrypted record reference and hash metadata.

## 17. Discussion

This framework demonstrates how AI and blockchain can be combined for secure IoT medical monitoring. The use of multiple models allows the system to handle different inference conditions. Isolation Forest provides fault-awareness, XGBoost provides strong classification for current readings, and LSTM provides temporal pattern recognition. The adaptive fusion gate improves flexibility by learning how to combine XGBoost and LSTM rather than relying on fixed voting.

The database and blockchain design separates operational monitoring from integrity protection. InfluxDB is suitable for time-series querying, while Ethereum provides tamper-evident anchoring. Encryption ensures that sensitive medical data are not exposed on-chain.

## 18. Limitations

This project is a prototype and has the following limitations:

- The generated labels are rule-based and derived from selected vitals rather than direct clinical diagnosis.
- Device-error examples are simulated.
- Clinical validation with real hospital deployment has not been performed.
- The blockchain layer stores metadata and references, not raw medical data.
- The ESP32 firmware requires hardware calibration before real clinical use.
- Model performance depends on preprocessing assumptions and dataset quality.

The system should not be used for real medical decision-making without clinical validation, regulatory review, and deployment hardening.

## 19. Conclusion

This work presents a layered secure IoT healthcare monitoring framework that integrates medical sensor data acquisition, AI-based classification, anomaly detection, adaptive fusion, time-series storage, encryption, and blockchain anchoring. The proposed approach classifies patient readings into normal, critical, and device-error categories while preserving data security and integrity. The adaptive fusion gate provides a flexible method for combining current-reading and temporal-sequence models. The complete implementation demonstrates the feasibility of combining IoT, machine learning, InfluxDB, and Ethereum blockchain for secure medical monitoring applications.

Future work can include real sensor deployment, larger clinical evaluation, improved device-fault modelling, explainable AI integration, blockchain gas optimization, and integration with healthcare interoperability standards such as HL7/FHIR.

## References

Add final formatted references before submission. Suggested reference categories:

1. PhysioNet/Computing in Cardiology Challenge 2019 dataset paper.
2. Isolation Forest original paper.
3. XGBoost original paper.
4. LSTM original paper.
5. Blockchain in healthcare survey papers.
6. IoT healthcare monitoring survey papers.
7. InfluxDB/time-series database documentation or technical references.
8. Ethereum smart contract references.

