# AI Governed Blockchain Architecture for Secure Healthcare IOT Systems

## Abstract

This project implements a secure IoT medical monitoring architecture for storing and verifying vital-sign data collected from medical sensors. The system processes temperature, heart rate, respiratory rate, and SpO2 readings, classifies each reading into `normal_vitals`, `critical_vitals`, or `device_error`, stores model outputs in InfluxDB, encrypts input vitals using asymmetric-key encryption, and stores verification metadata on an Ethereum smart contract.

The machine-learning pipeline uses three pretrained models: Isolation Forest, XGBoost, and LSTM. Their outputs are fused by a trained adaptive feed-forward fusion gate that predicts the alpha weight used to combine XGBoost and LSTM probabilities. The final trust score is the maximum probability from the Stage 3 fused output.

## Introduction

IoT medical sensors continuously generate sensitive physiological data. Such data must be monitored in real time, classified reliably, stored efficiently, and protected against tampering. Traditional storage systems can hold raw readings but do not provide strong tamper evidence. Public blockchains provide immutability, but storing complete medical records directly on-chain is expensive and inappropriate for privacy.

This project therefore uses a hybrid design:

- InfluxDB stores time-series vitals, model outputs, and encrypted vitals.
- Ethereum stores only hash and reference metadata.
- RSA-OAEP and AES-GCM protect vitals before off-chain storage.
- AI models classify sensor readings and produce confidence/trust values.

## Proposed Architecture

Runtime flow:

```text
IoT sensor / simulator
        |
        v
Flask API
        |
        v
Feature engineering
        |
        v
Stage 1: Isolation Forest -> anomaly score
Stage 2: XGBoost          -> 3-class probabilities
Stage 2: LSTM             -> 3-class probabilities when 12 readings exist
        |
        v
Stage 3: Adaptive Fusion Gate -> alpha
        |
        v
Final probabilities = alpha * XGBoost + (1 - alpha) * LSTM
        |
        v
Final label + trust score
        |
        v
Encrypt vitals
        |
        v
Store clean record in InfluxDB
        |
        v
Store hash/reference metadata on Ethereum
        |
        v
React dashboard
```

## Architecture Layers

### 1. IoT Data Acquisition Layer

The project supports real IoT sensor integration and a simulator for demonstration. The simulator generates all three classes:

- `normal_vitals`
- `critical_vitals`
- `device_error`

Simulator file:

```text
simulator/device_simulator.py
```

ESP32 firmware skeleton:

```text
esp32/
```

### 2. Data Preprocessing Layer

The preprocessing stage reads PhysioNet Challenge 2019 `.psv` files and extracts:

- Temperature
- Heart rate
- SpO2
- Respiratory rate

It cleans missing values, creates time-based and rolling features, balances labels, and simulates realistic device-error records when needed.

Main file:

```text
backend/ml/preprocess.py
```

Output:

```text
data/cleaned/cleaned_vitals.csv
```

### 3. Model Training Layer

The training stage creates:

- Isolation Forest model
- XGBoost model
- LSTM model
- Feature scalers
- Metadata

Main file:

```text
backend/ml/train_models.py
```

Artifacts:

```text
artifacts/
```

### 4. Stage 3 Adaptive Fusion Layer

Stage 3 is trained on outputs from previous models.

Training inputs:

- Original vitals
- XGBoost probabilities
- LSTM probabilities
- Isolation Forest anomaly score
- XGBoost confidence
- LSTM confidence
- XGBoost entropy
- LSTM entropy
- Sequence availability flag

Stage 3 predicts:

```text
alpha
```

Final probability formula:

```text
Final Prob = alpha * XGBoost + (1 - alpha) * LSTM
```

The final label is the class with the highest final probability. The final trust score is that maximum probability.

Main file:

```text
backend/ml/train_fusion_gate.py
```

Runtime fusion:

```text
backend/ml/fusion.py
backend/ml/inference.py
```

### 5. Database Layer

InfluxDB is the primary time-series database. The backend writes to InfluxDB first. Blockchain storage is attempted only after InfluxDB write succeeds.

InfluxDB measurement:

```text
medical_vitals
```

Stored fields match the Database tab:

```text
Time
Device
Temp
HR
Resp
SpO2
IF
XGB-N
XGB-C
XGB-D
LSTM-N
LSTM-C
LSTM-D
Alpha
Final-N
Final-C
Final-D
Label
Encrypted Vitals
```

Main file:

```text
backend/database/influx_client.py
```

### 6. Encryption Layer

Only the input vitals are encrypted:

- `heart_rate`
- `temperature`
- `respiratory_rate`
- `spo2`

Encryption algorithm:

```text
RSA-OAEP-256 + AES-256-GCM
```

Hash function:

```text
SHA-256
```

Hash formula:

```text
dataHash = SHA256(encrypted_vitals)
```

Key files:

```text
keys/public.pem
keys/private.pem
```

The public key encrypts vitals. The private key decrypts vitals.

Main file:

```text
backend/blockchain/crypto.py
```

Detailed guide:

```text
docs/ENCRYPTION_DECRYPTION_GUIDE.md
```

### 7. Blockchain Layer

Ethereum stores metadata only:

```text
ethereumRecordId
patientId
dataHash
storageId
doctorWalletAddress
timestamp
finalLabel
trustScore
```

The encrypted vitals stay in InfluxDB. Ethereum stores a tamper-evident hash and reference.

Smart contract:

```text
contracts/MedicalRecordStore.sol
```

Blockchain service:

```text
backend/blockchain/service.py
```

### 8. Server Layer

Flask provides the backend API.

Main file:

```text
backend/app.py
```

Main endpoints:

```text
GET  /health
POST /api/vitals
POST /api/predict
GET  /api/latest
GET  /api/db-records
GET  /api/blockchain-records
POST /api/blockchain/store
```

### 9. Dashboard Layer

The React dashboard contains:

- Live Monitoring
- Test Prediction
- Database Records
- Blockchain Records

Dashboard features:

- Live vitals graph with hover tooltip
- Final label and trust score
- InfluxDB records table
- Blockchain metadata cards
- Decrypt button for encrypted vitals

Frontend folder:

```text
frontend/
```

## Project Structure

```text
iot-health-monitor/
  backend/
    app.py
    blockchain/
    database/
    ml/
    services/
  contracts/
    MedicalRecordStore.sol
  data/
    raw/
    cleaned/
    processed/
  docs/
  esp32/
  frontend/
  keys/
  scripts/
  simulator/
  artifacts/
  hardhat.config.js
  package.json
  requirements.txt
  .env.example
```

## Environment Configuration

Create `.env` from `.env.example`:

```powershell
copy .env.example .env
```

Important values:

```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=1
DEFAULT_DEVICE_ID=ESP32-SIM-01
DEFAULT_PATIENT_ID=patient-demo

INFLUXDB_URL=http://localhost:8086
INFLUXDB_ORG=iot-health-monitor
INFLUXDB_BUCKET=vitals
INFLUXDB_TOKEN=your_influx_token

RPC_URL=http://127.0.0.1:8545
WEB3_PROVIDER_URL=http://127.0.0.1:8545
ETH_PRIVATE_KEY=hardhat_private_key
PRIVATE_KEY=same_hardhat_private_key
CONTRACT_ADDRESS=deployed_contract_address
DOCTOR_WALLET=doctor_or_hospital_wallet_address
PUBLIC_KEY_PATH=keys/public.pem
PRIVATE_KEY_PATH=keys/private.pem
VITE_API_URL=http://127.0.0.1:5000/api
```

## Installation

From project root:

```powershell
cd "D:\IOT_Project\iot-health-monitor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
cd frontend
npm install
cd ..
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Dataset Preprocessing

Place Challenge 2019 `.psv` files in:

```text
data/raw/
```

Run:

```powershell
python -m backend.ml.preprocess --raw-dir data/raw
```

## Model Training

Train Isolation Forest, XGBoost, and LSTM:

```powershell
python -m backend.ml.train_models --input data/cleaned/cleaned_vitals.csv
```

Train Stage 3 adaptive fusion gate:

```powershell
python -m backend.ml.train_fusion_gate --input data/cleaned/cleaned_vitals.csv
```

Models are saved in:

```text
artifacts/
```

## Running the Project

Use separate terminals.

### 1. Start InfluxDB

```powershell
docker start iot-influxdb
```

InfluxDB console:

```text
http://localhost:8086
```

### 2. Start Free Local Ethereum Blockchain

```powershell
npx hardhat node
```

Keep this terminal open.

### 3. Deploy Smart Contract

```powershell
npm run contract:deploy
```

Copy the deployed address into `.env`:

```env
CONTRACT_ADDRESS=0x...
```

For local Hardhat, use Account #0 private key and wallet address:

```env
ETH_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
DOCTOR_WALLET=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
```

### 4. Start Flask Backend

Restart Flask after editing `.env`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.app
```

Health check:

```text
http://127.0.0.1:5000/health
```

### 5. Start Simulator

```powershell
.\.venv\Scripts\Activate.ps1
python simulator/device_simulator.py --mode mixed
```

Other modes:

```powershell
python simulator/device_simulator.py --mode normal_vitals
python simulator/device_simulator.py --mode critical_vitals
python simulator/device_simulator.py --mode device_error
```

### 6. Start Dashboard

```powershell
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

## Viewing Data

### InfluxDB

Open:

```text
http://localhost:8086
```

Go to:

```text
Data Explorer -> bucket vitals -> measurement medical_vitals
```

Flux query:

```flux
from(bucket: "vitals")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "medical_vitals")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"], desc: true)
```

### Blockchain

Read local chain records:

```powershell
npm run contract:read
```

Dashboard:

```text
Blockchain Records tab
```

Decrypt vitals:

```text
Blockchain Records tab -> Decrypt
```

## Public Testnet Option

For a public free Ethereum testnet, use Sepolia with faucet test ETH.

Set `.env`:

```env
RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
WEB3_PROVIDER_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
ETH_PRIVATE_KEY=0xYOUR_SEPOLIA_PRIVATE_KEY
PRIVATE_KEY=0xYOUR_SEPOLIA_PRIVATE_KEY
DOCTOR_WALLET=0xDOCTOR_OR_HOSPITAL_WALLET
```

Deploy:

```powershell
npm run contract:deploy:sepolia
```

Read:

```powershell
npm run contract:read:sepolia
```

## Important Runtime Rules

- InfluxDB is written first.
- Blockchain metadata is written only after InfluxDB write succeeds.
- Runtime prediction uses trained model artifacts.
- Stage 3 alpha is predicted by the trained adaptive fusion gate.
- Final trust score is the maximum Stage 3 fused probability.
- Only input vitals are encrypted.
- Full medical/model records are not stored on Ethereum.

## Additional Documentation

```text
docs/SIMPLIFIED_RUN_AND_STAGE_IO.md
docs/ARCHITECTURE.md
docs/BLOCKCHAIN_LAYER.md
docs/ENCRYPTION_DECRYPTION_GUIDE.md
docs/IOT_SENSOR_SETUP.md
docs/JOURNAL_PAPER_DRAFT.md
```
