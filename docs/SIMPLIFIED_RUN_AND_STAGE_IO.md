# Simplified Project Run Guide and Stage Inputs/Outputs

This is the main guide for running the IoT Health Monitor project and explaining each architecture stage.

## Git-Ready Project Structure

Keep these files/folders in Git:

- `backend/`
- `frontend/`
- `contracts/`
- `scripts/`
- `simulator/`
- `esp32/`
- `docs/`
- `artifacts/` only if trained model artifacts are present and small enough for your Git host
- `data/cleaned/cleaned_vitals.csv` only if you want to commit the prepared training table
- `data/raw/.gitkeep`, not the raw PhysioNet `.psv` files
- `data/processed/.gitkeep`, not generated plots

Do not commit:

- `.venv/`
- `node_modules/`
- `.env`
- `keys/`
- `data/raw/*.psv`
- generated plots in `data/processed/`
- duplicate cleaned CSVs
- Hardhat cache/build output

## One-Time Setup

```powershell
cd "D:\IOT_Project\iot-health-monitor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
cd frontend
npm install
cd ..
copy .env.example .env
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Environment

Fill `.env` from `.env.example`.

For local demo:

```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=1
INFLUXDB_URL=http://localhost:8086
INFLUXDB_ORG=iot-health-monitor
INFLUXDB_BUCKET=vitals
RPC_URL=http://127.0.0.1:8545
WEB3_PROVIDER_URL=http://127.0.0.1:8545
VITE_API_URL=http://127.0.0.1:5000/api
```

## Dataset and Training

Put PhysioNet Challenge 2019 `.psv` files in:

```text
data/raw/
```

Then run:

```powershell
python -m backend.ml.preprocess --raw-dir data/raw
python -m backend.ml.train_models --input data/cleaned/cleaned_vitals.csv
python -m backend.ml.train_fusion_gate --input data/cleaned/cleaned_vitals.csv
```

The project only needs this cleaned training file:

```text
data/cleaned/cleaned_vitals.csv
```

The older files such as `cleaned_normal.csv`, `cleaned_critical.csv`, `normal_vitals.csv`, and similar split CSVs are not used by training or runtime.

## Daily Demo Run

Use separate terminals.

Backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.app
```

Simulator:

```powershell
.\.venv\Scripts\Activate.ps1
python simulator/device_simulator.py --mode mixed
```

If your terminal is already inside the `simulator` folder, run:

```powershell
cd "D:\IOT_Project\iot-health-monitor\simulator"
..\.venv\Scripts\Activate.ps1
python .\device_simulator.py --mode mixed
```

Dashboard:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

Optional local blockchain:

```powershell
npx hardhat node
npm run contract:deploy
```

Copy the deployed address into `.env` as `CONTRACT_ADDRESS`, then restart Flask.

For public Ethereum testnet storage, use Sepolia:

```powershell
npm run contract:deploy:sepolia
```

Before deploying to Sepolia, set these values in `.env`:

```env
RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
WEB3_PROVIDER_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
ETH_PRIVATE_KEY=0xYOUR_FUNDED_SEPOLIA_PRIVATE_KEY
PRIVATE_KEY=0xYOUR_FUNDED_SEPOLIA_PRIVATE_KEY
DOCTOR_WALLET=0xDOCTOR_OR_HOSPITAL_WALLET
```

Copy the deployed contract address from `deployment.json` into `.env` as `CONTRACT_ADDRESS`, then restart Flask.

## Dashboard JSON Testing

Open the `Test Prediction` tab and paste:

```json
{
  "temperature": 39.4,
  "heart_rate": 135,
  "spo2": 88,
  "respiratory_rate": 31,
  "device_id": "DASHBOARD-TEST",
  "patient_id": "patient-demo"
}
```

Click `Run Prediction`.

The dashboard displays:

- Input vitals
- XGBoost probabilities
- LSTM probabilities
- Final label

Click `Encrypt and Store Blockchain` to encrypt and store that exact prediction record.

## Stage Inputs and Outputs

### Stage 0: IoT Sensor or Simulator

Input:

- Temperature
- Heart rate
- SpO2
- Respiratory rate
- Device ID
- Patient ID
- Timestamp

Output:

- JSON payload sent to `POST /api/vitals`

### Stage 1: Preprocessing and Feature Engineering

Training input:

- Challenge 2019 `.psv` files from `data/raw/`
- Used columns: `Temp`, `HR`, `O2Sat`, `Resp`

Runtime input:

- Current JSON reading
- Previous readings for the same device

Output:

- Cleaned vitals
- Delta features
- 12-reading rolling mean/min/max features
- Balanced labels: `normal_vitals`, `critical_vitals`, `device_error`
- `data/cleaned/cleaned_vitals.csv`

### Stage 2A: Isolation Forest

Training input:

- Normal and critical feature rows only

Runtime input:

- Current feature vector

Output:

- `anomaly_score`
- Device-error flag when anomaly is high

### Stage 2B: XGBoost

Training input:

- Original vitals
- Delta features
- Rolling 12-reading features
- Three labels

Runtime output:

- Probabilities for `normal_vitals`, `critical_vitals`, and `device_error`

### Stage 2C: LSTM

Training input:

- 12-reading sequences
- Original vitals plus delta features
- Three labels

Runtime output:

- Three class probabilities when 12 readings are available
- XGBoost fallback probabilities when fewer than 12 readings exist

### Stage 3: Adaptive Fusion Gate

Actual Stage 3 training inputs used by this project:

- Original vitals:
  - `temperature`
  - `heart_rate`
  - `spo2`
  - `respiratory_rate`
- XGBoost probabilities:
  - `normal_vitals`
  - `critical_vitals`
  - `device_error`
- LSTM probabilities:
  - `normal_vitals`
  - `critical_vitals`
  - `device_error`
- Isolation Forest anomaly score

Additional reliability inputs:

- XGBoost confidence
- LSTM confidence
- XGBoost entropy
- LSTM entropy
- Sequence-available flag

Output:

- `alpha`
- Final fused probabilities
- Final label
- Final confidence

Formula:

```text
final_probability = alpha * xgboost_probability + (1 - alpha) * lstm_probability
```

### Stage 4: Database

Input:

- Original vitals
- Model outputs
- Fusion output
- Trust score
- Blockchain transaction hash when available
- Encrypted vitals
- Blockchain data hash
- InfluxDB storage id
- Doctor wallet

Output:

- InfluxDB record
- In-memory fallback record for demo

### Stage 5: Blockchain

Input:

- Final prediction record

Processing:

- AES-GCM encrypts the full record
- RSA-OAEP encrypts the AES key
- SHA-256 hashes the encrypted vitals
- InfluxDB stores the encrypted vitals
- Ethereum stores patient id, encrypted-data hash, InfluxDB storage id, doctor wallet, timestamp, label, and score

Output:

- Encrypted vitals
- Encrypted-data hash
- InfluxDB storage id
- Transaction hash or local marker
