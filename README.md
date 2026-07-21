# IoT Health Monitor

Secure IoT health monitoring system for classifying patient vital signs and storing tamper-evident medical records. The system accepts temperature, heart rate, SpO2, and respiratory-rate readings from a simulator or ESP32 device, predicts `normal_vitals`, `critical_vitals`, or `device_error`, stores records in InfluxDB, encrypts input vitals, and writes verification metadata to an Ethereum smart contract.

The ML pipeline uses Isolation Forest, XGBoost, LSTM, and an Adaptive Fusion Gate based on a feed-forward neural network/MLP. The dashboard shows live monitoring, prediction testing, database records, and blockchain records.

## Project Flow

```text
IoT sensor / simulator
-> Flask API
-> Feature engineering
-> Isolation Forest + XGBoost + LSTM
-> Adaptive Fusion Gate
-> Final label and trust score
-> Encrypt input vitals
-> Store full record in InfluxDB
-> Store hash/reference metadata on Ethereum
-> React dashboard
```

## Requirements

- Python 3.11
- Node.js and npm
- Docker, for InfluxDB
- Hardhat local Ethereum node, installed through npm dependencies

## 1. Open Project

```powershell
cd "D:\IOT_Project\iot-health-monitor"
```

## 2. Create And Activate Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## 3. Install Node Dependencies

From the project root:

```powershell
npm install
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## 4. Configure Environment

Create `.env`:

```powershell
copy .env.example .env
```

Update the important values in `.env`:

```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=1
DEFAULT_DEVICE_ID=ESP32-SIM-01
DEFAULT_PATIENT_ID=patient-demo

INFLUXDB_URL=http://localhost:8086
INFLUXDB_ORG=iot-health-monitor
INFLUXDB_BUCKET=vitals
INFLUXDB_TOKEN=your_influxdb_token

RPC_URL=http://127.0.0.1:8545
WEB3_PROVIDER_URL=http://127.0.0.1:8545
ETH_PRIVATE_KEY=your_hardhat_private_key
PRIVATE_KEY=your_hardhat_private_key
CONTRACT_ADDRESS=
DOCTOR_WALLET=your_wallet_address

PUBLIC_KEY_PATH=keys/public.pem
PRIVATE_KEY_PATH=keys/private.pem
VITE_API_URL=http://127.0.0.1:5000/api
```

## 5. Start InfluxDB

If the Docker container already exists:

```powershell
docker start iot-influxdb
```

Open InfluxDB:

```text
http://localhost:8086
```

Make sure the bucket name matches `.env`:

```text
vitals
```

## 6. Preprocess Dataset

Skip this step if `data/cleaned/cleaned_vitals.csv` already exists and you do not want to regenerate it.

Place PhysioNet Challenge 2019 `.psv` files in:

```text
data/raw/
```

Run:

```powershell
python -m backend.ml.preprocess --raw-dir data/raw
```

Output:

```text
data/cleaned/cleaned_vitals.csv
```

## 7. Train Models

Train Isolation Forest, XGBoost, LSTM, scalers, and metadata:

```powershell
python -m backend.ml.train_models --input data/cleaned/cleaned_vitals.csv
```

Train Stage 3 Adaptive Fusion Gate:

```powershell
python -m backend.ml.train_fusion_gate --input data/cleaned/cleaned_vitals.csv
```

Model artifacts are saved in:

```text
artifacts/
```

## 8. Evaluate Results

Generate accuracy, precision, recall, F1, ROC-AUC, runtime, encryption, database, and blockchain metrics:

```powershell
python scripts/evaluate_experimental_results.py
```

Output:

```text
experimental results.md
```

## 9. Start Local Ethereum Node

Open a new terminal:

```powershell
cd "D:\IOT_Project\iot-health-monitor"
npx hardhat node
```

Keep this terminal running.

## 10. Deploy Smart Contract

Open another terminal:

```powershell
cd "D:\IOT_Project\iot-health-monitor"
npm run contract:deploy
```

Copy the deployed contract address from the terminal or `deployment.json` into `.env`:

```env
CONTRACT_ADDRESS=0x...
```

For local Hardhat, Account #0 can be used:

```env
ETH_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
DOCTOR_WALLET=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
```

Restart Flask after editing `.env`.

## 11. Start Flask Backend

Open a new terminal:

```powershell
cd "D:\IOT_Project\iot-health-monitor"
.\.venv\Scripts\Activate.ps1
python -m backend.app
```

Health check:

```text
http://127.0.0.1:5000/health
```

## 12. Start Simulator

Open another terminal:

```powershell
cd "D:\IOT_Project\iot-health-monitor"
.\.venv\Scripts\Activate.ps1
python simulator/device_simulator.py --mode mixed
```

Other modes:

```powershell
python simulator/device_simulator.py --mode normal_vitals
python simulator/device_simulator.py --mode critical_vitals
python simulator/device_simulator.py --mode device_error
```

Send a fixed number of readings:

```powershell
python simulator/device_simulator.py --mode mixed --count 20
python simulator/device_simulator.py --mode mixed --device-id ESP32-SIM-01 --count 20
```

## 13. Start React Dashboard

Open another terminal:

```powershell
cd "D:\IOT_Project\iot-health-monitor\frontend"
npm run dev
```

Open:

```text
http://localhost:5173
```

Dashboard tabs:

- `Live Monitoring`
- `Test Prediction`
- `Database Records`
- `Blockchain Records`

## 14. Test Prediction JSON

Use this single-reading sample in the dashboard `Test Prediction` tab:

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

Use this sequence sample when you want LSTM to run with real 12-reading history instead of fallback mode:

```json
{
  "device_id": "DASHBOARD-SEQUENCE-TEST",
  "patient_id": "patient-demo",
  "history": [
    {
      "temperature": 36.7,
      "heart_rate": 76,
      "spo2": 98,
      "respiratory_rate": 16
    },
    {
      "temperature": 36.8,
      "heart_rate": 78,
      "spo2": 98,
      "respiratory_rate": 17
    },
    {
      "temperature": 36.9,
      "heart_rate": 80,
      "spo2": 97,
      "respiratory_rate": 17
    },
    {
      "temperature": 37.1,
      "heart_rate": 84,
      "spo2": 97,
      "respiratory_rate": 18
    },
    {
      "temperature": 37.5,
      "heart_rate": 92,
      "spo2": 95,
      "respiratory_rate": 20
    },
    {
      "temperature": 38.0,
      "heart_rate": 104,
      "spo2": 94,
      "respiratory_rate": 23
    },
    {
      "temperature": 38.4,
      "heart_rate": 116,
      "spo2": 92,
      "respiratory_rate": 25
    },
    {
      "temperature": 38.8,
      "heart_rate": 124,
      "spo2": 90,
      "respiratory_rate": 28
    },
    {
      "temperature": 39.1,
      "heart_rate": 131,
      "spo2": 89,
      "respiratory_rate": 30
    },
    {
      "temperature": 39.3,
      "heart_rate": 136,
      "spo2": 88,
      "respiratory_rate": 31
    },
    {
      "temperature": 39.5,
      "heart_rate": 140,
      "spo2": 87,
      "respiratory_rate": 33
    },
    {
      "temperature": 39.6,
      "heart_rate": 142,
      "spo2": 86,
      "respiratory_rate": 34
    }
  ],
  "temperature": 39.7,
  "heart_rate": 145,
  "spo2": 85,
  "respiratory_rate": 35
}
```

Click `Run Prediction`, then `Encrypt and Store Blockchain`.

## Common Run Order

Use separate terminals:

```text
1. docker start iot-influxdb
2. npx hardhat node
3. npm run contract:deploy
4. python -m backend.app
5. python simulator/device_simulator.py --mode mixed
6. cd frontend && npm run dev
```

## Useful Commands

Read blockchain records:

```powershell
npm run contract:read
```

Run backend health check:

```text
http://127.0.0.1:5000/health
```

View InfluxDB:

```text
http://localhost:8086
```

If models are retrained, restart Flask so the backend loads the new artifacts.

## Sepolia Testnet Option

Use Sepolia when you want blockchain records on a public Ethereum testnet instead of the local Hardhat chain.

1. Create or use an Ethereum wallet and get Sepolia test ETH from a faucet.

2. Create a Sepolia RPC URL using Alchemy, Infura, QuickNode, or another provider.

3. Update `.env`:

```env
RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
WEB3_PROVIDER_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
ETH_PRIVATE_KEY=0xYOUR_SEPOLIA_PRIVATE_KEY
PRIVATE_KEY=0xYOUR_SEPOLIA_PRIVATE_KEY
DOCTOR_WALLET=0xDOCTOR_OR_HOSPITAL_WALLET
CONTRACT_ADDRESS=
```

4. Deploy the contract to Sepolia:

```powershell
npm run contract:deploy:sepolia
```

5. Copy the deployed address from `deployment.json` into `.env`:

```env
CONTRACT_ADDRESS=0xDEPLOYED_SEPOLIA_CONTRACT_ADDRESS
```

6. Restart Flask:

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.app
```

7. Send data through the simulator or dashboard:

```powershell
python simulator/device_simulator.py --mode mixed --count 20
```

8. Read Sepolia records:

```powershell
npm run contract:read:sepolia
```

9. View the contract or transactions on Etherscan:

```text
https://sepolia.etherscan.io/address/YOUR_CONTRACT_ADDRESS
https://sepolia.etherscan.io/tx/YOUR_TRANSACTION_HASH
```

Only metadata is stored on Sepolia: patient id, encrypted-data hash, InfluxDB storage id, doctor wallet, timestamp, final label, and trust score. The encrypted vitals remain off-chain in InfluxDB.
