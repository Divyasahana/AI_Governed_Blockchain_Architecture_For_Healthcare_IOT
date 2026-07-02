# Blockchain Layer

This layer protects patient vitals before blockchain storage. Only the input vitals are encrypted and stored off-chain in InfluxDB. Ethereum stores only verification and lookup metadata.

## Security Workflow

1. Doctor or hospital creates a key pair.
   - Public key: shared with this system and saved as `keys/public.pem`.
   - Private key: kept secret by the doctor or hospital and saved as `keys/private.pem` only for local demo decryption.

2. Patient data is collected by the simulator or IoT device.

   Example:

   ```json
   {
     "patient_id": "P101",
     "heart_rate": 85,
     "temperature": 37.1,
     "prediction": "normal_vitals"
   }
   ```

3. The backend encrypts only the input vitals with the doctor public key.

   The implementation uses hybrid public-key encryption:
   - The vitals object is encrypted with AES-GCM.
   - The AES key is encrypted with RSA-OAEP using the doctor public key.
   - Only the matching doctor private key can decrypt the AES key and recover the original vitals.

   The encrypted JSON contains only:

   ```json
   {
     "heart_rate": 85,
     "temperature": 37.1,
     "respiratory_rate": 18,
     "spo2": 98
   }
   ```

   This is still asymmetric-key encryption because the AES key is protected by RSA-OAEP with `keys/public.pem`, and only `keys/private.pem` can decrypt it.

4. The backend generates the blockchain verification hash.

   ```text
   Data_Hash = SHA256(Encrypted_Data)
   ```

   The hash is calculated from the encrypted vitals value, not from plaintext patient data.

5. The encrypted data is stored off-chain in InfluxDB.

   The encrypted vitals value is written into the `medical_vitals` measurement as the `encrypted_vitals` field. The blockchain reference is:

   ```text
   influxdb://<bucket>/medical_vitals/<data_hash>
   ```

   For this project, `storage_id` points to the InfluxDB bucket and measurement where the encrypted vitals are stored.

6. Ethereum stores only metadata.

   The smart contract stores:

   | Field | Meaning |
   | --- | --- |
   | `patientId` | Patient id such as `P101` |
   | `dataHash` | SHA-256 hash of the encrypted vitals |
   | `storageId` | InfluxDB encrypted vitals reference |
   | `doctorWallet` | Doctor or hospital Ethereum wallet address |
   | `timestamp` | Unix timestamp of the reading |
   | `finalLabel` | Final AI label |
   | `trustScoreBps` | Final confidence/trust score in basis points |

7. Doctor decrypts the record.

   The doctor retrieves the encrypted vitals from the storage id and decrypts it with the private key:

   ```text
   Patient_Data = RSA_decrypt(Encrypted_AES_Key, Doctor_Private_Key)
   Patient_Vitals = AES_GCM_decrypt(Encrypted_Data, AES_Key)
   ```

## Files

| File | Purpose |
| --- | --- |
| `backend/blockchain/crypto.py` | Generates RSA key pair, encrypts/decrypts records, computes SHA-256 hash of encrypted data |
| `backend/blockchain/service.py` | Encrypts prediction records, creates the InfluxDB storage id, and stores Ethereum metadata |
| `backend/database/influx_client.py` | Stores vitals, model outputs, blockchain metadata, and encrypted vitals in InfluxDB |
| `contracts/MedicalRecordStore.sol` | Ethereum smart contract for patient id, data hash, storage id, doctor wallet, timestamp, label, and score |
| `scripts/deploy.js` | Deploys the smart contract and writes `deployment.json` |
| `scripts/read_records.js` | Reads stored records directly from Ethereum |
| `scripts/store_record.js` | Stores a prepared encrypted record into Ethereum from a JSON file |
| `scripts/decrypt_record.py` | Decrypts encrypted vitals using the doctor private key |
| `frontend/src/App.jsx` | Shows blockchain records in the dashboard |

## Environment Variables

Set these values in `.env`:

```text
RPC_URL=http://127.0.0.1:8545
WEB3_PROVIDER_URL=http://127.0.0.1:8545
ETH_PRIVATE_KEY=<hardhat_or_doctor_sender_private_key>
PRIVATE_KEY=<same_private_key_for_compatibility>
CONTRACT_ADDRESS=<deployed_contract_address>
DOCTOR_WALLET=<doctor_or_hospital_wallet_address>
PUBLIC_KEY_PATH=keys/public.pem
PRIVATE_KEY_PATH=keys/private.pem
```

If `DOCTOR_WALLET` is empty, the backend uses the wallet derived from `ETH_PRIVATE_KEY`. If no Ethereum private key is configured, records are still encrypted and stored in InfluxDB, but `stored_on_chain` will be `false`.

## How to Store Data

Normal simulator/live data is stored automatically by the backend:

```powershell
python -m backend.app
python simulator/device_simulator.py --mode mixed
```

When `/api/vitals` receives a reading, the backend:

1. Runs Isolation Forest, XGBoost, LSTM, and adaptive fusion.
2. Encrypts only heart rate, temperature, respiratory rate, and SpO2 using the doctor public key.
3. Generates `data_hash = SHA256(encrypted_vitals)`.
4. Stores the encrypted vitals, encrypted data reference, and metadata in InfluxDB.
5. Sends the metadata to Ethereum if the local Ethereum node and contract are configured.

For dashboard test inputs, open `Test Prediction`, run the JSON prediction, then click `Encrypt and Store Blockchain`.

## How to View Blockchain Data

### Dashboard

Open the dashboard and go to `Blockchain Records`.

The tab shows only:

- `ethereumRecordId`
- `patientId`
- `dataHash`
- `storageId`
- `doctorWalletAddress`
- `timestamp`
- `finalLabel`
- `trustScore`

The `Database Records` tab also includes an `Encrypted Vitals` column. The table displays a shortened value, and the full encrypted value is available in the cell tooltip/API response.

### API

View stored encrypted blockchain records:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/blockchain-records
```

View with local demo decryption:

```powershell
Invoke-RestMethod "http://127.0.0.1:5000/api/blockchain-records?decrypt=1"
```

### Local Ethereum Contract

Start a local Ethereum node:

```powershell
npx hardhat node
```

Deploy the contract:

```powershell
npm run contract:deploy
```

Copy the deployed address from `deployment.json` into `.env` as `CONTRACT_ADDRESS`, then restart Flask.

Read records directly from Ethereum:

```powershell
npm run contract:read
```

The contract output contains the metadata stored on-chain. The encrypted medical payload itself is not stored directly on Ethereum; Ethereum stores the hash and storage reference needed to verify and retrieve it.

## Real Ethereum Blockchain: Sepolia Testnet

Use Sepolia for journal/demo testing because it is a public Ethereum testnet and does not require real ETH. The same contract and backend flow also work on mainnet if you configure a mainnet RPC URL and fund the sender wallet with real ETH.

### 1. Create a Wallet

Create a wallet in MetaMask or another Ethereum wallet.

Keep the private key secret. For demo deployment, copy the private key into `.env`:

```text
ETH_PRIVATE_KEY=0xYOUR_WALLET_PRIVATE_KEY
PRIVATE_KEY=0xYOUR_WALLET_PRIVATE_KEY
DOCTOR_WALLET=0xDOCTOR_OR_HOSPITAL_WALLET
```

`ETH_PRIVATE_KEY` is the wallet that pays gas and sends contract transactions. `DOCTOR_WALLET` is the doctor/hospital address stored in the medical metadata.

### 2. Get a Sepolia RPC URL

Create a free endpoint from a provider such as Alchemy, Infura, QuickNode, or Chainstack.

Set `.env`:

```text
RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
WEB3_PROVIDER_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
```

### 3. Get Sepolia ETH

Use a Sepolia faucet to fund the sender wallet. You only need test ETH for gas.

### 4. Deploy the Contract to Sepolia

From the project root:

```powershell
npm run contract:deploy:sepolia
```

This writes `deployment.json`. Copy its `address` into `.env`:

```text
CONTRACT_ADDRESS=0xDEPLOYED_CONTRACT_ADDRESS
```

Restart Flask after editing `.env`.

### 5. Run the Backend and Store Records

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.app
```

Then send data from the simulator:

```powershell
.\.venv\Scripts\Activate.ps1
python simulator/device_simulator.py --mode mixed
```

Each `/api/vitals` request now:

1. Encrypts only heart rate, temperature, respiratory rate, and SpO2.
2. Stores the encrypted vitals in InfluxDB.
3. Stores `patientId`, `dataHash`, `storageId`, `doctorWallet`, timestamp, label, and score on Sepolia.
4. Saves the transaction hash back into InfluxDB.

### 6. View Sepolia Records

Read records through Hardhat:

```powershell
npm run contract:read:sepolia
```

View the contract and transactions in a browser:

```text
https://sepolia.etherscan.io/address/YOUR_CONTRACT_ADDRESS
https://sepolia.etherscan.io/tx/YOUR_TRANSACTION_HASH
```

The dashboard `Blockchain Records` tab shows only the Ethereum record id, patient id, data hash, storage id, doctor wallet address, timestamp, final label, and trust score.

## Integrity Check

To verify a record:

1. Fetch encrypted data from InfluxDB using `storage_id`.
2. Calculate SHA-256 of the encrypted vitals.
3. Compare it with `data_hash` from Ethereum.
4. If the values match, decrypt with the doctor private key.

This proves that the off-chain encrypted record has not been modified after the blockchain transaction was written.
