# Encryption and Decryption Guide

This project encrypts only the input vitals before blockchain metadata is stored.

Encrypted vitals:

- `heart_rate`
- `temperature`
- `respiratory_rate`
- `spo2`

The final label, trust score, patient id, hash, storage id, doctor wallet, and timestamp are stored as blockchain metadata.

## Keys Used

The asymmetric key pair is stored locally in:

```text
keys/public.pem
keys/private.pem
```

Meaning:

| File | Used for | Who should have it |
| --- | --- | --- |
| `keys/public.pem` | Encrypts vitals | Shared with the system |
| `keys/private.pem` | Decrypts vitals | Kept secret by doctor/hospital |

The backend automatically creates this RSA key pair if the files do not exist.

Key generation happens in:

```text
backend/blockchain/crypto.py
```

The key pair is generated using:

```text
RSA 2048-bit
public exponent 65537
PEM format
```

## Encryption Algorithm

The project uses hybrid asymmetric encryption:

```text
RSA-OAEP-256 + AES-256-GCM
```

Why hybrid encryption is used:

- RSA is used for asymmetric encryption.
- AES-GCM is used to encrypt the vitals JSON.
- RSA-OAEP encrypts the AES key using the doctor public key.
- The doctor private key decrypts the AES key.
- The AES key then decrypts the vitals.

This is the standard practical way to encrypt JSON data with asymmetric keys because RSA alone is not suitable for larger payloads.

Encrypted output format:

```json
{
  "alg": "RSA-OAEP-256+A256GCM",
  "encrypted_key": "...",
  "nonce": "...",
  "ciphertext": "..."
}
```

This envelope is base64 encoded and stored in InfluxDB as:

```text
encrypted_vitals
```

## Hash Function

The project uses:

```text
SHA-256
```

Hash calculation:

```text
dataHash = SHA256(encrypted_vitals)
```

Important:

- The hash is calculated from encrypted vitals.
- The hash is not calculated from plaintext vitals.
- Ethereum stores this hash to verify that the encrypted InfluxDB data has not changed.

## Storage Flow

1. IoT/simulator sends vitals to Flask.
2. ML models generate final label and trust score.
3. Backend encrypts only the four input vitals with `keys/public.pem`.
4. Backend calculates `dataHash = SHA256(encrypted_vitals)`.
5. Backend stores the dashboard database fields in InfluxDB.
6. Backend stores `encrypted_vitals` in InfluxDB.
7. Backend stores metadata on Ethereum:

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

## Decryption Flow

The doctor/hospital decrypts using the private key:

```text
keys/private.pem
```

Backend API:

```powershell
Invoke-RestMethod "http://127.0.0.1:5000/api/blockchain-records?decrypt=1"
```

Dashboard:

```text
Blockchain Records tab -> Decrypt
```

The decrypted output contains only:

```json
{
  "heart_rate": 85,
  "temperature": 37.1,
  "respiratory_rate": 18,
  "spo2": 98
}
```

## Blockchain Environment Variables

These values are configured in `.env`.

| Variable | Example | Function |
| --- | --- | --- |
| `RPC_URL` | `http://127.0.0.1:8545` | Ethereum RPC endpoint used by Hardhat/scripts/backend |
| `WEB3_PROVIDER_URL` | `http://127.0.0.1:8545` | Web3 provider URL used by Flask backend |
| `ETH_PRIVATE_KEY` | `0xac0974...ff80` | Ethereum wallet private key used to send transactions |
| `PRIVATE_KEY` | `0xac0974...ff80` | Same as `ETH_PRIVATE_KEY`, kept for script compatibility |
| `CONTRACT_ADDRESS` | `0x5FbDB...80aa3` | Deployed `MedicalRecordStore` smart contract address |
| `DOCTOR_WALLET` | `0xf39F...2266` | Doctor/hospital Ethereum wallet address stored on-chain |
| `PUBLIC_KEY_PATH` | `keys/public.pem` | RSA public key used to encrypt vitals |
| `PRIVATE_KEY_PATH` | `keys/private.pem` | RSA private key used to decrypt vitals |

For local Hardhat demo, `ETH_PRIVATE_KEY` and `PRIVATE_KEY` can be the same private key printed by:

```powershell
npx hardhat node
```

Example:

```text
Account #0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
Private Key: 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

Then:

```env
ETH_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
DOCTOR_WALLET=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
```

## How to Encrypt and Store

Start InfluxDB, Hardhat, backend, and simulator:

```powershell
docker start iot-influxdb
npx hardhat node
npm run contract:deploy
python -m backend.app
python simulator/device_simulator.py --mode mixed
```

For dashboard testing:

```text
Test Prediction tab -> Run Prediction -> Encrypt and Store Blockchain
```

The system stores data in this order:

```text
InfluxDB first -> Ethereum metadata second
```

If InfluxDB write fails, blockchain storage is stopped.

## How to Decrypt from Command Line

Copy an encrypted vitals value from InfluxDB or API, then run:

```powershell
python scripts/decrypt_record.py --ciphertext "PASTE_ENCRYPTED_VITALS_HERE" --private-key keys/private.pem
```

The output will be the original vitals JSON.

## Integrity Verification

To verify that InfluxDB encrypted vitals were not changed:

1. Copy `encrypted_vitals` from InfluxDB.
2. Calculate SHA-256 of that encrypted string.
3. Compare it with `dataHash` in the Blockchain tab or smart contract.
4. If they match, decrypt with `keys/private.pem`.

This proves the encrypted vitals stored in InfluxDB match the hash stored on Ethereum.
