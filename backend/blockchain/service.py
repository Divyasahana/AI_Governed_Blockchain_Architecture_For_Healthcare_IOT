from __future__ import annotations

import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, List

from .crypto import encrypt_record, encrypted_data_hash, generate_keypair

try:
    from web3 import Web3
except Exception:
    Web3 = None

CONTRACT_ABI = [{
    "inputs": [
        {"internalType": "string", "name": "patientId", "type": "string"},
        {"internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
        {"internalType": "string", "name": "storageId", "type": "string"},
        {"internalType": "address", "name": "doctorWallet", "type": "address"},
        {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
        {"internalType": "string", "name": "finalLabel", "type": "string"},
        {"internalType": "uint256", "name": "trustScoreBps", "type": "uint256"},
    ],
    "name": "storeRecord",
    "outputs": [{"internalType": "uint256", "name": "id", "type": "uint256"}],
    "stateMutability": "nonpayable",
    "type": "function",
}, {
    "inputs": [],
    "name": "recordCount",
    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function",
}]


def timestamp_to_unix(value: str) -> int:
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except Exception:
        return int(datetime.now(timezone.utc).timestamp())


class BlockchainService:
    def __init__(self):
        self.public_key_path = os.getenv("PUBLIC_KEY_PATH", "keys/public.pem")
        self.private_key_path = os.getenv("PRIVATE_KEY_PATH", "keys/private.pem")
        generate_keypair(self.private_key_path, self.public_key_path)
        self.memory: Deque[dict] = deque(maxlen=500)

    def _vitals_payload(self, record: dict) -> dict:
        input_data = record.get("input", {})
        return {
            "heart_rate": input_data.get("heart_rate"),
            "temperature": input_data.get("temperature"),
            "respiratory_rate": input_data.get("respiratory_rate"),
            "spo2": input_data.get("spo2"),
        }

    def prepare_record(self, record: dict) -> dict:
        encrypted_vitals = encrypt_record(self._vitals_payload(record), self.public_key_path)
        digest = encrypted_data_hash(encrypted_vitals)
        storage_id = f"influxdb://{os.getenv('INFLUXDB_BUCKET', 'vitals')}/medical_vitals/{digest}"
        input_data = record.get("input", {})
        timestamp = input_data.get("timestamp") or record.get("timestamp") or datetime.now(timezone.utc).isoformat()
        patient_id = input_data.get("patient_id") or input_data.get("device_id") or record.get("patient_id") or "unknown-patient"
        device_id = input_data.get("device_id") or record.get("device_id") or patient_id
        doctor_wallet = self._doctor_wallet()
        return {
            "patient_id": patient_id,
            "id": len(self.memory),
            "device_id": device_id,
            "data_hash": digest,
            "storage_id": storage_id,
            "doctor_wallet": doctor_wallet,
            "doctor_public_key_path": self.public_key_path,
            "record_hash": digest,
            "encrypted_data_reference": storage_id,
            "encrypted_vitals": encrypted_vitals,
            "encryption_algorithm": "RSA-OAEP-256+A256GCM",
            "encrypted_vitals_bytes": len(encrypted_vitals.encode("utf-8")),
            "timestamp": timestamp,
            "patient_device_id": device_id,
            "final_label": record.get("final_label", "unknown"),
            "trust_score": record.get("trust_score", 0),
            "transaction_hash": None,
            "stored_on_chain": False,
        }

    def commit_prepared(self, prepared: dict) -> dict:
        chain_result = self._store_on_chain(
            prepared["patient_id"],
            prepared["data_hash"],
            prepared["storage_id"],
            prepared["doctor_wallet"],
            prepared["timestamp"],
            prepared["final_label"],
            int(float(prepared.get("trust_score", 0)) * 10000),
        )
        stored = {
            **prepared,
            "id": chain_result.get("record_id", prepared.get("id", len(self.memory))),
            "transaction_hash": chain_result.get("transaction_hash"),
            "stored_on_chain": chain_result.get("stored_on_chain", False),
        }
        self.memory.appendleft(stored)
        return stored

    def store(self, record: dict) -> dict:
        return self.commit_prepared(self.prepare_record(record))

    def _doctor_wallet(self) -> str:
        configured = os.getenv("DOCTOR_WALLET")
        if configured:
            return configured
        private_key = os.getenv("ETH_PRIVATE_KEY") or os.getenv("PRIVATE_KEY")
        if Web3 is not None and private_key:
            try:
                return Web3().eth.account.from_key(private_key).address
            except Exception:
                pass
        return "0x0000000000000000000000000000000000000000"

    def _store_on_chain(self, patient_id: str, data_hash: str, storage_id: str, doctor_wallet: str, timestamp: str, final_label: str, trust_score: int) -> dict:
        if Web3 is None:
            return {"stored_on_chain": False, "transaction_hash": f"local-{int(time.time())}", "record_id": len(self.memory)}
        rpc_url = os.getenv("WEB3_PROVIDER_URL") or os.getenv("RPC_URL")
        contract_address = os.getenv("CONTRACT_ADDRESS")
        private_key = os.getenv("ETH_PRIVATE_KEY") or os.getenv("PRIVATE_KEY")
        if not (rpc_url and contract_address and private_key):
            return {"stored_on_chain": False, "transaction_hash": f"local-{int(time.time())}", "record_id": len(self.memory)}
        try:
            web3 = Web3(Web3.HTTPProvider(rpc_url))
            if not web3.is_connected():
                return {"stored_on_chain": False, "transaction_hash": f"pending-chain-{int(time.time())}", "record_id": len(self.memory)}
            account = web3.eth.account.from_key(private_key)
            contract = web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=CONTRACT_ABI)
            record_id = int(contract.functions.recordCount().call())
            transaction = contract.functions.storeRecord(
                patient_id,
                bytes.fromhex(data_hash.removeprefix("0x")),
                storage_id,
                Web3.to_checksum_address(doctor_wallet),
                timestamp_to_unix(timestamp),
                final_label,
                trust_score,
            ).build_transaction({
                "from": account.address,
                "nonce": web3.eth.get_transaction_count(account.address),
                "chainId": web3.eth.chain_id,
                "gas": 500000,
                "gasPrice": web3.eth.gas_price,
            })
            signed = web3.eth.account.sign_transaction(transaction, private_key)
            raw_transaction = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction")
            tx_hash = web3.eth.send_raw_transaction(raw_transaction)
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            return {"stored_on_chain": receipt.status == 1, "transaction_hash": receipt.transactionHash.hex(), "record_id": record_id}
        except Exception:
            return {"stored_on_chain": False, "transaction_hash": f"pending-chain-{int(time.time())}", "record_id": len(self.memory)}

    def list_records(self) -> List[dict]:
        return list(self.memory)
