from __future__ import annotations

import os
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
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
}, {
    "inputs": [{"internalType": "uint256", "name": "id", "type": "uint256"}],
    "name": "getRecord",
    "outputs": [{
        "components": [
            {"internalType": "string", "name": "patientId", "type": "string"},
            {"internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"internalType": "string", "name": "storageId", "type": "string"},
            {"internalType": "address", "name": "doctorWallet", "type": "address"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "string", "name": "finalLabel", "type": "string"},
            {"internalType": "uint256", "name": "trustScoreBps", "type": "uint256"},
        ],
        "internalType": "struct MedicalRecordStore.MedicalRecord",
        "name": "",
        "type": "tuple",
    }],
    "stateMutability": "view",
    "type": "function",
}, {
    "anonymous": False,
    "inputs": [
        {"indexed": True, "internalType": "uint256", "name": "id", "type": "uint256"},
        {"indexed": False, "internalType": "string", "name": "patientId", "type": "string"},
        {"indexed": True, "internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
        {"indexed": False, "internalType": "string", "name": "storageId", "type": "string"},
        {"indexed": True, "internalType": "address", "name": "doctorWallet", "type": "address"},
        {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
        {"indexed": False, "internalType": "string", "name": "finalLabel", "type": "string"},
        {"indexed": False, "internalType": "uint256", "name": "trustScoreBps", "type": "uint256"},
    ],
    "name": "MedicalRecordStored",
    "type": "event",
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
        self.project_root = Path(__file__).resolve().parents[2]
        generate_keypair(self.private_key_path, self.public_key_path)
        self.memory: Deque[dict] = deque(maxlen=500)

    def _contract_address(self) -> str | None:
        configured = os.getenv("CONTRACT_ADDRESS")
        if configured:
            return configured
        deployment_file = self.project_root / "deployment.json"
        try:
            deployment = json.loads(deployment_file.read_text(encoding="utf-8"))
            return deployment.get("address")
        except Exception:
            return None

    def _vitals_payload(self, record: dict) -> dict:
        input_data = record.get("input", {})
        return {
            "heart_rate": input_data.get("heart_rate"),
            "temperature": input_data.get("temperature"),
            "respiratory_rate": input_data.get("respiratory_rate"),
            "spo2": input_data.get("spo2"),
        }

    @staticmethod
    def _trust_score(record: dict) -> float:
        if record.get("trust_score") is not None:
            return float(record.get("trust_score") or 0)
        final_probabilities = record.get("fusion", {}).get("final_probabilities", {})
        values = [
            float(value)
            for value in final_probabilities.values()
            if value is not None
        ]
        return max(values) if values else 0.0

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
            "trust_score": self._trust_score(record),
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
            int(round(float(prepared.get("trust_score", 0)) * 10000)),
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
        contract_address = self._contract_address()
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

    def _contract_connection(self):
        if Web3 is None:
            return None, None
        rpc_url = os.getenv("WEB3_PROVIDER_URL") or os.getenv("RPC_URL")
        contract_address = self._contract_address()
        if not (rpc_url and contract_address):
            return None, None
        web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not web3.is_connected():
            return None, None
        contract = web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=CONTRACT_ABI)
        return web3, contract

    def _event_transaction_hashes(self, contract) -> dict[int, str]:
        tx_hashes: dict[int, str] = {}
        try:
            from_block = int(os.getenv("CONTRACT_DEPLOY_BLOCK", "0"))
            logs = contract.events.MedicalRecordStored().get_logs(fromBlock=from_block, toBlock="latest")
            for log in logs:
                tx_hashes[int(log["args"]["id"])] = log["transactionHash"].hex()
        except Exception:
            pass
        return tx_hashes

    @staticmethod
    def _hex_data_hash(value) -> str:
        if isinstance(value, str):
            return value
        return "0x" + bytes(value).hex()

    def list_on_chain_records(self) -> List[dict]:
        web3, contract = self._contract_connection()
        if web3 is None or contract is None:
            return []
        count = int(contract.functions.recordCount().call())
        tx_hashes = self._event_transaction_hashes(contract)
        records = []
        for record_id in range(count):
            raw = contract.functions.getRecord(record_id).call()
            patient_id, data_hash, storage_id, doctor_wallet, timestamp, final_label, trust_score_bps = raw
            data_hash = self._hex_data_hash(data_hash)
            local_match = next((item for item in self.memory if item.get("data_hash") == data_hash), {})
            records.append({
                "id": record_id,
                "ethereumRecordId": record_id,
                "patient_id": patient_id,
                "patientId": patient_id,
                "data_hash": data_hash,
                "dataHash": data_hash,
                "storage_id": storage_id,
                "storageId": storage_id,
                "doctor_wallet": doctor_wallet,
                "doctorWallet": doctor_wallet,
                "doctorWalletAddress": doctor_wallet,
                "timestamp": int(timestamp),
                "final_label": final_label,
                "finalLabel": final_label,
                "trust_score": int(trust_score_bps) / 10000,
                "trustScore": int(trust_score_bps) / 10000,
                "transaction_hash": tx_hashes.get(record_id) or local_match.get("transaction_hash", ""),
                "transactionHash": tx_hashes.get(record_id) or local_match.get("transaction_hash", ""),
                "stored_on_chain": True,
                "encrypted_vitals": local_match.get("encrypted_vitals"),
            })
        return records

    def list_records(self) -> List[dict]:
        on_chain = self.list_on_chain_records()
        if on_chain:
            return sorted(on_chain, key=lambda item: int(item.get("timestamp", 0)), reverse=True)
        return list(self.memory)
