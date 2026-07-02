from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, jsonify, request

try:
    from flask_cors import CORS
except Exception:
    def CORS(_app):
        return None

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from backend.blockchain.crypto import decrypt_record
from backend.blockchain.service import BlockchainService
from backend.database.influx_client import InfluxStore
from backend.ml.inference import ModelNotReadyError, ModelService
from backend.services.simulator_service import SimulatorService

app = Flask(__name__)
CORS(app)

models = ModelService(Path(__file__).resolve().parents[1] / "artifacts")
db = InfluxStore()
chain = BlockchainService()
simulator = SimulatorService()


def to_unix_timestamp(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except Exception:
        return int(datetime.now(timezone.utc).timestamp())


def normalize_blockchain_record(item: dict, fallback_id: int) -> dict:
    timestamp = item.get("timestamp", "")
    record_id = item.get("ethereumRecordId") if item.get("ethereumRecordId") is not None else item.get("id", fallback_id)
    patient_id = item.get("patientId") or item.get("patient_id") or item.get("patient_device_id") or ""
    data_hash = item.get("dataHash") or item.get("data_hash") or item.get("record_hash") or ""
    storage_id = item.get("storageId") or item.get("storage_id") or item.get("encrypted_data_reference") or ""
    doctor_wallet = item.get("doctorWalletAddress") or item.get("doctorWallet") or item.get("doctor_wallet") or ""
    final_label = item.get("finalLabel") or item.get("final_label") or ""
    trust_score = float(item.get("trustScore") if item.get("trustScore") is not None else item.get("trust_score", 0) or 0)
    return {
        **item,
        "ethereumRecordId": record_id,
        "id": record_id,
        "patientId": patient_id,
        "dataHash": data_hash,
        "storageId": storage_id,
        "doctorWalletAddress": doctor_wallet,
        "doctorWallet": doctor_wallet,
        "timestamp": to_unix_timestamp(timestamp),
        "finalLabel": final_label,
        "trustScore": trust_score,
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok", "influxdb_connected": db.connected, "influxdb": db.status()})


@app.errorhandler(ModelNotReadyError)
def model_not_ready(exc):
    return jsonify({"error": str(exc), "models": models.status}), 503


@app.post("/api/vitals")
def post_vitals():
    prediction = models.predict(request.get_json(force=True) or {})
    prepared = chain.prepare_record(prediction)
    prediction["encrypted_vitals"] = prepared["encrypted_vitals"]
    if not db.connected:
        return jsonify({"error": "InfluxDB is not connected. Record was not stored.", "influxdb": db.status()}), 503
    db.write_record(prediction)
    if db.error:
        return jsonify({"error": "InfluxDB write failed. Record was not stored on blockchain.", "influxdb": db.status()}), 503
    stored = chain.commit_prepared(prepared)
    prediction["blockchain_tx_hash"] = stored["transaction_hash"]
    prediction["blockchain_record_id"] = stored["id"]
    prediction["blockchain_record_hash"] = stored["record_hash"]
    prediction["blockchain_data_hash"] = stored["data_hash"]
    prediction["blockchain_storage_id"] = stored["storage_id"]
    prediction["doctor_wallet"] = stored["doctor_wallet"]
    prediction["stored_on_chain"] = stored["stored_on_chain"]
    prediction["encryption_algorithm"] = stored["encryption_algorithm"]
    prediction["encrypted_vitals_bytes"] = stored["encrypted_vitals_bytes"]
    return jsonify(prediction), 201


@app.post("/api/predict")
def predict_only():
    prediction = models.predict(request.get_json(force=True) or {})
    return jsonify(prediction), 200


@app.get("/api/latest")
def latest():
    return jsonify(db.latest() or {})


@app.get("/api/db-records")
def db_records():
    limit = int(request.args.get("limit", 100))
    return jsonify({"data": db.list_records(limit, require_influx=True)})


@app.get("/api/db-status")
def db_status():
    return jsonify(db.status())


@app.get("/api/blockchain-records")
def blockchain_records():
    include_decrypted = request.args.get("decrypt") == "1"
    records = chain.list_records()
    records = sorted(records, key=lambda item: to_unix_timestamp(item.get("timestamp", "")), reverse=True)
    normalized = [normalize_blockchain_record(item, index) for index, item in enumerate(records)]
    if include_decrypted:
        for item in normalized:
            try:
                item["decrypted"] = decrypt_record(item["encrypted_vitals"], chain.private_key_path)
            except Exception as exc:
                item["decryption_error"] = str(exc)
    return jsonify({"data": normalized})


@app.post("/api/blockchain/store")
def store_blockchain():
    record = request.get_json(silent=True) or db.latest()
    if not record:
        return jsonify({"error": "No record supplied and no latest database record exists"}), 400
    prepared = chain.prepare_record(record)
    record["encrypted_vitals"] = prepared["encrypted_vitals"]
    if not db.connected:
        return jsonify({"error": "InfluxDB is not connected. Record was not stored.", "influxdb": db.status()}), 503
    db.write_record(record)
    if db.error:
        return jsonify({"error": "InfluxDB write failed. Record was not stored on blockchain.", "influxdb": db.status()}), 503
    stored = chain.commit_prepared(prepared)
    record["blockchain_tx_hash"] = stored["transaction_hash"]
    record["blockchain_record_id"] = stored["id"]
    record["blockchain_record_hash"] = stored["record_hash"]
    record["blockchain_data_hash"] = stored["data_hash"]
    record["blockchain_storage_id"] = stored["storage_id"]
    record["doctor_wallet"] = stored["doctor_wallet"]
    record["stored_on_chain"] = stored["stored_on_chain"]
    record["encryption_algorithm"] = stored["encryption_algorithm"]
    record["encrypted_vitals_bytes"] = stored["encrypted_vitals_bytes"]
    return jsonify(stored), 201


@app.get("/api/models/status")
def model_status():
    return jsonify(models.status)


@app.post("/api/simulator/start")
def simulator_start():
    payload = request.get_json(silent=True) or {}
    host = payload.get("api_url") or f"http://127.0.0.1:{os.getenv('FLASK_PORT', '5000')}/api/vitals"
    status = simulator.start(
        api_url=host,
        mode=payload.get("mode", "normal_vitals"),
        interval=float(payload.get("interval", 3)),
        device_id=payload.get("device_id", os.getenv("DEFAULT_DEVICE_ID", "device-unknown")),
    )
    return jsonify(status)


@app.post("/api/simulator/stop")
def simulator_stop():
    return jsonify(simulator.stop())


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
