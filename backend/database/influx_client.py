from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from typing import Deque, List

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
except Exception:
    InfluxDBClient = None
    Point = None
    SYNCHRONOUS = None


def parse_timestamp(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


class InfluxStore:
    def __init__(self):
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.getenv("INFLUXDB_TOKEN", "")
        self.org = os.getenv("INFLUXDB_ORG", "iot-health-monitor")
        self.bucket = os.getenv("INFLUXDB_BUCKET", "vitals")
        self.memory: Deque[dict] = deque(maxlen=1000)
        self.client = None
        self.write_api = None
        self.query_api = None
        self.error = None
        if InfluxDBClient and self.token:
            try:
                self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.client.query_api()
            except Exception as exc:
                self.error = str(exc)
                self.client = None
                self.write_api = None
                self.query_api = None
        elif not self.token:
            self.error = "INFLUXDB_TOKEN is empty. Add token to .env and restart Flask."

    @property
    def connected(self) -> bool:
        return self.write_api is not None

    def status(self) -> dict:
        return {
            "connected": self.connected,
            "url": self.url,
            "org": self.org,
            "bucket": self.bucket,
            "token_configured": bool(self.token),
            "error": self.error,
        }

    def write_record(self, record: dict) -> dict:
        self.memory.appendleft(record)
        if not self.write_api or Point is None:
            return record
        vitals = record["input"]
        fusion = record["fusion"]
        point = (
            Point("medical_vitals")
            .tag("device_id", vitals["device_id"])
            .tag("final_label", record["final_label"])
            .time(parse_timestamp(vitals["timestamp"]))
            .field("temperature", float(vitals["temperature"]))
            .field("heart_rate", float(vitals["heart_rate"]))
            .field("spo2", float(vitals["spo2"]))
            .field("respiratory_rate", float(vitals["respiratory_rate"]))
            .field("anomaly_score", float(record["isolation_forest"]["anomaly_score"]))
            .field("alpha", float(fusion["alpha"]))
            .field("xgb_normal_vitals", float(record["xgboost"]["probabilities"]["normal_vitals"]))
            .field("xgb_critical_vitals", float(record["xgboost"]["probabilities"]["critical_vitals"]))
            .field("xgb_device_error", float(record["xgboost"]["probabilities"]["device_error"]))
            .field("lstm_normal_vitals", float(record["lstm"]["probabilities"]["normal_vitals"]))
            .field("lstm_critical_vitals", float(record["lstm"]["probabilities"]["critical_vitals"]))
            .field("lstm_device_error", float(record["lstm"]["probabilities"]["device_error"]))
            .field("lstm_sequence_available", bool(record.get("lstm", {}).get("sequence_available", False)))
            .field("fusion_normal_vitals", float(fusion["final_probabilities"]["normal_vitals"]))
            .field("fusion_critical_vitals", float(fusion["final_probabilities"]["critical_vitals"]))
            .field("fusion_device_error", float(fusion["final_probabilities"]["device_error"]))
        )
        if record.get("encrypted_vitals"):
            point = point.field("encrypted_vitals", record["encrypted_vitals"])
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            self.error = None
        except Exception as exc:
            self.error = str(exc)
        return record

    def latest(self) -> dict | None:
        return self.memory[0] if self.memory else None

    def list_records(self, limit: int = 100, require_influx: bool = False) -> List[dict]:
        if self.query_api:
            try:
                flux = f'''
from(bucket: "{self.bucket}")
  |> range(start: 0)
  |> filter(fn: (r) => r._measurement == "medical_vitals")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> group()
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: {int(limit)})
'''
                tables = self.query_api.query(flux, org=self.org)
                rows = []
                for table in tables:
                    for item in table.records:
                        values = item.values
                        timestamp = values.get("_time")
                        final_probabilities = {
                            "normal_vitals": values.get("fusion_normal_vitals"),
                            "critical_vitals": values.get("fusion_critical_vitals"),
                            "device_error": values.get("fusion_device_error"),
                        }
                        trust_values = [
                            float(value)
                            for value in final_probabilities.values()
                            if value is not None
                        ]
                        rows.append({
                            "input": {
                                "timestamp": timestamp.isoformat() if timestamp else "",
                                "device_id": values.get("device_id", ""),
                                "patient_id": "",
                                "temperature": values.get("temperature"),
                                "heart_rate": values.get("heart_rate"),
                                "spo2": values.get("spo2"),
                                "respiratory_rate": values.get("respiratory_rate"),
                            },
                            "final_label": values.get("final_label", ""),
                            "trust_score": max(trust_values) if trust_values else 0,
                            "isolation_forest": {"anomaly_score": values.get("anomaly_score")},
                            "fusion": {
                                "alpha": values.get("alpha"),
                                "final_probabilities": final_probabilities,
                            },
                            "xgboost": {"probabilities": {
                                "normal_vitals": values.get("xgb_normal_vitals"),
                                "critical_vitals": values.get("xgb_critical_vitals"),
                                "device_error": values.get("xgb_device_error"),
                            }},
                            "lstm": {"probabilities": {
                                "normal_vitals": values.get("lstm_normal_vitals"),
                                "critical_vitals": values.get("lstm_critical_vitals"),
                                "device_error": values.get("lstm_device_error"),
                            }, "sequence_available": bool(values.get("lstm_sequence_available", False))},
                            "encrypted_vitals": values.get("encrypted_vitals", ""),
                        })
                rows = sorted(
                    rows,
                    key=lambda row: parse_timestamp(row.get("input", {}).get("timestamp", "")),
                    reverse=True,
                )
                self.error = None
                return rows[:int(limit)]
            except Exception as exc:
                self.error = str(exc)
        if require_influx:
            return []
        return list(self.memory)[:limit]

    def attach_tx(self, timestamp: str, tx_hash: str):
        for record in self.memory:
            if record.get("input", {}).get("timestamp") == timestamp:
                record["blockchain_tx_hash"] = tx_hash
                return record
        return None

    def blockchain_records(self, limit: int = 100) -> List[dict]:
        return []
