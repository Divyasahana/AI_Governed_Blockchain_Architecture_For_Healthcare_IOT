from __future__ import annotations

import random
import threading
import os
from datetime import datetime, timezone

import requests


class SimulatorService:
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.mode = "normal_vitals"

    def start(self, api_url: str, mode: str = "normal_vitals", interval: float = 3.0, device_id: str = "ESP32-SIM-01"):
        self.stop()
        self.mode = mode
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(api_url, mode, interval, device_id), daemon=True)
        self.thread.start()
        return {"running": True, "mode": mode, "interval": interval, "device_id": device_id}

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=2)
        return {"running": False}

    def _run(self, api_url: str, mode: str, interval: float, device_id: str):
        timeout = float(os.getenv("SIMULATOR_REQUEST_TIMEOUT", "90"))
        while not self.stop_event.is_set():
            try:
                requests.post(api_url, json=generate(mode, device_id), timeout=timeout)
            except Exception:
                pass
            self.stop_event.wait(interval)


def generate(mode: str, device_id: str = "ESP32-SIM-01") -> dict:
    if mode == "critical_vitals":
        vals = {"temperature": random.gauss(39.2, 0.35), "heart_rate": random.gauss(132, 12), "spo2": random.gauss(88, 2), "respiratory_rate": random.gauss(30, 4)}
    elif mode == "device_error":
        pattern = random.choice(["impossible", "stuck_zero", "jump"])
        if pattern == "impossible":
            vals = {"temperature": 0, "heart_rate": 0, "spo2": 130, "respiratory_rate": 0}
        elif pattern == "jump":
            vals = {"temperature": 44.5, "heart_rate": 225, "spo2": 55, "respiratory_rate": 55}
        else:
            vals = {"temperature": 36.7, "heart_rate": 36.7, "spo2": 36.7, "respiratory_rate": 36.7}
    else:
        vals = {"temperature": random.gauss(36.8, 0.2), "heart_rate": random.gauss(76, 8), "spo2": random.gauss(98, 0.8), "respiratory_rate": random.gauss(16, 2)}
    vals = {key: round(float(value), 2) for key, value in vals.items()}
    vals.update({"timestamp": datetime.now(timezone.utc).isoformat(), "device_id": device_id, "patient_id": "patient-demo"})
    return vals
