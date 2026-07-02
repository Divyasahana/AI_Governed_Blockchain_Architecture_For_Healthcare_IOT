from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.simulator_service import generate


def main():
    parser = argparse.ArgumentParser(description="IoT medical vitals simulator")
    parser.add_argument("--url", default="http://127.0.0.1:5000/api/vitals")
    parser.add_argument("--mode", choices=["normal_vitals", "critical_vitals", "device_error", "mixed"], default="mixed")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--device-id", default="ESP32-SIM-01")
    args = parser.parse_args()
    cycle = ["normal_vitals", "critical_vitals", "device_error"]
    i = 0
    while True:
        mode = cycle[i % len(cycle)] if args.mode == "mixed" else args.mode
        payload = generate(mode, args.device_id)
        response = requests.post(args.url, json=payload, timeout=10)
        print(response.status_code, mode, response.json().get("final_label"))
        i += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
