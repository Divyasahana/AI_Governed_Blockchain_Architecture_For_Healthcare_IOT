from __future__ import annotations

import argparse
import sys
import time
import os
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
    parser.add_argument("--timeout", type=float, default=float(os.getenv("SIMULATOR_REQUEST_TIMEOUT", "90")))
    parser.add_argument("--count", type=int, help="Stop after sending this many readings.")
    args = parser.parse_args()
    cycle = ["normal_vitals", "critical_vitals", "device_error"]
    i = 0
    while args.count is None or i < args.count:
        mode = cycle[i % len(cycle)] if args.mode == "mixed" else args.mode
        payload = generate(mode, args.device_id)
        try:
            response = requests.post(args.url, json=payload, timeout=args.timeout)
            try:
                body = response.json()
            except ValueError:
                body = {}
            print(response.status_code, mode, body.get("final_label") or body.get("error") or response.text[:160])
        except requests.RequestException as exc:
            print("ERROR", mode, exc)
        i += 1
        if args.count is None or i < args.count:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
