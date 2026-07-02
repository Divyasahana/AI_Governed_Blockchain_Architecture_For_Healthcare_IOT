from __future__ import annotations

import argparse
import json

from backend.blockchain.crypto import decrypt_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciphertext", required=True)
    parser.add_argument("--private-key", default="keys/private.pem")
    args = parser.parse_args()
    print(json.dumps(decrypt_record(args.ciphertext, args.private_key), indent=2))


if __name__ == "__main__":
    main()
