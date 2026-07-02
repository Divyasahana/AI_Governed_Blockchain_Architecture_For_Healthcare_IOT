from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:
    AESGCM = hashes = serialization = padding = rsa = None


def canonical_json(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def record_hash(data: dict) -> str:
    return "0x" + hashlib.sha256(canonical_json(data)).hexdigest()


def encrypted_data_hash(encrypted_payload: str) -> str:
    return "0x" + hashlib.sha256(encrypted_payload.encode("utf-8")).hexdigest()


def generate_keypair(private_key_path: str, public_key_path: str):
    if rsa is None:
        return
    private_path = Path(private_key_path)
    public_path = Path(public_key_path)
    if private_path.exists() and public_path.exists():
        return
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public_path.write_bytes(private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))


def encrypt_record(data: dict, public_key_path: str) -> str:
    if serialization is None or AESGCM is None:
        raise RuntimeError("cryptography is required for asymmetric encryption. Install requirements.txt.")
    public_key = serialization.load_pem_public_key(Path(public_key_path).read_bytes())
    aes_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    encrypted_payload = AESGCM(aes_key).encrypt(nonce, canonical_json(data), None)
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    envelope = {
        "alg": "RSA-OAEP-256+A256GCM",
        "encrypted_key": base64.b64encode(encrypted_key).decode("ascii"),
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(encrypted_payload).decode("ascii"),
    }
    return base64.b64encode(json.dumps(envelope, separators=(",", ":")).encode("utf-8")).decode("ascii")


def decrypt_record(ciphertext_b64: str, private_key_path: str) -> dict:
    if serialization is None or AESGCM is None:
        raise RuntimeError("cryptography is required for asymmetric decryption. Install requirements.txt.")
    private_key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)
    envelope = json.loads(base64.b64decode(ciphertext_b64).decode("utf-8"))
    aes_key = private_key.decrypt(
        base64.b64decode(envelope["encrypted_key"]),
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    plaintext = AESGCM(aes_key).decrypt(
        base64.b64decode(envelope["nonce"]),
        base64.b64decode(envelope["ciphertext"]),
        None,
    )
    return json.loads(plaintext.decode("utf-8"))
