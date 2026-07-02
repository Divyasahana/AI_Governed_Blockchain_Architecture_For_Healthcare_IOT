const fs = require("fs");
const crypto = require("crypto");
const hre = require("hardhat");

async function main() {
  const deployment = JSON.parse(fs.readFileSync("deployment.json"));
  const contract = await hre.ethers.getContractAt("MedicalRecordStore", deployment.address);
  const file = process.argv[2] || "sample-record.json";
  const record = JSON.parse(fs.readFileSync(file));
  const encryptedPayload = record.encrypted_payload || record.encryptedPayload || JSON.stringify(record);
  const hash = record.data_hash || record.record_hash || "0x" + crypto.createHash("sha256").update(encryptedPayload).digest("hex");
  const storageId = record.storage_id || record.encrypted_data_reference || record.encryptedDataReference || `local://encrypted/${hash}`;
  const ts = Math.floor(new Date(record.timestamp || Date.now()).getTime() / 1000);
  const [owner] = await hre.ethers.getSigners();
  const doctorWallet = record.doctor_wallet || record.doctorWallet || owner.address;
  const tx = await contract.storeRecord(
    record.patient_id || record.patientId || record.patient_device_id || record.device_id || "P101",
    hash,
    storageId,
    doctorWallet,
    ts,
    record.final_label || "normal_vitals",
    Math.round((record.trust_score || 0) * 10000),
  );
  const receipt = await tx.wait();
  console.log(JSON.stringify({ transactionHash: receipt.hash, dataHash: hash, storageId, doctorWallet }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
