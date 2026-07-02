const fs = require("fs");
const hre = require("hardhat");

async function main() {
  const deployment = JSON.parse(fs.readFileSync("deployment.json"));
  const contract = await hre.ethers.getContractAt("MedicalRecordStore", deployment.address);
  const count = Number(await contract.recordCount());
  const records = [];
  for (let i = 0; i < count; i += 1) {
    const r = await contract.getRecord(i);
    records.push({
      id: i,
      patientId: r.patientId,
      dataHash: r.dataHash,
      storageId: r.storageId,
      doctorWallet: r.doctorWallet,
      timestamp: Number(r.timestamp),
      finalLabel: r.finalLabel,
      trustScore: Number(r.trustScoreBps) / 10000,
    });
  }
  console.log(JSON.stringify(records, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
