const fs = require("fs");
const hre = require("hardhat");

async function main() {
  const Contract = await hre.ethers.getContractFactory("MedicalRecordStore");
  const contract = await Contract.deploy();
  await contract.waitForDeployment();
  const address = await contract.getAddress();
  fs.writeFileSync("deployment.json", JSON.stringify({ address, network: hre.network.name }, null, 2));
  console.log(`MedicalRecordStore deployed to ${address}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
