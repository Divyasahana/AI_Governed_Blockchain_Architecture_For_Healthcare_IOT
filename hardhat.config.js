require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts-hardhat",
  },
  networks: {
    localhost: { url: "http://127.0.0.1:8545", chainId: 31337 },
    sepolia: {
      url: process.env.RPC_URL || "",
      accounts: process.env.ETH_PRIVATE_KEY ? [process.env.ETH_PRIVATE_KEY] : [],
    },
  },
};
