// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract MedicalRecordStore {
    struct MedicalRecord {
        string patientId;
        bytes32 dataHash;
        string storageId;
        address doctorWallet;
        uint256 timestamp;
        string finalLabel;
        uint256 trustScoreBps;
    }

    address public owner;
    MedicalRecord[] private records;

    event MedicalRecordStored(uint256 indexed id, string patientId, bytes32 indexed dataHash, string storageId, address indexed doctorWallet, uint256 timestamp, string finalLabel, uint256 trustScoreBps);

    error NotOwner();
    error BadIndex();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function storeRecord(string calldata patientId, bytes32 dataHash, string calldata storageId, address doctorWallet, uint256 timestamp, string calldata finalLabel, uint256 trustScoreBps) external onlyOwner returns (uint256 id) {
        records.push(MedicalRecord({
            patientId: patientId,
            dataHash: dataHash,
            storageId: storageId,
            doctorWallet: doctorWallet,
            timestamp: timestamp,
            finalLabel: finalLabel,
            trustScoreBps: trustScoreBps
        }));
        id = records.length - 1;
        emit MedicalRecordStored(id, patientId, dataHash, storageId, doctorWallet, timestamp, finalLabel, trustScoreBps);
    }

    function recordCount() external view returns (uint256) {
        return records.length;
    }

    function getRecord(uint256 id) external view returns (MedicalRecord memory) {
        if (id >= records.length) revert BadIndex();
        return records[id];
    }
}
