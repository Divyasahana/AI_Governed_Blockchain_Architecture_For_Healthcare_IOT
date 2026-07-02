const BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';

async function apiFetch(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export const fetchLatest = () => apiFetch('/latest');
export const submitVitals = (payload) => apiFetch('/vitals', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
export const predictVitals = (payload) => apiFetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
export const fetchDbRecords = (limit = 100) => apiFetch(`/db-records?limit=${limit}`).then((r) => r.data);
export const fetchDbStatus = () => apiFetch('/db-status');
export const fetchBlockchainRecords = (decrypt = false) => apiFetch(`/blockchain-records${decrypt ? '?decrypt=1' : ''}`).then((r) => r.data);
export const storeLatestOnChain = () => apiFetch('/blockchain/store', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
export const storeRecordOnChain = (record) => apiFetch('/blockchain/store', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(record) });
export const fetchModelStatus = () => apiFetch('/models/status');
export const startSimulator = (mode) => apiFetch('/simulator/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode, interval: 2 }) });
export const stopSimulator = () => apiFetch('/simulator/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
