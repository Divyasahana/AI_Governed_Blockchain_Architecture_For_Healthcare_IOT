import { useEffect, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import {
  fetchBlockchainRecords,
  fetchDbRecords,
  fetchDbStatus,
  fetchLatest,
  backfillBlockchainRecords,
  predictVitals,
  storeRecordOnChain,
} from './api/client.js';

const TABS = [
  ['live', 'Live Monitoring'],
  ['test', 'Test Prediction'],
  ['db', 'Database Records'],
  ['chain', 'Blockchain Records'],
];

const COLORS = {
  normal_vitals: '#16a34a',
  critical_vitals: '#dc2626',
  device_error: '#d97706',
};

const SAMPLE_JSON = `{
  "temperature": 39.4,
  "heart_rate": 135,
  "spo2": 88,
  "respiratory_rate": 31,
  "device_id": "DASHBOARD-TEST",
  "patient_id": "patient-demo"
}`;

export default function App() {
  const [tab, setTab] = useState('live');
  return (
    <div>
      <nav className="topbar">
        <strong>IoT Medical Monitor</strong>
        {TABS.map(([id, label]) => (
          <button className={tab === id ? 'active' : ''} key={id} onClick={() => setTab(id)}>{label}</button>
        ))}
      </nav>
      {tab === 'live' && <LiveMonitoring />}
      {tab === 'test' && <TestPrediction />}
      {tab === 'db' && <DatabaseRecords />}
      {tab === 'chain' && <BlockchainRecords />}
    </div>
  );
}

function LiveMonitoring() {
  const [latest, setLatest] = useState(null);
  const [records, setRecords] = useState([]);
  const [error, setError] = useState('');

  async function refresh() {
    try {
      const [lat, db] = await Promise.all([fetchLatest(), fetchDbRecords(30)]);
      setLatest(lat?.input ? lat : null);
      setRecords(db.reverse());
      setError('');
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, []);

  const input = latest?.input || {};
  const fusion = latest?.fusion || {};
  const label = latest?.final_label || 'waiting';
  return (
    <main className="page">
      {error && <p className="error">{error}</p>}
      <section className="summary">
        <Metric label="Temperature" value={input.temperature} unit="C" />
        <Metric label="Heart Rate" value={input.heart_rate} unit="bpm" />
        <Metric label="SpO2" value={input.spo2} unit="%" />
        <Metric label="Respiratory" value={input.respiratory_rate} unit="/min" />
      </section>
      <section className="panel">
        <div className="statusline">
          <span className="pill" style={{ borderColor: COLORS[label], color: COLORS[label] }}>{label.replaceAll('_', ' ')}</span>
          <span>Final Trust Score: {fmt(latest?.trust_score)}</span>
          <span>{input.timestamp || 'No readings yet'}</span>
          {fusion.warning && <b>{fusion.warning}</b>}
        </div>
        <Probability title="XGBoost" values={latest?.xgboost?.probabilities} />
        <Probability title="LSTM" values={latest?.lstm?.probabilities} />
        <Probability title="Fusion" values={fusion.final_probabilities} />
      </section>
      <VitalsTimeChart records={records} />
    </main>
  );
}

function TestPrediction() {
  const [jsonText, setJsonText] = useState(SAMPLE_JSON);
  const [prediction, setPrediction] = useState(null);
  const [chainResult, setChainResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function runPrediction() {
    setBusy(true);
    setError('');
    setChainResult(null);
    try {
      const result = await predictVitals(JSON.parse(jsonText));
      setPrediction(result);
    } catch (err) {
      setError(err instanceof SyntaxError ? 'Invalid JSON input' : err.message);
    } finally {
      setBusy(false);
    }
  }

  async function storePrediction() {
    if (!prediction) return;
    setBusy(true);
    setError('');
    try {
      setChainResult(await storeRecordOnChain(prediction));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const input = prediction?.input || {};
  const fusion = prediction?.fusion || {};
  const label = prediction?.final_label || 'waiting';

  return (
    <main className="page">
      <section className="panel">
        <h2>Sample JSON Input</h2>
        <textarea className="json-input" value={jsonText} onChange={(event) => setJsonText(event.target.value)} spellCheck="false" />
        <div className="toolbar compact">
          <button onClick={runPrediction} disabled={busy}>{busy ? 'Running...' : 'Run Prediction'}</button>
          <button onClick={storePrediction} disabled={busy || !prediction}>Encrypt and Store Blockchain</button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {prediction && (
        <>
          <section className="summary">
            <Metric label="Temperature" value={input.temperature} unit="C" />
            <Metric label="Heart Rate" value={input.heart_rate} unit="bpm" />
            <Metric label="SpO2" value={input.spo2} unit="%" />
            <Metric label="Respiratory" value={input.respiratory_rate} unit="/min" />
          </section>
          <section className="panel">
            <div className="statusline">
              <span className="pill" style={{ borderColor: COLORS[label], color: COLORS[label] }}>{label.replaceAll('_', ' ')}</span>
              <span>{input.timestamp}</span>
              {fusion.warning && <b>{fusion.warning}</b>}
            </div>
            <Probability title="XGBoost probabilities" values={prediction.xgboost?.probabilities} />
            <Probability title={`LSTM probabilities${prediction.lstm?.sequence_available ? '' : ' (sequence not available)'}`} values={prediction.lstm?.probabilities} />
            <Probability title="Adaptive fusion final probabilities" values={fusion.final_probabilities} />
          </section>
          <section className="panel">
            <h2>Full Prediction Output</h2>
            <pre>{JSON.stringify(prediction, null, 2)}</pre>
          </section>
        </>
      )}

      {chainResult && (
        <section className="panel">
          <h2>Encryption and Blockchain Storage Output</h2>
          <div className="kv-grid">
            <Metric label="Encrypted With" value={chainResult.encryption_algorithm} />
            <Metric label="Stored On Chain" value={String(chainResult.stored_on_chain)} />
            <Metric label="Patient ID" value={chainResult.patient_id} />
            <Metric label="Final Label" value={chainResult.final_label} />
            <Metric label="Doctor Wallet" value={shortHash(chainResult.doctor_wallet)} />
          </div>
          <p><b>Data hash SHA-256(encrypted vitals):</b> <code>{chainResult.data_hash}</code></p>
          <p><b>Off-chain storage ID:</b> <code>{chainResult.storage_id}</code></p>
          <p><b>Transaction hash:</b> <code>{chainResult.transaction_hash}</code></p>
          <p><b>Encrypted vitals size:</b> {chainResult.encrypted_vitals_bytes} bytes</p>
          <details>
            <summary>Encrypted vitals</summary>
            <pre>{chainResult.encrypted_vitals}</pre>
          </details>
        </section>
      )}
    </main>
  );
}

function DatabaseRecords() {
  const [records, setRecords] = useState([]);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState('');

  async function refresh() {
    try {
      const [dbRecords, dbStatus] = await Promise.all([fetchDbRecords(100), fetchDbStatus()]);
      setRecords(sortByRecentInputTime(dbRecords));
      setStatus(dbStatus);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => { refresh(); }, []);
  return (
    <main className="page">
      <section className="toolbar">
        <button onClick={refresh}>Refresh</button>
        <span>{status?.connected ? 'InfluxDB connected' : 'InfluxDB not connected'}</span>
      </section>
      {status?.error && <p className="error">{status.error}</p>}
      {error && <p className="error">{error}</p>}
      <RecordsTable records={records} />
    </main>
  );
}

function BlockchainRecords() {
  const [records, setRecords] = useState([]);
  const [decrypt, setDecrypt] = useState(false);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);

  async function refresh(nextDecrypt = decrypt) {
    const items = await fetchBlockchainRecords(nextDecrypt);
    setRecords(sortByRecentUnixTime(items));
  }

  useEffect(() => { refresh(false); }, []);

  async function toggleDecrypt() {
    const nextDecrypt = !decrypt;
    setDecrypt(nextDecrypt);
    await refresh(nextDecrypt);
  }

  async function storeMissingRecords() {
    setBusy(true);
    setStatus('Storing missing records on Sepolia...');
    try {
      const result = await backfillBlockchainRecords(100);
      setStatus(`Stored ${result.stored_count} missing records. Skipped ${result.skipped_existing_count}. Failed ${result.failed_count}.`);
      await refresh(decrypt);
    } catch (err) {
      setStatus(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="page">
      <section className="toolbar">
        <button onClick={toggleDecrypt}>{decrypt ? 'Hide Decrypted' : 'Decrypt'}</button>
        <button onClick={() => refresh()}>Refresh</button>
        <button onClick={storeMissingRecords} disabled={busy}>{busy ? 'Storing...' : 'Store Missing Records'}</button>
        {status && <span>{status}</span>}
      </section>
      <div className="grid">
        {records.map((r, i) => (
          <article className="record" key={`${r.dataHash || r.data_hash || r.record_hash}-${i}`}>
            <p><b>ethereumRecordId:</b> <code>{r.ethereumRecordId ?? r.id ?? i}</code></p>
            <p><b>patientId:</b> <code>{r.patientId || r.patient_id}</code></p>
            <p><b>dataHash:</b> <code>{r.dataHash || r.data_hash || r.record_hash}</code></p>
            <p><b>storageId:</b> <code>{r.storageId || r.storage_id || r.encrypted_data_reference}</code></p>
            <p><b>doctorWalletAddress:</b> <code>{r.doctorWalletAddress || r.doctorWallet || r.doctor_wallet}</code></p>
            <p><b>transactionHash:</b> <code>{r.transactionHash || r.transaction_hash || r.blockchain_tx_hash}</code></p>
            <p><b>timestamp:</b> <code>{r.timestamp}</code></p>
            <p><b>finalLabel:</b> <code>{r.finalLabel || r.final_label}</code></p>
            <p><b>trustScore:</b> <code>{fmt(r.trustScore ?? r.trust_score)}</code></p>
            {decrypt && r.decrypted && (
              <div className="decrypted">
                <p><b>Decrypted Vitals</b></p>
                <pre>{JSON.stringify(r.decrypted, null, 2)}</pre>
              </div>
            )}
            {decrypt && r.decryption_error && <p className="error">{r.decryption_error}</p>}
          </article>
        ))}
      </div>
    </main>
  );
}

function RecordsTable({ records }) {
  return (
    <table className="records-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Device</th>
          <th>Temp</th>
          <th>HR</th>
          <th>Resp</th>
          <th>SpO2</th>
          <th>IF</th>
          <th>XGB-N</th>
          <th>XGB-C</th>
          <th>XGB-D</th>
          <th>LSTM-N</th>
          <th>LSTM-C</th>
          <th>LSTM-D</th>
          <th>Alpha</th>
          <th>Final-N</th>
          <th>Final-C</th>
          <th>Final-D</th>
          <th>Label</th>
          <th>Encrypted Vitals</th>
        </tr>
      </thead>
      <tbody>
        {records.map((r, i) => (
          <tr key={`${r.input?.timestamp}-${i}`}>
            <td>{formatTime(r.input?.timestamp)}</td>
            <td>{r.input?.device_id}</td>
            <td className="num">{fmt(r.input?.temperature)}</td>
            <td className="num">{fmt(r.input?.heart_rate)}</td>
            <td className="num">{fmt(r.input?.respiratory_rate)}</td>
            <td className="num">{fmt(r.input?.spo2)}</td>
            <td className="num">{fmt(r.isolation_forest?.anomaly_score)}</td>
            <td className="num">{fmt(r.xgboost?.probabilities?.normal_vitals)}</td>
            <td className="num">{fmt(r.xgboost?.probabilities?.critical_vitals)}</td>
            <td className="num">{fmt(r.xgboost?.probabilities?.device_error)}</td>
            <td className="num">{fmt(r.lstm?.probabilities?.normal_vitals)}</td>
            <td className="num">{fmt(r.lstm?.probabilities?.critical_vitals)}</td>
            <td className="num">{fmt(r.lstm?.probabilities?.device_error)}</td>
            <td className="num">{fmt(r.fusion?.alpha)}</td>
            <td className="num">{fmt(r.fusion?.final_probabilities?.normal_vitals)}</td>
            <td className="num">{fmt(r.fusion?.final_probabilities?.critical_vitals)}</td>
            <td className="num">{fmt(r.fusion?.final_probabilities?.device_error)}</td>
            <td>{r.final_label}</td>
            <td><code title={r.encrypted_vitals || ''}>{shortHash(r.encrypted_vitals || '')}</code></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fmt(value) {
  if (value === null || value === undefined || value === '') return '';
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3).replace(/\.?0+$/, '') : value;
}

function formatTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function sortByRecentInputTime(items) {
  return [...items].sort((a, b) => new Date(b.input?.timestamp || 0) - new Date(a.input?.timestamp || 0));
}

function sortByRecentUnixTime(items) {
  return [...items].sort((a, b) => Number(b.timestamp || 0) - Number(a.timestamp || 0));
}

function shortHash(value) {
  if (!value) return '';
  return value.length > 14 ? `${value.slice(0, 8)}...${value.slice(-6)}` : value;
}

function VitalsTimeChart({ records }) {
  const data = records.map((record) => ({
    time: new Date(record.input?.timestamp).toLocaleTimeString(),
    timestamp: record.input?.timestamp,
    label: record.final_label,
    temperature: Number(record.input?.temperature ?? 0),
    heartbeat: Number(record.input?.heart_rate ?? 0),
    spo2: Number(record.input?.spo2 ?? 0),
    respiratory_rate: Number(record.input?.respiratory_rate ?? 0),
  }));

  return (
    <section className="panel chart-panel">
      <h2>Vitals Over Time</h2>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 10, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" minTickGap={24} />
          <YAxis />
          <Tooltip content={<VitalsTooltip />} />
          <Legend />
          <Line type="monotone" dataKey="temperature" name="Temperature" stroke="#dc2626" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="heartbeat" name="Heartbeat" stroke="#2563eb" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="spo2" name="SpO2" stroke="#16a34a" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="respiratory_rate" name="Respiratory Rate" stroke="#d97706" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}

function VitalsTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <b>{row.timestamp}</b>
      <span className="pill" style={{ borderColor: COLORS[row.label], color: COLORS[row.label] }}>{row.label?.replaceAll('_', ' ')}</span>
      <p>Temperature: {row.temperature} C</p>
      <p>Heartbeat: {row.heartbeat} bpm</p>
      <p>Respiratory Rate: {row.respiratory_rate}/min</p>
      <p>SpO2: {row.spo2}%</p>
    </div>
  );
}

function Metric({ label, value, unit = '' }) {
  return <div className="metric"><span>{label}</span><strong>{value ?? '--'} {unit}</strong></div>;
}

function Probability({ title, values = {} }) {
  return (
    <div className="prob">
      <h3>{title}</h3>
      {Object.entries(values).map(([label, value]) => (
        <label key={label}><span>{label}</span><progress max="1" value={value}></progress><b>{Math.round(value * 100)}%</b></label>
      ))}
    </div>
  );
}
