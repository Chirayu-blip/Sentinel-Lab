import { useEffect, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowUpRight, BadgeCheck, Bot, Check,
  ChevronRight, CircleHelp, FileCode2, FileText, Fingerprint,
  Gauge, LoaderCircle, LockKeyhole, Network, RotateCcw, ScanSearch,
  Shield, ShieldAlert, Sparkles, UploadCloud, X,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function verdictTone(verdict) {
  return verdict === "MALWARE" ? "danger" : verdict === "SUSPICIOUS" ? "warning" : "safe";
}

function DropFile({ label, hint, accept, file, onChange, icon: Icon, required }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const setFile = (candidate) => {
    if (candidate) onChange(candidate);
  };
  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        setFile(event.dataTransfer.files?.[0]);
      }}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label={`Choose ${label}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        hidden
        onChange={(event) => setFile(event.target.files?.[0])}
      />
      <div className={`upload-icon ${file ? "selected" : ""}`}>
        {file ? <Check size={18} /> : <Icon size={19} />}
      </div>
      <div className="drop-copy">
        <div className="drop-title">{file ? file.name : label}</div>
        <div className="drop-hint">
          {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · click to replace` : hint}
        </div>
      </div>
      {!file && <span className="file-requirement">{required ? "Required" : "Optional"}</span>}
      {file && (
        <button
          className="remove-file"
          type="button"
          aria-label={`Remove ${label}`}
          onClick={(event) => { event.stopPropagation(); onChange(null); }}
        >
          <X size={15} />
        </button>
      )}
    </div>
  );
}

function RiskGauge({ score, verdict }) {
  const tone = verdictTone(verdict);
  return (
    <div className={`risk-gauge ${tone}`} style={{ "--score": `${score * 100}%` }}>
      <div className="gauge-inner">
        <span className="gauge-caption">FUSED RISK</span>
        <strong>{percent(score)}</strong>
        <span className={`verdict-pill ${tone}`}>{verdict}</span>
      </div>
    </div>
  );
}

function EvidenceCard({ title, icon: Icon, evidence, color, emptyText }) {
  return (
    <section className="panel evidence-card">
      <div className="panel-heading">
        <div className={`heading-icon ${color}`}><Icon size={16} /></div>
        <div className="panel-title">{title}</div>
        {evidence && <span className="mini-score">{percent(evidence.score)} risk</span>}
      </div>
      {!evidence ? (
        <div className="empty-evidence"><CircleHelp size={15} />{emptyText}</div>
      ) : (
        <div className="shap-list">
          {evidence.shap.map((item) => {
            const positive = item.impact >= 0;
            const barWidth = Math.min(100, Math.max(8, Math.abs(item.impact) * 1000));
            return (
              <div className="shap-row" key={item.feature}>
                <div className="shap-label">
                  <span>{item.feature.replaceAll("_", " ")}</span>
                  <span className="shap-value">{Number(item.value).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                </div>
                <div className="impact-track">
                  <span className={`impact-bar ${positive ? "up" : "down"}`} style={{ width: `${barWidth}%` }} />
                </div>
                <div className={`impact-text ${positive ? "up" : "down"}`}>
                  {positive ? "+" : "−"}{Math.abs(item.impact).toFixed(3)}
                </div>
              </div>
            );
          })}
          <div className="shap-legend"><span><i className="legend-dot up" /> Pushes toward malware</span><span><i className="legend-dot down" /> Pushes away</span></div>
        </div>
      )}
    </section>
  );
}

function ResultView({ result }) {
  const tone = verdictTone(result.verdict);
  const staticWeight = result.weights.static;
  const dynamicWeight = result.weights.dynamic;
  return (
    <div className="results">
      <div className={`verdict-banner ${tone}`}>
        <div className={`verdict-emblem ${tone}`}>
          {tone === "safe" ? <BadgeCheck size={21} /> : <ShieldAlert size={21} />}
        </div>
        <div className="verdict-copy">
          <span>{result.demo_mode ? "DEMO MODEL · NOT FOR DETECTION" : "ANALYSIS COMPLETE"}</span>
          <strong>{result.file_name}</strong>
          <small>
            Entropy {result.entropy.toFixed(2)} bits/byte <span>·</span> 14 static indicators
            {result.demo_mode && <span> · synthetic training</span>}
          </small>
        </div>
        <RiskGauge score={result.score} verdict={result.verdict} />
      </div>

      <section className="panel fusion-panel">
        <div className="panel-heading">
          <div className="heading-icon purple"><Activity size={16} /></div>
          <div>
            <div className="panel-title">Entropy-adaptive fusion</div>
            <div className="panel-subtitle">Per-file evidence weighting · sigmoid threshold 6.2</div>
          </div>
          <span className="score-formula">P = {staticWeight.toFixed(2)}Pₛ + {dynamicWeight.toFixed(2)}Pᵈ</span>
        </div>
        <div className="weight-track">
          <div className="weight-static" style={{ width: `${staticWeight * 100}%` }} />
          <div className="weight-dynamic" style={{ width: `${dynamicWeight * 100}%` }} />
        </div>
        <div className="weight-legend">
          <span><i className="legend-dot static-dot" />Static <b>{percent(staticWeight)}</b></span>
          <span><i className="legend-dot dynamic-dot" />Dynamic <b>{percent(dynamicWeight)}</b></span>
          <span className="entropy-readout">H = {result.entropy.toFixed(2)}</span>
        </div>
      </section>

      <div className="evidence-grid">
        <EvidenceCard title="Static model · Random Forest" icon={FileCode2} evidence={result.static} color="blue" />
        <EvidenceCard
          title="Dynamic model · Gradient Boosting"
          icon={Network}
          evidence={result.dynamic}
          color="teal"
          emptyText="No API Monitor log provided. Dynamic model correctly excluded."
        />
      </div>

      <section className="panel analyst-panel">
        <div className="analyst-heading">
          <div className="ai-orb"><Sparkles size={17} /></div>
          <div>
            <div className="analyst-title">GenAI analyst</div>
            <div className="analyst-meta">
              {result.analysis_source === "genai" ? "Evidence-grounded · model-assisted" : "Evidence-grounded · local explanation"}
            </div>
          </div>
          <span className="grounded-badge"><LockKeyhole size={12} /> GROUNDED IN SHAP</span>
        </div>
        <p className="analyst-text">{result.analysis}</p>
        <div className="analyst-footnote">
          <AlertTriangle size={13} />
          The language model explains the classifier output; it does not determine or modify the verdict.
        </div>
      </section>
    </div>
  );
}

export default function App() {
  const [peFile, setPeFile] = useState(null);
  const [logFile, setLogFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((response) => response.ok ? response.json() : Promise.reject(new Error("API unavailable")))
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  async function analyze(event) {
    event.preventDefault();
    if (!peFile) {
      setError("Choose a PE file to start the analysis.");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    const body = new FormData();
    body.append("file", peFile);
    if (logFile) body.append("log", logFile);
    try {
      const response = await fetch(`${API_URL}/api/analyze`, { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || "Analysis request failed");
      setResult(payload);
    } catch (requestError) {
      setError(requestError.message || "Could not reach the analysis service.");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setPeFile(null);
    setLogFile(null);
    setResult(null);
    setError("");
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="Sentinel home">
          <span className="brand-mark"><Shield size={19} /></span>
          <span>SENTINEL<span className="brand-separator">/</span><b>LAB</b></span>
        </a>
        <div className="side-label">WORKSPACE</div>
        <div className="side-link active"><ScanSearch size={16} /><span>File analysis</span><ChevronRight size={14} /></div>
        <div className="side-label side-label-lower">INTELLIGENCE</div>
        <div className="side-link muted"><Fingerprint size={16} /><span>Model insights</span><ArrowUpRight size={13} /></div>
        <div className="sidebar-bottom">
          <div className="privacy-card">
            <div className="privacy-icon"><LockKeyhole size={15} /></div>
            <div><strong>Private by design</strong><span>Files processed in memory</span></div>
          </div>
          <div className="system-status">
            <span className={`status-dot ${health?.status === "ok" ? "online" : ""}`} />
            <span>{health?.status === "ok" ? "Analysis API connected" : health?.status === "offline" ? "Analysis API offline" : "Connecting to API"}</span>
          </div>
          <div className="version">SENTINEL LAB <span>v1.0.0</span></div>
        </div>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <ChevronRight size={13} /> <b>File analysis</b></div>
          <div className="topbar-right"><span className="environment-tag"><span /> DEFENSIVE RESEARCH</span><div className="avatar">SL</div></div>
        </header>

        <div className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> THREAT INTELLIGENCE WORKSPACE</div>
              <h1>Malware <span>analysis</span></h1>
              <p>Static and behavioral signals, fused by file entropy. Every score comes with its evidence.</p>
            </div>
            {result && <button className="reset-button" onClick={reset}><RotateCcw size={14} /> New analysis</button>}
          </div>

          <div className="workspace-grid">
            <div className="left-column">
              <form className="panel upload-panel" onSubmit={analyze}>
                <div className="section-title-row">
                  <div><div className="step-label">01 <span>·</span> INPUT</div><h2>Submit a sample</h2></div>
                  <div className="in-memory"><LockKeyhole size={12} /> IN-MEMORY ONLY</div>
                </div>
                <p className="upload-description">Analyze a Windows executable with static inspection. Add an API Monitor export to enable behavioral analysis.</p>
                {health?.demo_mode && (
                  <div className="demo-warning">
                    <AlertTriangle size={14} />
                    <span><b>Demo models active.</b> Results are synthetic and only verify the upload flow—not malware detection.</span>
                  </div>
                )}
                <div className="file-fields">
                  <DropFile label="Windows PE file" hint="Drop .exe, .dll or .sys here · max 100 MB" accept=".exe,.dll,.sys" file={peFile} onChange={setPeFile} icon={FileCode2} required />
                  <DropFile label="API Monitor log" hint="Drop a CSV export here · optional" accept=".csv" file={logFile} onChange={setLogFile} icon={FileText} />
                </div>
                {error && <div className="error-message"><AlertTriangle size={15} />{error}</div>}
                <button className="analyze-button" disabled={busy || !peFile} type="submit">
                  {busy ? <><LoaderCircle className="spin" size={16} /> Analyzing sample…</> : <><ScanSearch size={16} /> Analyze sample <ChevronRight size={15} /></>}
                </button>
                <div className="upload-note"><Shield size={13} /> No files are retained. Use only samples you are authorized to analyze.</div>
              </form>

              {result ? (
                <ResultView result={result} />
              ) : (
                <div className="empty-state">
                  <div className="empty-art"><div className="art-ring ring-one" /><div className="art-ring ring-two" /><div className="art-core"><Shield size={27} /></div><span className="art-spark spark-one" /><span className="art-spark spark-two" /></div>
                  <div className="empty-title">Your evidence, in context.</div>
                  <p>Run an analysis to see entropy-weighted model scores, per-feature SHAP signals and an AI-generated analyst brief.</p>
                  <div className="empty-chips"><span><Gauge size={13} /> Adaptive fusion</span><span><Activity size={13} /> SHAP explainability</span><span><Bot size={13} /> GenAI analyst</span></div>
                </div>
              )}
            </div>

            <aside className="right-column">
              <section className="panel how-panel">
                <div className="side-panel-title"><span className="how-icon"><Activity size={14} /></span> HOW FUSION WORKS</div>
                <p>Each file gets its own dynamic/static mix based on byte entropy.</p>
                <div className="equation-card">
                  <span>Dynamic weight</span>
                  <code>w<sub>d</sub> = σ((H − 6.2) × 2)</code>
                  <div className="equation-line"><i /><i /></div>
                  <div className="equation-labels"><span>STATIC-LEANING</span><span>DYNAMIC-LEANING</span></div>
                </div>
                <div className="how-divider" />
                <div className="verdict-scale">
                  <div><i className="scale-dot malware" /><span>Malware</span><b>≥ 0.65</b></div>
                  <div><i className="scale-dot suspicious" /><span>Suspicious</span><b>0.40 – 0.64</b></div>
                  <div><i className="scale-dot benign" /><span>Benign</span><b>&lt; 0.40</b></div>
                </div>
                <div className="static-fallback"><CircleHelp size={13} /><span>No log? Dynamic weight becomes <b>0%</b>. No synthetic zeros.</span></div>
              </section>

              <section className="panel genai-info">
                <div className="side-panel-title"><span className="ai-small"><Sparkles size={13} /></span> GROUNDED GENAI</div>
                <p>Turns model evidence into a readable analyst brief—not an unverified verdict.</p>
                <div className="genai-points">
                  <div><Check size={13} /><span>Uses model output + SHAP attributions</span></div>
                  <div><Check size={13} /><span>Never receives the uploaded file</span></div>
                  <div><Check size={13} /><span>Cannot modify the classifier verdict</span></div>
                </div>
                <div className="provider-status">
                  <span className={`status-dot ${health?.genai_enabled ? "online" : ""}`} />
                  {health?.genai_enabled ? "LLM provider configured" : "Local evidence explanation enabled"}
                </div>
              </section>

              <div className="research-note"><AlertTriangle size={14} /><p><b>Research use only.</b> Scores depend on model training data and are not proof of intent or a replacement for isolated analysis.</p></div>
            </aside>
          </div>
          <footer className="page-footer"><span>© 2026 SENTINEL LAB · DEFENSIVE MALWARE TRIAGE</span><span>MODEL OUTPUTS ARE PROBABILISTIC</span></footer>
        </div>
      </div>
    </main>
  );
}
