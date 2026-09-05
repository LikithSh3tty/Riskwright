import { useEffect, useState } from "react";
import { api, num, pct } from "../lib/api";
import { Band, ErrorNotice, Resource, Skeleton, Spinner, useResource } from "../components/Primitives";

/* Where the applicant sits between the two boundaries. The probability alone
 * does not say how close a call it was, and for a Medium that is the whole
 * question. */
function ThresholdRuler({ probability, tLow, tHigh }) {
  const ceiling = Math.max(tHigh * 2.4, probability * 1.15, 0.12);
  const at = (v) => `${Math.min(100, Math.max(0, (v / ceiling) * 100))}%`;
  return (
    <div className="ruler">
      <div className="ruler-track">
        <span className="ruler-tick" style={{ left: at(tLow) }} />
        <span className="ruler-tick" style={{ left: at(tHigh) }} />
        <span className="ruler-pin" style={{ left: at(probability) }} />
      </div>
      <div className="ruler-scale">
        <span>0%</span>
        <span>Low below {pct(tLow, 2)}</span>
        <span>High at or above {pct(tHigh, 2)}</span>
      </div>
    </div>
  );
}

function Verdict({ result }) {
  return (
    <>
      <div className={`verdict is-${result.risk_band}`}>
        <div>
          <Band band={`${result.risk_band}`} />
          <div className="verdict-prob" style={{ marginTop: 10 }}>
            {pct(result.default_probability, 2)}
          </div>
          <p className="verdict-action">{result.recommended_action}</p>
        </div>
      </div>
      <ThresholdRuler
        probability={result.default_probability}
        tLow={result.thresholds.t_low}
        tHigh={result.thresholds.t_high}
      />
      <p className="section-note">
        The upper boundary is the cost-optimal threshold, derived from a stated 10:1 cost ratio.
        The lower one is a business judgment about where automatic approval should stop, not an
        optimum.
      </p>
    </>
  );
}

function ExistingApplicant({ onScored }) {
  const listing = useResource(() => api.applicants(60), []);
  const [chosen, setChosen] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (listing.status === "ready" && chosen == null && listing.data.applicants.length) {
      setChosen(listing.data.applicants[0].sk_id_curr);
    }
  }, [listing.status, listing.data, chosen]);

  useEffect(() => {
    if (chosen == null) return;
    let alive = true;
    setBusy(true);
    setError(null);
    api.predict({ sk_id_curr: Number(chosen) })
      .then((r) => { if (alive) { setResult(r); onScored({ sk_id_curr: Number(chosen) }); } })
      .catch((e) => alive && setError(e))
      .finally(() => alive && setBusy(false));
    return () => { alive = false; };
  }, [chosen, onScored]);

  return (
    <Resource state={listing} skeleton={<Skeleton height={200} />}>
      {(data) => {
        const row = data.applicants.find((a) => a.sk_id_curr === Number(chosen));
        return (
          <div className="split-wide">
            <div>
              <div className="field" style={{ maxWidth: 260 }}>
                <label htmlFor="applicant">Applicant</label>
                <select id="applicant" className="control" value={chosen ?? ""}
                        onChange={(e) => setChosen(e.target.value)}>
                  {data.applicants.map((a) => (
                    <option key={a.sk_id_curr} value={a.sk_id_curr}>{a.sk_id_curr}</option>
                  ))}
                </select>
                <span className="field-hint">
                  {data.count} applicants, as /applicants returns them. The endpoint applies no
                  ORDER BY, so this is a stable but arbitrary window rather than the first rows
                  or a held-out sample.
                </span>
              </div>

              {row && (
                <div className="figures" style={{ marginTop: 28, borderBottom: 0, paddingBottom: 0 }}>
                  <div>
                    <div className="figure-label">Age</div>
                    <div className="metric-value is-mono">{row.age_years?.toFixed(0) ?? "n/a"}</div>
                  </div>
                  <div>
                    <div className="figure-label">Annual income</div>
                    <div className="metric-value is-mono">{num(row.amt_income_total)}</div>
                  </div>
                  <div>
                    <div className="figure-label">Loan amount</div>
                    <div className="metric-value is-mono">{num(row.amt_credit)}</div>
                  </div>
                </div>
              )}
            </div>

            <div>
              {error && <ErrorNotice error={error} />}
              {busy && !result && <Skeleton height={150} />}
              {result && !error && <Verdict result={result} />}
            </div>
          </div>
        );
      }}
    </Resource>
  );
}

const WHAT_IF_FIELDS = [
  { key: "amt_income_total", label: "Annual income", min: 25000, max: 1000000, step: 5000, value: 150000 },
  { key: "amt_credit", label: "Loan amount", min: 45000, max: 2000000, step: 5000, value: 500000 },
  { key: "amt_annuity", label: "Annual repayment", min: 2000, max: 200000, step: 1000, value: 25000 },
];

function WhatIf({ onScored }) {
  const [values, setValues] = useState(() =>
    Object.fromEntries(WHAT_IF_FIELDS.map((f) => [f.key, f.value]))
  );
  const [employed, setEmployed] = useState(5);
  const [ext, setExt] = useState(0.5);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const features = {
    ...values,
    days_employed: -Math.round(employed * 365.25),
    ext_source_1: ext,
    ext_source_2: ext,
    ext_source_3: ext,
  };

  const score = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.predict({ features });
      setResult(r);
      onScored({ features });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="split-wide">
      <div>
        <div className="form-grid">
          {WHAT_IF_FIELDS.map((f) => (
            <div className="field" key={f.key}>
              <label htmlFor={f.key}>{f.label}</label>
              <input id={f.key} className="control" type="number"
                     min={f.min} max={f.max} step={f.step} value={values[f.key]}
                     onChange={(e) => setValues((v) => ({ ...v, [f.key]: Number(e.target.value) }))} />
            </div>
          ))}
          <div className="field">
            <label htmlFor="employed">Years employed: {employed}</label>
            <input id="employed" className="control" type="range" min="0" max="45" step="1"
                   value={employed} onChange={(e) => setEmployed(Number(e.target.value))} />
          </div>
          <div className="field">
            <label htmlFor="ext">External credit score: {ext.toFixed(2)}</label>
            <input id="ext" className="control" type="range" min="0" max="1" step="0.05"
                   value={ext} onChange={(e) => setExt(Number(e.target.value))} />
            <span className="field-hint">Applied to all three bureau scores.</span>
          </div>
        </div>

        <button className="btn" style={{ marginTop: 20 }} onClick={score} disabled={busy}>
          {busy ? <><Spinner /> Scoring</> : "Score this applicant"}
        </button>
        <p className="section-note">
          Fields left unspecified stay missing rather than being guessed at. LightGBM handles
          missing values natively, so the score reflects only what was supplied. Age, gender and
          marital status are absent by design: see Fair lending in the README.
        </p>
      </div>

      <div>
        {error && <ErrorNotice error={error} />}
        {!result && !error && !busy && (
          <div className="notice">
            <h3>No score yet</h3>
            <p>Adjust the inputs and score to see a probability, a band, and the recommended action.</p>
          </div>
        )}
        {busy && <Skeleton height={150} />}
        {result && !error && !busy && <Verdict result={result} />}
      </div>
    </div>
  );
}

export default function RiskPrediction({ onScored }) {
  const [mode, setMode] = useState("existing");

  return (
    <div className="main-inner">
      <header className="page-head">
        <h1>Risk prediction</h1>
        <p>
          Probabilities are calibrated back from the imbalance weighting, so the number shown is a
          real probability of default rather than a ranking score.
        </p>
      </header>

      <div className="segmented" style={{ marginBottom: 28 }}>
        <button aria-pressed={mode === "existing"} onClick={() => setMode("existing")}>An existing applicant</button>
        <button aria-pressed={mode === "whatif"} onClick={() => setMode("whatif")}>A what-if applicant</button>
      </div>

      {mode === "existing" ? <ExistingApplicant onScored={onScored} /> : <WhatIf onScored={onScored} />}
    </div>
  );
}
