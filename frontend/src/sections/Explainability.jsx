import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, num, pct } from "../lib/api";
import { Band, Disclosure, Empty, Resource, Skeleton, useResource } from "../components/Primitives";

const UP = "#a03535";
const DOWN = "#4c78a8";
const AXIS = { stroke: "#6b6c75", fontSize: 11, tickLine: false };

function ContributionTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div style={{
      background: "#fff", border: "1px solid #cfcac0", borderRadius: 6, padding: "9px 11px",
      boxShadow: "0 4px 12px rgba(38,34,28,.10)", fontSize: 12, maxWidth: 280,
    }}>
      <div style={{ fontWeight: 550, marginBottom: 3 }}>{row.label}</div>
      <div style={{ color: "#45464d" }}>
        Value: <span style={{ fontVariantNumeric: "tabular-nums" }}>
          {row.value === null ? "missing" : typeof row.value === "number" ? num(row.value, 4) : row.value}
        </span>
      </div>
      <div style={{ color: row.shap_value > 0 ? UP : DOWN, fontVariantNumeric: "tabular-nums" }}>
        {row.shap_value > 0 ? "Raises risk" : "Lowers risk"} by {Math.abs(row.shap_value).toFixed(4)}
      </div>
    </div>
  );
}

/* The contract is {feature, value, shap_value} exactly as the API returns it.
 * Sorted ascending so the strongest downward push sits at the bottom and the
 * strongest upward push at the top, which is how a waterfall is read. */
function Contributions({ rows, baseValue }) {
  const data = [...rows].sort((a, b) => a.shap_value - b.shap_value);
  const height = Math.max(320, data.length * 30 + 60);

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 60, bottom: 24, left: 8 }}>
          <CartesianGrid stroke="#e2ded6" horizontal={false} />
          <XAxis type="number" {...AXIS}
                 label={{ value: "Pushes toward default", position: "insideBottom", offset: -12,
                          fill: "#6b6c75", fontSize: 11 }} />
          <YAxis type="category" dataKey="label" {...AXIS} width={188} interval={0} />
          <Tooltip content={<ContributionTip />} cursor={{ fill: "rgba(38,34,28,.04)" }} />
          <ReferenceLine x={0} stroke="#16161a" strokeWidth={1}
                         label={{ value: `base ${baseValue.toFixed(3)}`, position: "top",
                                  fill: "#6b6c75", fontSize: 10 }} />
          <Bar dataKey="shap_value" radius={[2, 2, 2, 2]} maxBarSize={18}>
            {data.map((row, i) => (
              <Cell key={i} fill={row.shap_value > 0 ? UP : DOWN} />
            ))}
            <LabelList dataKey="shap_value" position="right"
                       formatter={(v) => v.toFixed(3)}
                       style={{ fill: "#6b6c75", fontSize: 10, fontVariantNumeric: "tabular-nums" }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function GlobalImportance() {
  const state = useResource(() => api.explainGlobal(), []);
  return (
    <Resource state={state} skeleton={<Skeleton height={320} />}>
      {(data) => {
        const rows = data.features.slice(0, 14).reverse();
        return (
          <>
            <div style={{ height: Math.max(320, rows.length * 26 + 40) }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 8 }}>
                  <CartesianGrid stroke="#e2ded6" horizontal={false} />
                  <XAxis type="number" {...AXIS} />
                  <YAxis type="category" dataKey="label" {...AXIS} width={188} interval={0} />
                  <Tooltip cursor={{ fill: "rgba(38,34,28,.04)" }}
                           formatter={(v) => [v.toFixed(4), "Mean |SHAP|"]} />
                  <Bar dataKey="mean_abs_shap" fill="#4c78a8" radius={[2, 2, 2, 2]} maxBarSize={16} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="section-note">
              Mean absolute SHAP over a sample of {num(data.sampled_rows)} of {num(data.total_rows)}{" "}
              applicants. Sampled rather than exhaustive, which is a stated limitation.
            </p>
          </>
        );
      }}
    </Resource>
  );
}

export default function Explainability({ scored }) {
  const state = useResource(
    () => (scored ? api.explain(scored) : Promise.resolve(null)),
    [JSON.stringify(scored ?? null)]
  );

  if (!scored) {
    return (
      <div className="main-inner">
        <header className="page-head">
          <h1>Why this decision</h1>
        </header>
        <Empty title="Nothing scored yet">
          <p>Score an applicant on the Risk prediction page and the explanation appears here.</p>
        </Empty>
      </div>
    );
  }

  return (
    <div className="main-inner">
      <header className="page-head">
        <h1>Why this decision</h1>
        <p>
          SHAP values from a TreeExplainer, exact on tree models. The API returns them as JSON and
          this page draws them, so the explanation is not a picture rendered on the server.
        </p>
      </header>

      <Resource state={state} skeleton={<Skeleton height={420} />}>
        {(data) => data && (
          <>
            <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 12 }}>
              <Band band={data.risk_band} />
              <span style={{ fontSize: "1.5rem", fontWeight: 550, fontVariantNumeric: "tabular-nums" }}>
                {pct(data.default_probability, 2)}
              </span>
            </div>

            <p style={{ maxWidth: "68ch", color: "var(--ink)" }}>{data.narrative}</p>

            <section className="section">
              <h2>What moved this score</h2>
              <Contributions rows={data.contributions} baseValue={data.base_value} />
              <p className="section-note">{data.note}</p>
            </section>

            <section className="section">
              <Disclosure summary="Which features matter across all applicants">
                <GlobalImportance />
              </Disclosure>
            </section>
          </>
        )}
      </Resource>
    </div>
  );
}
