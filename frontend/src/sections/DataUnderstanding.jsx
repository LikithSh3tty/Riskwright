import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, num, pct } from "../lib/api";
import { Disclosure, Figure, FigureSkeleton, Resource, Skeleton, useResource } from "../components/Primitives";

const AXIS = { stroke: "#6b6c75", fontSize: 11, tickLine: false };
const GRID = "#e2ded6";

function ChartTip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#fff", border: "1px solid #cfcac0", borderRadius: 6,
      padding: "8px 10px", boxShadow: "0 4px 12px rgba(38,34,28,.10)", fontSize: 12,
    }}>
      <div style={{ color: "#45464d", marginBottom: 4 }}>{label}{unit}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color, fontVariantNumeric: "tabular-nums" }}>
          {p.name}: {typeof p.value === "number" && p.value < 1 && p.name === "Default rate"
            ? pct(p.value, 2)
            : num(p.value)}
        </div>
      ))}
    </div>
  );
}

function InsightEvidence({ evidence }) {
  const key = ["bands", "groups", "quartiles"].find((k) => k in (evidence ?? {}));
  if (!key) return null;
  const rows = evidence[key];
  const labelKey = Object.keys(rows[0]).find((k) => k !== "default_rate" && k !== "count");

  return (
    <div className="chart-scroll" style={{ height: 210, marginTop: 16 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={labelKey} {...AXIS} interval={0}
                 tickFormatter={(v) => String(v).length > 16 ? `${String(v).slice(0, 15)}...` : v} />
          <YAxis {...AXIS} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} width={44} />
          <Tooltip content={<ChartTip unit="" />} cursor={{ fill: "rgba(38,34,28,.04)" }} />
          <Bar dataKey="default_rate" name="Default rate" radius={[3, 3, 0, 0]} maxBarSize={54}>
            {rows.map((_, i) => <Cell key={i} fill="#4c78a8" />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Distribution({ columns }) {
  const [column, setColumn] = useState(columns.includes("age_years") ? "age_years" : columns[0]);
  const [bins, setBins] = useState(30);
  const dist = useResource(() => api.edaDistribution(column, bins), [column, bins]);

  return (
    <>
      <div style={{ display: "flex", gap: 24, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 20 }}>
        <div className="field" style={{ minWidth: 240 }}>
          <label htmlFor="dist-col">Column</label>
          <select id="dist-col" className="control" value={column} onChange={(e) => setColumn(e.target.value)}>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="field" style={{ minWidth: 200 }}>
          <label htmlFor="dist-bins">Bins: {bins}</label>
          <input id="dist-bins" className="control" type="range" min="10" max="60" step="5"
                 value={bins} onChange={(e) => setBins(Number(e.target.value))} />
        </div>
      </div>

      <Resource state={dist} skeleton={<Skeleton height={340} />}>
        {(data) => (
          <>
            <div className="chart-scroll" style={{ height: 340 }}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={data.bins} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="bin_start" {...AXIS}
                         tickFormatter={(v) => Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : Number(v).toFixed(1)} />
                  <YAxis yAxisId="n" {...AXIS} width={46}
                         tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v} />
                  <YAxis yAxisId="rate" orientation="right" {...AXIS} width={46}
                         tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip content={<ChartTip unit="" />} cursor={{ fill: "rgba(38,34,28,.04)" }} />
                  <Bar yAxisId="n" dataKey="repaid" name="Repaid" stackId="a" fill="#4c78a8" maxBarSize={30} />
                  <Bar yAxisId="n" dataKey="defaulted" name="Defaulted" stackId="a" fill="#a03535" maxBarSize={30} />
                  <Line yAxisId="rate" type="monotone" dataKey="default_rate" name="Default rate"
                        stroke="#16161a" strokeWidth={1.5} strokeDasharray="3 3" dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <p className="section-note">
              Bars are applicant counts, split by outcome. The dotted line is the default rate
              within each bin, read against the right axis. Range {data.note}.
            </p>
          </>
        )}
      </Resource>
    </>
  );
}

function ByCategory({ dimensions }) {
  const [dimension, setDimension] = useState(
    dimensions.includes("name_education_type") ? "name_education_type" : dimensions[0]
  );
  const grouped = useResource(() => api.edaGroupby(dimension), [dimension]);

  return (
    <>
      <div className="field" style={{ maxWidth: 320, marginBottom: 20 }}>
        <label htmlFor="dim">Dimension</label>
        <select id="dim" className="control" value={dimension} onChange={(e) => setDimension(e.target.value)}>
          {dimensions.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
      </div>

      <Resource state={grouped} skeleton={<Skeleton height={300} />}>
        {(data) => (
          <>
            <div className="chart-scroll" style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.groups} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="value" {...AXIS} interval={0}
                         tickFormatter={(v) => String(v).length > 14 ? `${String(v).slice(0, 13)}...` : v} />
                  <YAxis {...AXIS} width={46} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip content={<ChartTip unit="" />} cursor={{ fill: "rgba(38,34,28,.04)" }} />
                  <ReferenceLine y={data.base_default_rate} stroke="#16161a" strokeDasharray="4 3"
                                 label={{ value: "portfolio average", position: "right", fill: "#6b6c75", fontSize: 11 }} />
                  <Bar dataKey="default_rate" name="Default rate" radius={[3, 3, 0, 0]} maxBarSize={64}>
                    {data.groups.map((g, i) => (
                      <Cell key={i} fill={g.default_rate >= data.base_default_rate ? "#a03535" : "#4c78a8"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="section-note">
              Red sits above the {pct(data.base_default_rate, 2)} portfolio average, blue below it.
            </p>
          </>
        )}
      </Resource>
    </>
  );
}

export default function DataUnderstanding() {
  const summary = useResource(() => api.edaSummary(), []);
  const insights = useResource(() => api.edaInsights(), []);
  const columns = useResource(() => api.edaColumns(), []);

  return (
    <div className="main-inner">
      <header className="page-head">
        <h1>Data understanding</h1>
        <p>
          Every figure below is computed by the API from Postgres. The same module backs the
          notebook, so the numbers here and in the analysis cannot drift apart.
        </p>
      </header>

      <Resource state={summary} skeleton={<FigureSkeleton />}>
        {(data) => (
          <>
            <div className="figures">
              <Figure label="Applications" value={num(data.rows)} />
              <Figure label="Columns" value={num(data.columns)} />
              <Figure label="Default rate" value={pct(data.default_rate, 2)} />
              <Figure label="Imbalance" value={`${data.imbalance_ratio.toFixed(1)} to 1`} />
            </div>
            <p className="section-note">
              {num(data.defaulted)} defaulted, {num(data.repaid)} repaid.{" "}
              {data.columns_with_missing} of {data.columns} columns contain missing values.
            </p>
          </>
        )}
      </Resource>

      <section className="section">
        <h2>What the data says</h2>
        <Resource state={insights} skeleton={<Skeleton height={220} />}>
          {(data) => (
            <div>
              {data.insights.map((insight, i) => (
                <div className="insight" key={insight.title}>
                  <Disclosure summary={insight.title} open={i === 0}>
                    <p className="insight-finding">{insight.finding}</p>
                    <p className="insight-sowhat"><b>So what:</b> {insight.so_what}</p>
                    <InsightEvidence evidence={insight.evidence} />
                  </Disclosure>
                </div>
              ))}
            </div>
          )}
        </Resource>
      </section>

      <Resource state={columns} skeleton={<Skeleton height={300} style={{ marginTop: 48 }} />}>
        {(cols) => (
          <>
            <section className="section">
              <h2>Distribution explorer</h2>
              <Distribution columns={cols.numeric} />
            </section>
            <section className="section">
              <h2>Default rate by category</h2>
              <ByCategory dimensions={cols.categorical} />
            </section>
          </>
        )}
      </Resource>
    </div>
  );
}
