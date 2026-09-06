import { useCallback, useState } from "react";
import {
  ChartBar, ChatText, Gavel, Scales, TreeStructure,
} from "@phosphor-icons/react";
import { api } from "./lib/api";
import { useResource } from "./components/Primitives";
import DataUnderstanding from "./sections/DataUnderstanding";
import RiskPrediction from "./sections/RiskPrediction";
import Explainability from "./sections/Explainability";
import DerivedRules from "./sections/DerivedRules";
import AskTheData from "./sections/AskTheData";

const SECTIONS = [
  { key: "data", label: "Data understanding", Icon: ChartBar },
  { key: "predict", label: "Risk prediction", Icon: Scales },
  { key: "explain", label: "Why this decision", Icon: TreeStructure },
  { key: "rules", label: "Derived rules", Icon: Gavel },
  { key: "chat", label: "Ask the data", Icon: ChatText },
];

function StatusLine({ label, value, bad }) {
  return (
    <div className={`status-line${bad ? " is-bad" : ""}`}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function RailStatus() {
  const health = useResource(() => api.health(), []);
  if (health.status !== "ready") {
    return <StatusLine label="API" value={health.status === "error" ? "unreachable" : "checking"}
                       bad={health.status === "error"} />;
  }
  const h = health.data;
  return (
    <>
      <StatusLine label="Database" value={h.database.connected ? "connected" : "unavailable"}
                  bad={!h.database.connected} />
      <StatusLine label="Model" value={h.model_loaded ? "loaded" : "not trained"} bad={!h.model_loaded} />
      <StatusLine label="Assistant" value={h.llm_configured ? h.llm_model : "not configured"}
                  bad={!h.llm_configured} />
    </>
  );
}

export default function App() {
  const [active, setActive] = useState("data");
  // The scored applicant is held here so the explanation page can pick up
  // whatever prediction was last made, in either scoring mode.
  const [scored, setScored] = useState(null);

  const handleScored = useCallback((payload) => setScored(payload), []);

  return (
    <div className="app">
      <nav className="rail" aria-label="Sections">
        <div className="rail-mark">
          <strong>Riskwright</strong>
          <span>credit risk</span>
        </div>

        <div className="rail-nav">
          {SECTIONS.map(({ key, label, Icon }) => (
            <button
              key={key}
              className="rail-item"
              aria-current={active === key}
              onClick={() => setActive(key)}
            >
              <Icon size={16} weight={active === key ? "fill" : "regular"} />
              <span>{label}</span>
            </button>
          ))}
        </div>

        <div className="rail-foot">
          <RailStatus />
        </div>
      </nav>

      <main className="main">
        {active === "data" && <DataUnderstanding />}
        {active === "predict" && <RiskPrediction onScored={handleScored} />}
        {active === "explain" && <Explainability scored={scored} />}
        {active === "rules" && <DerivedRules />}
        {active === "chat" && <AskTheData />}
      </main>
    </div>
  );
}
