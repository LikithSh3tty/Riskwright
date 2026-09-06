import { useEffect, useRef, useState } from "react";
import { ArrowUp, Question, Prohibit, Table } from "@phosphor-icons/react";
import { api, ApiError, num } from "../lib/api";
import { Disclosure, Spinner } from "../components/Primitives";

const SUGGESTIONS = [
  "What is the overall default rate?",
  "Which education level has the highest default rate?",
  "What is the average credit score of applicants?",
  "How many applicants have a prior refused application?",
];

/* The session id lives here and the transcript lives on the server, keyed by
 * it. That is the contract the /chat endpoint returns, and it is why
 * swapping the frontend does not mean reimplementing conversation memory. */
function newSessionId() {
  return `react-${Math.random().toString(36).slice(2, 14)}`;
}

function ResultTable({ columns, rows }) {
  if (!rows?.length) return null;
  const shown = rows.slice(0, 50);
  return (
    <div className="panel panel-flush">
      <div className="table-wrap" style={{ maxHeight: 340 }}>
        <table className="data">
          <thead>
            <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {shown.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} className={typeof cell === "number" ? "right mono" : ""}>
                    {cell === null ? <span style={{ color: "var(--ink-3)" }}>null</span>
                      : typeof cell === "number" ? num(cell, Number.isInteger(cell) ? 0 : 4)
                      : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > shown.length && (
        <p className="section-note" style={{ padding: "0 1rem 0.75rem" }}>
          Showing {shown.length} of {rows.length} rows.
        </p>
      )}
    </div>
  );
}

function Answer({ turn }) {
  if (turn.action === "sql") {
    return (
      <div className="turn-body">
        <p className="turn-said">{turn.answer}</p>
        <ResultTable columns={turn.columns} rows={turn.rows} />
        {turn.sql && (
          <Disclosure summary={<><Table size={14} weight="bold" style={{ verticalAlign: "-2px" }} /> Generated SQL</>}>
            <pre className="sql">{turn.sql}</pre>
          </Disclosure>
        )}
      </div>
    );
  }

  /* Refusals and clarifications get a distinct surface. They are correct
   * answers to questions the data cannot support, not failures, and reading
   * one as an answer would be the failure. */
  const isRefusal = turn.action === "refuse";
  return (
    <div className="turn-body">
      <div className="refusal">
        <div className="refusal-head">
          {isRefusal ? <Prohibit size={15} weight="bold" /> : <Question size={15} weight="bold" />}
          {isRefusal ? "Cannot answer this" : "Needs clarification"}
        </div>
        <p>{turn.reason}</p>
        {turn.suggestion && (
          <p className="suggest"><b>Try instead:</b> {turn.suggestion}</p>
        )}
      </div>
    </div>
  );
}

export default function AskTheData() {
  const [sessionId, setSessionId] = useState(newSessionId);
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, busy]);

  const send = async (text) => {
    const question = (text ?? draft).trim();
    if (!question || busy) return;
    setDraft("");
    setError(null);
    setTurns((t) => [...t, { role: "user", question }]);
    setBusy(true);
    try {
      const result = await api.chat({ question, session_id: sessionId });
      setTurns((t) => [...t, { role: "assistant", ...result }]);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    try { await api.clearChat(sessionId); } catch { /* the session may not exist yet */ }
    setSessionId(newSessionId());
    setTurns([]);
    setError(null);
  };

  return (
    <div className="main-inner">
      <header className="page-head">
        <h1>Ask the data</h1>
        <p>
          Questions become SQL, which is parsed and checked against the live schema before it runs.
          Questions the data cannot answer are declined rather than guessed at, and the generated
          SQL is always shown.
        </p>
      </header>

      {turns.length === 0 && !busy && (
        <div className="notice">
          <h3>Ask a question about the portfolio</h3>
          <p>
            Follow-ups keep their context: ask about one education level, then simply ask
            "and for the next one down?".
          </p>
          <div className="prompt-chips">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="chip" onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        </div>
      )}

      <div className="chat" style={{ marginTop: turns.length ? 0 : 24 }}>
        {turns.map((turn, i) => (
          <div className="turn" key={i}>
            <div className="turn-who">{turn.role === "user" ? "You" : "Riskwright"}</div>
            {turn.role === "user"
              ? <div className="turn-body"><p className="turn-said">{turn.question}</p></div>
              : <Answer turn={turn} />}
          </div>
        ))}

        {busy && (
          <div className="turn">
            <div className="turn-who">Riskwright</div>
            <div className="turn-body" style={{ color: "var(--ink-3)", fontSize: "var(--t-small)",
                                                display: "flex", alignItems: "center", gap: 8 }}>
              <Spinner /> Writing and checking SQL
            </div>
          </div>
        )}

        {error && (
          <div className="notice is-error" role="alert">
            <h3>The request failed</h3>
            <p>{error.message}</p>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <textarea
          className="control"
          rows={1}
          placeholder="How does default rate vary by education level?"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
          }}
          aria-label="Ask a question about the data"
        />
        <button className="btn" onClick={() => send()} disabled={busy || !draft.trim()}
                aria-label="Send question">
          {busy ? <Spinner /> : <ArrowUp size={15} weight="bold" />}
        </button>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
                    marginTop: 12, gap: 16, flexWrap: "wrap" }}>
        <span className="section-note" style={{ margin: 0 }}>
          Session {sessionId.slice(-8)}. Conversation memory is held by the API, not by this page.
        </span>
        <button className="btn btn-quiet" onClick={clear} disabled={busy || turns.length === 0}>
          Clear conversation
        </button>
      </div>
    </div>
  );
}
