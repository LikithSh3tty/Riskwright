import { useState } from "react";
import { api, num, pct } from "../lib/api";
import { Figure, Resource, Skeleton, useResource } from "../components/Primitives";

const VARIANTS = [
  { key: "faithful", label: "Faithful to the model" },
  { key: "policy", label: "Policy usable" },
];

export default function DerivedRules() {
  const [variant, setVariant] = useState("faithful");
  const state = useResource(() => api.rules(variant), [variant]);

  return (
    <div className="main-inner">
      <header className="page-head">
        <h1>Derived credit rules</h1>
        <p>
          A depth-3 surrogate tree fitted to the model's behaviour, so its logic is short enough to
          sit in a policy document. These approximate the model. They are not the model.
        </p>
      </header>

      <div className="segmented" style={{ marginBottom: 28 }}>
        {VARIANTS.map((v) => (
          <button key={v.key} aria-pressed={variant === v.key} onClick={() => setVariant(v.key)}>
            {v.label}
          </button>
        ))}
      </div>

      <Resource state={state} skeleton={<Skeleton height={380} />}>
        {(data) => (
          <>
            <div className="figures">
              <Figure label="Agreement with the model" value={pct(data.surrogate_fidelity, 1)} />
              <Figure label="Base default rate" value={pct(data.base_default_rate, 2)} />
              <Figure label="Rules" value={data.rules.length} />
            </div>

            {variant === "policy" && (
              <p className="section-note" style={{ maxWidth: "72ch" }}>
                External bureau scores dominate the model, so a faithful surrogate keys off them
                almost entirely and tells a credit officer little they can act on. This set excludes
                those scores, trading agreement with the model for rules that apply to an applicant
                with no bureau history.
              </p>
            )}

            <section className="section">
              <div className="panel panel-flush">
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr>
                        <th style={{ width: 56 }}>Rule</th>
                        <th>Condition</th>
                        <th className="right" style={{ width: 110 }}>Applicants</th>
                        <th className="right" style={{ width: 104 }}>Default rate</th>
                        <th className="right" style={{ width: 84 }}>Lift</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.rules.map((rule) => (
                        <tr key={rule.rule_id}>
                          <td className="mono">{rule.rule_id}</td>
                          <td className="rule-cond">{rule.readable}</td>
                          <td className="right">
                            {rule.support_pct.toFixed(1)}%
                            <div style={{ color: "var(--ink-3)", fontSize: "0.75rem" }}>
                              {num(rule.support_n)}
                            </div>
                          </td>
                          <td className="right">{pct(rule.default_rate, 1)}</td>
                          <td className={`right lift ${rule.lift >= 1 ? "is-up" : "is-down"}`}>
                            {rule.lift.toFixed(2)}x
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <p className="section-note">
                Lift is the leaf default rate over the base rate. The leaves partition the
                population, so the shares sum to 100%. Leaf statistics are counted from the data
                rather than read off the tree's internal values, which are reweighted and would
                report a default rate several times the true one.
              </p>
            </section>

            {data.features_excluded?.length > 0 && (
              <p className="section-note">
                Excluded from this derivation as non-numeric: {data.features_excluded.join(", ")}.
              </p>
            )}
          </>
        )}
      </Resource>
    </div>
  );
}
