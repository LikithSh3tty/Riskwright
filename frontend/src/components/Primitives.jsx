import { useCallback, useEffect, useRef, useState } from "react";
import { CaretRight, CircleNotch, Warning } from "@phosphor-icons/react";

/* Data fetching with the three states that actually occur, so no section has
 * to reimplement loading, failure, and success. */
export function useResource(loader, deps = []) {
  const [state, setState] = useState({ status: "loading", data: null, error: null });
  const alive = useRef(true);

  const run = useCallback(() => {
    setState((prev) => ({ ...prev, status: "loading" }));
    loader()
      .then((data) => alive.current && setState({ status: "ready", data, error: null }))
      .catch((error) => alive.current && setState({ status: "error", data: null, error }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    alive.current = true;
    run();
    return () => { alive.current = false; };
  }, [run]);

  return { ...state, reload: run };
}

/* Skeletons match the shape of what is arriving, rather than a spinner parked
 * in the middle of empty space. */
export function Skeleton({ height = 16, width = "100%", style }) {
  return <div className="skeleton" style={{ height, width, ...style }} aria-hidden="true" />;
}

export function FigureSkeleton({ count = 4 }) {
  return (
    <div className="figures">
      {Array.from({ length: count }, (_, i) => (
        <div key={i}>
          <Skeleton height={11} width="52%" style={{ marginBottom: 10 }} />
          <Skeleton height={26} width="74%" />
        </div>
      ))}
    </div>
  );
}

export function ErrorNotice({ error, onRetry }) {
  return (
    <div className="notice is-error" role="alert">
      <h3>
        <Warning size={16} weight="bold" style={{ verticalAlign: "-2px", marginRight: 6 }} />
        Could not load this
      </h3>
      <p>{error?.message ?? "Something went wrong."}</p>
      {onRetry && (
        <button className="btn btn-quiet" style={{ marginTop: 12 }} onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

/* Renders the loading and failure paths so a caller only writes the happy one. */
export function Resource({ state, skeleton, children }) {
  if (state.status === "loading") return skeleton ?? <Skeleton height={120} />;
  if (state.status === "error") return <ErrorNotice error={state.error} onRetry={state.reload} />;
  return children(state.data);
}

export function Figure({ label, value, mono = false, tone }) {
  return (
    <div>
      <div className="figure-label">{label}</div>
      <div className={`metric-value${mono ? " is-mono" : ""}`} style={tone ? { color: tone } : undefined}>
        {value}
      </div>
    </div>
  );
}

export function Band({ band }) {
  return <span className={`band band-${band}`}>{band}</span>;
}

export function Disclosure({ summary, children, open = false }) {
  return (
    <details className="disclosure" open={open}>
      <summary>
        <CaretRight size={13} weight="bold" className="chev" />
        {summary}
      </summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}

export function Spinner({ size = 15 }) {
  return <CircleNotch size={size} weight="bold" className="spin" />;
}

export function Empty({ title, children }) {
  return (
    <div className="notice">
      <h3>{title}</h3>
      {children}
    </div>
  );
}
