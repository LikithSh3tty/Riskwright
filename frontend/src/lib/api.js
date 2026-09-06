/* The only route to data.
 *
 * Everything is reached at /api/*, proxied to the API service by nginx in the
 * container and by Vite in development. The frontend therefore never learns
 * the API's host, and no VITE_* variable is needed. That matters: Vite inlines
 * VITE_* values into the built bundle, so the Anthropic key stays in the api
 * service environment and is never referenced here.
 */

const BASE = "/api";
const DEFAULT_TIMEOUT = 30000;
export const CHAT_TIMEOUT = 90000; // two model calls plus a SQL execution

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, { method = "GET", body, timeout = DEFAULT_TIMEOUT } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  let response;
  try {
    response = await fetch(BASE + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timer);
    if (error.name === "AbortError") {
      throw new ApiError(`The request to ${path} timed out.`, 0);
    }
    throw new ApiError(`Could not reach the API at ${BASE}${path}.`, 0);
  }
  clearTimeout(timer);

  if (!response.ok) {
    let detail = `${method} ${path} returned ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) detail = typeof payload.detail === "string"
        ? payload.detail
        : JSON.stringify(payload.detail);
    } catch {
      /* response body was not JSON; the status line is all we have */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json();
}

export const get = (path) => request(path);
export const post = (path, body, timeout) => request(path, { method: "POST", body, timeout });
export const del = (path) => request(path, { method: "DELETE" });

/* Endpoints, exactly as the API already exposes them. Nothing here asks for a
 * contract the API does not already expose. */
export const api = {
  health: () => get("/health"),
  edaSummary: () => get("/eda/summary"),
  edaColumns: () => get("/eda/columns"),
  edaDistribution: (column, bins) => get(`/eda/distribution?column=${encodeURIComponent(column)}&bins=${bins}`),
  edaGroupby: (dimension) => get(`/eda/groupby?dimension=${encodeURIComponent(dimension)}`),
  edaInsights: () => get("/eda/insights"),
  applicants: (limit = 60) => get(`/applicants?limit=${limit}`),
  predict: (payload) => post("/predict", payload),
  explain: (payload) => post("/explain", payload),
  explainGlobal: () => get("/explain/global"),
  rules: (variant) => get(`/rules?variant=${variant}`),
  metrics: () => get("/model/metrics"),
  chat: (payload) => post("/chat", payload, CHAT_TIMEOUT),
  clearChat: (sessionId) => del(`/chat/${encodeURIComponent(sessionId)}`),
  chatSchema: () => get("/chat/schema"),
};

/* Formatting. Kept here so a percentage is written the same way everywhere. */
export const pct = (value, digits = 1) =>
  value == null || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(digits)}%`;

export const num = (value, digits = 0) =>
  value == null || Number.isNaN(value)
    ? "n/a"
    : Number(value).toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

export const compact = (value) =>
  value == null ? "n/a" : Number(value).toLocaleString("en-US", { notation: "compact", maximumFractionDigits: 1 });
