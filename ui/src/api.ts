import type {
  BusStats,
  FaultsState,
  ProtocolSummary,
  RackConfig,
  ReportSummary,
  ScenarioInfo,
} from "./types";

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  return (await resp.json()) as T;
}

export const api = {
  protocol: () => getJson<ProtocolSummary>("/api/protocol"),
  rack: () => getJson<RackConfig>("/api/rack"),
  report: () => getJson<ReportSummary>("/api/report"),
  busStats: () => getJson<BusStats>("/api/bus_stats"),

  // These 404 when the server was built with enable_demo=False (a real pipeline owns the bus);
  // callers treat that as "no demo controls available", not an error.
  async scenarios(): Promise<ScenarioInfo[] | null> {
    const resp = await fetch("/api/scenarios");
    if (!resp.ok) return null;
    return (await resp.json()) as ScenarioInfo[];
  },
  async demoStart(scenario: string): Promise<boolean> {
    const resp = await fetch(`/api/demo/start?scenario=${encodeURIComponent(scenario)}`, {
      method: "POST",
    });
    return resp.ok;
  },
  async demoStop(): Promise<boolean> {
    const resp = await fetch("/api/demo/stop", { method: "POST" });
    return resp.ok;
  },

  // "absent" on a 404 (the pipeline runs without fault injection; that never changes while the
  // server is up), null on a network error (try again later): no controls shown either way.
  async faults(): Promise<FaultsState | "absent" | null> {
    try {
      const resp = await fetch("/api/faults");
      if (resp.status === 404) return "absent";
      if (!resp.ok) return null;
      return (await resp.json()) as FaultsState;
    } catch {
      return null;
    }
  },
  // Never rejects: a network error or an HTTP error comes back as { ok: false, error }.
  async setFault(name: string, on: boolean): Promise<{ ok: boolean; error?: string }> {
    try {
      const resp = await fetch(`/api/faults/${encodeURIComponent(name)}?on=${on}`, {
        method: "POST",
      });
      if (resp.ok) return { ok: true };
      let detail = `HTTP ${resp.status}`;
      try {
        const body = (await resp.json()) as { detail?: unknown };
        if (body.detail !== undefined) detail = String(body.detail);
      } catch {
        // not JSON; keep the status line
      }
      return { ok: false, error: detail };
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  },
};
