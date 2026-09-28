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

  // 404 (or a network error) when the pipeline runs without fault injection: no controls shown.
  async faults(): Promise<FaultsState | null> {
    try {
      const resp = await fetch("/api/faults");
      if (!resp.ok) return null;
      return (await resp.json()) as FaultsState;
    } catch {
      return null;
    }
  },
  async setFault(name: string, on?: boolean): Promise<boolean> {
    const query = on === undefined ? "" : `?on=${on}`;
    const resp = await fetch(`/api/faults/${encodeURIComponent(name)}${query}`, { method: "POST" });
    return resp.ok;
  },
};
