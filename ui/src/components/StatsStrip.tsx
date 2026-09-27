import type { BusStats } from "../types";
import type { BusState } from "../useBusStore";

interface Props {
  stats: BusState["stats"];
  detectorBackend: string | undefined;
  busStats: BusStats | null;
}

function formatValue(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(1);
  return String(v);
}

/** Generic: every lane publishes its own `Stats(source, values)` (fps, drops, backend, ...) on
 * the `stats` topic under whatever keys make sense for it, so this renders whatever has arrived
 * rather than assuming fixed source names from lanes that haven't shipped yet. */
export function StatsStrip({ stats, detectorBackend, busStats }: Props) {
  const sources = Object.entries(stats);
  const wsSub = busStats?.subscribers.find((s) => s.name === "ui-ws");

  return (
    <div className="panel area-stats stats-strip">
      {detectorBackend && (
        <div className="stat-item">
          <span className="k">detector backend</span>
          <span className="v">{detectorBackend}</span>
        </div>
      )}
      {sources.length === 0 && !detectorBackend && (
        <span style={{ color: "var(--text-dim)" }}>No stage stats published yet.</span>
      )}
      {sources.map(([source, { values }]) => (
        <div className="stat-item" key={source}>
          <span className="k">{source}</span>
          <span className="v">
            {Object.entries(values)
              .slice(0, 3)
              .map(([k, v]) => `${k}=${formatValue(v)}`)
              .join(" ")}
          </span>
        </div>
      ))}
      {wsSub && (
        <div className="stat-item">
          <span className="k">ws dropped / delivered</span>
          <span className="v">
            {wsSub.dropped} / {wsSub.delivered}
          </span>
        </div>
      )}
    </div>
  );
}
