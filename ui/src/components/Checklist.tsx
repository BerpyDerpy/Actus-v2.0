import type { ProtocolStep } from "../types";
import type { BusState } from "../useBusStore";

interface Props {
  steps: ProtocolStep[];
  evidence: BusState["evidence"];
  events: BusState["events"];
}

function formatDuration(tStart: number | null, tEnd: number | null): string {
  if (tStart === null || tEnd === null) return "–";
  return `${Math.max(0, tEnd - tStart).toFixed(1)}s`;
}

export function Checklist({ steps, evidence, events }: Props) {
  return (
    <div className="panel area-checklist checklist">
      <div className="panel-title">Procedure checklist</div>
      {steps.map((step) => {
        const event = events[step.id];
        const status = event?.status ?? "pending";
        const pComplete = evidence[step.id]?.p_complete ?? 0;
        return (
          <div className="step-row" key={step.id}>
            <span className="step-id">{step.id}</span>
            <div className="step-main">
              <span className="step-text" title={step.text}>
                {step.text}
              </span>
              <div className="conf-bar-track">
                <div
                  className="conf-bar-fill"
                  style={{ width: `${Math.round(pComplete * 100)}%` }}
                />
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
              <span className={`chip ${status}`}>{status.replace(/_/g, " ")}</span>
              <span className="duration">
                {event ? formatDuration(event.t_start, event.t_end) : "–"}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
