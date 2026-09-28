import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { FaultInfo, FaultsState } from "../types";
import type { BusState } from "../useBusStore";

interface Props {
  stats: BusState["stats"];
}

const POLL_MS = 1500;

/** What the pipeline's fault controller last reported on the `stats` topic. */
function liveFaults(stats: BusState["stats"]) {
  const v = stats["faults"]?.values;
  if (!v) return null;
  const active = typeof v.active === "string" && v.active !== "none" ? v.active.split(",") : [];
  const detected = typeof v.detected === "string" && v.detected !== "none" ? v.detected : null;
  return { injected: v.injected === true, active, detected };
}

/** "FAULT INJECTED" pill: shown whenever a fault is on, and "FAULT DETECTED" when the system's
 * own stale-frame guard has tripped. Renders nothing otherwise. */
export function FaultBanner({ stats }: Props) {
  const live = liveFaults(stats);
  if (!live || (!live.injected && !live.detected)) return null;
  return (
    <span className="fault-banner" role="alert">
      {live.injected && (
        <>
          <span className="fault-tag">FAULT INJECTED</span>
          <span>{live.active.map((n) => n.replace(/_/g, " ")).join(", ")}</span>
        </>
      )}
      {live.detected && (
        <>
          <span className="fault-tag detected">FAULT DETECTED</span>
          <span>{live.detected.replace(/_/g, " ")}</span>
        </>
      )}
    </span>
  );
}

/** Toggles for the injected faults, also on the keys shown on the buttons (ignored while typing
 * in a field). Renders nothing when the pipeline runs without fault injection (/api/faults 404s).
 * The expected safe behaviour is in each button's tooltip. */
export function FaultControls({ stats }: Props) {
  const [state, setState] = useState<FaultsState | null>(null);
  const listRef = useRef<FaultInfo[]>([]);
  listRef.current = state?.faults ?? [];
  const available = state !== null;

  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      api.faults().then((s) => {
        if (!cancelled) setState(s);
      });
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (!available) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName) || el.isContentEditable)) {
        return;
      }
      const fault = listRef.current.find((f) => f.key === e.key.toLowerCase());
      if (fault) void api.setFault(fault.name);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [available]);

  if (!state) return null;
  const live = liveFaults(stats);
  const active = new Set(live ? live.active : state.active);

  return (
    <div className="fault-controls">
      <span className="fault-label">faults</span>
      {state.faults.map((f) => (
        <button
          key={f.name}
          className={`fault-btn ${active.has(f.name) ? "on" : ""}`}
          title={`${f.summary} Expected: ${f.expected}${f.refused ? ` (refused: ${f.refused})` : ""}`}
          onClick={() => void api.setFault(f.name)}
        >
          <kbd>{f.key}</kbd> {f.name.replace(/_/g, " ")}
        </button>
      ))}
    </div>
  );
}
