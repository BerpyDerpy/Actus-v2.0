import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { FaultInfo, FaultsState } from "../types";
import type { BusState } from "../useBusStore";

interface Props {
  stats: BusState["stats"];
}

const POLL_MS = 1500;

/** What the pipeline's fault controller last reported on the `stats` topic. The store drops
 * this entry when the WebSocket disconnects, so a finished run never leaves it lit. */
function liveFaults(stats: BusState["stats"]) {
  const v = stats["faults"]?.values;
  if (!v) return null;
  const active = typeof v.active === "string" && v.active !== "none" ? v.active.split(",") : [];
  const detected = typeof v.detected === "string" && v.detected !== "none" ? v.detected : null;
  return {
    injected: v.injected === true,
    active,
    detected,
    simulated: v.simulated === true,
    networkSimulated: v.network_simulated === true,
  };
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
          <span>
            {live.active
              .map((n) =>
                n === "kill_network" && live.networkSimulated
                  ? "kill network (SIMULATED target)"
                  : n.replace(/_/g, " "),
              )
              .join(", ")}
          </span>
        </>
      )}
      {(live.injected || live.detected) && live.simulated && (
        <span className="fault-tag detected">SIMULATED</span>
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
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<FaultInfo[]>([]);
  const activeRef = useRef<Set<string>>(new Set());
  listRef.current = state?.faults ?? [];
  const available = state !== null;

  useEffect(() => {
    let cancelled = false;
    let id: ReturnType<typeof setInterval> | undefined;
    const poll = () =>
      api.faults().then((s) => {
        if (cancelled) return;
        if (s === "absent") {
          // The routes are fixed when the server starts: a 404 now is a 404 for good.
          setState(null);
          if (id !== undefined) clearInterval(id);
          id = undefined;
          return;
        }
        setState(s);
      });
    void poll();
    id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      if (id !== undefined) clearInterval(id);
    };
  }, []);

  const live = liveFaults(stats);
  const active = new Set(live ? live.active : (state?.active ?? []));
  activeRef.current = active;

  // Always send the state the button shows the user wants, never a server-side "toggle".
  const set = (name: string, on: boolean) => {
    setError(null);
    void api.setFault(name, on).then((r) => {
      if (!r.ok) setError(`${name.replace(/_/g, " ")}: ${r.error}`);
    });
  };

  useEffect(() => {
    if (!available) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return; // a held key must not queue a burst of toggles
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName) || el.isContentEditable)) {
        return;
      }
      const fault = listRef.current.find((f) => f.key === e.key.toLowerCase());
      if (fault) set(fault.name, !activeRef.current.has(fault.name));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [available]);

  if (!state) return null;

  return (
    <div className="fault-controls">
      <span className="fault-label">faults{state.simulated ? " (SIMULATED)" : ""}</span>
      {state.faults.map((f) => (
        <button
          key={f.name}
          className={`fault-btn ${active.has(f.name) ? "on" : ""}`}
          title={`${f.summary} Expected: ${f.expected}${f.refused ? ` (refused: ${f.refused})` : ""}`}
          onClick={() => set(f.name, !active.has(f.name))}
        >
          <kbd>{f.key}</kbd> {f.name.replace(/_/g, " ")}
          {f.name === "kill_network" && state.network.target?.kind === "simulated"
            ? " (SIMULATED)"
            : ""}
        </button>
      ))}
      {error && (
        <span className="fault-tag detected" role="status">
          {error}
        </span>
      )}
    </div>
  );
}
