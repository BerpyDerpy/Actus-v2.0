import { useEffect, useState } from "react";
import { api } from "../api";
import type { ScenarioInfo } from "../types";

/** Feeds a P0 replay scenario onto the server's bus so the dashboard has something to show
 * without a live pipeline. Renders nothing once /api/scenarios 404s (a real pipeline owns the
 * bus and enable_demo=False was passed to create_app), so this never appears once T-INT wires
 * ACTUS up for real. */
export function DemoControls() {
  const [scenarios, setScenarios] = useState<ScenarioInfo[] | null>(null);
  const [selected, setSelected] = useState("correct");
  const [running, setRunning] = useState(false);

  useEffect(() => {
    api.scenarios().then((list) => {
      setScenarios(list);
      if (list && list.length > 0) setSelected((s) => (list.some((x) => x.name === s) ? s : list[0].name));
    });
  }, []);

  if (!scenarios) return null;

  return (
    <div className="demo-controls">
      <span className="simulated-tag">DEMO / SIMULATED</span>
      <select value={selected} onChange={(e) => setSelected(e.target.value)}>
        {scenarios.map((s) => (
          <option key={s.name} value={s.name} title={s.description}>
            {s.name}
          </option>
        ))}
      </select>
      {!running ? (
        <button
          onClick={async () => {
            if (await api.demoStart(selected)) setRunning(true);
          }}
        >
          Start replay
        </button>
      ) : (
        <button
          onClick={async () => {
            await api.demoStop();
            setRunning(false);
          }}
        >
          Stop
        </button>
      )}
    </div>
  );
}
