import { useEffect, useState } from "react";
import { api } from "./api";
import { AlertBanner } from "./components/AlertBanner";
import { AskCard } from "./components/AskCard";
import { Checklist } from "./components/Checklist";
import { DemoControls } from "./components/DemoControls";
import { EventLogTail } from "./components/EventLogTail";
import { LivePane } from "./components/LivePane";
import { NextStepCard } from "./components/NextStepCard";
import { RackMiniMap } from "./components/RackMiniMap";
import { StatsStrip } from "./components/StatsStrip";
import { ValidationTab } from "./components/ValidationTab";
import type { BusStats, ProtocolSummary, RackConfig } from "./types";
import { useBusStore } from "./useBusStore";

type Tab = "live" | "validation";

export function App() {
  const { state, sendAnswer } = useBusStore();
  const [tab, setTab] = useState<Tab>("live");
  const [protocol, setProtocol] = useState<ProtocolSummary | null>(null);
  const [rack, setRack] = useState<RackConfig | null>(null);
  const [busStats, setBusStats] = useState<BusStats | null>(null);

  useEffect(() => {
    api.protocol().then(setProtocol);
    api.rack().then(setRack);
  }, []);

  useEffect(() => {
    const poll = () => api.busStats().then(setBusStats).catch(() => {});
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-title">
          <span className="accent">ACTUS</span> mission console
        </span>
        <span className="protocol-title">{protocol?.available ? protocol.title : "…"}</span>
        <DemoControls />
        <span>
          <span className={`conn-dot ${state.connected ? "on" : "off"}`} />
          {state.connected ? "connected" : "disconnected"}
        </span>
        <div className="tabs">
          <button className={`tab ${tab === "live" ? "active" : ""}`} onClick={() => setTab("live")}>
            Live
          </button>
          <button
            className={`tab ${tab === "validation" ? "active" : ""}`}
            onClick={() => setTab("validation")}
          >
            Validation
          </button>
        </div>
      </header>

      <div className="overlay-stack">
        <AlertBanner alerts={state.alerts} />
        <AskCard ask={state.ask} onAnswer={sendAnswer} />
      </div>

      {tab === "live" ? (
        <div className="dashboard">
          <div className="panel area-live">
            <div className="panel-title">Live feed</div>
            <LivePane predicates={state.predicates} />
          </div>
          <div className="area-next">
            <NextStepCard prompt={state.prompt} />
          </div>
          <Checklist steps={protocol?.steps ?? []} evidence={state.evidence} events={state.events} />
          <RackMiniMap
            rack={rack}
            protocol={protocol}
            detections={state.detections}
            poses={state.poses}
          />
          <EventLogTail log={state.log} />
          <StatsStrip
            stats={state.stats}
            detectorBackend={state.detections?.backend}
            busStats={busStats}
          />
        </div>
      ) : (
        <ValidationTab />
      )}
    </div>
  );
}
