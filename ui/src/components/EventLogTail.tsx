import type { LogEntry } from "../useBusStore";

interface Props {
  log: LogEntry[];
}

export function EventLogTail({ log }: Props) {
  return (
    <div className="panel area-log event-log">
      <div className="panel-title">Event log</div>
      {log.length === 0 && <div style={{ color: "var(--text-dim)" }}>Nothing logged yet.</div>}
      {log.map((entry) => (
        <div className={`log-row ${entry.kind}`} key={entry.id}>
          <span className="kind">{entry.kind}</span>
          <span>{entry.text}</span>
        </div>
      ))}
    </div>
  );
}
