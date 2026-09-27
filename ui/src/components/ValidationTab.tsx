import { useEffect, useState } from "react";
import { api } from "../api";
import type { ReportSummary } from "../types";

function isPlainNumberOrString(v: unknown): v is number | string {
  return typeof v === "number" || typeof v === "string";
}

/** Renders runs/eval/report.json (T-EVAL) as-is: scalar metrics as cards, the first array field
 * (the take-level table) as a table. T-EVAL's exact schema isn't fixed yet, so this stays
 * generic rather than assuming field names that don't exist until that lane ships. */
export function ValidationTab() {
  const [report, setReport] = useState<ReportSummary | null>(null);

  useEffect(() => {
    api.report().then(setReport);
  }, []);

  if (!report) {
    return (
      <div className="validation-page">
        <div className="empty-state">Loading…</div>
      </div>
    );
  }

  if (!report.available) {
    return (
      <div className="validation-page">
        <div className="empty-state">
          No evaluation report yet — T-EVAL hasn't produced <code>runs/eval/report.json</code>{" "}
          on this machine.
        </div>
      </div>
    );
  }

  const scalarEntries = Object.entries(report).filter(
    ([k, v]) => k !== "available" && isPlainNumberOrString(v),
  ) as Array<[string, number | string]>;
  const tableEntries = Object.entries(report).filter(([, v]) => Array.isArray(v)) as Array<
    [string, Array<Record<string, unknown>>]
  >;

  return (
    <div className="validation-page">
      <div className="metric-grid">
        {scalarEntries.map(([k, v]) => (
          <div className="metric-card" key={k}>
            <div className="v">{typeof v === "number" ? v.toFixed(3) : v}</div>
            <div className="k">{k.replace(/_/g, " ")}</div>
          </div>
        ))}
      </div>
      {tableEntries.map(([name, rows]) => {
        const columns = rows.length > 0 ? Object.keys(rows[0]) : [];
        return (
          <div key={name} style={{ marginBottom: 24 }}>
            <div className="panel-title">{name.replace(/_/g, " ")}</div>
            <table className="validation-table">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c}>{String(row[c])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
      {scalarEntries.length === 0 && tableEntries.length === 0 && (
        <div className="empty-state">Report has no recognisable fields to render.</div>
      )}
    </div>
  );
}
