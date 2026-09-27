import type {
  DetectionSet,
  PoseSet,
  ProtocolSummary,
  RackConfig,
} from "../types";

interface Props {
  rack: RackConfig | null;
  protocol: ProtocolSummary | null;
  detections: DetectionSet | null;
  poses: PoseSet | null;
}

const PX_PER_M = 260;
const DOT_COLORS: Record<string, string> = {
  red_box: "#ff4d5e",
  yellow_box: "#ffd23f",
  outer_box_open: "#4d9dff",
  outer_box_closed: "#4d9dff",
  lid: "#8899b8",
};

/** Top-down view in the rack frame (metres, origin top-left, +x right, +y down — the same
 * orientation as SVG), so this reads the same whether the camera itself is upright, rolled 90°
 * or rolled 180°: object and torso positions are already rack-relative, not camera-relative. */
export function RackMiniMap({ rack, protocol, detections, poses }: Props) {
  const width = rack?.board?.width ?? 0.9;
  const height = rack?.board?.height ?? 0.8;
  const w = width * PX_PER_M;
  const h = height * PX_PER_M;
  const pad = 24;

  const torso = poses?.poses?.[0]?.torso_vec_rack ?? null;
  const anchor: [number, number] = [width / 2, height + 0.12];

  return (
    <div className="panel area-map">
      <div className="panel-title">
        Rack mini-map
        {rack?.available && rack.measured === false && (
          <span className="unmeasured-tag">UNMEASURED PLACEHOLDER</span>
        )}
      </div>
      <div className="rack-map">
        <svg viewBox={`0 0 ${w + pad * 2} ${h + pad * 2 + 60}`}>
          <g transform={`translate(${pad},${pad})`}>
            <rect
              x={0}
              y={0}
              width={w}
              height={h}
              fill="#171f2e"
              stroke="#24304a"
              rx={4}
            />
            {protocol?.zones &&
              Object.entries(protocol.zones).map(([name, pts]) => {
                const xs = pts.map((p) => p[0] * PX_PER_M);
                const ys = pts.map((p) => p[1] * PX_PER_M);
                const x0 = Math.min(...xs);
                const y0 = Math.min(...ys);
                const zw = Math.max(...xs) - x0;
                const zh = Math.max(...ys) - y0;
                return (
                  <g key={name}>
                    <rect
                      x={x0}
                      y={y0}
                      width={zw}
                      height={zh}
                      fill="none"
                      stroke="#3a7d5c"
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                    />
                    <text x={x0 + 4} y={y0 + 12} fill="#3a7d5c" fontSize={10}>
                      Zone {name}
                    </text>
                  </g>
                );
              })}
            {rack?.outer_box_footprint && (
              <rect
                x={rack.outer_box_footprint[0][0] * PX_PER_M}
                y={rack.outer_box_footprint[0][1] * PX_PER_M}
                width={
                  (rack.outer_box_footprint[1][0] -
                    rack.outer_box_footprint[0][0]) *
                  PX_PER_M
                }
                height={
                  (rack.outer_box_footprint[1][1] -
                    rack.outer_box_footprint[0][1]) *
                  PX_PER_M
                }
                fill="#24304a"
                opacity={0.6}
              />
            )}
            {detections?.detections
              .filter((d) => d.bbox_rack_xy)
              .map((d, i) => {
                const [x, y] = d.bbox_rack_xy as [number, number];
                return (
                  <g key={`${d.track_id ?? i}`}>
                    <circle
                      cx={x * PX_PER_M}
                      cy={y * PX_PER_M}
                      r={7}
                      fill={DOT_COLORS[d.cls] ?? "#c8d3e6"}
                      stroke="#0a0d13"
                      strokeWidth={1.5}
                    />
                    <text
                      x={x * PX_PER_M + 9}
                      y={y * PX_PER_M + 3}
                      fill="#c8d3e6"
                      fontSize={9}
                    >
                      {d.cls}
                    </text>
                  </g>
                );
              })}
            {torso && (
              <g>
                <circle
                  cx={anchor[0] * PX_PER_M}
                  cy={anchor[1] * PX_PER_M}
                  r={5}
                  fill="#c8d3e6"
                />
                <line
                  x1={anchor[0] * PX_PER_M}
                  y1={anchor[1] * PX_PER_M}
                  x2={anchor[0] * PX_PER_M + torso[0] * 40}
                  y2={anchor[1] * PX_PER_M + torso[1] * 40}
                  stroke="var(--accent)"
                  strokeWidth={2}
                  markerEnd="url(#arrowhead)"
                />
              </g>
            )}
            <defs>
              <marker
                id="arrowhead"
                markerWidth={6}
                markerHeight={6}
                refX={5}
                refY={3}
                orient="auto"
              >
                <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent)" />
              </marker>
            </defs>
          </g>
        </svg>
      </div>
    </div>
  );
}
