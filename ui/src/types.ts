// Mirrors the JSON shape of `to_jsonable()` in actus/core/contracts.py. Kept intentionally loose
// (most numeric/string fields, no branded types) since the wire format is the source of truth,
// not this file — a lane changing a contract only has to touch Python; the UI degrades to
// showing "?" for a field it doesn't recognise rather than crashing on a strict schema mismatch.

export type StepStatus =
  | "pending"
  | "verified"
  | "operator_confirmed"
  | "skipped"
  | "out_of_order"
  | "timed_out";

export type AlertKind = "skip" | "out_of_order" | "unexpected" | "timeout" | "too_fast";

export interface Detection {
  type: "Detection";
  track_id: number | null;
  cls: string;
  conf: number;
  bbox_xyxy: [number, number, number, number];
  bbox_rack_xy: [number, number] | null;
}

export interface DetectionSet {
  type: "DetectionSet";
  frame_idx: number;
  t: number;
  detections: Detection[];
  backend: string;
}

export interface PoseObs {
  type: "PoseObs";
  torso_vec_rack: [number, number, number] | null;
  conf: number;
  rotation_used: number;
}

export interface PoseSet {
  type: "PoseSet";
  frame_idx: number;
  t: number;
  poses: PoseObs[];
}

export interface RackFrame {
  type: "RackFrame";
  valid: boolean;
  markers_seen: number[];
  reproj_err: number | null;
  frame_idx: number;
  t: number;
  age_frames: number;
}

export interface Predicates {
  type: "Predicates";
  frame_idx: number;
  t: number;
  values: Record<string, number>;
  visibility: number;
}

export interface StepEvidence {
  type: "StepEvidence";
  frame_idx: number;
  t: number;
  step_id: string;
  p_complete: number;
  sources: Record<string, number>;
}

export interface ActionScores {
  type: "ActionScores";
  frame_idx: number;
  t: number;
  scores: Record<string, number>;
}

export interface StepEvent {
  type: "StepEvent";
  step_id: string;
  status: StepStatus;
  t_start: number | null;
  t_end: number | null;
  confidence: number;
  evidence: Record<string, number>;
  frame_idx: number;
  t: number;
  retroactive: boolean;
}

export interface Alert {
  type: "Alert";
  kind: AlertKind;
  message: string;
  step_ids: string[];
  frame_idx: number;
  t: number;
}

export interface Prompt {
  type: "Prompt";
  next_step_id: string | null;
  text: string;
  frame_idx: number;
  t: number;
}

export interface Ask {
  type: "Ask";
  question_id: string;
  step_id: string;
  text: string;
  frame_idx: number;
  t: number;
}

export interface Stats {
  type: "Stats";
  source: string;
  values: Record<string, number | string | boolean | null>;
  t: number;
  frame_idx: number | null;
}

export interface AnswerMsg {
  type: "Answer";
  question_id: string;
  value: "yes" | "no";
  via: "voice" | "gui";
  t: number;
  frame_idx: number | null;
}

export interface CommandMsg {
  type: "Command";
  name: string;
  via: "voice" | "gui";
  t: number;
  frame_idx: number | null;
}

export type Payload =
  | DetectionSet
  | PoseSet
  | RackFrame
  | Predicates
  | StepEvidence
  | ActionScores
  | StepEvent
  | Alert
  | Prompt
  | Ask
  | Stats
  | AnswerMsg
  | CommandMsg
  | { type: string; [key: string]: unknown };

export interface Envelope<P = Payload> {
  topic: string;
  seq: number;
  t_pub: number;
  payload: P;
}

// -- REST payloads ----------------------------------------------------------------------------

export interface ProtocolStep {
  id: string;
  text: string;
  pre: Record<string, boolean>;
  post: Record<string, boolean>;
  requires_hoi: string | null;
  ask: string | null;
}

export interface ProtocolSummary {
  available: boolean;
  id?: string;
  title?: string;
  objects?: Record<string, { label: string; classes: string[] }>;
  zones?: Record<string, [number, number][]>;
  steps?: ProtocolStep[];
  order?: "strict" | [string, string][];
  timing?: { dwell_s: number; timeout_s: number; min_duration_s: Record<string, number> };
  decision?: { p_enter: number; p_exit: number; ask_band: [number, number]; ask_after_s: number };
}

export interface RackConfig {
  available: boolean;
  measured?: boolean;
  board?: { width: number; height: number };
  outer_box_footprint?: [number, number][];
  [key: string]: unknown;
}

export interface ScenarioInfo {
  name: string;
  description: string;
  duration_s: number;
}

export interface ReportSummary {
  available: boolean;
  [key: string]: unknown;
}

export interface BusStats {
  published: Record<string, number>;
  subscribers: Array<{
    name: string;
    topics: string[];
    depth: number;
    maxsize: number;
    policy: string;
    delivered: number;
    dropped: number;
  }>;
  dropped_total: number;
}

// -- fault injection (T-FAULT; GET /api/faults, 404 unless the pipeline runs with faults on) ------

export interface FaultInfo {
  name: string;
  label: string;
  key: string;
  summary: string;
  expected: string;
  active: boolean;
  refused: string | null;
}

export interface FaultsState {
  available: boolean;
  simulated: boolean;
  faults: FaultInfo[];
  active: string[];
  detected: string[];
  network: { state: string; error: string | null };
  guard: { enabled: boolean; stale_after: number | null };
}
