import { useEffect, useMemo, useReducer, useRef } from "react";
import type {
  ActionScores,
  Alert,
  AnswerMsg,
  Ask,
  CommandMsg,
  DetectionSet,
  Envelope,
  Payload,
  PoseSet,
  Predicates,
  Prompt,
  RackFrame,
  StepEvent,
  StepEvidence,
} from "./types";

const LOG_CAP = 80;
const ALERT_CAP = 6;
const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 8000;

export interface LogEntry {
  id: string;
  t: number; // pipeline clock, for display only
  receivedAt: number; // wall clock (Date.now()), for sorting / relative "Xs ago"
  kind: "event" | "alert" | "prompt" | "ask" | "answer" | "command";
  text: string;
}

export interface AlertEntry {
  alert: Alert;
  receivedAt: number;
}

export interface BusState {
  connected: boolean;
  predicates: Predicates | null;
  evidence: Record<string, StepEvidence>;
  actions: ActionScores | null;
  detections: DetectionSet | null;
  poses: PoseSet | null;
  rack: RackFrame | null;
  events: Record<string, StepEvent>;
  alerts: AlertEntry[];
  prompt: Prompt | null;
  ask: Ask | null;
  stats: Record<string, { values: Record<string, unknown>; t: number }>;
  log: LogEntry[];
}

const initialState: BusState = {
  connected: false,
  predicates: null,
  evidence: {},
  actions: null,
  detections: null,
  poses: null,
  rack: null,
  events: {},
  alerts: [],
  prompt: null,
  ask: null,
  stats: {},
  log: [],
};

type Action =
  | { kind: "connected" }
  | { kind: "disconnected" }
  | { kind: "envelope"; envelope: Envelope }
  | { kind: "sent_answer"; questionId: string; value: "yes" | "no" }
  | { kind: "sent_command"; name: string };

let logSeq = 0;
function pushLog(log: LogEntry[], entry: Omit<LogEntry, "id">): LogEntry[] {
  const next = [{ ...entry, id: `${Date.now()}-${logSeq++}` }, ...log];
  return next.length > LOG_CAP ? next.slice(0, LOG_CAP) : next;
}

function statusWord(status: string): string {
  return status.replace(/_/g, " ");
}

function reducer(state: BusState, action: Action): BusState {
  switch (action.kind) {
    case "connected":
      return { ...state, connected: true };
    case "disconnected":
      return { ...state, connected: false };
    case "sent_answer":
      // Dismiss the ask card immediately; the log entry comes from the server's own echo of
      // this Answer on the `answers` topic (below), so it isn't duplicated here.
      return {
        ...state,
        ask: state.ask?.question_id === action.questionId ? null : state.ask,
      };
    case "sent_command":
      return state;
    case "envelope":
      return applyEnvelope(state, action.envelope);
    default:
      return state;
  }
}

function applyEnvelope(state: BusState, env: Envelope): BusState {
  const receivedAt = Date.now();
  const payload = env.payload as Payload & { [k: string]: unknown };
  switch (env.topic) {
    case "predicates":
      return { ...state, predicates: payload as Predicates };
    case "evidence": {
      const ev = payload as StepEvidence;
      return { ...state, evidence: { ...state.evidence, [ev.step_id]: ev } };
    }
    case "actions":
      return { ...state, actions: payload as ActionScores };
    case "detections":
      return { ...state, detections: payload as DetectionSet };
    case "pose":
      return { ...state, poses: payload as PoseSet };
    case "rack":
      return { ...state, rack: payload as RackFrame };
    case "events": {
      const se = payload as StepEvent;
      return {
        ...state,
        events: { ...state.events, [se.step_id]: se },
        log: pushLog(state.log, {
          t: se.t,
          receivedAt,
          kind: "event",
          text: `${se.step_id}: ${statusWord(se.status)}${se.retroactive ? " (retroactive)" : ""}`,
        }),
      };
    }
    case "alerts": {
      const al = payload as Alert;
      const alerts = [{ alert: al, receivedAt }, ...state.alerts].slice(0, ALERT_CAP);
      return {
        ...state,
        alerts,
        log: pushLog(state.log, {
          t: al.t,
          receivedAt,
          kind: "alert",
          text: `ALERT (${al.kind}): ${al.message}`,
        }),
      };
    }
    case "prompts": {
      const pr = payload as Prompt;
      return {
        ...state,
        prompt: pr,
        log: pushLog(state.log, {
          t: pr.t,
          receivedAt,
          kind: "prompt",
          text: pr.next_step_id ? `Next: ${pr.text}` : "Procedure complete",
        }),
      };
    }
    case "asks": {
      const ask = payload as Ask;
      return {
        ...state,
        ask,
        log: pushLog(state.log, { t: ask.t, receivedAt, kind: "ask", text: `Asked: ${ask.text}` }),
      };
    }
    case "answers": {
      const ans = payload as AnswerMsg;
      return {
        ...state,
        ask: state.ask?.question_id === ans.question_id ? null : state.ask,
        log: pushLog(state.log, {
          t: 0,
          receivedAt,
          kind: "answer",
          text: `Answer (${ans.via}): ${ans.value}`,
        }),
      };
    }
    case "commands": {
      const cmd = payload as CommandMsg;
      return {
        ...state,
        log: pushLog(state.log, {
          t: 0,
          receivedAt,
          kind: "command",
          text: `Command (${cmd.via}): ${cmd.name}`,
        }),
      };
    }
    case "stats": {
      const st = payload as { source: string; values: Record<string, unknown>; t: number };
      return { ...state, stats: { ...state.stats, [st.source]: { values: st.values, t: st.t } } };
    }
    default:
      return state;
  }
}

export interface BusStore {
  state: BusState;
  sendAnswer: (questionId: string, value: "yes" | "no") => void;
  sendCommand: (name: string) => void;
}

/** Owns the single WebSocket connection to /ws, with auto-reconnect. */
export function useBusStore(): BusStore {
  const [state, dispatch] = useReducer(reducer, initialState);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retryDelay = RECONNECT_MIN_MS;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      if (cancelled) return;
      const proto = window.location.protocol === "https:" ? "wss://" : "ws://";
      const ws = new WebSocket(`${proto}${window.location.host}/ws`);
      wsRef.current = ws;

      ws.onopen = () => {
        retryDelay = RECONNECT_MIN_MS;
        dispatch({ kind: "connected" });
      };
      ws.onclose = () => {
        dispatch({ kind: "disconnected" });
        if (cancelled) return;
        retryTimer = setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, RECONNECT_MAX_MS);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (event) => {
        try {
          const envelope = JSON.parse(event.data) as Envelope;
          dispatch({ kind: "envelope", envelope });
        } catch {
          // malformed frame from the server; drop it rather than crash the dashboard
        }
      };
    }

    connect();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, []);

  return useMemo(
    () => ({
      state,
      sendAnswer: (questionId, value) => {
        wsRef.current?.send(JSON.stringify({ type: "Answer", question_id: questionId, value }));
        dispatch({ kind: "sent_answer", questionId, value });
      },
      sendCommand: (name) => {
        wsRef.current?.send(JSON.stringify({ type: "Command", name }));
        dispatch({ kind: "sent_command", name });
      },
    }),
    [state],
  );
}
