import type { Ask } from "../types";

interface Props {
  ask: Ask | null;
  onAnswer: (questionId: string, value: "yes" | "no") => void;
}

/** The question the protocol engine couldn't resolve from evidence alone. Yes/No here is the
 * GUI path; a spoken "yes"/"no" answers the same open Ask through T-VOICE's ASR instead. */
export function AskCard({ ask, onAnswer }: Props) {
  if (!ask) return null;
  return (
    <div className="ask-card">
      <div className="text">{ask.text}</div>
      <div className="buttons">
        <button className="yes" onClick={() => onAnswer(ask.question_id, "yes")}>
          Yes
        </button>
        <button className="no" onClick={() => onAnswer(ask.question_id, "no")}>
          No
        </button>
      </div>
    </div>
  );
}
