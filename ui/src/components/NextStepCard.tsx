import type { Prompt } from "../types";

interface Props {
  prompt: Prompt | null;
}

export function NextStepCard({ prompt }: Props) {
  const complete = prompt !== null && prompt.next_step_id === null;
  return (
    <div className={`panel next-step-card${complete ? " complete" : ""}`}>
      <div className="label">Next step</div>
      <div className="text">
        {prompt ? prompt.text : "Waiting for the protocol engine’s first prompt…"}
      </div>
    </div>
  );
}
