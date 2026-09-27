import { useState } from "react";
import type { Predicates } from "../types";

interface Props {
  predicates: Predicates | null;
}

/** MJPEG fallback of the annotated stream (boxes / hand skeletons / zones are drawn server-side
 * by whichever lane publishes `annotated` frames). WebRTC via MediaMTX (T-IO) is the primary
 * path in the finished pipeline; this pane only ever does the MJPEG fallback, so it works with
 * nothing more than this server running. */
export function LivePane({ predicates }: Props) {
  const [failed, setFailed] = useState(false);
  const [key, setKey] = useState(0);

  return (
    <div className="live-pane">
      {!failed ? (
        <img
          key={key}
          src="/stream/mjpeg"
          alt="Live annotated feed"
          onError={() => setFailed(true)}
        />
      ) : (
        <div className="placeholder">
          No video yet.
          <br />
          Waiting for frames on the <code>annotated</code> / <code>frames</code> topic.
          <br />
          <button
            onClick={() => {
              setFailed(false);
              setKey((k) => k + 1);
            }}
          >
            Retry
          </button>
        </div>
      )}
      {predicates && (
        <span className="visibility-chip">visibility {(predicates.visibility * 100).toFixed(0)}%</span>
      )}
    </div>
  );
}
