import { useEffect, useRef, useState } from "react";
import type { AlertEntry } from "../useBusStore";

const VISIBLE_MS = 10000;

function beep() {
  try {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.4);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  } catch {
    // no audio output available (autoplay policy, headless environment, ...) - visual banner
    // still carries the alert, so this is a soft failure.
  }
}

interface Props {
  alerts: AlertEntry[];
}

export function AlertBanner({ alerts }: Props) {
  const [, setTick] = useState(0);
  const lastSeenRef = useRef<number>(0);

  useEffect(() => {
    if (alerts.length === 0) return;
    const newest = alerts[0].receivedAt;
    if (newest !== lastSeenRef.current) {
      lastSeenRef.current = newest;
      beep();
    }
  }, [alerts]);

  useEffect(() => {
    if (alerts.length === 0) return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [alerts.length]);

  const visible = alerts.filter((a) => Date.now() - a.receivedAt < VISIBLE_MS);
  if (visible.length === 0) return null;

  return (
    <>
      {visible.map((entry) => (
        <div className="alert-banner" key={`${entry.alert.frame_idx}-${entry.receivedAt}`}>
          <span className="kind-tag">{entry.alert.kind.replace(/_/g, " ")}</span>
          <span>{entry.alert.message}</span>
        </div>
      ))}
    </>
  );
}
