const INTERHAND_DELTA_START = 138;
const LEFT_PRESENT_INDEX = 187;
const RIGHT_PRESENT_INDEX = 188;

function median(values) {
  if (!values.length) return 0;
  const xs = [...values].sort((a, b) => a - b);
  const m = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
}

export function interhandSeparationGain(frames) {
  const sep = [];
  for (const row of frames ?? []) {
    if (!row || row.length < 190) continue;
    if (Number(row[LEFT_PRESENT_INDEX]) < 0.5 || Number(row[RIGHT_PRESENT_INDEX]) < 0.5) continue;
    const dx = Number(row[INTERHAND_DELTA_START] ?? 0);
    const dy = Number(row[INTERHAND_DELTA_START + 1] ?? 0);
    const dz = Number(row[INTERHAND_DELTA_START + 2] ?? 0);
    const d = Math.hypot(dx, dy, dz);
    if (Number.isFinite(d)) sep.push(d);
  }

  if (sep.length < 6) {
    return { valid: false, gain: 0, early: 0, late: 0, max: 0, frames: sep.length };
  }

  const window = Math.max(3, Math.min(10, Math.floor(sep.length * 0.22)));
  const early = median(sep.slice(0, window));
  const late = median(sep.slice(-window));
  const max = Math.max(...sep);
  return {
    valid: true,
    gain: late - early,
    early,
    late,
    max,
    maxGain: max - early,
    frames: sep.length,
  };
}

export function applyMotionPairRule(result, frames, config) {
  if (!result || !config?.enabled) return result;

  const sourceLabel = String(config.source_label || "");
  const targetLabel = String(config.target_label || "");
  if (!sourceLabel || !targetLabel || result.label !== sourceLabel) return result;

  const metric = interhandSeparationGain(frames);
  if (!metric.valid) return result;

  const threshold = Number(config.threshold);
  if (!Number.isFinite(threshold)) return result;

  const direction = config.direction === "less_is_target" ? "less" : "greater";
  const matched = direction === "greater"
    ? metric.gain >= threshold
    : metric.gain <= threshold;

  if (!matched) return result;

  return {
    ...result,
    label: targetLabel,
    motionOverride: {
      from: sourceLabel,
      to: targetLabel,
      metric: "interhand_separation_gain",
      value: metric.gain,
      threshold,
      frames: metric.frames,
    },
  };
}
