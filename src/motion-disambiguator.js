const INTERHAND_DELTA_START = 138;
const LEFT_WRIST_FACE_START = 66;
const RIGHT_WRIST_FACE_START = 135;
const LEFT_PRESENT_INDEX = 187;
const RIGHT_PRESENT_INDEX = 188;

function median(values) {
  if (!values.length) return 0;
  const xs = [...values].sort((a, b) => a - b);
  const m = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
}

function validRows(frames) {
  return (frames ?? []).filter(row => row && row.length >= 190);
}

function interhandSeparationSeries(frames) {
  const sep = [];
  for (const row of validRows(frames)) {
    if (Number(row[LEFT_PRESENT_INDEX]) < 0.5 || Number(row[RIGHT_PRESENT_INDEX]) < 0.5) continue;
    const dx = Number(row[INTERHAND_DELTA_START] ?? 0);
    const dy = Number(row[INTERHAND_DELTA_START + 1] ?? 0);
    const dz = Number(row[INTERHAND_DELTA_START + 2] ?? 0);
    const d = Math.hypot(dx, dy, dz);
    if (Number.isFinite(d)) sep.push(d);
  }
  return sep;
}

function wristMeanYSeries(frames) {
  const ys = [];
  for (const row of validRows(frames)) {
    const vals = [];
    if (Number(row[LEFT_PRESENT_INDEX]) >= 0.5) {
      const y = Number(row[LEFT_WRIST_FACE_START + 1] ?? 0);
      if (Number.isFinite(y)) vals.push(y);
    }
    if (Number(row[RIGHT_PRESENT_INDEX]) >= 0.5) {
      const y = Number(row[RIGHT_WRIST_FACE_START + 1] ?? 0);
      if (Number.isFinite(y)) vals.push(y);
    }
    if (vals.length) ys.push(vals.reduce((a, b) => a + b, 0) / vals.length);
  }
  return ys;
}

function handOpenness(row, offset) {
  // Local hand coordinates are wrist-centered and palm-scale normalized.
  // Mean fingertip-to-wrist distance is a simple open-vs-clasp descriptor.
  const tips = [4, 8, 12, 16, 20];
  const ds = [];
  for (const lm of tips) {
    const i = offset + lm * 3;
    const x = Number(row[i] ?? 0);
    const y = Number(row[i + 1] ?? 0);
    const z = Number(row[i + 2] ?? 0);
    const d = Math.hypot(x, y, z);
    if (Number.isFinite(d)) ds.push(d);
  }
  return ds.length ? ds.reduce((a, b) => a + b, 0) / ds.length : null;
}

function handOpennessSeries(frames) {
  const out = [];
  for (const row of validRows(frames)) {
    const vals = [];
    if (Number(row[LEFT_PRESENT_INDEX]) >= 0.5) {
      const v = handOpenness(row, 0);
      if (v != null) vals.push(v);
    }
    if (Number(row[RIGHT_PRESENT_INDEX]) >= 0.5) {
      const v = handOpenness(row, 69);
      if (v != null) vals.push(v);
    }
    if (vals.length) out.push(vals.reduce((a, b) => a + b, 0) / vals.length);
  }
  return out;
}

function edgeStats(series) {
  if (series.length < 6) return null;
  const window = Math.max(3, Math.min(10, Math.floor(series.length * 0.22)));
  const early = median(series.slice(0, window));
  const late = median(series.slice(-window));
  return {
    early,
    late,
    gain: late - early,
    mean: series.reduce((a, b) => a + b, 0) / series.length,
    max: Math.max(...series),
    min: Math.min(...series),
    frames: series.length,
  };
}

export function motionMetric(frames, metric) {
  if (metric === "interhand_separation_gain") {
    const s = edgeStats(interhandSeparationSeries(frames));
    return s ? { valid: true, value: s.gain, ...s } : { valid: false, value: 0, frames: 0 };
  }

  if (metric === "wrist_downward_gain") {
    // MediaPipe normalized Y grows downward in the image.
    const s = edgeStats(wristMeanYSeries(frames));
    return s ? { valid: true, value: s.gain, ...s } : { valid: false, value: 0, frames: 0 };
  }

  if (metric === "hand_openness_early") {
    const s = edgeStats(handOpennessSeries(frames));
    return s ? { valid: true, value: s.early, ...s } : { valid: false, value: 0, frames: 0 };
  }

  if (metric === "hand_openness_mean") {
    const s = edgeStats(handOpennessSeries(frames));
    return s ? { valid: true, value: s.mean, ...s } : { valid: false, value: 0, frames: 0 };
  }

  if (metric === "interhand_separation_early") {
    const s = edgeStats(interhandSeparationSeries(frames));
    return s ? { valid: true, value: s.early, ...s } : { valid: false, value: 0, frames: 0 };
  }

  return { valid: false, value: 0, frames: 0 };
}

// Backward-compatible export used by existing tests/tools.
export function interhandSeparationGain(frames) {
  const m = motionMetric(frames, "interhand_separation_gain");
  return {
    valid: m.valid,
    gain: m.value,
    early: m.early ?? 0,
    late: m.late ?? 0,
    max: m.max ?? 0,
    maxGain: (m.max ?? 0) - (m.early ?? 0),
    frames: m.frames ?? 0,
  };
}

function applyOneRule(result, frames, config) {
  if (!result || !config?.enabled) return result;

  const sourceLabel = String(config.source_label || "");
  const targetLabel = String(config.target_label || "");
  if (!sourceLabel || !targetLabel || result.label !== sourceLabel) return result;

  const metricName = String(config.metric || "interhand_separation_gain");
  const metric = motionMetric(frames, metricName);
  if (!metric.valid) return result;

  const threshold = Number(config.threshold);
  if (!Number.isFinite(threshold)) return result;

  const direction = config.direction === "less_is_target" ? "less" : "greater";
  const matched = direction === "greater"
    ? metric.value >= threshold
    : metric.value <= threshold;

  if (!matched) return result;

  return {
    ...result,
    label: targetLabel,
    motionOverride: {
      from: sourceLabel,
      to: targetLabel,
      metric: metricName,
      value: metric.value,
      threshold,
      frames: metric.frames,
    },
  };
}

export function applyMotionPairRule(result, frames, config) {
  if (!config) return result;

  // New format: multiple ordered rules. Keep the old single-rule JSON valid.
  const rules = Array.isArray(config.rules) ? config.rules : [config];
  let out = result;
  for (const rule of rules) {
    out = applyOneRule(out, frames, rule);
  }
  return out;
}
