import assert from "node:assert/strict";
import { motionMetric, interhandSeparationGain, applyMotionPairRule } from "../src/motion-disambiguator.js";

function frame({ sep = 0.3, y = 0.2, open = 1.0 } = {}) {
  const x = new Array(190).fill(0);
  x[138] = sep;
  x[187] = 1;
  x[188] = 1;
  x[189] = 1;
  x[67] = y;
  x[136] = y;

  for (const offset of [0, 69]) {
    for (const lm of [4, 8, 12, 16, 20]) {
      x[offset + lm * 3] = open;
    }
  }
  return x;
}

const stable = Array.from({ length: 20 }, (_, i) => frame({ sep: 0.35 + 0.01 * Math.sin(i), y: 0.2 }));
const split = Array.from({ length: 20 }, (_, i) => frame({ sep: 0.25 + i * 0.035, y: 0.2 }));
const downward = Array.from({ length: 20 }, (_, i) => frame({ sep: 0.8, y: 0.1 + i * 0.025, open: 1.3 }));

assert.ok(interhandSeparationGain(stable).gain < 0.1);
assert.ok(interhandSeparationGain(split).gain > 0.35);
assert.ok(motionMetric(downward, "wrist_downward_gain").value > 0.3);
assert.ok(motionMetric(downward, "hand_openness_early").value > 1.0);

const result = { label: "아메리카노", score: 0.95, margin: 0.8, candidates: [] };
const multi = {
  rules: [
    {
      enabled: true,
      source_label: "아메리카노",
      target_label: "뜨거운",
      metric: "wrist_downward_gain",
      direction: "greater_is_target",
      threshold: 0.2,
    },
    {
      enabled: true,
      source_label: "아메리카노",
      target_label: "영수증",
      metric: "interhand_separation_gain",
      direction: "greater_is_target",
      threshold: 0.25,
    },
  ],
};

assert.equal(applyMotionPairRule(result, stable, multi).label, "아메리카노");
assert.equal(applyMotionPairRule(result, split, multi).label, "영수증");
assert.equal(applyMotionPairRule(result, downward, multi).label, "뜨거운");

console.log("motion disambiguator smoke test OK");
