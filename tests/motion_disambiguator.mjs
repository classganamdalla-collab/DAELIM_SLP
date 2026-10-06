import assert from "node:assert/strict";
import { interhandSeparationGain, applyMotionPairRule } from "../src/motion-disambiguator.js";

function frame(sep) {
  const x = new Array(190).fill(0);
  x[138] = sep;
  x[187] = 1;
  x[188] = 1;
  x[189] = 1;
  return x;
}

const stable = Array.from({ length: 20 }, (_, i) => frame(0.35 + 0.01 * Math.sin(i)));
const split = Array.from({ length: 20 }, (_, i) => frame(0.25 + i * 0.035));

assert.ok(interhandSeparationGain(stable).gain < 0.1);
assert.ok(interhandSeparationGain(split).gain > 0.35);

const cfg = {
  enabled: true,
  source_label: "아메리카노",
  target_label: "영수증",
  direction: "greater_is_target",
  threshold: 0.25,
};

const result = { label: "아메리카노", score: 0.95, margin: 0.8, candidates: [] };
assert.equal(applyMotionPairRule(result, stable, cfg).label, "아메리카노");
assert.equal(applyMotionPairRule(result, split, cfg).label, "영수증");

console.log("motion disambiguator smoke test OK");
