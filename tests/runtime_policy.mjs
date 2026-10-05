import assert from "node:assert/strict";
import { getRuntimePolicy } from "../src/runtime-policy.js";

const base8 = getRuntimePolicy("v2", "base8");
assert.deepEqual(base8, {
  id: "v2-base8-legacy-collector",
  handStride: 2,
  sampleEveryVideoFrame: true,
  noHandEndFrames: 3,
  cooldownMs: 650,
  autoLockWhileHandVisible: false,
});

const full = getRuntimePolicy("v2", "full");
assert.equal(full.handStride, 2);
assert.equal(full.sampleEveryVideoFrame, false);
assert.equal(full.noHandEndFrames, 4);
assert.equal(full.autoLockWhileHandVisible, true);

const v1 = getRuntimePolicy("v1", "full");
assert.equal(v1.handStride, 3);
assert.equal(v1.sampleEveryVideoFrame, true);

console.log("runtime policy smoke test OK", base8.id);
