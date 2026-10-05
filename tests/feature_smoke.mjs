import assert from "node:assert/strict";
import {
  FEATURE_SCHEMAS,
  extractV2FeaturesFromRawFrame,
} from "../src/feature-schema.js";
import { decodeCafeWords } from "../src/cafe-decoder.js";

const hand = {
  handedness: "Left",
  landmarks: Array.from({ length: 21 }, (_, i) => ({
    x: i / 100,
    y: i / 120,
    z: -i / 300,
  })),
};

const faceIndices = [4,10,13,14,33,61,105,133,145,152,159,234,263,291,334,362,374,386,454];
const faceExtended = Object.fromEntries(
  faceIndices.map((idx, i) => [String(idx), { x: 0.45 + i / 1000, y: 0.35 + i / 1200, z: 0 }])
);

const x = extractV2FeaturesFromRawFrame({
  hands: [hand],
  faceExtended,
  faceBlendshapes: { jawOpen: 0.2, browInnerUp: 0.1 },
  facePresent: true,
});

assert.equal(x.length, FEATURE_SCHEMAS.v2.featureDim);
assert.ok(x.every(Number.isFinite));

const phrase = decodeCafeWords(["아이스", "아메리카노", "2잔", "테이크아웃"]);
assert.match(phrase, /아메리카노/);
assert.match(phrase, /두 잔/);
assert.match(phrase, /테이크아웃/);

console.log("JS feature/decoder smoke test OK", x.length, phrase);
