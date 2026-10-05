import assert from "node:assert/strict";
import {
  FEATURE_SCHEMAS,
  extractV2FeaturesFromRawFrame,
  extractV2FeaturesFromResults,
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

const liveHandResults = {
  landmarks: [hand.landmarks],
  handedness: [[{ categoryName: "Left", score: 0.99 }]],
};
const liveFaceLandmarks = Array.from({ length: 478 }, () => ({ x: 0, y: 0, z: 0 }));
for (const [idx, point] of Object.entries(faceExtended)) {
  liveFaceLandmarks[Number(idx)] = point;
}
const liveFaceResults = {
  faceLandmarks: [liveFaceLandmarks],
  // This is the MediaPipe Tasks Vision Classifications-object shape seen in browsers.
  faceBlendshapes: [{
    categories: [
      { categoryName: "jawOpen", score: 0.2 },
      { categoryName: "browInnerUp", score: 0.1 },
    ],
  }],
};

const liveBase8 = extractV2FeaturesFromResults(
  liveHandResults,
  liveFaceResults,
  { profile: "base8" },
);
assert.equal(liveBase8.length, FEATURE_SCHEMAS.v2Base8.featureDim);
assert.ok(liveBase8.every(Number.isFinite));


const cases = [
  [["아이스", "아메리카노", "주세요"], /아이스 아메리카노 주세요/],
  [["뜨거운", "아메리카노", "주세요"], /따뜻한 아메리카노 주세요/],
  [["아메리카노", "2잔", "주세요"], /아메리카노 두 잔 주세요/],
  [["제일", "큰걸로", "주세요"], /제일 큰 사이즈로 주세요/],
  [["테이크아웃", "해주세요"], /테이크아웃으로 해주세요/],
  [["카드"], /카드로 결제할게요/],
  [["영수증"], /영수증 주세요/],
  [["와이파이", "있나요"], /와이파이 있나요/],
  [["감사합니다"], /감사합니다/],
];

for (const [tokens, expected] of cases) {
  const phrase = decodeCafeWords(tokens);
  assert.match(phrase, expected, `${tokens.join(" / ")} -> ${phrase}`);
}

// Unknown/non-domain input must not be hallucinated into a cafe phrase.
assert.equal(decodeCafeWords(["임의표현"]), "임의표현");

console.log("JS feature/decoder smoke test OK", x.length, "live-base8", liveBase8.length);
