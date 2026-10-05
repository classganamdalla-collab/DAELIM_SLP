import fs from "node:fs";
import { extractV2FeaturesFromRawFrame } from "../src/feature-schema.js";

const path = new URL("./feature_fixture.json", import.meta.url);
const frame = JSON.parse(fs.readFileSync(path, "utf8"));

const legacyFrame = {
  hands: frame.hands,
  facePresent: true,
  face: [4,10,13,33,152,234,263,454].map(idx => frame.faceExtended[String(idx)]),
};

const payload = {
  full: extractV2FeaturesFromRawFrame(frame, { profile: "full" }),
  base8: extractV2FeaturesFromRawFrame(legacyFrame, { profile: "base8" }),
};

process.stdout.write(JSON.stringify(payload));
