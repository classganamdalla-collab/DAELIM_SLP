import fs from "node:fs";
import { extractV2FeaturesFromRawFrame } from "../src/feature-schema.js";

const path = new URL("./feature_fixture.json", import.meta.url);
const frame = JSON.parse(fs.readFileSync(path, "utf8"));
const features = extractV2FeaturesFromRawFrame(frame);
process.stdout.write(JSON.stringify(features));
