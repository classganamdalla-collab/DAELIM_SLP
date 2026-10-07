import fs from "node:fs";
import assert from "node:assert/strict";

const css = fs.readFileSync(new URL("../style.css", import.meta.url), "utf8");
const js = fs.readFileSync(new URL("../script.js", import.meta.url), "utf8");

assert.match(css, /100dvh/);
assert.match(css, /safe-area-inset-top/);
assert.match(css, /@media \(max-height: 520px\)/);
assert.match(js, /function syncCameraSurface\(\)/);
assert.match(js, /orientationchange/);
assert.match(js, /resetTrackingAfterViewportChange/);

console.log("mobile viewport/orientation smoke test OK");
