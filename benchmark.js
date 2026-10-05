const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const parityEl = document.getElementById("parity");

const TFJS_URL = "./model/tfjs_model/model.json";
const ONNX_FP32_URL = "./model/onnx/sign_language_fp32.onnx";
const ONNX_INT8_URL = "./model/onnx/sign_language_int8.onnx";
const N = 80;
const WARMUP = 10;
const SHAPE = [1, 100, 90];

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

function seededRandom(seed = 20261005) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makeInput() {
  const rand = seededRandom();
  const arr = new Float32Array(SHAPE[0] * SHAPE[1] * SHAPE[2]);
  const valid = 72;
  for (let i = 0; i < valid * SHAPE[2]; i++) {
    // Approximate zero-centered relative-coordinate range.
    arr[i] = (rand() - 0.5) * 0.36;
  }
  return arr;
}

function stats(xs) {
  const a = [...xs].sort((x, y) => x - y);
  const mean = a.reduce((s, x) => s + x, 0) / a.length;
  const median = a[Math.floor(a.length / 2)];
  const p95 = a[Math.min(a.length - 1, Math.ceil(a.length * 0.95) - 1)];
  return { mean, median, p95 };
}

function bytesText(n) {
  if (!Number.isFinite(n)) return "-";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(2) + " MB";
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  return { buf, bytes: buf.byteLength };
}

function maxAbsDiff(a, b) {
  let max = 0, sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    sum += d;
    if (d > max) max = d;
  }
  return { mean: sum / n, max };
}

function argmax(a) {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

async function benchmarkTfjs(model, input) {
  const tensor = tf.tensor3d(input, SHAPE);
  for (let i = 0; i < WARMUP; i++) {
    const y = model.predict(tensor);
    await y.data();
    y.dispose();
  }
  const times = [];
  let last = null;
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    const y = model.predict(tensor);
    last = await y.data();
    times.push(performance.now() - t0);
    y.dispose();
  }
  tensor.dispose();
  return { times, output: Float32Array.from(last) };
}

async function benchmarkOrt(session, input) {
  const name = session.inputNames[0];
  const tensor = new ort.Tensor("float32", input, SHAPE);
  for (let i = 0; i < WARMUP; i++) await session.run({ [name]: tensor });
  const times = [];
  let last = null;
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    const out = await session.run({ [name]: tensor });
    times.push(performance.now() - t0);
    last = out[session.outputNames[0]].data;
  }
  return { times, output: Float32Array.from(last) };
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  rowsEl.innerHTML = "";
  parityEl.textContent = "-";
  try {
    statusEl.textContent = " 모델과 런타임을 불러오는 중...";

    await tf.setBackend("webgl");
    await tf.ready();

    const fetchStarted = performance.now();
    const [fp32File, int8File] = await Promise.all([
      fetchBytes(ONNX_FP32_URL),
      fetchBytes(ONNX_INT8_URL),
    ]);
    const fetchMs = performance.now() - fetchStarted;

    const tfLoadStart = performance.now();
    const tfModel = await tf.loadLayersModel(TFJS_URL);
    const tfLoadMs = performance.now() - tfLoadStart;

    const fpLoadStart = performance.now();
    const fp32Session = await ort.InferenceSession.create(fp32File.buf.slice(0), { executionProviders: ["wasm"] });
    const fpLoadMs = performance.now() - fpLoadStart;

    const iqLoadStart = performance.now();
    const int8Session = await ort.InferenceSession.create(int8File.buf.slice(0), { executionProviders: ["wasm"] });
    const iqLoadMs = performance.now() - iqLoadStart;

    let webgpuSession = null;
    let webgpuLoadMs = null;
    let webgpuError = null;
    if (navigator.gpu) {
      try {
        const t = performance.now();
        webgpuSession = await ort.InferenceSession.create(fp32File.buf.slice(0), {
          executionProviders: ["webgpu"],
        });
        webgpuLoadMs = performance.now() - t;
      } catch (e) {
        webgpuError = String(e?.message || e);
      }
    }

    const input = makeInput();

    statusEl.textContent = " TF.js 측정 중...";
    const tfRes = await benchmarkTfjs(tfModel, input);
    statusEl.textContent = " ONNX FP32 측정 중...";
    const fpRes = await benchmarkOrt(fp32Session, input);
    statusEl.textContent = " ONNX INT8 측정 중...";
    const iqRes = await benchmarkOrt(int8Session, input);

    let wgRes = null;
    if (webgpuSession) {
      statusEl.textContent = " ONNX WebGPU 측정 중...";
      try {
        wgRes = await benchmarkOrt(webgpuSession, input);
      } catch (e) {
        webgpuError = String(e?.message || e);
      }
    }

    const sTf = stats(tfRes.times), sFp = stats(fpRes.times), sIq = stats(iqRes.times);
    const dTfFp = maxAbsDiff(tfRes.output, fpRes.output);
    const dFpIq = maxAbsDiff(fpRes.output, iqRes.output);

    const rows = [
      ["TF.js WebGL FP32", tfLoadMs, sTf, NaN],
      ["ONNX WASM FP32", fpLoadMs + fetchMs, sFp, fp32File.bytes],
      ["ONNX WASM INT8", iqLoadMs + fetchMs, sIq, int8File.bytes],
    ];
    if (wgRes) rows.push(["ONNX WebGPU FP32", webgpuLoadMs + fetchMs, stats(wgRes.times), fp32File.bytes]);

    rowsEl.innerHTML = rows.map(([name, loadMs, s, bytes]) =>
      `<tr><td>${name}</td><td>${Number(loadMs).toFixed(1)}</td><td>${s.mean.toFixed(2)}</td><td>${s.median.toFixed(2)}</td><td>${s.p95.toFixed(2)}</td><td>${bytesText(bytes)}</td></tr>`
    ).join("");

    parityEl.textContent = JSON.stringify({
      tfjs_vs_onnx_fp32: {
        mean_abs_error: dTfFp.mean,
        max_abs_error: dTfFp.max,
        argmax_match: argmax(tfRes.output) === argmax(fpRes.output),
      },
      onnx_fp32_vs_int8: {
        mean_abs_error: dFpIq.mean,
        max_abs_error: dFpIq.max,
        argmax_match: argmax(fpRes.output) === argmax(iqRes.output),
      },
      webgpu: {
        available: Boolean(navigator.gpu),
        session_created: Boolean(webgpuSession),
        error: webgpuError,
      },
      note: "합성 입력의 수치 비교입니다. 실제 인식 정확도는 별도 파일럿/검증 데이터로 측정해야 합니다."
    }, null, 2);

    statusEl.textContent = " 완료";
  } catch (e) {
    console.error(e);
    statusEl.textContent = " 실패";
    parityEl.textContent = String(e?.stack || e);
  } finally {
    runBtn.disabled = false;
  }
});
