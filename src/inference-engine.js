const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

export class TfjsEngine {
  constructor(model, label = "TensorFlow.js FP32") {
    this.model = model;
    this.name = label;
    this.kind = "tfjs";
  }

  static async load(url) {
    if (!globalThis.tf) throw new Error("TensorFlow.js is not loaded");
    await tf.setBackend("webgl");
    await tf.ready();
    const model = await tf.loadLayersModel(url);
    return new TfjsEngine(model);
  }

  async predict(data, shape) {
    const input = tf.tensor3d(data, shape);
    const output = this.model.predict(input);
    try {
      const probs = await output.data();
      return Float32Array.from(probs);
    } finally {
      input.dispose();
      output.dispose();
    }
  }
}

export class OnnxEngine {
  constructor(session, label) {
    this.session = session;
    this.name = label;
    this.kind = "onnx";
  }

  static async load(url, { int8 = false } = {}) {
    if (!globalThis.ort) throw new Error("ONNX Runtime Web is not loaded");
    ort.env.wasm.wasmPaths = ORT_CDN;

    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) throw new Error(`ONNX HTTP ${response.status}: ${url}`);
    const buffer = await response.arrayBuffer();

    const session = await ort.InferenceSession.create(buffer, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    return new OnnxEngine(
      session,
      int8 ? "ONNX Runtime Web INT8" : "ONNX Runtime Web FP32"
    );
  }

  async predict(data, shape) {
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const tensor = new ort.Tensor("float32", data, shape);
    const results = await this.session.run({ [inputName]: tensor });
    return Float32Array.from(results[outputName].data);
  }
}

export async function createV1Engine(preference = "auto") {
  const query = String(preference || "auto").toLowerCase();

  if (query === "tfjs") {
    return TfjsEngine.load("./model/tfjs_model/model.json");
  }
  if (query === "onnx-int8" || query === "int8") {
    return OnnxEngine.load("./model/onnx/sign_language_int8.onnx", { int8: true });
  }
  if (query === "onnx-fp32" || query === "onnx") {
    return OnnxEngine.load("./model/onnx/sign_language_fp32.onnx");
  }

  // Auto: use the validated ONNX FP32 model first; retain TF.js as a compatibility fallback.
  try {
    return await OnnxEngine.load("./model/onnx/sign_language_fp32.onnx");
  } catch (onnxError) {
    console.warn("ONNX FP32 load failed; falling back to TF.js", onnxError);
    return TfjsEngine.load("./model/tfjs_model/model.json");
  }
}

export async function createV2Engine(preference = "auto") {
  const query = String(preference || "auto").toLowerCase();
  if (query === "onnx-int8" || query === "int8") {
    return OnnxEngine.load("./model/v2/best_model_int8.onnx", { int8: true });
  }
  return OnnxEngine.load("./model/v2/best_model_fp32.onnx");
}
