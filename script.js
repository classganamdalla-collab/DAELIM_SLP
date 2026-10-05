import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

import {
  FACE_BASE_POINTS,
  FEATURE_SCHEMAS,
  extractV1FeaturesFromResults,
  extractV2FeaturesFromResults,
} from "./src/feature-schema.js";

import {
  createV1Engine,
  createV2Engine,
} from "./src/inference-engine.js";

import { decodeCafeWords } from "./src/cafe-decoder.js";
import { StudyLogger } from "./src/study-logger.js";

const $ = id => document.getElementById(id);

const video = $("webcam");
const canvas = $("outputCanvas");
const canvasCtx = canvas.getContext("2d");
const startBtn = $("startBtn");
const addWordBtn = $("addWordBtn");
const clearSentenceBtn = $("clearSentenceBtn");
const speakSentenceBtn = $("speakSentenceBtn");
const autoSpeakToggle = $("autoSpeakToggle");
const autoAddToggle = $("autoAddToggle");
const listenBtn = $("listenBtn");
const fallbackBtn = $("fallbackBtn");
const speechPanel = $("speechPanel");
const speechFinal = $("speechFinal");
const speechInterim = $("speechInterim");
const clearSpeechBtn = $("clearSpeechBtn");
const datasetStatus = $("datasetStatus");
const cameraStatus = $("cameraStatus");
const handStatus = $("handStatus");
const engineStatus = $("engineStatus");
const runStatus = $("runStatus");
const currentPrediction = $("currentPrediction");
const predictionScore = $("predictionScore");
const holdStatus = $("holdStatus");
const holdBarFill = $("holdBarFill");
const sentenceOutput = $("sentenceOutput");
const sentenceNatural = $("sentenceNatural");
const repairCandidates = $("repairCandidates");
const labelChips = $("labelChips");
const launchBtn = $("launchBtn");
const modelBadge = $("modelBadge");
const brandBtn = $("brandBtn");
const statusDot = $("statusDot");
const manualPanel = $("manualPanel");
const manualCloseBtn = $("manualCloseBtn");
const manualText = $("manualText");
const manualSpeakBtn = $("manualSpeakBtn");
const studyControls = $("studyControls");
const studyParticipant = $("studyParticipant");
const studyStartBtn = $("studyStartBtn");
const studySuccessBtn = $("studySuccessBtn");
const studyFailBtn = $("studyFailBtn");
const studyExportBtn = $("studyExportBtn");

const pageParams = new URLSearchParams(location.search);
const studyLogger = new StudyLogger({
  enabled: pageParams.get("study") === "1",
  participant: pageParams.get("participant") || "P00",
});

const MIN_SCORE_SHOW = 0.70;
const MIN_SCORE_LOCK = 0.75;
const MIN_MARGIN = 0.20;
const MIN_GESTURE_FRAMES = 15;
const NO_HAND_END_FRAMES = 8;
const COOLDOWN_MS = 1500;
const LOCK_HOLD_MS = 700;

let LIVE_BUFFER_MAX = 100;
let featureDim = FEATURE_SCHEMAS.v1.featureDim;
let featureMode = "v1";

let handLandmarker = null;
let faceLandmarker = null;
let drawingUtils = null;
let inferenceEngine = null;
let modelMeta = null;

let webcamRunning = false;
let lastVideoTime = -1;
let appStarted = false;
let autoAddEnabled = true;
let autoSpeakEnabled = false;
let autoSpeakTimer = null;

let labelsConfig = [];
let targetLabels = [];
let sentenceWords = [];
let latestHandCount = 0;

let gestureState = "waiting";
let gestureBuffer = [];
let noHandCount = 0;
let lastResult = null;
let liveInferCount = 0;
let lockStartTime = 0;
let lockLabel = null;

let lastFaceResults = null;
let lastHandResults = null;
let faceFrameCount = 0;
let handFrameCount = 0;
let activeInferencePromise = null;

let recognition = null;
let isListening = false;
let landmarkersReady = false;
let sttReady = false;

const inferCanvas = document.createElement("canvas");
inferCanvas.width = 640;
inferCanvas.height = 480;
const inferCtx = inferCanvas.getContext("2d");

async function fetchJsonOptional(url) {
  try {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function loadLabels() {
  const json = await fetchJsonOptional("./labels.json");
  labelsConfig = json?.labels ?? [];
  renderLabelChips();
}

function renderLabelChips() {
  if (!labelChips) return;
  labelChips.innerHTML = labelsConfig
    .filter(l => l.id !== "기타")
    .map(l => `<span class="chip">${l.korean}</span>`)
    .join("");
}

async function loadModelStack() {
  const preference = pageParams.get("engine") || "auto";
  datasetStatus.textContent = "AI 모델 로딩 중...";
  modelBadge.textContent = "AI 모델 로딩 중...";

  const v2Meta = await fetchJsonOptional("./model/v2/metadata.json");
  if (v2Meta?.feature_schema === FEATURE_SCHEMAS.v2.id) {
    try {
      inferenceEngine = await createV2Engine(preference);
      modelMeta = v2Meta;
      featureMode = "v2";
      featureDim = Number(v2Meta.feature_dim || FEATURE_SCHEMAS.v2.featureDim);
      LIVE_BUFFER_MAX = Number(v2Meta.max_sequence_length || 100);
      targetLabels = v2Meta.labels || [];
    } catch (e) {
      console.warn("v2 model detected but could not be loaded; falling back to v1", e);
    }
  }

  if (!inferenceEngine) {
    modelMeta = await fetchJsonOptional("./model/metadata.json");
    if (!modelMeta) throw new Error("model/metadata.json을 불러올 수 없습니다.");

    featureMode = "v1";
    featureDim = Number(modelMeta.feature_dim || FEATURE_SCHEMAS.v1.featureDim);
    LIVE_BUFFER_MAX = Number(modelMeta.max_sequence_length || 100);
    targetLabels = modelMeta.labels || labelsConfig.filter(l => l.id !== "기타").map(l => l.id);
    inferenceEngine = await createV1Engine(preference);
  }

  if (!targetLabels.length) throw new Error("모델 라벨 정보가 없습니다.");

  datasetStatus.textContent = `AI 모델 준비 완료: ${inferenceEngine.name}`;
  engineStatus.textContent = `${inferenceEngine.name} · ${featureMode === "v2" ? FEATURE_SCHEMAS.v2.id : FEATURE_SCHEMAS.v1.id}`;
  modelBadge.textContent = `준비 완료 · ${inferenceEngine.name}`;
}

async function createLandmarkers() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });

  drawingUtils = new DrawingUtils(canvasCtx);
}

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => { video.onloadedmetadata = resolve; });
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    webcamRunning = true;
    cameraStatus.textContent = "카메라 연결 완료";
  } catch (e) {
    cameraStatus.textContent = "카메라 연결 실패";
    throw e;
  }
}

function drawResults(handResults, faceResults) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  canvasCtx.translate(canvas.width, 0);
  canvasCtx.scale(-1, 1);

  for (const lm of handResults?.landmarks ?? []) {
    drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, {
      lineWidth: 2,
      color: "rgba(95,212,196,0.65)"
    });
    drawingUtils.drawLandmarks(lm, {
      radius: 3,
      color: "rgba(79,182,245,0.78)"
    });
  }

  const faceLM = faceResults?.faceLandmarks?.[0];
  if (faceLM?.length) {
    for (const idx of FACE_BASE_POINTS) {
      const pt = faceLM[idx];
      if (!pt) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, Math.PI * 2);
      canvasCtx.fillStyle = "rgba(79,182,245,0.45)";
      canvasCtx.fill();
    }
  }

  canvasCtx.restore();
}

function extractCurrentFrame(handResults, faceResults) {
  return featureMode === "v2"
    ? extractV2FeaturesFromResults(handResults, faceResults)
    : extractV1FeaturesFromResults(handResults, faceResults);
}

function predictWebcam() {
  if (!appStarted || !webcamRunning || !handLandmarker) return;

  const now = performance.now();
  drawResults(lastHandResults, lastFaceResults);

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    inferCtx.drawImage(video, 0, 0, 640, 480);

    handFrameCount++;
    if (handFrameCount % 3 === 0) {
      lastHandResults = handLandmarker.detectForVideo(inferCanvas, now);
      const cur = lastHandResults?.landmarks?.length ?? 0;
      if (cur !== latestHandCount) {
        latestHandCount = cur;
        handStatus.textContent = cur > 0 ? `손 감지됨: ${cur}개` : "손 감지되지 않음";
      }
    }

    faceFrameCount++;
    if (faceFrameCount % 6 === 0) {
      lastFaceResults = faceLandmarker.detectForVideo(inferCanvas, now);
    }

    updateGestureBuffer(lastHandResults, lastFaceResults);
  }

  requestAnimationFrame(predictWebcam);
}

function resetLock() {
  lockStartTime = 0;
  lockLabel = null;
}

function updateGestureBuffer(handResults, faceResults) {
  if (gestureState === "cooldown" || gestureState === "classifying") return;

  const hasHand = latestHandCount > 0;

  if (gestureState === "waiting") {
    if (hasHand) {
      gestureState = "recording";
      gestureBuffer = [];
      noHandCount = 0;
      liveInferCount = 0;
      resetLock();
      hideRepairCandidates();
      holdStatus.textContent = "동작 중...";
      holdBarFill.style.width = "0%";
      currentPrediction.textContent = "—";
      predictionScore.textContent = "";
    }
    return;
  }

  if (gestureState !== "recording") return;

  if (hasHand) {
    noHandCount = 0;
    gestureBuffer.push(extractCurrentFrame(handResults, faceResults));
    if (gestureBuffer.length > LIVE_BUFFER_MAX) gestureBuffer.shift();

    liveInferCount++;
    if (
      liveInferCount % 8 === 0 &&
      gestureBuffer.length >= MIN_GESTURE_FRAMES &&
      !activeInferencePromise
    ) {
      const snapshot = gestureBuffer.map(row => [...row]);
      activeInferencePromise = inferFrames(snapshot)
        .then(result => {
          if (gestureState === "recording") applyLivePrediction(result);
        })
        .catch(err => console.error("live inference", err))
        .finally(() => {
          activeInferencePromise = null;
        });
    }
  } else {
    noHandCount++;
    resetLock();
    if (noHandCount >= NO_HAND_END_FRAMES) {
      if (gestureBuffer.length >= MIN_GESTURE_FRAMES) {
        void classifyGesture();
      } else {
        resetGesture();
        holdStatus.textContent = "동작이 너무 짧습니다. 다시 해주세요";
      }
    }
  }
}

function makeModelInput(frames) {
  const data = new Float32Array(LIVE_BUFFER_MAX * featureDim);
  const len = Math.min(frames.length, LIVE_BUFFER_MAX);

  for (let i = 0; i < len; i++) {
    const row = frames[i];
    for (let j = 0; j < featureDim; j++) {
      const value = Number(row?.[j] ?? 0);
      data[i * featureDim + j] = Number.isFinite(value) ? value : 0;
    }
  }

  return data;
}

function analyzeProbabilities(probs) {
  const entries = Array.from(probs, (score, i) => ({
    label: targetLabels[i] ?? `class_${i}`,
    score: Number(score),
  })).sort((a, b) => b.score - a.score);

  const best = entries[0] ?? { label: "", score: 0 };
  const second = entries[1] ?? { score: 0 };
  const margin = best.score - second.score;

  let entropy = 0;
  for (const p of probs) {
    const x = Math.max(Number(p), 1e-12);
    entropy -= x * Math.log(x);
  }
  entropy /= Math.log(Math.max(2, probs.length));

  return {
    ...best,
    margin,
    entropy,
    candidates: entries.slice(0, 3),
  };
}

async function inferFrames(frames) {
  if (!inferenceEngine || frames.length < MIN_GESTURE_FRAMES) return null;
  const data = makeModelInput(frames);
  const t0 = performance.now();
  const probs = await inferenceEngine.predict(data, [1, LIVE_BUFFER_MAX, featureDim]);
  const latency = performance.now() - t0;
  studyLogger.recordInferenceLatency(latency);
  if (studyLogger.enabled && studyLogger.inferenceLatencies.length % 12 === 0) {
    const recent = studyLogger.inferenceLatencies.slice(-12);
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
    engineStatus.textContent = `${inferenceEngine.name} · ${featureMode === "v2" ? FEATURE_SCHEMAS.v2.id : FEATURE_SCHEMAS.v1.id} · ${avg.toFixed(1)}ms`;
  }
  return analyzeProbabilities(probs);
}

function isDisplayable(result) {
  return Boolean(
    result &&
    result.label !== "기타" &&
    result.score >= MIN_SCORE_SHOW &&
    result.margin >= MIN_MARGIN
  );
}

function isLockable(result) {
  return Boolean(
    isDisplayable(result) &&
    result.score >= MIN_SCORE_LOCK
  );
}

function displayName(labelId) {
  return labelsConfig.find(l => l.id === labelId)?.korean ?? labelId;
}

function applyLivePrediction(result) {
  if (!isDisplayable(result)) {
    resetLock();
    currentPrediction.textContent = "...";
    predictionScore.textContent = "";
    holdStatus.textContent = "동작 중...";
    holdBarFill.style.width = "0%";
    return;
  }

  const word = displayName(result.label);
  currentPrediction.textContent = word;
  predictionScore.textContent =
    `신뢰도 ${(result.score * 100).toFixed(1)}% · 마진 ${(result.margin * 100).toFixed(1)}%`;
  holdBarFill.style.width = `${Math.round(result.score * 100)}%`;

  if (!isLockable(result)) {
    resetLock();
    holdStatus.textContent = "손을 내리면 최종 확인합니다";
    return;
  }

  const now = performance.now();
  if (lockLabel !== result.label) {
    lockLabel = result.label;
    lockStartTime = now;
  }

  const elapsed = now - lockStartTime;
  const remaining = Math.max(0, LOCK_HOLD_MS - elapsed);
  holdStatus.textContent = remaining > 50
    ? `동작 유지... (${(remaining / 1000).toFixed(1)}초)`
    : "확정 중...";

  if (elapsed >= LOCK_HOLD_MS) {
    void classifyGesture();
  }
}

async function classifyGesture() {
  if (gestureState === "classifying" || gestureState === "cooldown") return;

  const snapshot = gestureBuffer.map(row => [...row]);
  gestureState = "classifying";
  gestureBuffer = [];
  resetLock();
  holdStatus.textContent = "최종 확인 중...";

  try {
    if (activeInferencePromise) {
      try { await activeInferencePromise; } catch { /* final inference below */ }
    }

    const result = await inferFrames(snapshot);
    gestureState = "cooldown";

    const accepted = isDisplayable(result);
    studyLogger.recordFinalPrediction(result, accepted);

    if (!accepted) {
      currentPrediction.textContent = "...";
      predictionScore.textContent = result
        ? `최고 ${(result.score * 100).toFixed(1)}% · 마진 ${(result.margin * 100).toFixed(1)}%`
        : "";
      holdStatus.textContent = "확신이 낮습니다. 후보를 선택하거나 다시 시도하세요";
      holdBarFill.style.width = "0%";
      lastResult = null;
      showRepairCandidates(result?.candidates ?? []);
      return;
    }

    hideRepairCandidates();
    const word = displayName(result.label);
    currentPrediction.textContent = word;
    predictionScore.textContent =
      `신뢰도 ${(result.score * 100).toFixed(1)}% · 마진 ${(result.margin * 100).toFixed(1)}%`;
    holdBarFill.style.width = "100%";
    lastResult = result;

    if (autoAddEnabled && result.score >= MIN_SCORE_LOCK) {
      addWordToSentence(result.label);
    } else if (result.score >= MIN_SCORE_LOCK) {
      holdStatus.textContent = `“${word}” 확인됨 — + 버튼으로 추가`;
    } else {
      holdStatus.textContent = "신뢰도가 충분하지 않습니다. 필요하면 후보를 선택하세요";
      showRepairCandidates(result.candidates ?? []);
    }
  } catch (e) {
    console.error(e);
    gestureState = "cooldown";
    holdStatus.textContent = "모델 추론 오류 — 다시 시도해주세요";
  } finally {
    setTimeout(resetGesture, COOLDOWN_MS);
  }
}

function showRepairCandidates(candidates) {
  repairCandidates.innerHTML = "";
  const usable = candidates
    .filter(c => c.label && c.label !== "기타" && c.score >= 0.12)
    .slice(0, 3);

  if (!usable.length) {
    repairCandidates.classList.add("hidden");
    return;
  }

  const label = document.createElement("div");
  label.className = "repair-label";
  label.textContent = "혹시 이 표현인가요?";
  repairCandidates.appendChild(label);

  for (const candidate of usable) {
    const btn = document.createElement("button");
    btn.className = "repair-btn";
    btn.textContent = `${displayName(candidate.label)} ${Math.round(candidate.score * 100)}%`;
    btn.addEventListener("click", () => {
      studyLogger.recordCandidateRepair(candidate.label);
      addWordToSentence(candidate.label);
      lastResult = candidate;
      hideRepairCandidates();
    });
    repairCandidates.appendChild(btn);
  }
  repairCandidates.classList.remove("hidden");
}

function hideRepairCandidates() {
  repairCandidates.classList.add("hidden");
  repairCandidates.innerHTML = "";
}

function addWordToSentence(labelId) {
  const word = displayName(labelId);
  sentenceWords.push(word);
  updateSentenceUI();
  scheduleAutoSpeak();
  holdStatus.textContent = `“${word}” 추가됨`;
}

function resetGesture() {
  if (!appStarted) return;
  gestureState = "waiting";
  gestureBuffer = [];
  noHandCount = 0;
  resetLock();
  holdBarFill.style.width = "0%";
  holdStatus.textContent = "손을 카메라에 보여주세요";
}

function addWord() {
  if (!lastResult || lastResult.label === "기타") {
    holdStatus.textContent = "추가할 인식 결과가 없습니다";
    return;
  }
  addWordToSentence(lastResult.label);
  lastResult = null;
}

function clearSentence() {
  clearTimeout(autoSpeakTimer);
  sentenceWords = [];
  updateSentenceUI();
}

function updateSentenceUI() {
  sentenceOutput.innerHTML = "";

  if (!sentenceWords.length) {
    const empty = document.createElement("span");
    empty.className = "sentence-empty";
    empty.textContent = "수어로 단어를 추가해보세요";
    sentenceOutput.appendChild(empty);
    sentenceNatural.textContent = "";
    return;
  }

  sentenceWords.forEach((word, i) => {
    const chip = document.createElement("span");
    chip.className = "sentence-chip";

    const text = document.createElement("span");
    text.textContent = word;

    const del = document.createElement("button");
    del.className = "sentence-chip-del";
    del.textContent = "×";
    del.addEventListener("click", () => {
      sentenceWords.splice(i, 1);
      updateSentenceUI();
    });

    chip.append(text, del);
    sentenceOutput.appendChild(chip);
  });

  sentenceNatural.textContent = decodeCafeWords(sentenceWords);
}

function speakText(text) {
  const value = String(text || "").trim();
  if (!value) return;
  const utt = new SpeechSynthesisUtterance(value);
  utt.lang = "ko-KR";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utt);
}

function speakSentence() {
  if (!sentenceWords.length) {
    holdStatus.textContent = "읽을 문장이 없습니다";
    return;
  }
  clearTimeout(autoSpeakTimer);
  const text = decodeCafeWords(sentenceWords);
  studyLogger.recordSentenceSpoken("recognized", text.length);
  speakText(text);
}

function scheduleAutoSpeak() {
  if (!autoSpeakEnabled || !sentenceWords.length) return;
  clearTimeout(autoSpeakTimer);
  autoSpeakTimer = setTimeout(speakSentence, 2800);
}

startBtn.addEventListener("click", () => {
  appStarted = !appStarted;
  lastResult = null;
  hideRepairCandidates();

  if (appStarted) {
    startBtn.textContent = "■";
    startBtn.classList.add("recording");
    statusDot.classList.add("active");
    runStatus.textContent = autoAddEnabled ? "인식 중" : "인식 중 · 수동추가";
    currentPrediction.textContent = "—";
    predictionScore.textContent = "";
    resetGesture();
    predictWebcam();
  } else {
    gestureState = "waiting";
    gestureBuffer = [];
    startBtn.textContent = "▶";
    startBtn.classList.remove("recording");
    statusDot.classList.remove("active");
    runStatus.textContent = "중지됨";
    currentPrediction.textContent = "—";
    predictionScore.textContent = "";
    holdStatus.textContent = "안정화 대기 중";
    holdBarFill.style.width = "0%";
  }
});

addWordBtn.addEventListener("click", addWord);
clearSentenceBtn.addEventListener("click", clearSentence);
speakSentenceBtn.addEventListener("click", speakSentence);

autoSpeakToggle.addEventListener("click", () => {
  autoSpeakEnabled = !autoSpeakEnabled;
  autoSpeakToggle.classList.toggle("accent", autoSpeakEnabled);
  autoSpeakToggle.classList.toggle("ghost", !autoSpeakEnabled);
  autoSpeakToggle.textContent = autoSpeakEnabled ? "자동읽기 ON" : "자동읽기";
  if (!autoSpeakEnabled) clearTimeout(autoSpeakTimer);
});

autoAddToggle.addEventListener("click", () => {
  autoAddEnabled = !autoAddEnabled;
  autoAddToggle.classList.toggle("accent", autoAddEnabled);
  autoAddToggle.classList.toggle("ghost", !autoAddEnabled);
  autoAddToggle.textContent = autoAddEnabled ? "자동추가" : "수동추가";
  if (appStarted) runStatus.textContent = autoAddEnabled ? "인식 중" : "인식 중 · 수동추가";
});

function initSTT() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    listenBtn.textContent = "🎤 미지원";
    listenBtn.disabled = true;
    return;
  }

  recognition = new SR();
  recognition.lang = "ko-KR";
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onresult = event => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        speechFinal.textContent = t.trim();
        speechInterim.textContent = "";
      } else {
        interim += t;
      }
    }
    if (interim) speechInterim.textContent = interim;
    speechPanel.classList.toggle("has-content", Boolean(speechFinal.textContent.trim() || interim));
  };

  recognition.onend = () => {
    if (isListening) {
      try { recognition.start(); } catch { /* browser may still be transitioning */ }
    }
  };

  recognition.onerror = e => {
    if (e.error !== "no-speech") speechInterim.textContent = `오류: ${e.error}`;
  };
}

function toggleListening() {
  if (!recognition) return;
  isListening = !isListening;

  if (isListening) {
    speechFinal.textContent = "";
    speechInterim.textContent = "";
    speechPanel.classList.remove("hidden");
    speechPanel.classList.remove("has-content");
    listenBtn.textContent = "🔴 듣기 중지";
    listenBtn.classList.add("listen-active");
    try { recognition.start(); } catch {}
  } else {
    try { recognition.stop(); } catch {}
    speechPanel.classList.add("hidden");
    speechPanel.classList.remove("has-content");
    listenBtn.textContent = "🎤 듣기";
    listenBtn.classList.remove("listen-active");
  }
}

listenBtn.addEventListener("click", toggleListening);
clearSpeechBtn.addEventListener("click", () => {
  speechFinal.textContent = "";
  speechInterim.textContent = "";
  speechPanel.classList.remove("has-content");
});

function openManualPanel() {
  studyLogger.recordManualFallback();
  manualPanel.classList.remove("hidden");
  setTimeout(() => manualText.focus(), 0);
}

function closeManualPanel() {
  manualPanel.classList.add("hidden");
}

fallbackBtn.addEventListener("click", openManualPanel);
manualCloseBtn.addEventListener("click", closeManualPanel);
manualPanel.addEventListener("click", e => {
  if (e.target === manualPanel) closeManualPanel();
});
manualSpeakBtn.addEventListener("click", () => {
  const text = manualText.value.trim();
  if (!text) return;
  studyLogger.recordSentenceSpoken("manual", text.length);
  speakText(text);
});
document.querySelectorAll(".quick-phrase").forEach(btn => {
  btn.addEventListener("click", () => {
    manualText.value = btn.textContent.trim();
  });
});

function initStudyControls() {
  if (!studyLogger.enabled) return;
  studyControls.classList.remove("hidden");
  studyParticipant.textContent = `파일럿 · ${studyLogger.participant}`;

  const setTaskActive = active => {
    studyStartBtn.disabled = active;
    studySuccessBtn.disabled = !active;
    studyFailBtn.disabled = !active;
  };

  studyStartBtn.addEventListener("click", () => {
    clearSentence();
    hideRepairCandidates();
    const id = studyLogger.startTask();
    studyParticipant.textContent = `파일럿 · ${studyLogger.participant} · ${id}`;
    setTaskActive(true);
  });

  studySuccessBtn.addEventListener("click", () => {
    const task = studyLogger.finishTask(true);
    studyParticipant.textContent = `파일럿 · ${studyLogger.participant} · ${task?.id ?? ""} 성공`;
    setTaskActive(false);
  });

  studyFailBtn.addEventListener("click", () => {
    const task = studyLogger.finishTask(false);
    studyParticipant.textContent = `파일럿 · ${studyLogger.participant} · ${task?.id ?? ""} 실패`;
    setTaskActive(false);
  });

  studyExportBtn.addEventListener("click", () => studyLogger.download());
}

async function preload() {
  initStudyControls();
  try {
    await loadLabels();
    await loadModelStack();
    launchBtn.textContent = "시작하기";
    launchBtn.disabled = false;
  } catch (e) {
    console.error(e);
    modelBadge.textContent = "모델 로딩 실패";
    launchBtn.textContent = "모델 오류";
    launchBtn.disabled = true;
  }
}

async function launchApp() {
  if (!landmarkersReady) {
    cameraStatus.textContent = "MediaPipe 준비 중...";
    await createLandmarkers();
    landmarkersReady = true;
  }

  cameraStatus.textContent = "카메라 연결 중...";
  await setupCamera();
  runStatus.textContent = "대기 중";

  if (!sttReady) {
    initSTT();
    sttReady = true;
  }
}

launchBtn.addEventListener("click", async () => {
  launchBtn.disabled = true;
  try {
    await launchApp();
    $("startScreen").classList.add("hidden");
    $("mainApp").classList.remove("hidden");
  } catch (e) {
    console.error(e);
    modelBadge.textContent = "카메라 권한 또는 장치 상태를 확인해주세요";
  } finally {
    launchBtn.disabled = false;
  }
});

brandBtn.addEventListener("click", () => {
  appStarted = false;
  gestureState = "waiting";
  gestureBuffer = [];
  startBtn.textContent = "▶";
  startBtn.classList.remove("recording");
  statusDot.classList.remove("active");

  if (isListening) toggleListening();
  window.speechSynthesis.cancel();

  webcamRunning = false;
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  $("mainApp").classList.add("hidden");
  $("startScreen").classList.remove("hidden");
});

preload();
