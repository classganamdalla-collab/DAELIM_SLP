import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

import {
  FEATURE_SCHEMAS,
  FACE_EXTENDED_POINTS,
  cloneHandsFromResults,
  cloneFaceDataFromResults,
} from "./src/feature-schema.js";

const $ = id => document.getElementById(id);

const video = $("webcam");
const canvas = $("outputCanvas");
const canvasCtx = canvas.getContext("2d");
const labelGrid = $("labelGrid");
const situationBadge = $("situationBadge");
const sampleNote = $("sampleNote");
const participantCode = $("participantCode");
const sessionCode = $("sessionCode");
const dominantHand = $("dominantHand");
const environmentTag = $("environmentTag");
const startRecordBtn = $("startRecordBtn");
const stopRecordBtn = $("stopRecordBtn");
const exportJsonBtn = $("exportJsonBtn");
const mergeJsonBtn = $("mergeJsonBtn");
const mergeFileInput = $("mergeFileInput");
const clearSamplesBtn = $("clearSamplesBtn");
const deleteLastBtn = $("deleteLastBtn");
const autoRepeatBtn = $("autoRepeatBtn");
const cameraStatus = $("cameraStatus");
const handStatus = $("handStatus");
const recordStatus = $("recordStatus");
const frameCountStatus = $("frameCountStatus");
const qualityStatus = $("qualityStatus");
const sampleCountStatus = $("sampleCountStatus");
const samplesList = $("samplesList");
const labelSummary = $("labelSummary");
const hudLabel = $("hudLabel");
const hudRecord = $("hudRecord");

const MIN_SAMPLE_FRAMES = 15;
const MAX_SAMPLE_FRAMES = 140;
const NO_HAND_STOP_THRESHOLD = 4;

let handLandmarker = null;
let faceLandmarker = null;
let drawingUtils = null;
let webcamRunning = false;
let lastVideoTime = -1;
let latestHandCount = 0;

let labelsConfig = [];
let selectedLabel = null;
let collectedSamples = [];
let autoRepeat = false;

let isArmed = false;
let isRecording = false;
let currentRecording = null;
let noHandFrameStreak = 0;

let lastHandResults = null;
let lastFaceResults = null;
let handFrameCount = 0;
let faceFrameCount = 0;

const inferCanvas = document.createElement("canvas");
inferCanvas.width = 640;
inferCanvas.height = 480;
const inferCtx = inferCanvas.getContext("2d");

async function loadLabels() {
  try {
    const res = await fetch("./labels.json");
    if (!res.ok) throw new Error(`labels.json HTTP ${res.status}`);
    const json = await res.json();
    labelsConfig = json.labels || [];
    situationBadge.textContent = json.situation || "기본";
  } catch (e) {
    console.error(e);
    situationBadge.textContent = "labels.json 오류";
    labelsConfig = [];
  }
  buildLabelGrid();
}

function buildLabelGrid() {
  labelGrid.innerHTML = "";
  for (const label of labelsConfig) {
    const btn = document.createElement("button");
    btn.className = "label-btn";
    btn.textContent = label.korean;
    btn.dataset.id = label.id;
    btn.addEventListener("click", () => selectLabel(label.id));
    labelGrid.appendChild(btn);
  }
  if (labelsConfig.length) selectLabel(labelsConfig[0].id);
}

function selectLabel(id) {
  selectedLabel = id;
  document.querySelectorAll(".label-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.id === id);
  });
  const found = labelsConfig.find(l => l.id === id);
  hudLabel.textContent = found?.korean ?? id;
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
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => { video.onloadedmetadata = resolve; });
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    webcamRunning = true;
    cameraStatus.textContent = "📷 카메라 연결 완료";
  } catch (e) {
    console.error(e);
    cameraStatus.textContent = "📷 카메라 연결 실패";
  }
}

function drawResults(handResults, faceResults) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  canvasCtx.translate(canvas.width, 0);
  canvasCtx.scale(-1, 1);

  for (const lm of handResults?.landmarks ?? []) {
    drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, {
      lineWidth: 3, color: "#4f8cff"
    });
    drawingUtils.drawLandmarks(lm, { radius: 4, color: "#5fd4c4" });
  }

  const faceLM = faceResults?.faceLandmarks?.[0];
  if (faceLM?.length) {
    for (const idx of FACE_EXTENDED_POINTS) {
      const pt = faceLM[idx];
      if (!pt) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(pt.x * canvas.width, pt.y * canvas.height, 2.5, 0, Math.PI * 2);
      canvasCtx.fillStyle = "rgba(79,182,245,0.55)";
      canvasCtx.fill();
    }
  }
  canvasCtx.restore();
}

function predictWebcam() {
  if (!webcamRunning || !handLandmarker || !faceLandmarker) return;
  const now = performance.now();

  drawResults(lastHandResults, lastFaceResults);

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    inferCtx.drawImage(video, 0, 0, 640, 480);

    faceFrameCount++;
    if (faceFrameCount % 6 === 0) {
      lastFaceResults = faceLandmarker.detectForVideo(inferCanvas, now);
    }

    handFrameCount++;
    if (handFrameCount % 2 === 0) {
      lastHandResults = handLandmarker.detectForVideo(inferCanvas, now);
      const prev = latestHandCount;
      latestHandCount = lastHandResults?.landmarks?.length ?? 0;
      if (prev !== latestHandCount) {
        handStatus.textContent = latestHandCount
          ? `✋ 손 감지됨: ${latestHandCount}개`
          : "✋ 손 감지되지 않음";
      }
      handleAutoRecordingFlow(lastHandResults, lastFaceResults);
    }
  }

  requestAnimationFrame(predictWebcam);
}

function sessionMeta() {
  return {
    participantCode: participantCode.value.trim(),
    sessionCode: sessionCode.value.trim(),
    dominantHand: dominantHand.value,
    environmentTag: environmentTag.value,
  };
}

function addFrameToRecording(handResults, faceResults) {
  if (!isRecording || !currentRecording) return;
  const hands = cloneHandsFromResults(handResults);
  if (!hands.length) return;

  const face = cloneFaceDataFromResults(faceResults);
  currentRecording.frames.push({
    timestamp: Date.now(),
    hands,
    face: face.keyPoints,
    faceExtended: face.extended,
    faceBlendshapes: face.blendshapes,
    facePresent: face.present,
  });

  frameCountStatus.textContent = `프레임: ${currentRecording.frames.length}`;

  const twoHand = currentRecording.frames.filter(f => f.hands.length >= 2).length;
  const faceCount = currentRecording.frames.filter(f => f.facePresent).length;
  const total = currentRecording.frames.length;
  qualityStatus.textContent =
    `품질: 양손 ${Math.round(100 * twoHand / total)}% · 얼굴 ${Math.round(100 * faceCount / total)}%`;

  if (currentRecording.frames.length >= MAX_SAMPLE_FRAMES) {
    finalizeRecording("최대 프레임 도달");
  }
}

function armRecording() {
  if (isArmed || isRecording) return alert("이미 기록 대기 중이거나 기록 중입니다.");
  if (!selectedLabel) return alert("라벨을 선택해주세요.");
  if (!participantCode.value.trim()) return alert("참여자 코드를 입력해주세요. 실명 대신 P01처럼 입력하세요.");
  if (!sessionCode.value.trim()) return alert("세션 코드를 입력해주세요. 예: S01");

  isArmed = true;
  noHandFrameStreak = 0;
  currentRecording = null;
  recordStatus.textContent = `⏺ 손이 보이면 자동 시작 (${selectedLabel})`;
  recordStatus.className = "stat-line rec-idle";
  frameCountStatus.textContent = "프레임: 0";
  qualityStatus.textContent = "품질: -";
}

function beginActualRecording() {
  currentRecording = {
    id: crypto.randomUUID(),
    label: selectedLabel,
    note: sampleNote.value.trim(),
    createdAt: new Date().toISOString(),
    ...sessionMeta(),
    collectorVersion: "3.0",
    rawSchema: "raw-landmarks-v3",
    targetFeatureSchema: FEATURE_SCHEMAS.v2.id,
    frames: [],
  };

  isRecording = true;
  isArmed = false;
  noHandFrameStreak = 0;
  hudRecord.classList.remove("hidden");
  recordStatus.textContent = `⏺ 기록 중: ${selectedLabel}`;
  recordStatus.className = "stat-line rec-active";
}

function handleAutoRecordingFlow(handResults, faceResults) {
  const hasHand = latestHandCount > 0;

  if (isArmed && !isRecording) {
    if (hasHand) {
      beginActualRecording();
      addFrameToRecording(handResults, faceResults);
    }
    return;
  }

  if (!isRecording) return;

  if (hasHand) {
    noHandFrameStreak = 0;
    addFrameToRecording(handResults, faceResults);
  } else {
    noHandFrameStreak++;
    if (noHandFrameStreak >= NO_HAND_STOP_THRESHOLD) {
      finalizeRecording("손이 내려가 자동 종료");
    }
  }
}

function sampleQuality(sample) {
  const frames = sample.frames ?? [];
  const total = Math.max(1, frames.length);
  const twoHand = frames.filter(f => (f.hands?.length ?? 0) >= 2).length;
  const face = frames.filter(f => f.facePresent).length;
  const handCounts = frames.map(f => f.hands?.length ?? 0);
  return {
    twoHandRatio: twoHand / total,
    faceRatio: face / total,
    meanDetectedHands: handCounts.reduce((a, b) => a + b, 0) / total,
  };
}

function finalizeRecording(reasonText = "자동 종료") {
  if (!currentRecording) return resetRecordingState();

  const frameCount = currentRecording.frames.length;
  if (frameCount < MIN_SAMPLE_FRAMES) {
    resetRecordingState();
    recordStatus.textContent = `⚠️ ${frameCount}프레임: 너무 짧아 저장하지 않음 (최소 ${MIN_SAMPLE_FRAMES})`;
    return;
  }

  const quality = sampleQuality(currentRecording);
  collectedSamples.push({ ...currentRecording, frameCount, quality, endReason: reasonText });
  resetRecordingState();
  updateSamplesUI();

  recordStatus.textContent = `✅ 저장됨 (${frameCount}f) — 총 ${collectedSamples.length}개`;
  qualityStatus.textContent =
    `품질: 양손 ${Math.round(quality.twoHandRatio * 100)}% · 얼굴 ${Math.round(quality.faceRatio * 100)}%`;

  if (autoRepeat) {
    setTimeout(() => {
      if (!isArmed && !isRecording) armRecording();
    }, 700);
  }
}

function resetRecordingState() {
  isArmed = false;
  isRecording = false;
  currentRecording = null;
  noHandFrameStreak = 0;
  hudRecord.classList.add("hidden");
  recordStatus.classList.remove("rec-active");
  recordStatus.classList.add("rec-idle");
  frameCountStatus.textContent = "프레임: 0";
}

function stopRecordingManually() {
  if (isRecording && currentRecording) return finalizeRecording("수동 종료");
  if (isArmed) {
    resetRecordingState();
    recordStatus.textContent = "기록 대기 취소됨";
    return;
  }
  alert("중지할 기록이 없습니다.");
}

function deleteLastSample() {
  if (isArmed || isRecording) return alert("기록 중에는 삭제할 수 없습니다.");
  if (!collectedSamples.length) return alert("삭제할 표본이 없습니다.");
  collectedSamples.pop();
  updateSamplesUI();
}

function deleteSampleById(sampleId) {
  if (isArmed || isRecording) return alert("기록 중에는 삭제할 수 없습니다.");
  collectedSamples = collectedSamples.filter(s => s.id !== sampleId);
  updateSamplesUI();
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function updateSamplesUI() {
  sampleCountStatus.textContent = `저장된 표본: ${collectedSamples.length}개`;

  const summary = {};
  for (const s of collectedSamples) summary[s.label] = (summary[s.label] || 0) + 1;
  labelSummary.innerHTML = Object.entries(summary)
    .map(([label, count]) => `<span class="label-chip">${escapeHtml(label)} × ${count}</span>`)
    .join("");

  if (!collectedSamples.length) {
    samplesList.innerHTML = "아직 저장된 표본이 없습니다.";
    return;
  }

  samplesList.innerHTML = collectedSamples.slice().reverse().map(s => {
    const q = s.quality ?? sampleQuality(s);
    return `
      <div class="sample-item">
        <div style="display:flex;justify-content:space-between;gap:8px;align-items:flex-start">
          <div>
            <div class="s-label">${escapeHtml(s.label)}</div>
            <div class="s-meta">
              ${s.frameCount}f · ${escapeHtml(s.participantCode || "-")} / ${escapeHtml(s.sessionCode || "-")}
              · 양손 ${Math.round((q.twoHandRatio || 0) * 100)}% · 얼굴 ${Math.round((q.faceRatio || 0) * 100)}%
            </div>
            <div class="s-meta">${s.note ? escapeHtml(s.note) : "메모없음"}</div>
          </div>
          <button class="delete-sample-btn" data-sample-id="${s.id}">삭제</button>
        </div>
      </div>`;
  }).join("");

  document.querySelectorAll(".delete-sample-btn").forEach(btn => {
    btn.addEventListener("click", () => deleteSampleById(btn.dataset.sampleId));
  });
}

function getLabelsSummary() {
  const s = {};
  for (const sample of collectedSamples) s[sample.label] = (s[sample.label] || 0) + 1;
  return s;
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportJson() {
  if (!collectedSamples.length) return alert("내보낼 표본이 없습니다.");

  const meta = sessionMeta();
  const data = {
    exportedAt: new Date().toISOString(),
    version: "3.0-raw-landmarks",
    rawSchema: "raw-landmarks-v3",
    targetFeatureSchema: FEATURE_SCHEMAS.v2,
    situation: situationBadge.textContent,
    ...meta,
    totalSamples: collectedSamples.length,
    labelsSummary: getLabelsSummary(),
    samples: collectedSamples,
  };

  const safeP = meta.participantCode.replace(/[^A-Za-z0-9_-]/g, "_") || "P";
  const safeS = meta.sessionCode.replace(/[^A-Za-z0-9_-]/g, "_") || "S";
  downloadJson(data, `ieum_${safeP}_${safeS}_${Date.now()}.json`);
}

mergeJsonBtn.addEventListener("click", () => mergeFileInput.click());
mergeFileInput.addEventListener("change", async e => {
  const files = Array.from(e.target.files ?? []);
  let merged = [...collectedSamples];
  const existingIds = new Set(merged.map(s => s.id));
  let added = 0;

  for (const file of files) {
    try {
      const json = JSON.parse(await file.text());
      for (const s of json.samples ?? []) {
        if (!s.id || existingIds.has(s.id)) continue;
        merged.push(s);
        existingIds.add(s.id);
        added++;
      }
    } catch (err) {
      console.error(file.name, err);
    }
  }
  collectedSamples = merged;
  mergeFileInput.value = "";
  updateSamplesUI();
  alert(`병합 완료: ${added}개 추가됨`);
});

function clearSamples() {
  if (isArmed || isRecording) return alert("기록 중에는 전체 삭제할 수 없습니다.");
  if (!collectedSamples.length) return;
  if (!confirm("현재 브라우저에 모인 표본을 모두 지울까요? 내보내지 않은 데이터는 복구할 수 없습니다.")) return;
  collectedSamples = [];
  updateSamplesUI();
}

startRecordBtn.addEventListener("click", armRecording);
stopRecordBtn.addEventListener("click", stopRecordingManually);
exportJsonBtn.addEventListener("click", exportJson);
clearSamplesBtn.addEventListener("click", clearSamples);
deleteLastBtn.addEventListener("click", deleteLastSample);
autoRepeatBtn.addEventListener("click", () => {
  autoRepeat = !autoRepeat;
  autoRepeatBtn.textContent = autoRepeat ? "연속 모드 ON" : "연속 모드 OFF";
  autoRepeatBtn.classList.toggle("btn-primary", autoRepeat);
  autoRepeatBtn.classList.toggle("btn-ghost", !autoRepeat);
});

async function initApp() {
  await loadLabels();
  cameraStatus.textContent = "📷 MediaPipe 준비 중...";
  await createLandmarkers();
  await setupCamera();
  resetRecordingState();
  updateSamplesUI();
  predictWebcam();
}

initApp();
