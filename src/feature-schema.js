export const FACE_BASE_POINTS = [4, 10, 13, 33, 152, 234, 263, 454];

export const FACE_EXTENDED_POINTS = [
  4, 10, 13, 14, 33, 61, 105, 133, 145, 152, 159,
  234, 263, 291, 334, 362, 374, 386, 454
];

export const V2_BLENDSHAPES = [
  "browInnerUp",
  "browDownLeft",
  "browDownRight",
  "browOuterUpLeft",
  "browOuterUpRight",
  "eyeBlinkLeft",
  "eyeBlinkRight",
  "eyeWideLeft",
  "eyeWideRight",
  "jawOpen",
  "mouthFunnel",
  "mouthPucker",
  "mouthSmileLeft",
  "mouthSmileRight",
];

export const FEATURE_SCHEMAS = {
  v1: {
    id: "ieum_v1_90",
    featureDim: 90,
    description: "첫 번째 감지 손 66 + 얼굴 핵심점 24 (기존 모델 호환)",
  },
  v2: {
    id: "ieum_v2_190",
    featureDim: 190,
    description: "양손 정규화 + 얼굴 상대 위치/표정 + 선택적 blendshape + 존재 마스크",
  },
};

const ZERO_POINT = Object.freeze({ x: 0, y: 0, z: 0 });

function finite(n) {
  return Number.isFinite(n) ? n : 0;
}

function copyPoint(p) {
  if (!p) return { ...ZERO_POINT };
  return { x: finite(p.x), y: finite(p.y), z: finite(p.z) };
}

function sub(a, b) {
  return [finite(a?.x) - finite(b?.x), finite(a?.y) - finite(b?.y), finite(a?.z) - finite(b?.z)];
}

function norm3(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function dist3(a, b) {
  return norm3(sub(a, b));
}

function dist2(a, b) {
  return Math.hypot(finite(a?.x) - finite(b?.x), finite(a?.y) - finite(b?.y));
}

function safeDiv(x, d) {
  return Math.abs(d) > 1e-6 ? x / d : 0;
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalized(v) {
  const n = norm3(v);
  return n > 1e-6 ? v.map(x => x / n) : [0, 0, 0];
}

function handednessCategory(handResults, i) {
  const groups = handResults?.handedness ?? handResults?.handednesses ?? [];
  const cat = groups?.[i]?.[0] ?? null;
  return cat?.categoryName ?? cat?.displayName ?? cat?.label ?? "Unknown";
}

export function cloneHandsFromResults(handResults) {
  if (!handResults?.landmarks) return [];
  return handResults.landmarks.map((lm, i) => ({
    handedness: handednessCategory(handResults, i),
    landmarks: lm.map(copyPoint),
  }));
}

export function cloneFaceDataFromResults(faceResults) {
  const faceLM = faceResults?.faceLandmarks?.[0];
  if (!faceLM?.length) {
    return {
      present: false,
      keyPoints: FACE_BASE_POINTS.map(() => ({ ...ZERO_POINT })),
      extended: {},
      blendshapes: {},
    };
  }

  const extended = {};
  for (const idx of FACE_EXTENDED_POINTS) {
    extended[String(idx)] = copyPoint(faceLM[idx]);
  }

  const blendshapes = {};
  for (const cat of faceResults?.faceBlendshapes?.[0] ?? []) {
    const name = cat?.categoryName ?? cat?.displayName;
    if (name) blendshapes[name] = finite(cat.score);
  }

  return {
    present: true,
    keyPoints: FACE_BASE_POINTS.map(idx => copyPoint(faceLM[idx])),
    extended,
    blendshapes,
  };
}

function baseFaceMap(frame) {
  const out = {};
  for (const [k, p] of Object.entries(frame?.faceExtended ?? {})) {
    out[String(k)] = copyPoint(p);
  }

  const old = Array.isArray(frame?.face) ? frame.face : [];
  FACE_BASE_POINTS.forEach((idx, i) => {
    if (!out[String(idx)] && old[i]) out[String(idx)] = copyPoint(old[i]);
  });
  return out;
}

function chooseHandSlots(hands) {
  const valid = (hands ?? []).filter(h => Array.isArray(h?.landmarks) && h.landmarks.length >= 21);
  let left = valid.find(h => String(h.handedness).toLowerCase().startsWith("left")) ?? null;
  let right = valid.find(h => String(h.handedness).toLowerCase().startsWith("right")) ?? null;

  const leftovers = valid.filter(h => h !== left && h !== right)
    .sort((a, b) => finite(a.landmarks[0]?.x) - finite(b.landmarks[0]?.x));

  if (!left && leftovers.length) left = leftovers.shift();
  if (!right && leftovers.length) right = leftovers.pop() ?? leftovers.shift() ?? null;

  if (!left && !right && valid.length === 1) left = valid[0];
  if (left === right) right = null;
  return { left, right };
}

function handPalmScale(hand) {
  const lm = hand?.landmarks;
  if (!lm || lm.length < 21) return 1;
  const wrist = lm[0];
  const d5 = dist3(lm[5], wrist);
  const d9 = dist3(lm[9], wrist);
  const d17 = dist3(lm[17], wrist);
  return Math.max((d5 + d9 + d17) / 3, 1e-4);
}

function faceScale(faceMap) {
  const temple = dist3(faceMap["234"], faceMap["454"]);
  const eyes = dist3(faceMap["33"], faceMap["263"]);
  return Math.max(temple || eyes || 0, 1e-4);
}

function encodeHand(hand, nose, fScale, facePresent) {
  const out = [];
  if (!hand?.landmarks || hand.landmarks.length < 21) {
    return new Array(69).fill(0);
  }

  const lm = hand.landmarks;
  const wrist = lm[0];
  const scale = handPalmScale(hand);

  for (const pt of lm.slice(0, 21)) {
    const d = sub(pt, wrist);
    out.push(safeDiv(d[0], scale), safeDiv(d[1], scale), safeDiv(d[2], scale));
  }

  const v1 = sub(lm[5], wrist);
  const v2 = sub(lm[17], wrist);
  out.push(...normalized(cross(v1, v2)));

  if (facePresent) {
    const wr = sub(wrist, nose);
    out.push(safeDiv(wr[0], fScale), safeDiv(wr[1], fScale), safeDiv(wr[2], fScale));
  } else {
    out.push(0, 0, 0);
  }

  return out;
}

function encodeFace(faceMap, blendshapes, present) {
  if (!present) return new Array(46).fill(0);

  const nose = faceMap["4"] ?? ZERO_POINT;
  const scale = faceScale(faceMap);
  const out = [];

  for (const idx of FACE_BASE_POINTS) {
    const p = faceMap[String(idx)] ?? ZERO_POINT;
    const d = sub(p, nose);
    out.push(safeDiv(d[0], scale), safeDiv(d[1], scale), safeDiv(d[2], scale));
  }

  const mouthOpen = safeDiv(dist2(faceMap["13"], faceMap["14"]), scale);
  const mouthWidth = safeDiv(dist2(faceMap["61"], faceMap["291"]), scale);

  const leftEyeWidth = Math.max(dist2(faceMap["33"], faceMap["133"]), 1e-4);
  const rightEyeWidth = Math.max(dist2(faceMap["263"], faceMap["362"]), 1e-4);
  const leftEyeOpen = safeDiv(dist2(faceMap["159"], faceMap["145"]), leftEyeWidth);
  const rightEyeOpen = safeDiv(dist2(faceMap["386"], faceMap["374"]), rightEyeWidth);

  const leftBrowRaise = safeDiv(dist2(faceMap["105"], faceMap["159"]), scale);
  const rightBrowRaise = safeDiv(dist2(faceMap["334"], faceMap["386"]), scale);

  const eyeA = faceMap["33"] ?? ZERO_POINT;
  const eyeB = faceMap["263"] ?? ZERO_POINT;
  const dx = finite(eyeB.x) - finite(eyeA.x);
  const dy = finite(eyeB.y) - finite(eyeA.y);
  const angle = Math.atan2(dy, dx);

  out.push(
    mouthOpen,
    mouthWidth,
    leftEyeOpen,
    rightEyeOpen,
    leftBrowRaise,
    rightBrowRaise,
    Math.sin(angle),
    Math.cos(angle)
  );

  for (const name of V2_BLENDSHAPES) {
    out.push(finite(blendshapes?.[name]));
  }

  return out;
}

export function extractV2FeaturesFromRawFrame(frame) {
  const hands = frame?.hands ?? [];
  const { left, right } = chooseHandSlots(hands);
  const fmap = baseFaceMap(frame);
  const facePresent = Boolean(frame?.facePresent ?? Object.keys(fmap).length);
  const nose = fmap["4"] ?? ZERO_POINT;
  const fScale = faceScale(fmap);

  const features = [
    ...encodeHand(left, nose, fScale, facePresent),
    ...encodeHand(right, nose, fScale, facePresent),
  ];

  if (left && right) {
    const delta = sub(right.landmarks[0], left.landmarks[0]);
    const scale = facePresent
      ? fScale
      : Math.max((handPalmScale(left) + handPalmScale(right)) / 2, 1e-4);
    features.push(safeDiv(delta[0], scale), safeDiv(delta[1], scale), safeDiv(delta[2], scale));
  } else {
    features.push(0, 0, 0);
  }

  features.push(...encodeFace(fmap, frame?.faceBlendshapes ?? {}, facePresent));
  features.push(left ? 1 : 0, right ? 1 : 0, facePresent ? 1 : 0);

  if (features.length !== FEATURE_SCHEMAS.v2.featureDim) {
    throw new Error(`v2 feature dimension mismatch: ${features.length}`);
  }
  return features;
}

export function extractV2FeaturesFromResults(handResults, faceResults) {
  const face = cloneFaceDataFromResults(faceResults);
  return extractV2FeaturesFromRawFrame({
    hands: cloneHandsFromResults(handResults),
    face: face.keyPoints,
    faceExtended: face.extended,
    faceBlendshapes: face.blendshapes,
    facePresent: face.present,
  });
}

export function extractV1FeaturesFromResults(handResults, faceResults) {
  const features = [];

  if (handResults?.landmarks?.length) {
    const lm = handResults.landmarks[0];
    const wrist = lm[0];
    for (const pt of lm) {
      features.push(
        finite(pt.x) - finite(wrist.x),
        finite(pt.y) - finite(wrist.y),
        finite(pt.z) - finite(wrist.z)
      );
    }

    const v1 = sub(lm[5], wrist);
    const v2 = sub(lm[17], wrist);
    features.push(...normalized(cross(v1, v2)));
  } else {
    features.push(...new Array(66).fill(0));
  }

  const faceLM = faceResults?.faceLandmarks?.[0];
  if (faceLM?.length) {
    const nose = faceLM[FACE_BASE_POINTS[0]];
    for (const idx of FACE_BASE_POINTS) {
      const pt = faceLM[idx];
      features.push(
        pt ? finite(pt.x) - finite(nose.x) : 0,
        pt ? finite(pt.y) - finite(nose.y) : 0,
        pt ? finite(pt.z) - finite(nose.z) : 0
      );
    }
  } else {
    features.push(...new Array(24).fill(0));
  }

  if (features.length !== FEATURE_SCHEMAS.v1.featureDim) {
    throw new Error(`v1 feature dimension mismatch: ${features.length}`);
  }
  return features;
}
