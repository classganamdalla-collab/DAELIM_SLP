from __future__ import annotations

import math
from typing import Any

import numpy as np

FACE_BASE_POINTS = [4, 10, 13, 33, 152, 234, 263, 454]
FACE_EXTENDED_POINTS = [4, 10, 13, 14, 33, 61, 105, 133, 145, 152, 159, 234, 263, 291, 334, 362, 374, 386, 454]
V2_BLENDSHAPES = [
    "browInnerUp", "browDownLeft", "browDownRight", "browOuterUpLeft", "browOuterUpRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight", "jawOpen",
    "mouthFunnel", "mouthPucker", "mouthSmileLeft", "mouthSmileRight",
]
V2_FEATURE_DIM = 190
V2_SCHEMA_ID = "ieum_v2_190"
V2_BASE8_SCHEMA_ID = "ieum_v2_base8_190"


def _f(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pt(p: Any) -> np.ndarray:
    if not isinstance(p, dict):
        return np.zeros(3, dtype=np.float32)
    return np.asarray([_f(p.get("x")), _f(p.get("y")), _f(p.get("z"))], dtype=np.float32)


def _dist3(a: Any, b: Any) -> float:
    return float(np.linalg.norm(_pt(a) - _pt(b)))


def _dist2(a: Any, b: Any) -> float:
    pa, pb = _pt(a), _pt(b)
    return float(np.linalg.norm(pa[:2] - pb[:2]))


def _face_map(frame: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ext = frame.get("faceExtended") or {}
    if isinstance(ext, dict):
        for k, p in ext.items():
            out[str(k)] = p if isinstance(p, dict) else {}

    old = frame.get("face") or []
    if isinstance(old, list):
        for idx, p in zip(FACE_BASE_POINTS, old):
            out.setdefault(str(idx), p if isinstance(p, dict) else {})
    return out


def _choose_hands(hands: list[dict]) -> tuple[dict | None, dict | None]:
    valid = [h for h in hands if isinstance(h, dict) and len(h.get("landmarks") or []) >= 21]
    left = next((h for h in valid if str(h.get("handedness", "")).lower().startswith("left")), None)
    right = next((h for h in valid if str(h.get("handedness", "")).lower().startswith("right")), None)

    leftovers = [h for h in valid if h is not left and h is not right]
    leftovers.sort(key=lambda h: _f((h.get("landmarks") or [{}])[0].get("x")))

    if left is None and leftovers:
        left = leftovers.pop(0)
    if right is None and leftovers:
        right = leftovers.pop(-1)

    if left is None and right is None and len(valid) == 1:
        left = valid[0]
    if left is right:
        right = None
    return left, right


def _palm_scale(hand: dict | None) -> float:
    if not hand:
        return 1.0
    lm = hand.get("landmarks") or []
    if len(lm) < 21:
        return 1.0
    wrist = lm[0]
    return max((_dist3(lm[5], wrist) + _dist3(lm[9], wrist) + _dist3(lm[17], wrist)) / 3.0, 1e-4)


def _face_scale(fmap: dict[str, dict]) -> float:
    temple = _dist3(fmap.get("234"), fmap.get("454"))
    eyes = _dist3(fmap.get("33"), fmap.get("263"))
    return max(temple or eyes or 0.0, 1e-4)


def _encode_hand(hand: dict | None, nose: np.ndarray, fscale: float, face_present: bool) -> list[float]:
    if not hand:
        return [0.0] * 69

    lm = hand.get("landmarks") or []
    if len(lm) < 21:
        return [0.0] * 69

    wrist = _pt(lm[0])
    scale = _palm_scale(hand)
    out: list[float] = []

    for p in lm[:21]:
        d = (_pt(p) - wrist) / scale
        out.extend(map(float, d))

    v1 = _pt(lm[5]) - wrist
    v2 = _pt(lm[17]) - wrist
    normal = np.cross(v1, v2)
    n = float(np.linalg.norm(normal))
    if n > 1e-6:
        normal /= n
    else:
        normal[:] = 0
    out.extend(map(float, normal))

    if face_present:
        out.extend(map(float, (wrist - nose) / fscale))
    else:
        out.extend([0.0, 0.0, 0.0])
    return out


def _has_face_points(fmap: dict[str, dict], *ids: int) -> bool:
    return all(str(idx) in fmap for idx in ids)


def _encode_face(
    fmap: dict[str, dict],
    blend: dict[str, Any],
    present: bool,
    profile: str = "full",
) -> list[float]:
    if not present:
        return [0.0] * 46

    nose = _pt(fmap.get("4"))
    scale = _face_scale(fmap)
    out: list[float] = []

    for idx in FACE_BASE_POINTS:
        d = (_pt(fmap.get(str(idx))) - nose) / scale
        out.extend(map(float, d))

    mouth_open = mouth_width = 0.0
    left_eye_open = right_eye_open = 0.0
    left_brow_raise = right_brow_raise = 0.0

    if profile != "base8":
        if _has_face_points(fmap, 13, 14):
            mouth_open = _dist2(fmap["13"], fmap["14"]) / scale
        if _has_face_points(fmap, 61, 291):
            mouth_width = _dist2(fmap["61"], fmap["291"]) / scale
        if _has_face_points(fmap, 33, 133, 159, 145):
            left_eye_width = max(_dist2(fmap["33"], fmap["133"]), 1e-4)
            left_eye_open = _dist2(fmap["159"], fmap["145"]) / left_eye_width
        if _has_face_points(fmap, 263, 362, 386, 374):
            right_eye_width = max(_dist2(fmap["263"], fmap["362"]), 1e-4)
            right_eye_open = _dist2(fmap["386"], fmap["374"]) / right_eye_width
        if _has_face_points(fmap, 105, 159):
            left_brow_raise = _dist2(fmap["105"], fmap["159"]) / scale
        if _has_face_points(fmap, 334, 386):
            right_brow_raise = _dist2(fmap["334"], fmap["386"]) / scale

    sin_roll = cos_roll = 0.0
    if _has_face_points(fmap, 33, 263):
        eye_a = _pt(fmap["33"])
        eye_b = _pt(fmap["263"])
        angle = math.atan2(float(eye_b[1] - eye_a[1]), float(eye_b[0] - eye_a[0]))
        sin_roll, cos_roll = math.sin(angle), math.cos(angle)

    out.extend([
        mouth_open, mouth_width, left_eye_open, right_eye_open,
        left_brow_raise, right_brow_raise, sin_roll, cos_roll,
    ])
    out.extend(
        0.0 if profile == "base8" else _f((blend or {}).get(name))
        for name in V2_BLENDSHAPES
    )
    return out


def extract_v2_frame(frame: dict, profile: str = "full") -> np.ndarray:
    hands = frame.get("hands") or []
    left, right = _choose_hands(hands)
    fmap = _face_map(frame)
    face_present = bool(frame.get("facePresent", bool(fmap)))
    nose = _pt(fmap.get("4"))
    fscale = _face_scale(fmap)

    features: list[float] = []
    features.extend(_encode_hand(left, nose, fscale, face_present))
    features.extend(_encode_hand(right, nose, fscale, face_present))

    if left and right:
        lw = _pt((left.get("landmarks") or [{}])[0])
        rw = _pt((right.get("landmarks") or [{}])[0])
        scale = fscale if face_present else max((_palm_scale(left) + _palm_scale(right)) / 2.0, 1e-4)
        features.extend(map(float, (rw - lw) / scale))
    else:
        features.extend([0.0, 0.0, 0.0])

    features.extend(_encode_face(fmap, frame.get("faceBlendshapes") or {}, face_present, profile))
    features.extend([1.0 if left else 0.0, 1.0 if right else 0.0, 1.0 if face_present else 0.0])

    arr = np.asarray(features, dtype=np.float32)
    if arr.shape != (V2_FEATURE_DIM,):
        raise ValueError(f"feature dimension mismatch: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite feature generated")
    return arr


def pad_or_trim(frames: list[dict], max_len: int = 100, profile: str = "full") -> tuple[np.ndarray, int]:
    seq = [extract_v2_frame(f, profile=profile) for f in frames[:max_len]]
    valid = len(seq)
    out = np.zeros((max_len, V2_FEATURE_DIM), dtype=np.float32)
    if seq:
        out[:valid] = np.stack(seq)
    return out, valid


def _self_test() -> None:
    hand = {
        "handedness": "Left",
        "landmarks": [{"x": i / 100.0, "y": i / 120.0, "z": -i / 300.0} for i in range(21)],
    }
    frame = {
        "hands": [hand],
        "face": [{"x": 0.5 + i / 1000.0, "y": 0.4 + i / 1200.0, "z": 0.0} for i in range(8)],
        "facePresent": True,
    }
    x = extract_v2_frame(frame)
    assert x.shape == (190,)
    assert np.isfinite(x).all()
    seq, valid = pad_or_trim([frame] * 12)
    assert seq.shape == (100, 190) and valid == 12
    print("feature self-test OK", V2_SCHEMA_ID, x.shape, seq.shape)


if __name__ == "__main__":
    _self_test()
