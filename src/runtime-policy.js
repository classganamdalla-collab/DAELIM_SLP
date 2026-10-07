export function getRuntimePolicy(featureMode, v2FeatureProfile = "full") {
  if (featureMode === "v1") {
    return {
      id: "v1-legacy",
      handStride: 3,
      sampleEveryVideoFrame: true,
      noHandEndFrames: 8,
      cooldownMs: 1500,
      autoLockWhileHandVisible: true,
    };
  }

  if (featureMode === "v2" && v2FeatureProfile === "base8") {
    // Match the original 2.0-facemesh collector:
    // - hand landmarks were refreshed every 2 video frames
    // - the latest hand result was nevertheless appended on every video frame
    // - a gesture ended after 3 consecutive no-hand frames
    //
    // Also avoid finalizing while the hand is still visible. The legacy
    // training samples were segmented by putting the hand down, so final
    // classification should use the completed gesture instead of an early
    // mid-gesture snapshot.
    return {
      id: "v2-base8-legacy-collector",
      handStride: 2,
      sampleEveryVideoFrame: true,
      noHandEndFrames: 3,
      cooldownMs: 650,
      autoLockWhileHandVisible: false,
    };
  }

  // Current v2 collector records one feature frame per fresh hand-landmarker
  // result and uses a 4-update no-hand stop threshold.
  return {
    id: "v2-full",
    handStride: 2,
    sampleEveryVideoFrame: false,
    noHandEndFrames: 4,
    cooldownMs: 900,
    autoLockWhileHandVisible: true,
  };
}
