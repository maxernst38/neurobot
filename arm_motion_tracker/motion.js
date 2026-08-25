// Pure geometry + classification helpers for turning MediaPipe pose/hand
// landmarks into joint angles and extend/flex-style motion predictions.
// No DOM or MediaPipe imports here so this stays easy to reuse/test standalone.

export const POSE = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
};

export const HAND = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

// Metric keys, matched (where possible) to the SO-101 joint names used
// elsewhere in this project (shoulder_pan, shoulder_lift, elbow_flex,
// wrist_flex, wrist_roll) so this can later feed a teleoperation mapping.
// highLabel/midLabel/lowLabel describe the three discrete states each
// metric is classified into (top/middle/bottom band of `range`).
export const METRICS = [
  { key: "shoulder_rotation", label: "Shoulder Rotation", unit: "deg", range: [-180, 180], highLabel: "Rotated Out", midLabel: "Neutral", lowLabel: "Rotated In" },
  { key: "shoulder_flexion", label: "Shoulder Ext/Flex", unit: "deg", range: [0, 180], highLabel: "Flexed (raised)", midLabel: "Neutral", lowLabel: "Extended (lowered)" },
  { key: "elbow_flexion", label: "Elbow Ext/Flex", unit: "deg", range: [0, 180], highLabel: "Extended", midLabel: "Neutral", lowLabel: "Flexed" },
  { key: "wrist_flexion", label: "Wrist Ext/Flex", unit: "deg", range: [-90, 90], highLabel: "Extended", midLabel: "Neutral", lowLabel: "Flexed" },
  { key: "wrist_rotation", label: "Wrist Rotation", unit: "deg", range: [-180, 180], highLabel: "Supinated", midLabel: "Neutral", lowLabel: "Pronated" },
  { key: "grip", label: "Hand Grip", unit: "", range: [0, 1], highLabel: "Gripping", midLabel: "Half", lowLabel: "Open" },
];

function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z ?? 0) - (a.z ?? 0) };
}

function dot(u, v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

function mag(v) {
  return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) || 1e-6;
}

function cross(u, v) {
  return {
    x: u.y * v.z - u.z * v.y,
    y: u.z * v.x - u.x * v.z,
    z: u.x * v.y - u.y * v.x,
  };
}

// Angle at vertex `b`, formed by rays b->a and b->c, in degrees.
function angleAt(a, b, c) {
  const v1 = vec(b, a);
  const v2 = vec(b, c);
  const cos = Math.min(1, Math.max(-1, dot(v1, v2) / (mag(v1) * mag(v2))));
  return (Math.acos(cos) * 180) / Math.PI;
}

function dist(a, b) {
  return mag(vec(a, b));
}

/**
 * Compute the six joint metrics for one side of the body.
 * @param {Array} pose - 33 pose landmarks (normalized x,y,z), or null.
 * @param {Array} hand - 21 hand landmarks matching the same side, or null.
 * @param {"LEFT"|"RIGHT"} side
 * @returns {Object<string, number|null>} raw metric values, null if not computable
 */
export function computeJointMetrics(pose, hand, side) {
  const shoulder = pose?.[POSE[`${side}_SHOULDER`]];
  const elbow = pose?.[POSE[`${side}_ELBOW`]];
  const wrist = pose?.[POSE[`${side}_WRIST`]];
  const hip = pose?.[POSE[`${side}_HIP`]];

  const out = {
    shoulder_rotation: null,
    shoulder_flexion: null,
    elbow_flexion: null,
    wrist_flexion: null,
    wrist_rotation: null,
    grip: null,
  };

  if (shoulder && elbow && wrist && hip) {
    out.elbow_flexion = angleAt(shoulder, elbow, wrist);
    out.shoulder_flexion = angleAt(hip, shoulder, elbow);
    // Rough axial-rotation proxy: swing angle of the forearm through the
    // depth (z) axis while the upper arm is roughly fixed. True shoulder
    // rotation isn't fully recoverable from a single monocular camera, so
    // treat this as an approximation, not a precise IMU-grade reading.
    out.shoulder_rotation = (Math.atan2(wrist.z - elbow.z, wrist.x - elbow.x) * 180) / Math.PI;
  }

  if (hand) {
    const handWrist = hand[HAND.WRIST];
    const middleMcp = hand[HAND.MIDDLE_MCP];
    if (elbow && handWrist && middleMcp) {
      // Signed deviation from "straight" (0 deg), not the unsigned angleAt():
      // bending the wrist palm-down and bending it back both shrink the
      // unsigned angle the same way, so that form can't tell flexion from
      // extension apart. The cross-product sign gives the bend direction.
      const forearmVec = vec(elbow, handWrist);
      const handVec = vec(handWrist, middleMcp);
      const cross2d = forearmVec.x * handVec.y - forearmVec.y * handVec.x;
      const dot2d = forearmVec.x * handVec.x + forearmVec.y * handVec.y;
      out.wrist_flexion = (Math.atan2(cross2d, dot2d) * 180) / Math.PI;
    }

    const indexMcp = hand[HAND.INDEX_MCP];
    const pinkyMcp = hand[HAND.PINKY_MCP];
    if (handWrist && indexMcp && pinkyMcp) {
      const normal = cross(vec(handWrist, indexMcp), vec(handWrist, pinkyMcp));
      out.wrist_rotation = (Math.atan2(normal.x, normal.z) * 180) / Math.PI;
    }

    const fingers = [
      [HAND.INDEX_MCP, HAND.INDEX_PIP, HAND.INDEX_DIP, HAND.INDEX_TIP],
      [HAND.MIDDLE_MCP, HAND.MIDDLE_PIP, HAND.MIDDLE_DIP, HAND.MIDDLE_TIP],
      [HAND.RING_MCP, HAND.RING_PIP, HAND.RING_DIP, HAND.RING_TIP],
      [HAND.PINKY_MCP, HAND.PINKY_PIP, HAND.PINKY_DIP, HAND.PINKY_TIP],
    ];
    if (handWrist) {
      let curlSum = 0;
      let n = 0;
      for (const [mcp, pip, dip, tip] of fingers) {
        const m = hand[mcp], p = hand[pip], d = hand[dip], t = hand[tip];
        if (!m || !p || !d || !t) continue;
        const straight = dist(handWrist, t);
        const path = dist(handWrist, m) + dist(m, p) + dist(p, d) + dist(d, t);
        const extension = straight / (path || 1e-6);
        curlSum += 1 - Math.min(1, Math.max(0, extension));
        n += 1;
      }
      if (n > 0) out.grip = curlSum / n;
    }
  }

  return out;
}

/**
 * Smooths each metric's raw value, then classifies it into one of three
 * discrete bands: "high" (top bandFraction of range), "low" (bottom
 * bandFraction), or "mid" (everything between). Hysteresis keeps a value
 * sitting near a threshold from flickering between two states every frame.
 */
export class JointStateClassifier {
  constructor(smoothing = 0.35, bandFraction = 0.3, hysteresis = 0.05) {
    this.smoothing = smoothing;
    this.bandFraction = bandFraction;
    this.hysteresis = hysteresis;
    this.state = {}; // key -> { smoothed, band }
  }

  reset() {
    this.state = {};
  }

  /**
   * @param {Object<string, number|null>} rawMetrics
   * @returns {Object<string, { value: number|null, state: "high"|"mid"|"low"|"none" }>}
   */
  update(rawMetrics) {
    const results = {};

    for (const def of METRICS) {
      const raw = rawMetrics[def.key];
      const prev = this.state[def.key];

      if (raw === null || raw === undefined || Number.isNaN(raw)) {
        results[def.key] = { value: null, state: "none" };
        continue;
      }

      let smoothed = raw;
      if (prev && prev.smoothed !== null) {
        const isDeg = def.unit === "deg";
        const delta = isDeg ? shortestDeltaDeg(raw, prev.smoothed) : raw - prev.smoothed;
        smoothed = prev.smoothed + this.smoothing * delta;
      }

      const [lo, hi] = def.range;
      const span = hi - lo;
      const highThreshold = hi - this.bandFraction * span;
      const lowThreshold = lo + this.bandFraction * span;
      const margin = this.hysteresis * span;

      const prevBand = prev?.band ?? "mid";
      let band;
      if (prevBand === "high") {
        band = smoothed >= highThreshold - margin ? "high" : smoothed <= lowThreshold ? "low" : "mid";
      } else if (prevBand === "low") {
        band = smoothed <= lowThreshold + margin ? "low" : smoothed >= highThreshold ? "high" : "mid";
      } else {
        band = smoothed >= highThreshold ? "high" : smoothed <= lowThreshold ? "low" : "mid";
      }

      this.state[def.key] = { smoothed, band };
      results[def.key] = { value: smoothed, state: band };
    }

    return results;
  }
}

// Shortest signed difference between two degree values, e.g. shortestDeltaDeg(-179, 179) === 2,
// so a rotation crossing the +-180 wraparound doesn't register as a ~360deg spike.
function shortestDeltaDeg(curr, prev) {
  return (((curr - prev + 180) % 360) + 360) % 360 - 180;
}

export function stateLabel(def, state) {
  if (state === "high") return def.highLabel;
  if (state === "low") return def.lowLabel;
  if (state === "mid") return def.midLabel;
  return "No data";
}
