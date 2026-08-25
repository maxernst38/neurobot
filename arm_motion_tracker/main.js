import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import { computeJointMetrics, JointStateClassifier, METRICS, stateLabel, POSE } from "./motion.js";
import { RobotLink } from "./robotLink.js";

const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const sideSelect = document.getElementById("sideSelect");
const statusEl = document.getElementById("status");
const primaryEl = document.getElementById("primary");
const metricsEl = document.getElementById("metrics");
const robotHostInput = document.getElementById("robotHost");
const robotEnable = document.getElementById("robotEnable");
const robotStatusEl = document.getElementById("robotStatus");

robotHostInput.value = `${location.hostname || "localhost"}:8765`;

const classifier = new JointStateClassifier();

const robotLink = new RobotLink((status) => {
  const labels = {
    connecting: "Connecting...",
    connected: "Connected — sending motion",
    disconnected: "Not connected",
    error: "Connection error",
  };
  robotStatusEl.textContent = labels[status] || status;
  robotStatusEl.className = "robot-status" + (status === "connected" ? " connected" : status === "error" ? " error" : "");
  if (status === "disconnected" || status === "error") {
    robotEnable.checked = false;
  }
});

robotEnable.addEventListener("change", () => {
  if (robotEnable.checked) {
    robotLink.connect(`ws://${robotHostInput.value}/ws`);
  } else {
    robotLink.disconnect();
    robotStatusEl.textContent = "Not connected";
    robotStatusEl.className = "robot-status";
  }
});

const metricEls = {};
for (const def of METRICS) {
  const row = document.createElement("div");
  row.className = "metric";
  row.innerHTML = `
    <div class="metric-head">
      <span class="metric-label">${def.label}</span>
      <span class="metric-value" data-value>--</span>
    </div>
    <div class="metric-bar"><div class="metric-bar-fill" data-fill></div></div>
    <div class="metric-direction" data-direction>No data</div>
  `;
  metricsEl.appendChild(row);
  metricEls[def.key] = {
    value: row.querySelector("[data-value]"),
    fill: row.querySelector("[data-fill]"),
    direction: row.querySelector("[data-direction]"),
  };
}

const POSE_CONNECTIONS = [
  [POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER],
  [POSE.LEFT_SHOULDER, POSE.LEFT_ELBOW], [POSE.LEFT_ELBOW, POSE.LEFT_WRIST],
  [POSE.RIGHT_SHOULDER, POSE.RIGHT_ELBOW], [POSE.RIGHT_ELBOW, POSE.RIGHT_WRIST],
  [POSE.LEFT_SHOULDER, POSE.LEFT_HIP], [POSE.RIGHT_SHOULDER, POSE.RIGHT_HIP],
  [POSE.LEFT_HIP, POSE.RIGHT_HIP],
];
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

let poseLandmarker = null;
let handLandmarker = null;
let running = false;

function setStatus(text) {
  statusEl.textContent = text;
}

async function initModels() {
  setStatus("Loading models...");
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });

  setStatus("Models ready");
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// Match a detected hand to LEFT/RIGHT pose wrist by nearest image-space distance,
// since MediaPipe's own handedness label can disagree with pose left/right when
// the camera feed isn't mirrored the same way the model expects.
function matchHandToSide(handResults, pose) {
  const matches = { LEFT: null, RIGHT: null };
  if (!handResults.landmarks || handResults.landmarks.length === 0) return matches;

  const targets = {
    LEFT: pose?.[POSE.LEFT_WRIST],
    RIGHT: pose?.[POSE.RIGHT_WRIST],
  };

  for (const landmarks of handResults.landmarks) {
    const handWrist = landmarks[0];
    let bestSide = null;
    let bestDist = Infinity;
    for (const side of ["LEFT", "RIGHT"]) {
      const t = targets[side];
      if (!t) continue;
      const d = Math.hypot(handWrist.x - t.x, handWrist.y - t.y);
      if (d < bestDist) {
        bestDist = d;
        bestSide = side;
      }
    }
    if (bestSide && bestDist < 0.15) {
      matches[bestSide] = landmarks;
    }
  }
  return matches;
}

function drawSkeleton(pose, hands) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;

  if (pose) {
    ctx.strokeStyle = "#4f8cff";
    for (const [a, b] of POSE_CONNECTIONS) {
      const pa = pose[a], pb = pose[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x * canvas.width, pa.y * canvas.height);
      ctx.lineTo(pb.x * canvas.width, pb.y * canvas.height);
      ctx.stroke();
    }
    ctx.fillStyle = "#e6e9ef";
    for (const idx of [POSE.LEFT_SHOULDER, POSE.RIGHT_SHOULDER, POSE.LEFT_ELBOW, POSE.RIGHT_ELBOW, POSE.LEFT_WRIST, POSE.RIGHT_WRIST]) {
      const p = pose[idx];
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  for (const landmarks of hands) {
    if (!landmarks) continue;
    ctx.strokeStyle = "#33c481";
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = landmarks[a], pb = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(pa.x * canvas.width, pa.y * canvas.height);
      ctx.lineTo(pb.x * canvas.width, pb.y * canvas.height);
      ctx.stroke();
    }
  }
}

function updateUI(results) {
  const active = [];

  for (const def of METRICS) {
    const m = results[def.key];
    const el = metricEls[def.key];
    if (!m || m.value === null) {
      el.value.textContent = "--";
      el.direction.textContent = "No data";
      el.fill.style.width = "0%";
      el.fill.className = "metric-bar-fill";
      continue;
    }
    el.value.textContent = def.unit === "deg" ? `${m.value.toFixed(0)}°` : m.value.toFixed(2);
    el.direction.textContent = stateLabel(def, m.state);

    const [lo, hi] = def.range;
    const pct = ((m.value - lo) / (hi - lo)) * 100;
    el.fill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    el.fill.className = "metric-bar-fill" + (m.state === "high" ? " state-high" : m.state === "low" ? " state-low" : "");

    if (m.state === "high" || m.state === "low") {
      active.push(`${def.label.replace(" Ext/Flex", "")}: ${stateLabel(def, m.state)}`);
    }
  }

  if (active.length > 0) {
    primaryEl.textContent = active.join("  ·  ");
    primaryEl.className = "primary-banner active";
  } else {
    primaryEl.textContent = "All joints neutral";
    primaryEl.className = "primary-banner";
  }
}

function renderLoop() {
  if (!running) return;

  const nowMs = performance.now();
  const poseResult = poseLandmarker.detectForVideo(video, nowMs);
  const handResult = handLandmarker.detectForVideo(video, nowMs);

  const pose = poseResult.landmarks?.[0] ?? null;
  const hands = matchHandToSide(handResult, pose);

  drawSkeleton(pose, [hands.LEFT, hands.RIGHT]);

  const side = sideSelect.value;
  const raw = computeJointMetrics(pose, hands[side], side);
  const results = classifier.update(raw);
  updateUI(results);

  const states = {};
  for (const def of METRICS) states[def.key] = results[def.key]?.state ?? "none";
  robotLink.send(states, nowMs);

  requestAnimationFrame(renderLoop);
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  try {
    await initModels();
    await startCamera();
    running = true;
    setStatus("Tracking");
    requestAnimationFrame(renderLoop);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    startBtn.disabled = false;
  }
});

sideSelect.addEventListener("change", () => classifier.reset());
