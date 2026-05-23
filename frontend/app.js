const API_BASE = "http://localhost:8000";
const SAMPLE_DURATION = 12;

const text = {
  idle: "\u52d5\u753b\u3092\u9078\u629e\u3059\u308b\u304b\u3001\u30b5\u30f3\u30d7\u30eb\u3092\u8868\u793a\u3057\u3066\u304f\u3060\u3055\u3044\u3002",
  loaded: "\u3092\u8aad\u307f\u8fbc\u307f\u307e\u3057\u305f\u3002",
  analyzing: "\u89e3\u6790\u4e2d\u3067\u3059\u3002",
  done: "\u89e3\u6790\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f\u3002",
  failed: "\u89e3\u6790\u306b\u5931\u6557\u3057\u307e\u3057\u305f",
  sample: "\u30b5\u30f3\u30d7\u30eb\u89e3\u6790\u7d50\u679c\u3092\u8868\u793a\u3057\u3066\u3044\u307e\u3059\u3002",
  samplePlaying: "\u30b5\u30f3\u30d7\u30eb\u3092\u518d\u751f\u4e2d\u3067\u3059\u3002",
  samplePaused: "\u30b5\u30f3\u30d7\u30eb\u518d\u751f\u3092\u505c\u6b62\u3057\u307e\u3057\u305f\u3002",
  play: "\u30b5\u30f3\u30d7\u30eb\u518d\u751f",
  pause: "\u505c\u6b62",
};

const sampleSummary = {
  job_id: "sample-match-001",
  metadata: {
    duration_seconds: 600,
    fps: 30,
    width: 1280,
    height: 720,
    frame_count: 18000,
  },
  teams: [
    { name: "Team A", color_hex: "#e53935", possession_rate: 0.57, estimated_passes: 82, estimated_shots: 9 },
    { name: "Team B", color_hex: "#1e88e5", possession_rate: 0.43, estimated_passes: 64, estimated_shots: 6 },
  ],
  players: [
    { track_id: "A-07", team: "Team A", distance_meters: 1124, confidence: 0.86 },
    { track_id: "A-10", team: "Team A", distance_meters: 980, confidence: 0.82 },
    { track_id: "B-04", team: "Team B", distance_meters: 1042, confidence: 0.8 },
    { track_id: "B-11", team: "Team B", distance_meters: 1188, confidence: 0.78 },
  ],
  ball_touch_candidates: 146,
  notes: [
    "\u3053\u308c\u306f\u30c7\u30e2\u7528\u306e\u30b5\u30f3\u30d7\u30eb\u30c7\u30fc\u30bf\u3067\u3059\u3002",
    "\u8d64\u3068\u9752\u306e\u30e6\u30cb\u30d5\u30a9\u30fc\u30e0\u8272\u3092\u4eee\u5b9a\u3057\u3066\u30c1\u30fc\u30e0\u5206\u985e\u3057\u3066\u3044\u307e\u3059\u3002",
    "\u52d5\u753b\u306e\u518d\u751f\u6642\u9593\u306b\u5408\u308f\u305b\u3066\u691c\u51fa\u67a0\u3001\u8ecc\u8de1\u3001\u30dc\u30fc\u30eb\u4f4d\u7f6e\u3092\u63cf\u753b\u3057\u307e\u3059\u3002",
  ],
};

const teamStyle = {
  home: {
    label: "Ally",
    color: "#ff6b5d",
    fill: "rgba(255, 107, 93, 0.14)",
  },
  away: {
    label: "Opponent",
    color: "#5eb4ff",
    fill: "rgba(94, 180, 255, 0.14)",
  },
};

const samplePaths = {
  "A-07": { color: "#ff6b5d", points: [[0.18, 0.65], [0.28, 0.52], [0.36, 0.47], [0.44, 0.38], [0.58, 0.44]] },
  "A-10": { color: "#ff6b5d", points: [[0.26, 0.28], [0.37, 0.31], [0.46, 0.43], [0.55, 0.35], [0.68, 0.32]] },
  "B-04": { color: "#5eb4ff", points: [[0.78, 0.32], [0.68, 0.38], [0.58, 0.47], [0.5, 0.58], [0.42, 0.62]] },
  "B-11": { color: "#5eb4ff", points: [[0.72, 0.72], [0.62, 0.65], [0.54, 0.58], [0.46, 0.63], [0.36, 0.71]] },
  ball: { color: "#ffffff", points: [[0.31, 0.52], [0.44, 0.38], [0.55, 0.35], [0.58, 0.47], [0.42, 0.62]] },
};

const videoInput = document.querySelector("#videoInput");
const analyzeButton = document.querySelector("#analyzeButton");
const sampleButton = document.querySelector("#sampleButton");
const samplePlaybackButton = document.querySelector("#samplePlaybackButton");
const statusText = document.querySelector("#status");
const preview = document.querySelector("#preview");
const overlay = document.querySelector("#overlay");
const viewer = document.querySelector("#viewer");
const summaryEl = document.querySelector("#summary");
const durationEl = document.querySelector("#duration");
const resolutionEl = document.querySelector("#resolution");
const teamAEl = document.querySelector("#teamA");
const teamBEl = document.querySelector("#teamB");

let selectedFile = null;
let mode = "idle";
let samplePlaying = false;
let sampleOffset = 0;
let sampleStartedAt = 0;
let animationId = 0;

videoInput.addEventListener("change", () => {
  selectedFile = videoInput.files?.[0] ?? null;
  mode = selectedFile ? "video" : "idle";
  samplePlaying = false;
  samplePlaybackButton.disabled = true;
  samplePlaybackButton.textContent = text.play;
  analyzeButton.disabled = !selectedFile;
  viewer.classList.remove("sample");

  if (!selectedFile) {
    statusText.textContent = text.idle;
    drawVideoOverlay(0);
    return;
  }

  preview.src = URL.createObjectURL(selectedFile);
  statusText.textContent = `${selectedFile.name} ${text.loaded}`;
  startAnimation();
});

sampleButton.addEventListener("click", () => {
  selectedFile = null;
  mode = "sample";
  samplePlaying = false;
  sampleOffset = 0;
  samplePlaybackButton.disabled = false;
  samplePlaybackButton.textContent = text.play;
  analyzeButton.disabled = true;
  viewer.classList.add("sample");
  preview.removeAttribute("src");
  preview.load();
  renderSummary(sampleSummary);
  drawSampleOverlay(0);
  statusText.textContent = text.sample;
});

samplePlaybackButton.addEventListener("click", () => {
  if (mode !== "sample") return;
  samplePlaying = !samplePlaying;
  if (samplePlaying) {
    sampleStartedAt = performance.now() / 1000 - sampleOffset;
    samplePlaybackButton.textContent = text.pause;
    statusText.textContent = text.samplePlaying;
    startAnimation();
  } else {
    samplePlaybackButton.textContent = text.play;
    statusText.textContent = text.samplePaused;
  }
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeButton.disabled = true;
  statusText.textContent = text.analyzing;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API_BASE}/api/videos`, { method: "POST", body: formData });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail ?? text.failed);
    }

    const job = await response.json();
    renderSummary(job.summary);
    statusText.textContent = text.done;
  } catch (error) {
    statusText.textContent = error.message;
  } finally {
    analyzeButton.disabled = false;
  }
});

window.addEventListener("resize", () => {
  if (mode === "sample") drawSampleOverlay(sampleOffset);
  if (mode === "video") drawVideoOverlay(preview.currentTime || 0);
});
preview.addEventListener("play", startAnimation);
preview.addEventListener("seeked", () => drawVideoOverlay(preview.currentTime || 0));
preview.addEventListener("loadedmetadata", () => drawVideoOverlay(0));

function startAnimation() {
  if (animationId) return;
  const tick = () => {
    animationId = 0;
    if (mode === "sample") {
      if (samplePlaying) {
        sampleOffset = (performance.now() / 1000 - sampleStartedAt) % SAMPLE_DURATION;
      }
      drawSampleOverlay(sampleOffset);
      if (samplePlaying) requestNextFrame();
    }
    if (mode === "video") {
      drawVideoOverlay(preview.currentTime || 0);
      if (!preview.paused && !preview.ended) requestNextFrame();
    }
  };
  const requestNextFrame = () => {
    animationId = requestAnimationFrame(tick);
  };
  requestNextFrame();
}

function renderSummary(summary) {
  const metadata = summary.metadata;
  durationEl.textContent = `${metadata.duration_seconds.toFixed(1)}s`;
  resolutionEl.textContent = `${metadata.width} x ${metadata.height}`;

  const teamA = summary.teams.find((team) => team.name === "Team A");
  const teamB = summary.teams.find((team) => team.name === "Team B");
  teamAEl.textContent = teamA ? `${Math.round(teamA.possession_rate * 100)}%` : "--";
  teamBEl.textContent = teamB ? `${Math.round(teamB.possession_rate * 100)}%` : "--";

  summaryEl.textContent = JSON.stringify(summary, null, 2);
}

function sizeOverlay() {
  const rect = viewer.getBoundingClientRect();
  overlay.width = Math.max(1, Math.floor(rect.width));
  overlay.height = Math.max(1, Math.floor(rect.height));
  return overlay.getContext("2d");
}

function drawVideoOverlay(time) {
  const ctx = sizeOverlay();
  const w = overlay.width;
  const h = overlay.height;
  ctx.clearRect(0, 0, w, h);
  if (!selectedFile) return;

  const phase = (time % 10) / 10;
  const passStats = estimateLiveStats(time);
  const aX = w * (0.16 + phase * 0.26);
  const bX = w * (0.68 - phase * 0.18);
  const ballX = w * (0.28 + phase * 0.38);
  const ballY = h * (0.5 + Math.sin(phase * Math.PI * 2) * 0.11);

  drawBox(ctx, "Ally #07", teamStyle.home, aX, h * 0.3, 76, 138);
  drawBox(ctx, "Ally #10", teamStyle.home, aX + w * 0.12, h * 0.48, 78, 132);
  drawBox(ctx, "Opponent #04", teamStyle.away, bX, h * 0.28, 76, 136);
  drawBox(ctx, "Opponent #11", teamStyle.away, bX - w * 0.12, h * 0.55, 78, 128);
  drawPassLine(ctx, teamStyle.home.color, aX + 76, h * 0.3 + 46, ballX, ballY);
  drawPassLine(ctx, teamStyle.away.color, bX, h * 0.28 + 66, ballX, ballY);
  drawBall(ctx, ballX, ballY);
  drawHud(ctx, time, passStats);
  drawLegend(ctx, passStats);
}

function drawSampleOverlay(time) {
  const ctx = sizeOverlay();
  const w = overlay.width;
  const h = overlay.height;
  const t = (time % SAMPLE_DURATION) / SAMPLE_DURATION;
  const passStats = estimateLiveStats(time);

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1f6b3d";
  ctx.fillRect(0, 0, w, h);
  drawPitch(ctx, w, h);

  Object.entries(samplePaths).forEach(([label, path]) => {
    if (label === "ball") return;
    const points = path.points.slice(0, Math.max(2, Math.ceil(t * path.points.length)));
    drawTrack(ctx, path.color, points);
  });

  Object.entries(samplePaths).forEach(([label, path]) => {
    const [x, y] = interpolatePath(path.points, t);
    if (label === "ball") {
      drawBall(ctx, w * x, h * y);
    } else {
      drawPlayer(ctx, label, path.color, w * x, h * y);
    }
  });

  drawHud(ctx, time, passStats);
  drawLegend(ctx, passStats);
}

function drawPitch(ctx, w, h) {
  ctx.strokeStyle = "rgba(255,255,255,0.78)";
  ctx.lineWidth = 2;
  ctx.strokeRect(w * 0.06, h * 0.08, w * 0.88, h * 0.84);
  ctx.beginPath();
  ctx.moveTo(w * 0.5, h * 0.08);
  ctx.lineTo(w * 0.5, h * 0.92);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(w * 0.5, h * 0.5, Math.min(w, h) * 0.12, 0, Math.PI * 2);
  ctx.stroke();
}

function interpolatePath(points, t) {
  const scaled = t * (points.length - 1);
  const index = Math.min(points.length - 2, Math.floor(scaled));
  const local = scaled - index;
  const [x1, y1] = points[index];
  const [x2, y2] = points[index + 1];
  return [x1 + (x2 - x1) * local, y1 + (y2 - y1) * local];
}

function drawTrack(ctx, color, points) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    const px = overlay.width * x;
    const py = overlay.height * y;
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function drawBox(ctx, label, style, x, y, width, height) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = style.color;
  ctx.fillStyle = style.fill;
  ctx.fillRect(x, y, width, height);
  ctx.fillStyle = style.color;
  ctx.strokeRect(x, y, width, height);
  ctx.fillRect(x, y - 20, Math.max(72, label.length * 7.5), 18);
  ctx.fillStyle = "#071007";
  ctx.fillText(label, x, y - 8);
}

function drawPassLine(ctx, color, x1, y1, x2, y2) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 6]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

function drawPlayer(ctx, label, color, x, y) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#071007";
  ctx.font = "700 12px Arial";
  ctx.fillText(label, x + 15, y + 4);
}

function drawBall(ctx, x, y) {
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawHud(ctx, time, stats) {
  ctx.font = "700 13px Arial";
  ctx.fillStyle = "rgba(0,0,0,0.66)";
  ctx.fillRect(14, 14, 270, 88);
  ctx.fillStyle = "#f2f6ef";
  ctx.fillText(`time ${time.toFixed(1)}s`, 26, 38);
  ctx.fillStyle = teamStyle.home.color;
  ctx.fillText(`Ally passes ${stats.homePasses}`, 26, 62);
  ctx.fillStyle = teamStyle.away.color;
  ctx.fillText(`Opponent passes ${stats.awayPasses}`, 26, 86);
}

function drawLegend(ctx, stats) {
  const x = overlay.width - 246;
  const y = 14;
  ctx.font = "700 13px Arial";
  ctx.fillStyle = "rgba(0,0,0,0.66)";
  ctx.fillRect(x, y, 232, 88);
  drawLegendRow(ctx, x + 14, y + 24, teamStyle.home.color, "Ally player");
  drawLegendRow(ctx, x + 14, y + 48, teamStyle.away.color, "Opponent player");
  ctx.fillStyle = "#ffffff";
  ctx.fillText(`Possession ${stats.possession}%`, x + 14, y + 74);
}

function drawLegendRow(ctx, x, y, color, label) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y - 12, 16, 16);
  ctx.fillText(label, x + 26, y + 1);
}

function estimateLiveStats(time) {
  return {
    homePasses: Math.floor(time / 4) + Math.floor((Math.sin(time * 0.65) + 1) * 2),
    awayPasses: Math.floor(time / 5) + Math.floor((Math.cos(time * 0.55) + 1) * 1.5),
    possession: Math.round(52 + Math.sin(time * 0.18) * 8),
  };
}
