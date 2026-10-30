"use strict";

// ---------------------------------------------------------------- constants
const N_CHANNELS = 12;
const TAU = 2 * Math.PI;
// Parameters of data/generated/kuramoto_propagation_series.csv
// (see synthetic_datasets_metadata.json, "kuramoto_propagation").
const NOTEBOOK = {
  n_points: 1500, time_step: 0.02, coupling: 0.7, seed: 42, source_channel: 0,
  noise: {
    enabled: true, event_onset: 932, event_duration: 75, event_magnitude_in_sigma: 2.0,
    hurst_exponent: 0.75, delay_per_hop: 5, decay_per_hop: 0.65,
  },
};

const state = structuredClone(NOTEBOOK);
state.n_channels = N_CHANNELS;
state.natural_frequencies = new Array(N_CHANNELS).fill(1);
state.initial_phases = new Array(N_CHANNELS).fill(0);

let data = null;          // last generator output
let cursor = 0;           // sample index shown on the ring
let playing = false;
let firstRun = true;

const $ = (id) => document.getElementById(id);
const chName = (i) => `channel_${String(i).padStart(2, "0")}`;
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "–");

// ---------------------------------------------------------------- worker
const worker = new Worker("worker.js");
const pending = new Map();
let nextId = 0;

function call(message) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...message, id });
  });
}

worker.onmessage = ({ data: msg }) => {
  if (msg.type === "ready") {
    setStatus("ok", `Ready · numpy ${msg.numpyVersion} in Pyodide ${msg.pyodideVersion}`);
    applyPreset("notebook");
    return;
  }
  if (msg.type === "fatal") { setStatus("error", "Failed to start Python: " + msg.message); return; }
  const job = pending.get(msg.id);
  if (!job) return;
  pending.delete(msg.id);
  if (msg.type === "result") job.resolve(msg.result); else job.reject(new Error(msg.message));
};
worker.onerror = (event) => setStatus("error", "Worker error: " + event.message);

function setStatus(kind, text) {
  $("status").className = "status " + kind;
  $("status-text").textContent = text;
}

// Run the generator for the latest state; coalesce bursts of slider events.
let inFlight = false, dirty = false;
function schedule() {
  dirty = true;
  if (!inFlight) pump();
}
async function pump() {
  dirty = false;
  inFlight = true;
  setStatus("busy", "Running data_generator.py…");
  try {
    const result = await call({ kind: "run", params: snapshot() });
    data = result;
    if (firstRun) {
      cursor = state.noise.event_onset + Math.round(state.noise.event_duration * 0.6);
      firstRun = false;
    }
    cursor = Math.min(cursor, data.meta.n_points - 1);
    invalidate();
    setStatus("ok", `Generated ${data.meta.n_points} × ${data.meta.n_channels} samples in ${Math.round(result.elapsed)} ms`);
  } catch (error) {
    setStatus("error", error.message);
  }
  inFlight = false;
  if (dirty) pump();
}

function snapshot() {
  return {
    n_points: state.n_points, n_channels: N_CHANNELS, source_channel: state.source_channel,
    seed: state.seed, coupling: state.coupling, time_step: state.time_step,
    natural_frequencies: state.natural_frequencies, initial_phases: state.initial_phases,
    noise: { ...state.noise },
  };
}

async function drawSeededOscillators() {
  const seeded = await call({
    kind: "defaults", n_channels: N_CHANNELS, source_channel: state.source_channel, seed: state.seed,
  });
  state.natural_frequencies = seeded.natural_frequencies;
  state.initial_phases = seeded.initial_phases;
  syncNodes();
}

// ---------------------------------------------------------------- controls
const wrapPhase = (p) => Math.atan2(Math.sin(p), Math.cos(p));

function buildNodes() {
  const host = $("nodes");
  for (let i = 0; i < N_CHANNELS; i++) {
    const row = document.createElement("div");
    row.className = "node";
    row.id = `node-${i}`;
    row.innerHTML = `
      <span class="name">${String(i).padStart(2, "0")}</span>
      <label>
        <span>ω <output id="w-out-${i}"></output></span>
        <input id="w-${i}" type="range" min="0.2" max="2" step="0.001" aria-label="${chName(i)} natural frequency">
      </label>
      <label>
        <span>φ <output id="p-out-${i}"></output></span>
        <input id="p-${i}" type="range" min="${-Math.PI}" max="${Math.PI}" step="0.01" aria-label="${chName(i)} initial phase">
      </label>`;
    host.appendChild(row);
    row.querySelector(`#w-${i}`).addEventListener("input", (e) => {
      state.natural_frequencies[i] = Number(e.target.value);
      $(`w-out-${i}`).textContent = fmt(state.natural_frequencies[i], 3);
      schedule();
    });
    row.querySelector(`#p-${i}`).addEventListener("input", (e) => {
      state.initial_phases[i] = Number(e.target.value);
      $(`p-out-${i}`).textContent = fmt(state.initial_phases[i], 2);
      schedule();
    });
  }
}

function syncNodes() {
  for (let i = 0; i < N_CHANNELS; i++) {
    state.initial_phases[i] = wrapPhase(state.initial_phases[i]);
    $(`w-${i}`).value = state.natural_frequencies[i];
    $(`p-${i}`).value = state.initial_phases[i];
    $(`w-out-${i}`).textContent = fmt(state.natural_frequencies[i], 3);
    $(`p-out-${i}`).textContent = fmt(state.initial_phases[i], 2);
    $(`node-${i}`).classList.toggle("source", i === state.source_channel);
  }
}

// Keep the noise window inside what generate_kuramoto_propagation_series accepts.
const maxDelay = () => Math.floor(N_CHANNELS / 2) * state.noise.delay_per_hop;
function applyLimits() {
  const n = state.n_points, nz = state.noise, floorQuarter = Math.floor(n / 4);
  const durationMax = Math.max(5, Math.min(400, n - maxDelay() - floorQuarter - 1));
  nz.event_duration = Math.min(nz.event_duration, durationMax);
  const delayMax = Math.max(0, Math.min(30, Math.floor((n - nz.event_duration - floorQuarter - 1) / Math.floor(N_CHANNELS / 2))));
  nz.delay_per_hop = Math.min(nz.delay_per_hop, delayMax);
  const onsetMax = n - nz.event_duration - maxDelay();
  nz.event_onset = Math.max(0, Math.min(nz.event_onset, onsetMax));
  $("event_duration").max = durationMax;
  $("delay_per_hop").max = delayMax;
  $("event_onset").max = onsetMax;
  $("cursor").max = n - 1;
}

const NOISE_FIELDS = {
  event_onset: (v) => `${v} · t = ${fmt(v * state.time_step)} s`,
  event_duration: (v) => `${v} · ${fmt(v * state.time_step)} s`,
  event_magnitude_in_sigma: (v) => fmt(v, 1),
  hurst_exponent: (v) => fmt(v, 2),
  delay_per_hop: (v) => String(v),
  decay_per_hop: (v) => fmt(v, 2),
};

function syncControls() {
  applyLimits();
  $("coupling").value = state.coupling;
  $("coupling-out").textContent = fmt(state.coupling);
  $("n_points").value = state.n_points;
  $("time_step").value = state.time_step;
  $("seed").value = state.seed;
  $("source_channel").value = state.source_channel;
  $("noise-enabled").checked = state.noise.enabled;
  $("noise-fields").classList.toggle("off", !state.noise.enabled);
  for (const [key, label] of Object.entries(NOISE_FIELDS)) {
    $(key).value = state.noise[key];
    $(`${key}-out`).textContent = label(state.noise[key]);
  }
  $("hint-delay").textContent = state.noise.delay_per_hop;
  $("hint-decay").textContent = fmt(state.noise.decay_per_hop);
  $("source-title").textContent = `Source ${chName(state.source_channel)}: background + noise`;
  syncNodes();
  syncOverlayChips();
}

function bindControls() {
  const sel = $("source_channel");
  for (let i = 0; i < N_CHANNELS; i++) sel.add(new Option(chName(i), i));

  $("coupling").addEventListener("input", (e) => { state.coupling = Number(e.target.value); syncControls(); schedule(); });
  $("n_points").addEventListener("change", (e) => {
    state.n_points = clampInt(e.target.value, 300, 4000, state.n_points); syncControls(); schedule();
  });
  $("time_step").addEventListener("change", (e) => {
    const v = Number(e.target.value);
    if (v >= 0.005 && v <= 0.1) state.time_step = v;
    syncControls(); schedule();
  });
  $("seed").addEventListener("change", async (e) => {
    state.seed = clampInt(e.target.value, 0, 99999, state.seed);
    await drawSeededOscillators(); syncControls(); schedule();
  });
  $("redraw").addEventListener("click", async () => { await drawSeededOscillators(); schedule(); });
  sel.addEventListener("change", (e) => { state.source_channel = Number(e.target.value); syncControls(); schedule(); });
  $("noise-enabled").addEventListener("change", (e) => { state.noise.enabled = e.target.checked; syncControls(); schedule(); });
  for (const key of Object.keys(NOISE_FIELDS)) {
    $(key).addEventListener("input", (e) => { state.noise[key] = Number(e.target.value); syncControls(); schedule(); });
  }
  $("onset-at-cursor").addEventListener("click", () => { state.noise.event_onset = cursor; syncControls(); schedule(); });

  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => applyPreset(b.dataset.preset)));

  $("cursor").addEventListener("input", (e) => { cursor = Number(e.target.value); drawFrame(); });
  $("play").addEventListener("click", togglePlay);
}

function clampInt(value, lo, hi, fallback) {
  const v = Math.round(Number(value));
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
}

async function applyPreset(name) {
  if (name === "notebook") {
    const { noise, ...rest } = structuredClone(NOTEBOOK);
    Object.assign(state, rest);
    state.noise = noise;
    await drawSeededOscillators();
  } else if (name === "sync") {
    state.natural_frequencies = new Array(N_CHANNELS).fill(1);
    state.initial_phases = new Array(N_CHANNELS).fill(0);
  } else if (name === "strong") {
    state.coupling = 3;
  } else if (name === "uncoupled") {
    state.coupling = 0;
  }
  syncControls();
  schedule();
}

// ---------------------------------------------------------------- playback
let lastTick = 0;
function togglePlay() {
  playing = !playing;
  $("play").textContent = playing ? "❚❚ Pause" : "▶ Play";
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
  if (playing) { lastTick = 0; requestAnimationFrame(tick); }
}
function tick(now) {
  if (!playing) return;
  if (data && now - lastTick > 30) {
    lastTick = now;
    cursor = (cursor + Number($("speed").value)) % data.meta.n_points;
    drawFrame();
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- drawing helpers
let theme = {};
function readTheme() {
  const css = getComputedStyle(document.documentElement);
  for (const key of ["surface", "ink", "ink-2", "muted", "grid", "axis", "series-1", "series-2", "div-neg", "div-mid", "div-pos", "band"]) {
    theme[key] = css.getPropertyValue(`--${key}`).trim();
  }
}

function hexToRgb(hex) {
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
function diverging(value) {
  const t = Math.max(-1, Math.min(1, value));
  const [a, b] = t < 0 ? [theme["div-mid"], theme["div-neg"]] : [theme["div-mid"], theme["div-pos"]];
  const ca = hexToRgb(a), cb = hexToRgb(b), k = Math.abs(t);
  return `rgb(${ca.map((c, i) => Math.round(c + (cb[i] - c) * k)).join(",")})`;
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h, dpr };
}

function niceTicks(lo, hi, count) {
  const span = hi - lo, raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function drawTimeAxis(ctx, plot, n, dt) {
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = theme.muted;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of niceTicks(0, (n - 1) * dt, Math.max(2, Math.floor(plot.w / 90)))) {
    const x = plot.x + (t / dt / (n - 1)) * plot.w;
    ctx.fillText(`${t}s`, x, plot.y + plot.h + 6);
  }
  ctx.strokeStyle = theme.axis;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.x, plot.y + plot.h + 0.5);
  ctx.lineTo(plot.x + plot.w, plot.y + plot.h + 0.5);
  ctx.stroke();
}

function strokeSeries(ctx, plot, n, get, yOf, color, width) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = plot.x + (i / (n - 1)) * plot.w, y = yOf(get(i));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// Static layers are cached per chart and only the cursor is redrawn per frame.
const charts = {};
function chart(id, layout, drawStatic, hover) {
  const canvas = $(id);
  const layer = document.createElement("canvas");
  charts[id] = { canvas, layer, layout, drawStatic, hover, stale: true, plot: null };
  canvas.addEventListener("click", (e) => {
    const c = charts[id];
    if (!data || !c.plot) return;
    const idx = indexAt(c, e);
    if (idx !== null) { cursor = idx; drawFrame(); }
  });
  canvas.addEventListener("mousemove", (e) => showTip(charts[id], e));
  canvas.addEventListener("mouseleave", () => { $(`${id}-tip`).hidden = true; });
}

function indexAt(c, e) {
  const rect = c.canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const n = data.meta.n_points;
  if (x < c.plot.x || x > c.plot.x + c.plot.w) return null;
  return Math.round(((x - c.plot.x) / c.plot.w) * (n - 1));
}

function showTip(c, e) {
  const tip = $(`${c.canvas.id}-tip`);
  if (!data || !c.plot) return;
  const idx = indexAt(c, e);
  if (idx === null) { tip.hidden = true; return; }
  const rect = c.canvas.getBoundingClientRect();
  const html = c.hover(idx, e.clientY - rect.top);
  if (!html) { tip.hidden = true; return; }
  tip.innerHTML = html;
  tip.hidden = false;
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const flip = x + 16 + tip.offsetWidth > rect.width;
  tip.style.left = `${flip ? x - 16 - tip.offsetWidth : x + 16}px`;
  tip.style.top = `${Math.max(0, Math.min(y - 10, rect.height - tip.offsetHeight))}px`;
}

const tipRow = (label, value, color) =>
  `<div class="row"><span>${color ? `<span class="swatch line" style="background:${color}"></span>` : ""}${label}</span><b>${value}</b></div>`;

function renderChart(c) {
  const { ctx, w, h, dpr } = fitCanvas(c.canvas);
  if (c.stale || c.layer.width !== c.canvas.width || c.layer.height !== c.canvas.height) {
    c.layer.width = c.canvas.width;
    c.layer.height = c.canvas.height;
    const lctx = c.layer.getContext("2d");
    lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    lctx.clearRect(0, 0, w, h);
    c.plot = c.layout(w, h);
    c.drawStatic(lctx, c.plot, w, h);
    c.stale = false;
  }
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(c.layer, 0, 0, w, h);
  const n = data.meta.n_points;
  const x = Math.round(c.plot.x + (cursor / (n - 1)) * c.plot.w) + 0.5;
  ctx.strokeStyle = theme.ink;
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, c.plot.y);
  ctx.lineTo(x, c.plot.y + c.plot.h);
  ctx.stroke();
  ctx.globalAlpha = 1;
}

// ---------------------------------------------------------------- charts
const at = (arr, i, ch) => arr[i * N_CHANNELS + ch];
function eventWindow(ch) {
  if (!data.meta.event_onset && data.meta.event_onset !== 0) return null;
  const start = data.meta.event_onset + data.arrival[ch];
  return [start, start + data.meta.event_duration];
}

let laneScale = 1;
chart("lanes",
  (w, h) => ({ x: 74, y: 6, w: w - 74 - 10, h: h - 6 - 26 }),
  (ctx, plot) => {
    const n = data.meta.n_points, lh = plot.h / N_CHANNELS;
    laneScale = 1;
    for (const v of data.values) laneScale = Math.max(laneScale, Math.abs(v));
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      const mid = plot.y + lh * (ch + 0.5);
      const win = eventWindow(ch);
      if (win) {
        const x0 = plot.x + (win[0] / (n - 1)) * plot.w, x1 = plot.x + (win[1] / (n - 1)) * plot.w;
        ctx.fillStyle = theme.band;
        ctx.fillRect(x0, plot.y + lh * ch + 1, x1 - x0, lh - 2);
      }
      ctx.strokeStyle = theme.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.x, Math.round(mid) + 0.5);
      ctx.lineTo(plot.x + plot.w, Math.round(mid) + 0.5);
      ctx.stroke();
      const isSource = ch === state.source_channel && state.noise.enabled;
      ctx.fillStyle = isSource ? theme.ink : theme["ink-2"];
      ctx.textAlign = "right";
      ctx.font = `${isSource ? "600 " : ""}11px system-ui, sans-serif`;
      ctx.fillText(chName(ch).replace("channel_", "ch "), plot.x - 8, mid - (isSource ? 6 : 0));
      if (isSource) { ctx.fillStyle = theme.muted; ctx.fillText("source", plot.x - 8, mid + 7); }
      strokeSeries(ctx, plot, n, (i) => at(data.values, i, ch), (v) => mid - (v / laneScale) * lh * 0.46, theme["series-1"], 1);
    }
    drawTimeAxis(ctx, plot, n, data.meta.time_step);
  },
  (idx, y) => {
    const c = charts.lanes, lh = c.plot.h / N_CHANNELS;
    const ch = Math.floor((y - c.plot.y) / lh);
    if (ch < 0 || ch >= N_CHANNELS) return null;
    const v = at(data.values, idx, ch), bg = at(data.background, idx, ch);
    return `<b>${chName(ch)}</b> · t = ${fmt(idx * data.meta.time_step)} s<br>` +
      tipRow("observed", fmt(v, 3), theme["series-1"]) +
      tipRow("noise", fmt(v - bg, 3), theme["series-2"]) +
      (state.noise.enabled ? tipRow("arrival delay", `${data.arrival[ch]} samples`) : "");
  },
);

// Highlighted channel in the overlay; null follows the source channel.
let overlayPick = null;
const overlayChannel = () => (overlayPick ?? state.source_channel);

function buildOverlayChips() {
  const host = $("overlay-chips");
  for (let ch = 0; ch < N_CHANNELS; ch++) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = String(ch).padStart(2, "0");
    b.title = chName(ch);
    b.addEventListener("click", () => { overlayPick = ch; syncOverlayChips(); charts.overlay.stale = true; drawFrame(); });
    host.appendChild(b);
  }
}
function syncOverlayChips() {
  [...$("overlay-chips").children].forEach((b, ch) => b.setAttribute("aria-pressed", String(ch === overlayChannel())));
}

let overlayRange = [-1, 1];
chart("overlay",
  (w, h) => ({ x: 44, y: 10, w: w - 44 - 10, h: h - 10 - 26 }),
  (ctx, plot) => {
    const n = data.meta.n_points;
    let lo = -1, hi = 1;
    for (const v of data.values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.05;
    overlayRange = [lo - pad, hi + pad];
    const [y0, y1] = overlayRange;
    const yOf = (v) => plot.y + plot.h - ((v - y0) / (y1 - y0)) * plot.h;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of niceTicks(y0, y1, 4)) {
      ctx.strokeStyle = theme.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(plot.x, Math.round(yOf(t)) + 0.5); ctx.lineTo(plot.x + plot.w, Math.round(yOf(t)) + 0.5); ctx.stroke();
      ctx.fillStyle = theme.muted; ctx.fillText(String(t), plot.x - 6, yOf(t));
    }
    if (state.noise.enabled && data.meta.event_onset !== null) {
      const first = data.meta.event_onset, last = first + Math.max(...data.arrival) + data.meta.event_duration;
      const x0 = plot.x + (first / (n - 1)) * plot.w, x1 = plot.x + (Math.min(last, n - 1) / (n - 1)) * plot.w;
      ctx.fillStyle = theme.band;
      ctx.fillRect(x0, plot.y, x1 - x0, plot.h);
    }
    const pick = overlayChannel();
    ctx.globalAlpha = 0.55;
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      if (ch !== pick) strokeSeries(ctx, plot, n, (i) => at(data.values, i, ch), yOf, theme.muted, 0.8);
    }
    ctx.globalAlpha = 1;
    strokeSeries(ctx, plot, n, (i) => at(data.values, i, pick), yOf, theme["series-1"], 1.6);
    ctx.fillStyle = theme.ink;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillText(chName(pick) + (pick === state.source_channel && state.noise.enabled ? " (source)" : ""), plot.x + 6, plot.y + 2);
    drawTimeAxis(ctx, plot, n, data.meta.time_step);
  },
  (idx, y) => {
    const c = charts.overlay, [y0, y1] = overlayRange;
    const v = y0 + ((c.plot.y + c.plot.h - y) / c.plot.h) * (y1 - y0);
    let nearest = 0;
    for (let ch = 1; ch < N_CHANNELS; ch++) {
      if (Math.abs(at(data.values, idx, ch) - v) < Math.abs(at(data.values, idx, nearest) - v)) nearest = ch;
    }
    const pick = overlayChannel();
    return `t = ${fmt(idx * data.meta.time_step)} s<br>` +
      tipRow(chName(pick), fmt(at(data.values, idx, pick), 3), theme["series-1"]) +
      (nearest !== pick ? tipRow(`nearest: ${chName(nearest)}`, fmt(at(data.values, idx, nearest), 3), theme.muted) : "");
  },
);

chart("source",
  (w, h) => ({ x: 44, y: 10, w: w - 44 - 10, h: h - 10 - 26 }),
  (ctx, plot) => {
    const n = data.meta.n_points, ch = state.source_channel;
    let lo = -1, hi = 1;
    for (let i = 0; i < n; i++) { const v = at(data.values, i, ch); lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const pad = (hi - lo) * 0.06;
    lo -= pad; hi += pad;
    const yOf = (v) => plot.y + plot.h - ((v - lo) / (hi - lo)) * plot.h;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of niceTicks(lo, hi, 4)) {
      ctx.strokeStyle = theme.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(plot.x, Math.round(yOf(t)) + 0.5); ctx.lineTo(plot.x + plot.w, Math.round(yOf(t)) + 0.5); ctx.stroke();
      ctx.fillStyle = theme.muted; ctx.fillText(String(t), plot.x - 6, yOf(t));
    }
    const win = eventWindow(ch);
    if (win) {
      const x0 = plot.x + (win[0] / (n - 1)) * plot.w, x1 = plot.x + (win[1] / (n - 1)) * plot.w;
      ctx.fillStyle = theme.band;
      ctx.fillRect(x0, plot.y, x1 - x0, plot.h);
      ctx.fillStyle = theme["ink-2"];
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`onset ${data.meta.event_onset} (t = ${fmt(data.meta.event_onset * data.meta.time_step)} s)`, Math.min(x0 + 4, plot.x + plot.w - 150), plot.y + 2);
    }
    strokeSeries(ctx, plot, n, (i) => at(data.background, i, ch), yOf, theme.muted, 1.5);
    strokeSeries(ctx, plot, n, (i) => at(data.values, i, ch) - at(data.background, i, ch), yOf, theme["series-2"], 1.5);
    strokeSeries(ctx, plot, n, (i) => at(data.values, i, ch), yOf, theme["series-1"], 1);
    drawTimeAxis(ctx, plot, n, data.meta.time_step);
  },
  (idx) => {
    const ch = state.source_channel, v = at(data.values, idx, ch), bg = at(data.background, idx, ch);
    return `<b>${chName(ch)}</b> · t = ${fmt(idx * data.meta.time_step)} s<br>` +
      tipRow("background", fmt(bg, 3), theme.muted) +
      tipRow("noise", fmt(v - bg, 3), theme["series-2"]) +
      tipRow("observed", fmt(v, 3), theme["series-1"]);
  },
);

chart("coherence",
  (w, h) => ({ x: 44, y: 8, w: w - 44 - 10, h: h - 8 - 26 }),
  (ctx, plot) => {
    const n = data.meta.n_points, yOf = (v) => plot.y + plot.h - v * plot.h;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of [0, 0.5, 1]) {
      ctx.strokeStyle = theme.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(plot.x, Math.round(yOf(t)) + 0.5); ctx.lineTo(plot.x + plot.w, Math.round(yOf(t)) + 0.5); ctx.stroke();
      ctx.fillStyle = theme.muted; ctx.fillText(String(t), plot.x - 6, yOf(t));
    }
    strokeSeries(ctx, plot, n, (i) => data.coherence[i], yOf, theme["series-1"], 2);
    drawTimeAxis(ctx, plot, n, data.meta.time_step);
  },
  (idx) => `t = ${fmt(idx * data.meta.time_step)} s<br>` + tipRow("r(t)", fmt(data.coherence[idx], 3), theme["series-1"]),
);

function drawRing() {
  const { ctx, w, h } = fitCanvas($("ring"));
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2, R = Math.max(60, Math.min(w, h) / 2 - 66), nodeR = Math.max(12, Math.min(22, R / 7));
  const pos = Array.from({ length: N_CHANNELS }, (_, i) => {
    const a = -Math.PI / 2 + (TAU * i) / N_CHANNELS;
    return [cx + R * Math.cos(a), cy + R * Math.sin(a), a];
  });

  // Global (all-to-all) coupling: faint chords whose opacity follows K.
  ctx.strokeStyle = theme.grid;
  ctx.globalAlpha = Math.min(1, 0.15 + state.coupling / 3);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < N_CHANNELS; i++) for (let j = i + 1; j < N_CHANNELS; j++) {
    ctx.moveTo(pos[i][0], pos[i][1]); ctx.lineTo(pos[j][0], pos[j][1]);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
  // Ring path the injected event travels along.
  ctx.strokeStyle = theme.axis;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, TAU);
  ctx.stroke();

  let peak = 1e-9;
  if (state.noise.enabled) for (let k = 0; k < data.values.length; k++) peak = Math.max(peak, Math.abs(data.values[k] - data.background[k]));

  const i = cursor;
  for (let ch = 0; ch < N_CHANNELS; ch++) {
    const [x, y, a] = pos[ch];
    const bg = at(data.background, i, ch);
    const noise = at(data.values, i, ch) - bg;
    const win = eventWindow(ch);
    const active = win && i >= win[0] && i < win[1];
    if (active && Math.abs(noise) > 1e-9) {
      ctx.strokeStyle = theme["series-2"];
      ctx.lineWidth = 2 + 12 * Math.min(1, Math.abs(noise) / peak);
      ctx.beginPath(); ctx.arc(x, y, nodeR + ctx.lineWidth / 2 + 2, 0, TAU); ctx.stroke();
    }
    ctx.fillStyle = diverging(bg);
    ctx.strokeStyle = ch === state.source_channel && state.noise.enabled ? theme.ink : theme.surface;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, nodeR, 0, TAU); ctx.fill(); ctx.stroke();

    ctx.fillStyle = Math.abs(bg) > 0.6 ? "#fff" : theme.ink;
    ctx.font = `600 ${Math.round(nodeR * 0.62)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(ch).padStart(2, "0"), x, y);

    const lx = cx + (R + nodeR + 26) * Math.cos(a), ly = cy + (R + nodeR + 26) * Math.sin(a);
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = theme["ink-2"];
    ctx.fillText(`ω ${fmt(state.natural_frequencies[ch])}`, lx, ly - (state.noise.enabled ? 6 : 0));
    if (state.noise.enabled) {
      ctx.fillStyle = theme.muted;
      ctx.fillText(ch === state.source_channel ? "source" : `+${data.arrival[ch]}`, lx, ly + 7);
    }
  }

  ctx.textAlign = "center";
  ctx.fillStyle = theme.ink;
  const big = Math.round(Math.min(28, R / 4.5)), small = big < 24 ? 10 : 12;
  ctx.font = `600 ${big}px system-ui, sans-serif`;
  ctx.fillText(`r = ${fmt(data.coherence[i])}`, cx, cy - 10);
  ctx.font = `${small}px system-ui, sans-serif`;
  ctx.fillStyle = theme["ink-2"];
  ctx.fillText(`t = ${fmt(i * data.meta.time_step)} s · sample ${i}`, cx, cy + 6 + small);
  const active = Array.from({ length: N_CHANNELS }, (_, ch) => eventWindow(ch)).filter((win) => win && i >= win[0] && i < win[1]).length;
  if (state.noise.enabled) {
    ctx.fillStyle = active ? theme["series-2"] : theme.muted;
    ctx.fillText(active ? `noise present on ${active} of ${N_CHANNELS} nodes` : "no noise at this moment", cx, cy + 10 + 2.4 * small);
  }
}

function invalidate() {
  readTheme();
  for (const c of Object.values(charts)) c.stale = true;
  drawFrame();
}

function drawFrame() {
  if (!data) return;
  $("cursor").value = cursor;
  $("cursor-label").textContent = `t = ${fmt(cursor * data.meta.time_step)} s · #${cursor}`;
  let mean = 0;
  for (const r of data.coherence) mean += r;
  $("coherence-stat").textContent = `mean r = ${fmt(mean / data.coherence.length)} · now ${fmt(data.coherence[cursor])}`;
  drawRing();
  for (const c of Object.values(charts)) renderChart(c);
}

// ---------------------------------------------------------------- boot
buildNodes();
buildOverlayChips();
bindControls();
syncControls();
readTheme();
new ResizeObserver(() => { for (const c of Object.values(charts)) c.stale = true; drawFrame(); }).observe(document.body);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", invalidate);
