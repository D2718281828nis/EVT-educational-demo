import * as C from "./core.js";
import { LinePlot, BarPlot, Heatmap, RingGraph, fmt, color } from "./charts.js";

const $ = (id) => document.getElementById(id);
const root = document.documentElement;
let lang = root.dataset.lang === "ru" ? "ru" : "en";
const L = (en, ru) => (lang === "ru" ? ru : en);
const ch = (j) => C.channelName(j);
const short = (j) => String(j).padStart(2, "0");
const pct = (v) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : "–");
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

// ------------------------------------------------------------------ state
// Defaults are the notebooks' values: 1-d/01 + 1-d/03 for the 1-D series,
// scripts/generate_datasets.py + n-d/03 for the n-D recording. The n-D seed
// is 73 because this page's RNG is not numpy's: 73 gives the notebook's
// behaviour (alarm shortly after onset, correct source at persistence 1,
// a ring neighbour at persistence 2).
const DEFAULTS = {
  case: "nd",
  sig: {
    n: 1500, seed: 42, periods: [300, 97, 41], amplitudes: [1, 0.6, 0.3], phases: [0, 0.3, 0.6],
    sigma: 0.3, hurst: 0.75, magnitude: 8, onset: 1000, duration: 150, envelope: "exp",
    use: { periodic: true, stochastic: true, fractal: true },
  },
  nd: { n: 1500, p: 12, source: 0, seed: 73, coupling: 0.7, dt: 0.02, hurst: 0.75, magnitude: 2, delay: 5, decay: 0.65, onset: 932, duration: 75 },
  evt1: { oracle: true, baseline: 1000, tailQuantile: 0.95, alarmProbability: 0.999, topK: 1, persistence: 3 },
  evtN: { oracle: true, baseline: 932, tailQuantile: 0.85, alarmProbability: 0.97, topK: 3, persistence: 1 },
  graph: { method: "ar", edgeThreshold: 0.35, arOrder: 2, surrogateCount: 99 },
  mp: { mode: "gat", node: null },
  useModel: false,
  train: { recordings: 16, validation: 0.25, hidden: 8, epochs: 200, lr: 0.02, seed: 42 },
  evalb: { cases: 24, share1d: 0.3, shareNone: 0.25, oracle: true, earlyTolerance: 0, k: 3, seed: 7 },
  tg: { windowSize: 32, stride: 8, neighborhood: 1, context: true, records: 48, epochs: 200, hidden: 8 },
};
const state = structuredClone(DEFAULTS);
let model = null, modelVersion = 0, modelReport = null;
let temporalModel = null, temporalReport = null;

const resolve = (key) => (key.startsWith("@evt.") ? `${state.case === "1d" ? "evt1" : "evtN"}.${key.slice(5)}` : key);
function getPath(path) { return path.split(".").reduce((o, k) => o?.[k], state); }
function setPath(path, value) {
  const keys = path.split(".");
  const last = keys.pop();
  keys.reduce((o, k) => o[k], state)[last] = value;
}

// ----------------------------------------------------------- computations
function memo(fn) {
  let key = null, value;
  return (...args) => {
    const k = JSON.stringify(args);
    if (k !== key) { value = fn(...args); key = k; }
    return value;
  };
}
const getSig = memo((p) => C.additiveSeries(p));
const getNd = memo((p) => C.propagationSeries(p));
const sig = () => getSig(state.sig);
const nd = () => getNd(state.nd);
const graphConfig = () => ({ ...C.GRAPH_DEFAULTS, ...state.graph });
function evtConfig(which) {
  const e = which === "1d" ? state.evt1 : state.evtN;
  const onset = which === "1d" ? sig().onset : nd().onset;
  return { baseline: e.oracle ? onset : e.baseline, tailQuantile: e.tailQuantile, alarmProbability: e.alarmProbability, topK: e.topK, persistence: e.persistence };
}
const getRes1 = memo((sigP, evt) => C.runPipeline(C.fromSeries(getSig(sigP).values), { evt }));
const getResN = memo((ndP, evt, graph, version) => C.runPipeline(getNd(ndP), { evt, graph, processModel: version ? model : null }));
const res1 = () => getRes1(state.sig, evtConfig("1d"));
const resN = (withModel = false) => getResN(state.nd, evtConfig("nd"), graphConfig(), withModel && model ? modelVersion : 0);
const activeRes = () => (state.case === "1d" ? res1() : resN());

// --------------------------------------------------------------- controls
function bilingual(en, ru, target) {
  const a = el("span", null, en); a.lang = "en";
  const b = el("span", null, ru ?? en); b.lang = "ru";
  target.append(a, b);
}
function parseOptions(spec) {
  const split = (s) => { const i = s.indexOf(":"); return [s.slice(0, i), s.slice(i + 1)]; };
  return spec.split(";").map((item) => {
    const [enPart, ruPart] = item.split("|");
    const [value, en] = split(enPart);
    const ru = ruPart ? split(ruPart)[1] : en;
    return { value, en, ru };
  });
}
const decimals = (step) => (String(step).includes(".") ? String(step).split(".")[1].length : 0);
const DYNAMIC_MAX = {
  "sig.onset": () => state.sig.n - state.sig.duration,
  "sig.duration": () => Math.min(400, state.sig.n - 20),
  "nd.source": () => state.nd.p - 1,
  "nd.onset": () => state.nd.n - state.nd.duration - Math.floor(state.nd.p / 2) * state.nd.delay,
  "@evt.baseline": () => (state.case === "1d" ? state.sig.n : state.nd.n) - 50,
};
const controls = [];

function buildControl(host) {
  const key = host.dataset.k, type = host.dataset.type || "range";
  const label = el("label");
  if (type === "range") {
    const top = el("span", "ctl-top");
    const name = el("span", "ctl-name");
    bilingual(host.dataset.en, host.dataset.ru, name);
    const out = el("output");
    top.append(name, out);
    const input = el("input");
    Object.assign(input, { type: "range", min: host.dataset.min, max: host.dataset.max, step: host.dataset.step });
    input.addEventListener("input", () => commit(key, parseFloat(input.value)));
    label.append(top, input);
    host.append(label);
    const digits = decimals(host.dataset.step);
    host._sync = () => {
      if (DYNAMIC_MAX[key]) input.max = DYNAMIC_MAX[key]();
      // With the oracle on, the effective baseline follows the true onset.
      const v = key === "@evt.baseline" ? evtConfig(state.case).baseline : getPath(resolve(key));
      input.value = v;
      out.textContent = Number(v).toFixed(digits);
    };
  } else if (type === "number") {
    const name = el("span", "ctl-name");
    bilingual(host.dataset.en, host.dataset.ru, name);
    const input = el("input");
    Object.assign(input, { type: "number", min: host.dataset.min, max: host.dataset.max, step: host.dataset.step });
    input.addEventListener("change", () => {
      const v = Math.min(+host.dataset.max, Math.max(+host.dataset.min, Math.round(+input.value || 0)));
      commit(key, v);
    });
    label.append(name, input);
    host.append(label);
    host._sync = () => { input.value = getPath(resolve(key)); };
  } else if (type === "toggle") {
    label.className = "switch";
    const input = el("input");
    input.type = "checkbox";
    input.setAttribute("role", "switch");
    input.addEventListener("change", () => commit(key, input.checked));
    label.append(input);
    if (host.dataset.en) { const name = el("span"); bilingual(host.dataset.en, host.dataset.ru, name); label.append(name); }
    else input.setAttribute("aria-label", key);
    host.append(label);
    host._sync = () => { input.checked = !!getPath(resolve(key)); };
  } else if (type === "seg") {
    host.classList.add("seg-ctl");
    if (host.dataset.en) { const name = el("span"); bilingual(host.dataset.en, host.dataset.ru, name); host.append(name); }
    const group = el("div", "seg");
    group.setAttribute("role", "group");
    const buttons = parseOptions(host.dataset.options).map((o) => {
      const b = el("button");
      b.type = "button";
      bilingual(o.en, o.ru, b);
      b.addEventListener("click", () => commit(key, o.value));
      group.append(b);
      return [b, o.value];
    });
    host.append(group);
    host._sync = () => { const v = getPath(resolve(key)); for (const [b, value] of buttons) b.setAttribute("aria-pressed", String(value === v)); };
  } else if (type === "select") {
    const name = el("span", "ctl-name");
    bilingual(host.dataset.en, host.dataset.ru, name);
    const select = el("select");
    const options = parseOptions(host.dataset.options);
    for (const o of options) { const opt = el("option"); opt.value = o.value; select.append(opt); }
    select.addEventListener("change", () => commit(key, select.value));
    label.append(name, select);
    host.append(label);
    host._sync = () => {
      options.forEach((o, i) => { select.options[i].textContent = lang === "ru" ? o.ru : o.en; });
      select.value = getPath(resolve(key));
    };
  }
  controls.push(host);
}
function syncControls() {
  for (const c of controls) c._sync();
  const e = state.case === "1d" ? state.evt1 : state.evtN;
  for (const c of controls) if (c.dataset.k === "@evt.baseline") c.classList.toggle("disabled", e.oracle);
}

function clampState() {
  const s = state.sig, n = state.nd;
  s.duration = Math.min(s.duration, s.n - 20);
  s.onset = Math.max(0, Math.min(s.onset, s.n - s.duration));
  n.source = Math.min(n.source, n.p - 1);
  n.onset = Math.max(0, Math.min(n.onset, DYNAMIC_MAX["nd.onset"]()));
  state.evt1.baseline = Math.max(50, Math.min(state.evt1.baseline, s.n - 50));
  state.evtN.baseline = Math.max(50, Math.min(state.evtN.baseline, n.n - 50));
  if (state.evt1.alarmProbability <= state.evt1.tailQuantile) state.evt1.alarmProbability = Math.min(0.9995, state.evt1.tailQuantile + 0.005);
  if (state.evtN.alarmProbability <= state.evtN.tailQuantile) state.evtN.alarmProbability = Math.min(0.9995, state.evtN.tailQuantile + 0.005);
}

function commit(key, value) {
  const path = resolve(key);
  setPath(path, value);
  if (path.endsWith(".baseline")) setPath(path.replace("baseline", "oracle"), false);
  if (path.endsWith(".oracle") && value) {
    const which = path.startsWith("evt1") ? "1d" : "nd";
    setPath(path.replace("oracle", "baseline"), which === "1d" ? sig().onset : nd().onset);
  }
  if (path === "mp.mode" || path === "mp.node") { syncControls(); invalidate("s5"); return; }
  if (path.startsWith("evalb.")) { syncControls(); evalDirty = true; invalidate("s6"); return; }
  if (path.startsWith("train.")) { syncControls(); return; }
  if (path.startsWith("tg.")) { syncControls(); invalidate("s8"); return; }
  clampState();
  syncControls();
  applyCase();
  invalidate();
}
function applyCase() { document.body.dataset.case = state.case; }

// ----------------------------------------------------- lazy section render
const visible = new Set(), dirty = new Set();
const renderers = {};
let frame = 0;
function invalidate(...ids) {
  for (const id of ids.length ? ids : Object.keys(renderers)) dirty.add(id);
  if (!frame) frame = requestAnimationFrame(flush);
}
function flush() {
  frame = 0;
  for (const id of [...dirty]) {
    if (!visible.has(id)) continue;
    dirty.delete(id);
    try { renderers[id](); } catch (error) { console.error(id, error); }
  }
}

// ------------------------------------------------------------- utilities
function seqColor(f) {
  const parse = (hex) => [1, 3, 5].map((i) => parseInt(hex.trim().slice(i, i + 2), 16));
  const lo = parse(color("--seq-lo")), hi = parse(color("--seq-hi"));
  f = Math.max(0, Math.min(1, f));
  return `rgb(${lo.map((v, k) => Math.round(v + (hi[k] - v) * f)).join(",")})`;
}
const seqInk = (f) => (f > 0.55 ? (root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches) ? "#0b0b0b" : "#ffffff") : "--ink");
function tile(label, value, sub, cls = "") {
  const t = el("div", "tile");
  t.append(el("div", "t-label", label), el("div", `t-value ${cls}`, value));
  if (sub) t.append(el("div", "t-sub", sub));
  return t;
}
function table(target, head, rows, highlight = () => false) {
  const t = $(target);
  t.replaceChildren();
  const thead = el("thead"), tr = el("tr");
  for (const h of head) tr.append(el("th", null, h));
  thead.append(tr);
  const tbody = el("tbody");
  rows.forEach((row, r) => {
    const line = el("tr", highlight(r) ? "hl" : null);
    for (const cell of row) {
      const td = el("td");
      if (cell && typeof cell === "object" && "text" in cell) { td.textContent = cell.text; if (cell.cls) td.className = cell.cls; } else td.textContent = cell;
      line.append(td);
    }
    tbody.append(line);
  });
  t.append(thead, tbody);
}
const argmax = (a) => Array.from(a).reduce((best, v, i, arr) => (v > arr[best] ? i : best), 0);

// ============================================================ 1 — signal
const p1 = {
  periodic: new LinePlot("p1-periodic", { height: 90, group: "s1", hideX: true }),
  noise: new LinePlot("p1-noise", { height: 80, group: "s1", hideX: true }),
  fractal: new LinePlot("p1-fractal", { height: 110, group: "s1", hideX: true }),
  sum: new LinePlot("p1-sum", { height: 170, group: "s1" }),
  psd: new LinePlot("p1-psd", { height: 200 }),
  dfa: new LinePlot("p1-dfa", { height: 200 }),
};
renderers.s1 = () => {
  const s = sig(), n = s.n, use = state.sig.use;
  const tooltip = (i) => ({
    title: `t = ${i}`,
    rows: [
      { value: fmt(s.periodic[i]), label: L("periodic", "периодика"), color: "--series-3" },
      { value: fmt(s.stochastic[i]), label: L("stochastic", "стохастика"), color: "--muted" },
      { value: fmt(s.fractal[i]), label: L("fractal", "фрактал"), color: "--series-2" },
      { value: fmt(s.values[i]), label: "x(t)", color: "--series-1" },
    ],
  });
  const band = [{ x0: s.onset, x1: s.onset + s.duration, color: "--band" }];
  const off = (on) => (on ? 1 : 0.3);
  p1.periodic.update({ n, series: [{ y: s.periodic, color: "--series-3", alpha: off(use.periodic) }], tooltip });
  p1.noise.update({ n, series: [{ y: s.stochastic, color: "--muted", width: 1, alpha: off(use.stochastic) }], tooltip });
  p1.fractal.update({
    n, series: [{ y: s.fractal, color: "--series-2", alpha: off(use.fractal) }], bands: band, tooltip,
    vlines: [{ x: s.onset, color: "--ink", dash: [4, 3], label: `t₀ = ${s.onset}` }],
  });
  p1.sum.update({ n, series: [{ y: s.values, color: "--series-1", width: 1.25 }], bands: band, tooltip, xLabel: L("sample t", "отсчёт t") });

  const long = C.coloredNoise(2048, state.sig.hurst, C.makeRng(state.sig.seed, "hurst-demo"));
  const sp = C.spectrum(long), d = C.dfa(long, [8, 16, 32, 64, 128, 256]);
  const H = state.sig.hurst, theory = 1 - 2 * H;
  const mid = Math.floor(sp.bins.length / 2), anchor = sp.bins[mid];
  p1.psd.update({
    n: sp.bins.length, x: (i) => sp.bins[i].f, logX: true, logY: true,
    series: [
      { y: sp.bins.map((b) => 10 ** (sp.intercept + sp.slope * Math.log10(b.f))), color: "--series-2", width: 2 },
      { y: sp.bins.map((b) => anchor.power * (b.f / anchor.f) ** theory), color: "--ink", width: 1.25, dash: [5, 4] },
    ],
    points: sp.bins.map((b) => ({ x: b.f, y: b.power, color: "--series-2", r: 3.5 })),
    xLabel: L("frequency f", "частота f"),
    tooltip: (i) => ({ title: `f = ${fmt(sp.bins[i].f, 4)}`, rows: [{ value: fmt(sp.bins[i].power, 3), label: L("mean power", "средняя мощность"), color: "--series-2" }] }),
  });
  p1.dfa.update({
    n: d.points.length, x: (i) => d.points[i].s, logX: true, logY: true,
    series: [
      { y: d.points.map((q) => 10 ** (d.intercept + d.alpha * Math.log10(q.s))), color: "--series-2", width: 2 },
      { y: d.points.map((q) => d.points[0].F * (q.s / d.points[0].s) ** H), color: "--ink", width: 1.25, dash: [5, 4] },
    ],
    points: d.points.map((q) => ({ x: q.s, y: q.F, color: "--series-2", r: 4 })),
    xLabel: L("scale s", "масштаб s"),
    tooltip: (i) => ({ title: `s = ${d.points[i].s}`, rows: [{ value: fmt(d.points[i].F, 3), label: "F(s)", color: "--series-2" }] }),
  });
  $("s1-hurst-stat").textContent = L(
    `PSD slope ${fmt(sp.slope)} (theory ${fmt(theory)}) · DFA α = ${fmt(d.alpha)} (H = ${fmt(H)}) · dashed = theory`,
    `наклон PSD ${fmt(sp.slope)} (теория ${fmt(theory)}) · DFA α = ${fmt(d.alpha)} (H = ${fmt(H)}) · пунктир — теория`,
  );
};

// ============================================================ 2 — cases
const p2 = {
  one: new LinePlot("p2-1d", { height: 200 }),
  ring: new RingGraph("p2-ring", { height: 300 }),
  lanes: new LinePlot("p2-lanes", { height: 380, left: 70 }),
};
renderers.s2 = () => {
  const s = sig();
  p2.one.update({
    n: s.n, series: [{ y: s.values, color: "--series-1", width: 1.25 }],
    bands: [{ x0: s.onset, x1: s.onset + s.duration, color: "--band" }], xLabel: L("sample t", "отсчёт t"),
    tooltip: (i) => ({ title: `t = ${i}`, rows: [{ value: fmt(s.values[i]), label: "x(t)", color: "--series-1" }] }),
  });
  const d = nd(), p = d.p;
  const gains = d.hops.map((h) => state.nd.decay ** h);
  p2.ring.update({
    n: p, label: short,
    fill: (i) => seqColor(gains[i]), labelColor: (i) => seqInk(gains[i]),
    ring: (i) => (i === d.source ? { color: "--ink", width: 3 } : null),
    badge: (i) => `+${d.delays[i]}`,
    edges: Array.from({ length: p }, (_, i) => ({ a: i, b: (i + 1) % p, w: 0.3, color: "--axis" })),
    center: [L("ring topology", "кольцевая топология"), `Δ = ${state.nd.delay}, γ = ${state.nd.decay}`],
    tooltip: (i) => ({
      title: ch(i) + (i === d.source ? L(" · source", " · источник") : ""),
      rows: [
        { value: `${d.hops[i]}`, label: L("ring hops from source", "хопов от источника") },
        { value: `+${d.delays[i]}`, label: L("arrival delay, samples", "задержка прихода, отсчёты") },
        { value: fmt(gains[i], 3), label: L("burst gain γ^hops", "усиление γ^hops") },
        { value: `${d.onset + d.delays[i]}`, label: L("burst arrives at t", "всплеск приходит в t") },
      ],
    }),
  });
  let peak = 1;
  for (const v of d.v) peak = Math.max(peak, Math.abs(v));
  const spacing = Math.max(2.4, 1.15 * peak);
  const offset = (j) => (p - 1 - j) * spacing;
  const columns = Array.from({ length: p }, (_, j) => C.column(d, j));
  p2.lanes.update({
    n: d.n, series: columns.map((y, j) => ({ y, offset: offset(j), color: j === d.source ? "--series-1" : "--ink-2", width: j === d.source ? 1.4 : 1, noDot: true })),
    bands: columns.map((_, j) => ({ x0: d.onset + d.delays[j], x1: d.onset + d.delays[j] + d.duration, y0: offset(j) - spacing / 2, y1: offset(j) + spacing / 2, color: "--band" })),
    yDomain: [-spacing / 2 - 0.2, offset(0) + spacing / 2 + 0.2],
    yTicks: columns.map((_, j) => ({ v: offset(j), label: (j === d.source ? "★ " : "") + ch(j).replace("channel_", "ch ") })),
    xLabel: L("sample t", "отсчёт t"),
    tooltip: (i, ypx) => {
      const g = p2.lanes.g;
      const value = g.y1 - ((ypx - g.top) / (g.height - g.top - g.bottom)) * (g.y1 - g.y0);
      const j = Math.max(0, Math.min(p - 1, p - 1 - Math.round(value / spacing)));
      const inside = d.mask[i * p + j] === 1;
      return {
        title: `${ch(j)} · t = ${i}`,
        rows: [
          { value: fmt(d.v[i * p + j]), label: L("observed", "наблюдение"), color: "--series-1" },
          { value: fmt(d.background[i * p + j]), label: L("Kuramoto background", "фон Курамото"), color: "--muted" },
          { value: fmt(d.injected[i * p + j]), label: inside ? L("burst (in window)", "всплеск (в окне)") : L("burst", "всплеск"), color: "--series-2" },
        ],
      };
    },
  });
};

// ============================================================ 3 — EVT
const p3 = {
  indicator: new LinePlot("p3-indicator", { height: 250 }),
  gpd: new LinePlot("p3-gpd", { height: 200 }),
  rank: new BarPlot("p3-rank", { height: 210 }),
  heat: new Heatmap("p3-heat", { height: 260, left: 92 }),
};
// Synthetic ground truth of the active case; null when nothing was injected.
function truth() {
  if (state.case === "1d") {
    const injected = state.sig.use.fractal && state.sig.magnitude > 0;
    return { onset: injected ? sig().onset : null, source: null };
  }
  const injected = state.nd.magnitude > 0;
  return { onset: injected ? nd().onset : null, source: injected ? nd().source : null };
}
renderers.s3 = () => {
  const r = activeRes(), det = r.detection, tr = truth(), B = det.config.baseline;
  const isNd = state.case === "nd";
  const tiles = $("s3-tiles");
  tiles.replaceChildren();
  const delay = det.detected && tr.onset != null ? det.timeIndex - tr.onset : null;
  tiles.append(
    tile(L("Alarm t_EVT", "Тревога t_EVT"), det.detected ? String(det.timeIndex) : L("none", "нет"), L(`baseline B = ${B}`, `фон B = ${B}`), det.detected === (tr.onset != null) ? "" : "bad"),
    tile(L("Delay vs true onset", "Задержка от истинного начала"), delay == null ? "–" : `${delay >= 0 ? "+" : ""}${delay}`,
      tr.onset == null ? L("nothing injected", "ничего не впрыснуто") : delay == null ? "" : delay < 0 ? L("alarm before the onset", "тревога раньше начала") : L("samples after t₀", "отсчётов после t₀"), delay != null && delay < 0 ? "bad" : ""),
    tile(L("Alarm threshold q_EVT", "Порог тревоги q_EVT"), fmt(det.alarmThreshold, 3), `u = ${fmt(det.u, 3)}`),
  );
  if (isNd && state.nd.magnitude === 0) {
    // Nothing injected: there is no true source, only a false alarm or a correct rejection.
    tiles.append(
      tile(L("Estimated source ŝ", "Оценка источника ŝ"), r.ranking ? ch(r.ranking.source) : "–", L("ranking of background drift", "ранжирование дрейфа фона")),
      tile(L("True source", "Истинный источник"), "—", L("nothing injected (magnitude 0)", "ничего не впрыснуто (амплитуда 0)")),
      tile(L("Verdict", "Вердикт"), det.detected ? L("! false alarm", "! ложная тревога") : L("✓ correct rejection", "✓ верный отказ"),
        det.detected ? L("background drift is rare relative to the baseline", "дрейф фона редок относительно фона") : L("no alarm without an event", "нет тревоги без события"), det.detected ? "bad" : "ok"),
    );
  } else if (isNd) {
    const est = r.ranking ? r.ranking.source : null;
    const ok = est === tr.source;
    tiles.append(
      tile(L("Estimated source ŝ", "Оценка источника ŝ"), est == null ? "–" : ch(est), est == null ? L("no ranking without an alarm", "нет ранжирования без тревоги") : `P = ${fmt(r.ranking.probabilities[est], 3)}`),
      tile(L("True source", "Истинный источник"), ch(tr.source), L("synthetic label", "синтетическая метка")),
      tile(L("Verdict", "Вердикт"), est == null ? "–" : ok ? L("✓ correct", "✓ верно") : L("✕ wrong", "✕ неверно"),
        est == null ? "" : ok ? L("argmax hits the source", "argmax попал в источник") : L(`true source rank ${1 + Array.from(r.ranking.probabilities).filter((v) => v > r.ranking.probabilities[tr.source]).length}`, `ранг истинного источника ${1 + Array.from(r.ranking.probabilities).filter((v) => v > r.ranking.probabilities[tr.source]).length}`), est == null ? "" : ok ? "ok" : "bad"),
    );
  } else {
    tiles.append(tile(L("Planner stop", "Остановка планировщика"), "1-D", "single_channel_cannot_localize"));
  }

  const ind = det.indicator, n = ind.length;
  let top = det.alarmThreshold * 1.6;
  for (let t = 0; t < n; t++) top = Math.max(top, ind[t]);
  p3.indicator.update({
    n, series: [{ y: ind, color: "--series-1", width: 1.25 }],
    bands: [{ x0: 0, x1: B, color: "--band-neutral" }],
    hlines: [{ y: det.u, color: "--muted", dash: [4, 3], label: `u = ${fmt(det.u)}`, side: "left", below: true }, { y: det.alarmThreshold, color: "--critical", width: 1.5, label: `q_EVT = ${fmt(det.alarmThreshold)}` }],
    vlines: [
      tr.onset != null ? { x: tr.onset, color: "--ink", dash: [4, 3], label: `t₀ = ${tr.onset}` } : {},
      det.detected ? { x: det.timeIndex, color: "--critical", width: 2, label: `t_EVT = ${det.timeIndex}`, row: 1 } : {},
    ],
    yDomain: [0, top * 1.05], xLabel: L("sample t", "отсчёт t"),
    tooltip: (i) => ({
      title: `t = ${i}${i < B ? L(" · baseline", " · фон") : ""}`,
      rows: [{ value: fmt(ind[i], 3), label: isNd ? `I_t (top-${Math.min(det.config.topK, r.profile.n_channels)})` : "z_t", color: "--series-1" },
        { value: ind[i] > det.alarmThreshold ? L("above q_EVT", "выше q_EVT") : L("below q_EVT", "ниже q_EVT"), label: "" }],
    }),
  });

  // GPD tail fit on a shared grid: histogram density (step) + fitted density
  const ex = det.excess, gpd = det.gpd;
  if (ex.length) {
    const xmax = Math.max(Math.max(...ex) * 1.1, gpd ? det.alarmThreshold - det.u : 0) * 1.05 || 1;
    const G = 160, bins = Math.max(6, Math.min(16, Math.round(Math.sqrt(ex.length) * 1.5)));
    const counts = new Float64Array(bins);
    for (const y of ex) counts[Math.min(bins - 1, Math.floor((y / xmax) * bins))]++;
    const width = xmax / bins;
    const hist = new Float64Array(G), dens = new Float64Array(G);
    for (let i = 0; i < G; i++) {
      const x = (i / (G - 1)) * xmax;
      hist[i] = counts[Math.min(bins - 1, Math.floor((x / xmax) * bins))] / (ex.length * width);
      dens[i] = gpd ? C.gpdPdf(x, gpd.shape, gpd.scale) : NaN;
    }
    p3.gpd.update({
      n: G, x: (i) => (i / (G - 1)) * xmax,
      series: [{ y: hist, color: "--series-1", width: 1.25, alpha: 0.8 }, { y: dens, color: "--series-2", width: 2 }],
      vlines: gpd ? [{ x: det.alarmThreshold - det.u, color: "--critical", label: "q_EVT − u" }] : [],
      xLabel: "y = I − u",
      tooltip: (i) => ({ title: `y = ${fmt((i / (G - 1)) * xmax, 3)}`, rows: [{ value: fmt(hist[i], 3), label: L("empirical density", "эмпирическая плотность"), color: "--series-1" }, { value: fmt(dens[i], 3), label: L("GPD density", "плотность GPD"), color: "--series-2" }] }),
    });
  }
  const bounded = gpd && gpd.shape < -0.9
    ? L(" ξ ≈ −1 means a bounded tail (GPD → uniform): oscillator values never exceed |sin| ≤ 1, so the baseline has no heavy tail to extrapolate.",
      " ξ ≈ −1 означает ограниченный хвост (GPD → равномерное): значения осцилляторов не превышают |sin| ≤ 1, поэтому у фона нет тяжёлого хвоста для экстраполяции.")
    : "";
  $("s3-gpd-text").textContent = gpd
    ? L(`${ex.length} baseline excesses over u. MLE: ξ = ${fmt(gpd.shape, 3)}, σ = ${fmt(gpd.scale, 3)}. The alarm level is the GPD quantile at (p − q)/(1 − q) = ${fmt(gpd.conditional, 3)}; blue = empirical density, orange = fitted GPD.${bounded}`,
      `${ex.length} превышений фона над u. MLE: ξ = ${fmt(gpd.shape, 3)}, σ = ${fmt(gpd.scale, 3)}. Уровень тревоги — квантиль GPD при (p − q)/(1 − q) = ${fmt(gpd.conditional, 3)}; синяя линия — эмпирическая плотность, оранжевая — подогнанная GPD.${bounded}`)
    : L("Fewer than 5 excesses: the empirical p-quantile of the baseline indicator is used instead of a GPD.", "Меньше 5 превышений: вместо GPD используется эмпирический p-квантиль фонового индикатора.");

  if (isNd && r.ranking) {
    const d = nd(), prob = r.ranking.probabilities, est = r.ranking.source;
    p3.rank.update({
      labels: Array.from({ length: d.p }, (_, j) => short(j)),
      groups: [{ values: prob, color: "--series-1" }],
      emphasis: (i) => (i === est ? 1 : 0.5),
      valueLabels: (i) => i === est || i === d.source,
      marks: (i) => [i === est ? "ŝ" : "", i === d.source ? "★" : ""].filter(Boolean).join(" "),
      tooltip: (i) => ({ title: ch(i), rows: [{ value: fmt(prob[i], 3), label: "P(source)" }, ...C.FEATURE_NAMES.map((f, k) => ({ value: fmt(r.ranking.features[i * 5 + k], 2), label: f }))] }),
    });
    const t0 = det.timeIndex, a = Math.max(0, t0 - 40), b = Math.min(d.n, t0 + 60);
    let zmax = 2;
    for (let t = a; t < b; t++) for (let j = 0; j < d.p; j++) zmax = Math.max(zmax, det.z[t * d.p + j]);
    $("s3-heat-scale").textContent = `|z|: 0 → ${fmt(zmax, 1)}`;
    p3.heat.update({
      rows: d.p, cols: b - a, vmax: zmax,
      value: (row, c) => det.z[(a + c) * d.p + row],
      rowLabel: (row) => (row === d.source ? "★ " : "") + ch(row),
      colLabel: (c) => { const rel = a + c - t0; return rel === 0 ? "t_EVT" : `${rel > 0 ? "+" : ""}${rel}`; },
      xTicks: Array.from({ length: b - a }, (_, c) => c).filter((c) => (a + c - t0) % 20 === 0),
      marks: [{ c: t0 - a, color: "--critical" }, ...(tr.onset != null && tr.onset >= a && tr.onset < b ? [{ c: tr.onset - a, color: "--ink", dash: [3, 3] }] : [])],
      tooltip: (row, c) => ({
        title: `${ch(row)} · t = ${a + c} (${a + c - t0 >= 0 ? "+" : ""}${a + c - t0})`,
        rows: [{ value: fmt(det.z[(a + c) * d.p + row], 2), label: "|z|" }, { value: d.mask[(a + c) * d.p + row] ? L("yes", "да") : L("no", "нет"), label: L("inside the channel's event window", "внутри окна события канала") }],
      }),
    });
    table("s3-features", [L("channel", "канал"), ...C.FEATURE_NAMES, "P"],
      Array.from({ length: d.p }, (_, j) => [ch(j) + (j === est ? " ŝ" : "") + (j === d.source ? " ★" : ""), ...C.FEATURE_NAMES.map((_, k) => fmt(r.ranking.features[j * 5 + k], 2)), fmt(prob[j], 3)]),
      (j) => j === est);
  } else if (isNd) {
    p3.rank.update({ labels: [], groups: [{ values: [], color: "--series-1" }] });
  }

  // Specification sensitivity, as in notebooks */05
  const note = $("s3-spec-note");
  if (isNd) {
    const parts = [1, 2, 3].map((r0) => {
      const res = C.runPipeline(nd(), { evt: { ...evtConfig("nd"), persistence: r0 }, graph: graphConfig() });
      if (!res.detection.detected) return `r=${r0}: ${L("no alarm", "нет тревоги")}`;
      const s0 = res.ranking.source;
      return `r=${r0}: t=${res.detection.timeIndex}, ŝ=${short(s0)} ${s0 === nd().source ? "✓" : "✕"}`;
    });
    note.textContent = L(`Persistence sensitivity (n-d/05): ${parts.join(" · ")}`, `Чувствительность к устойчивости (n-d/05): ${parts.join(" · ")}`);
  } else {
    note.textContent = L("1-D notebook spec (1-d/03): q = 0.95, p = 0.999, r = 3; the baseline ends at the known onset (oracle-assisted).",
      "Спецификация 1-D (1-d/03): q = 0.95, p = 0.999, r = 3; фон заканчивается на известном начале (oracle-assisted).");
  }
};

// ============================================================ 4 — pipeline
let animating = 0;
function flowNodes(svg) { return Object.fromEntries([...svg.querySelectorAll("[data-node]")].map((g) => [g.dataset.node, g])); }
function flowEdges(svg) { return Object.fromEntries([...svg.querySelectorAll("[data-edge]")].map((p) => [p.dataset.edge, p])); }
function pipelineRoute(r) {
  const nodes = ["input", "inspect_input", "check_stationarity", "detect_evt", "decision"];
  const edges = ["input-inspect", "inspect-stationarity", "stationarity-evt", "evt-decision"];
  if (r.plan.stop_reason) { nodes.push("stop"); edges.push("decision-stop"); return { nodes, edges }; }
  nodes.push("build_baseline_graph", "rank_source");
  edges.push("decision-graph", "graph-rank");
  if (r.process) { nodes.push("infer_gnn_gat_process"); edges.push("rank-learned", "learned-output"); } else edges.push("rank-output");
  nodes.push("output");
  return { nodes, edges };
}
function paintFlow(r, upto = Infinity) {
  const svg = $("flow"), nodes = flowNodes(svg), edges = flowEdges(svg), route = pipelineRoute(r);
  for (const [name, g] of Object.entries(nodes)) {
    const index = route.nodes.indexOf(name);
    g.classList.remove("done", "active", "skipped", "stopped");
    if (index < 0) g.classList.add("skipped");
    else if (index < upto) g.classList.add(name === "stop" ? "stopped" : "done");
    else if (index === upto) g.classList.add("active");
  }
  for (const [name, p] of Object.entries(edges)) {
    const index = route.edges.indexOf(name);
    p.classList.toggle("on", index >= 0 && index < upto);
  }
  $("flow-stop-reason").textContent = r.plan.stop_reason || "—";
  return route;
}
function pipelineResult() { return state.case === "1d" ? res1() : resN(state.useModel); }
function renderTrace(r, upto = Infinity) {
  const list = $("s4-trace");
  list.replaceChildren();
  r.trace.forEach((step, i) => {
    const li = el("li", i >= upto ? "pending" : step.action === "stop" ? "stop" : "");
    li.append(el("span", "i", String(i + 1)), el("span", "a", step.action), el("span", "m", step.message));
    list.append(li);
  });
}
function renderPlan(r) {
  const plan = r.plan;
  $("s4-plan").textContent = JSON.stringify({
    route: plan.route, baseline_stationary: plan.baseline_stationary, initial_actions: plan.initial_actions,
    completed_actions: plan.completed_actions, stop_reason: plan.stop_reason,
  }, null, 2);
  const e = r.detection.config, isNd = r.profile.n_channels > 1;
  const lines = [
    "from graph_evt_agent import (",
    "    EVTConfig, GraphConfig, GraphEVTPipeline)",
    "",
    "pipeline = GraphEVTPipeline(",
    `    EVTConfig(baseline_size=${e.baseline},`,
    `              tail_quantile=${e.tailQuantile},`,
    `              alarm_probability=${e.alarmProbability},`,
    `              top_k=${e.topK}, persistence=${e.persistence}),`,
    `    graph=GraphConfig(method="${state.graph.method}",`,
    `                      edge_threshold=${state.graph.edgeThreshold}),`,
  ];
  if (isNd && r.process) lines.push("    process_model=model,  # section 5 / n-d/09");
  lines.push(")", "result = pipeline.run_detailed(values)", `# values.shape == (${r.profile.n_time}, ${r.profile.n_channels})`, "result.plan.completed_actions");
  $("s4-code").textContent = lines.join("\n");
  const d = r.detection;
  let status = d.detected ? L(`alarm at t = ${d.timeIndex}`, `тревога в t = ${d.timeIndex}`) : L("no alarm", "нет тревоги");
  if (state.case === "nd" && state.nd.magnitude === 0) {
    status += d.detected ? L(" — nothing was injected: a false alarm from background drift", " — ничего не впрыснуто: ложная тревога из-за дрейфа фона") : L(" — correct: nothing was injected", " — верно: ничего не впрыснуто");
  }
  if (r.ranking) status += ` · ŝ = ${r.names[r.ranking.source]}`;
  if (r.process) status += ` · GNN ${short(r.process.gnnSource)} / GAT ${short(r.process.gatSource)}`;
  $("s4-status").textContent = status;
}
renderers.s4 = () => {
  if (animating) return;
  const r = pipelineResult();
  paintFlow(r);
  renderTrace(r);
  renderPlan(r);
};
function animatePipeline() {
  const r = pipelineResult(), token = ++animating;
  const route = pipelineRoute(r);
  renderPlan(r);
  let step = 0;
  const tick = () => {
    if (token !== animating) return;
    paintFlow(r, step);
    const traceIndex = Math.max(0, route.nodes.slice(0, step + 1).filter((n) => !["input", "decision", "output"].includes(n)).length);
    renderTrace(r, traceIndex);
    if (step <= route.nodes.length) { step++; setTimeout(tick, 480); } else { animating = 0; renderers.s4(); }
  };
  tick();
}
$("s4-run").addEventListener("click", animatePipeline);
document.querySelectorAll("[data-scenario]").forEach((b) => b.addEventListener("click", () => {
  const kind = b.dataset.scenario;
  animating = 0;
  if (kind === "1d") state.case = "1d";
  else {
    state.case = "nd";
    if (kind === "nd-none") state.nd.magnitude = 0;
    else if (state.nd.magnitude === 0) state.nd.magnitude = DEFAULTS.nd.magnitude;
  }
  state.useModel = kind === "nd-model";
  if (state.useModel && !model) trainProcessModel(true);
  // Switching the case re-lays out section 3 above; keep this section still.
  const anchor = $("s4"), before = anchor.getBoundingClientRect().top;
  syncControls(); applyCase(); invalidate();
  requestAnimationFrame(() => {
    flush();
    window.scrollBy(0, anchor.getBoundingClientRect().top - before);
    animatePipeline();
  });
}));

// ============================================================ 5 — GNN/GAT
const p5 = {
  ring: new RingGraph("p5-ring", { height: 320 }),
  loss: new LinePlot("p5-loss", { height: 190 }),
  acc: new LinePlot("p5-acc", { height: 190 }),
  compare: new BarPlot("p5-compare", { height: 220 }),
};
let previewModel = null, previewKey = null;
function currentEpisode() {
  const r = resN();
  if (!r.ranking) return null;
  return { r, ep: C.episodeFromDense(r.ranking.features, 5, r.graph.adjacency, nd().source) };
}
function modelFor(ep) {
  if (model) return { m: model, trained: true };
  const key = JSON.stringify([state.nd, evtConfig("nd"), state.graph]);
  if (key !== previewKey) { previewModel = new C.GraphProcessModel({ hiddenDim: state.train.hidden, randomState: state.train.seed }).begin([ep]); previewKey = key; }
  return { m: previewModel, trained: false };
}
function renderMessagePassing() {
  const cur = currentEpisode();
  const info = $("s5-node");
  info.replaceChildren();
  if (!cur) {
    p5.ring.update({ n: 0, label: short, edges: [] });
    info.append(el("p", "hint", L("No alarm on the current n-D recording, so there is no episode to pass messages on. Raise the magnitude in section 2 or relax the spec in section 3.",
      "На текущей n-D записи нет тревоги, поэтому нет эпизода для передачи сообщений. Увеличьте амплитуду в разделе 2 или ослабьте спецификацию в разделе 3.")));
    return;
  }
  const { r, ep } = cur, p = ep.N, source = nd().source;
  const { m, trained } = modelFor(ep);
  const pred = m.predict(ep);
  const mode = state.mp.mode;
  const node = state.mp.node != null && state.mp.node < p ? state.mp.node : r.ranking.source;
  const prob = mode === "gat" ? pred.gatProbabilities : pred.gnnProbabilities;
  const best = argmax(prob), pmax = Math.max(...prob);
  const weightsOf = (i) => {
    const { idx, w } = ep.nbr[i];
    if (mode === "gat") return Array.from(idx, (j, k) => ({ j, w: pred.attention[i][k] }));
    const total = w.reduce((a, b) => a + b, 0);
    return Array.from(idx, (j, k) => ({ j, w: w[k] / total }));
  };
  const focus = weightsOf(node);
  const edges = [];
  for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) {
    if (!r.graph.adjacency[a * p + b]) continue;
    const touching = a === node || b === node;
    const other = a === node ? b : a;
    const w = touching ? focus.find((f) => f.j === other)?.w ?? 0 : 0;
    edges.push({ a, b, w: touching ? Math.min(1, 0.15 + 2 * w) : 0.05, color: touching ? (mode === "gat" ? "--series-3" : "--series-2") : "--axis" });
  }
  p5.ring.update({
    n: p, label: short, selected: node,
    fill: (i) => seqColor(prob[i] / pmax), labelColor: (i) => seqInk(prob[i] / pmax),
    ring: (i) => (i === best ? { color: "--ink", width: 3 } : null),
    badge: (i) => (i === source ? "★" : ""),
    edges,
    center: [mode === "gat" ? (trained ? L("GAT attention", "внимание GAT") : L("GAT, untrained", "GAT, не обучена")) : (trained ? "GNN" : L("GNN, untrained", "GNN, не обучена")), L("click a node", "кликните узел")],
    onClick: (i) => commit("mp.node", i),
    tooltip: (i) => ({ title: ch(i) + (i === source ? " ★" : ""), rows: [
      { value: fmt(pred.gnnProbabilities[i], 3), label: "P_GNN", color: "--series-2" },
      { value: fmt(pred.gatProbabilities[i], 3), label: "P_GAT", color: "--series-3" },
      { value: fmt(C.heuristicProbabilities(ep.x, 5)[i], 3), label: L("P_heuristic", "P_эвристика"), color: "--series-1" },
    ] }),
  });
  $("s5-mp-hint").textContent = mode === "gat"
    ? L(`Edges from the selected node are drawn with width ∝ α_ij — what the GAT ${trained ? "learned" : "starts with (random a₁, a₂)"} for this episode. Fill = P_GAT, ring = argmax, ★ = true source.`,
      `Рёбра выбранного узла — ширина ∝ α_ij: то, что GAT ${trained ? "выучила" : "имеет на старте (случайные a₁, a₂)"} на этом эпизоде. Заливка = P_GAT, обводка = argmax, ★ = истинный источник.`)
    : L(`GNN neighbours count equally: Â = D⁻¹(A + I) gives every neighbour weight 1/|N(i)|. Fill = P_GNN, ring = argmax, ★ = true source.${trained ? "" : " Train in the card below."}`,
      `В GNN соседи равноправны: Â = D⁻¹(A + I) даёт каждому вес 1/|N(i)|. Заливка = P_GNN, обводка = argmax, ★ = истинный источник.${trained ? "" : " Обучите в карточке ниже."}`);
  info.append(el("h4", null, `${ch(node)}${node === source ? " ★" : ""}${node === best ? "  (argmax)" : ""}`));
  const raw = Array.from({ length: 5 }, (_, k) => ep.x[node * 5 + k]);
  const agg = new Float64Array(5);
  for (const { j, w } of focus) for (let k = 0; k < 5; k++) agg[k] += w * ep.x[j * 5 + k];
  const tbl = el("table", "data");
  const head = el("tr");
  for (const h of [L("feature", "признак"), "x_i", mode === "gat" ? "Σ α_ij x_j" : "(Âx)_i"]) head.append(el("th", null, h));
  const thead = el("thead"); thead.append(head);
  const tbody = el("tbody");
  C.FEATURE_NAMES.forEach((f, k) => { const tr = el("tr"); tr.append(el("td", null, f), el("td", null, fmt(raw[k], 2)), el("td", null, fmt(agg[k], 2))); tbody.append(tr); });
  tbl.append(thead, tbody);
  info.append(tbl, el("div", "panel-label", mode === "gat" ? L("Attention α_ij over N(i) (sums to 1)", "Внимание α_ij по N(i) (сумма 1)") : L("Mean weights 1/|N(i)|", "Веса среднего 1/|N(i)|")));
  const maxW = Math.max(...focus.map((f) => f.w));
  for (const { j, w } of [...focus].sort((a, b) => b.w - a.w)) {
    const row = el("div", `wbar ${mode}`);
    const bar = el("div", "bar");
    bar.style.width = `${(w / maxW) * 100}%`;
    const holder = el("div"); holder.append(bar);
    row.append(el("span", null, (j === node ? L("self ", "сам ") : "") + ch(j).replace("channel_", "ch ")), holder, el("span", "v", fmt(w, 3)));
    info.append(row);
  }
  // Three rankers on the current episode
  const heur = C.heuristicProbabilities(ep.x, 5);
  const groups = [{ values: heur, color: "--series-1", label: L("heuristic", "эвристика") }];
  if (trained) groups.push({ values: pred.gnnProbabilities, color: "--series-2", label: "GNN" }, { values: pred.gatProbabilities, color: "--series-3", label: "GAT" });
  p5.compare.update({
    labels: Array.from({ length: p }, (_, j) => short(j)), groups, yMax: 1,
    marks: (i) => (i === source ? "★" : ""),
    tooltip: (i) => ({ title: ch(i) + (i === source ? " ★" : ""), rows: groups.map((g) => ({ value: fmt(g.values[i], 3), label: g.label, color: g.color })) }),
  });
  const verdict = (w) => (argmax(w) === source ? "✓" : "✕");
  $("s5-compare-note").textContent = trained
    ? L(`argmax: heuristic ${short(argmax(heur))} ${verdict(heur)} · GNN ${short(pred.gnnSource)} ${verdict(pred.gnnProbabilities)} · GAT ${short(pred.gatSource)} ${verdict(pred.gatProbabilities)}. One episode is an anecdote — the validation table above is the evidence.`,
      `argmax: эвристика ${short(argmax(heur))} ${verdict(heur)} · GNN ${short(pred.gnnSource)} ${verdict(pred.gnnProbabilities)} · GAT ${short(pred.gatSource)} ${verdict(pred.gatProbabilities)}. Один эпизод — анекдот; доказательство — таблица валидации выше.`)
    : L("Only the transparent heuristic until a model is trained.", "Пока модель не обучена, показана только прозрачная эвристика.");
}
function renderCurves(m, plots, statusId, metricsFn) {
  const h = m.history;
  if (!h || !h.epoch.length) return;
  const n = h.epoch.length, x = (i) => h.epoch[i];
  plots.loss.update({
    n, x, xLabel: L("epoch", "эпоха"),
    series: [
      { y: h.gnn_train_loss, color: "--series-2" }, { y: h.gnn_val_loss, color: "--series-2", dash: [5, 4] },
      { y: h.gat_train_loss, color: "--series-3" }, { y: h.gat_val_loss, color: "--series-3", dash: [5, 4] },
    ],
    tooltip: (i) => ({ title: L(`epoch ${h.epoch[i]}`, `эпоха ${h.epoch[i]}`), rows: [
      { value: fmt(h.gnn_train_loss[i], 3), label: L("GNN train", "GNN обучение"), color: "--series-2" }, { value: fmt(h.gnn_val_loss[i], 3), label: L("GNN val", "GNN валидация"), color: "--series-2" },
      { value: fmt(h.gat_train_loss[i], 3), label: L("GAT train", "GAT обучение"), color: "--series-3" }, { value: fmt(h.gat_val_loss[i], 3), label: L("GAT val", "GAT валидация"), color: "--series-3" },
    ] }),
  });
  const heurAcc = metricsFn ? metricsFn() : null;
  plots.acc.update({
    n, x, xLabel: L("epoch", "эпоха"), yDomain: [0, 1.02],
    series: [
      { y: h.gnn_train_accuracy, color: "--series-2" }, { y: h.gnn_val_accuracy, color: "--series-2", dash: [5, 4] },
      { y: h.gat_train_accuracy, color: "--series-3" }, { y: h.gat_val_accuracy, color: "--series-3", dash: [5, 4] },
    ],
    hlines: heurAcc != null ? [{ y: heurAcc, color: "--series-1", dash: [2, 3], label: L(`heuristic (val) ${fmt(heurAcc)}`, `эвристика (вал.) ${fmt(heurAcc)}`) }] : [],
    tooltip: (i) => ({ title: L(`epoch ${h.epoch[i]}`, `эпоха ${h.epoch[i]}`), rows: [
      { value: pct(h.gnn_train_accuracy[i]), label: L("GNN train", "GNN обучение"), color: "--series-2" }, { value: pct(h.gnn_val_accuracy[i]), label: L("GNN val", "GNN валидация"), color: "--series-2" },
      { value: pct(h.gat_train_accuracy[i]), label: L("GAT train", "GAT обучение"), color: "--series-3" }, { value: pct(h.gat_val_accuracy[i]), label: L("GAT val", "GAT валидация"), color: "--series-3" },
    ] }),
  });
  $(statusId).textContent = L(`epoch ${m.step}/${m.config.epochs} · ${m.train.length} train / ${m.validation.length} validation episodes`,
    `эпоха ${m.step}/${m.config.epochs} · ${m.train.length} обучающих / ${m.validation.length} валидационных эпизодов`);
}
function metricsTable(target, report, extraCols = null) {
  if (!report) { $(target).replaceChildren(); return; }
  const methods = [["heuristic", L("heuristic", "эвристика")], ["gnn", "GNN"], ["gat", "GAT"]];
  const best = Math.max(...methods.map(([k]) => report[k].top1));
  table(target, [L("validation", "валидация"), "n", "Top-1", "Top-3", "MRR", "NLL", ...(extraCols ? extraCols.head : [])],
    methods.map(([k, name]) => [name, String(report[k].n), { text: pct(report[k].top1), cls: report[k].top1 === best ? "best" : "" }, pct(report[k].topk), fmt(report[k].mrr, 3), fmt(report[k].nll, 2), ...(extraCols ? extraCols.row(k) : [])]));
}
function buildTrainingEpisodes() {
  const episodes = [];
  const t = state.train, n0 = state.nd;
  const evt = { baseline: 300, tailQuantile: 0.9, alarmProbability: 0.99, topK: 3, persistence: 2 };
  for (let r = 0; r < t.recordings; r++) {
    const rec = C.propagationRecording({ p: n0.p, seed: t.seed * 1000 + r, coupling: n0.coupling, dt: n0.dt, hurst: n0.hurst, magnitude: Math.max(n0.magnitude, 0.5), delay: n0.delay, decay: n0.decay });
    for (const e of C.labelRecording(rec, evt, graphConfig(), `rec_${r}`)) if (e.source !== null) episodes.push(C.episodeFromDense(e.features, 5, e.adjacency, e.source, e.group));
  }
  return episodes;
}
let training = 0;
function trainProcessModel(sync = false) {
  const t = state.train;
  const episodes = buildTrainingEpisodes();
  if (episodes.length < 4) {
    $("s5-train-status").textContent = L("Too few detected episodes — raise the magnitude in section 2.", "Слишком мало обнаруженных эпизодов — увеличьте амплитуду в разделе 2.");
    return;
  }
  const split = C.splitEpisodes(episodes, t.validation, t.seed);
  const m = new C.GraphProcessModel({ hiddenDim: t.hidden, epochs: t.epochs, learningRate: t.lr, randomState: t.seed }).begin(split.train, split.validation);
  const finish = () => {
    const sources = split.validation.map((e) => e.source);
    const preds = split.validation.map((e) => m.predict(e));
    modelReport = {
      heuristic: C.rankingMetrics(split.validation.map((e) => C.heuristicProbabilities(e.x, 5)), sources),
      gnn: C.rankingMetrics(preds.map((q) => q.gnnProbabilities), sources),
      gat: C.rankingMetrics(preds.map((q) => q.gatProbabilities), sources),
    };
    model = m; modelVersion++;
    renderCurves(m, p5, "s5-train-status", () => modelReport.heuristic.top1);
    metricsTable("s5-metrics", modelReport);
    $("s5-train").disabled = false;
    evalDirty = true;
    invalidate("s4", "s5", "s6");
  };
  if (sync) { while (m.epoch()) { /* train now */ } finish(); return; }
  const token = ++training;
  $("s5-train").disabled = true;
  const step = () => {
    if (token !== training) return;
    const start = performance.now();
    let more = true;
    while (more && performance.now() - start < 22) more = m.epoch();
    renderCurves(m, p5, "s5-train-status", null);
    if (more) requestAnimationFrame(step); else finish();
  };
  requestAnimationFrame(step);
}
$("s5-train").addEventListener("click", () => trainProcessModel());
let s5Started = false;
renderers.s5 = () => {
  renderMessagePassing();
  if (model) { renderCurves(model, p5, "s5-train-status", () => modelReport.heuristic.top1); metricsTable("s5-metrics", modelReport); }
  if (!s5Started) { s5Started = true; trainProcessModel(); }
};

// ============================================================ 6 — orchestration
const GUARD = "Опирайся только на JSON. Не изменяй численные результаты. Отделяй статистическую редкость от причинности и явно отмечай ограничения.";
function evidence(r) {
  const d = r.detection;
  let mn = Infinity, mx = -Infinity;
  for (const v of d.indicator) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const round = (v, k = 4) => +v.toFixed(k);
  const payload = {
    input_profile: { kind: r.profile.kind, n_time: r.profile.n_time, n_channels: r.profile.n_channels, channel_names: r.profile.n_channels > 4 ? [...r.profile.channel_names.slice(0, 3), "…"] : r.profile.channel_names, missing: 0, regular_time: true },
    execution_plan: r.plan,
    detection: { detected: d.detected, time_index: d.timeIndex, alarm_threshold: round(d.alarmThreshold), indicator_min: round(mn), indicator_max: round(mx), indicator_length: d.indicator.length },
    graph: null, ranking: null,
  };
  if (r.graph) {
    let lo = Infinity, hi = -Infinity;
    for (const v of r.graph.dependence) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    payload.graph = { method: r.graph.method, node_count: r.graph.p, undirected_edge_count: C.edgeCount(r.graph), dependence_min: round(lo), dependence_max: round(hi) };
  }
  if (r.ranking) {
    const k = Math.min(10, r.ranking.order.length);
    payload.ranking = { node_ids: r.ranking.order.slice(0, k), probabilities: r.ranking.order.slice(0, k).map((j) => round(r.ranking.probabilities[j])) };
  }
  if (r.process) payload.learned_process = { gnn_source: r.process.gnnSource, gat_source: r.process.gatSource, gnn_probabilities: Array.from(r.process.gnnProbabilities, (v) => round(v, 3)), gat_probabilities: Array.from(r.process.gatProbabilities, (v) => round(v, 3)) };
  return payload;
}
function teamMessages(r) {
  const ev = evidence(r), d = r.detection, tr = truth(), isNd = r.profile.n_channels > 1;
  const skipReason = !isNd ? ["одномерный вход", "one-dimensional input"] : !d.detected ? ["экстремальное событие не обнаружено", "no extreme event detected"] : null;
  const delay = d.detected ? d.timeIndex - tr.onset : null;
  const top = r.ranking ? r.ranking.order.slice(0, 3).map((j) => `${short(j)} (${fmt(r.ranking.probabilities[j], 2)})`).join(", ") : "";
  const sorted = r.ranking ? Array.from(r.ranking.probabilities).sort((a, b) => b - a) : [];
  const msgs = [
    { node: "input", who: L("Input agent", "Агент входа"), sys: "Ты агент проверки входных данных.",
      reply: L(`Route ${r.plan.route}: ${r.profile.kind} input, ${r.profile.n_time} samples × ${r.profile.n_channels} channel(s); regular time axis, no missing values. Baseline of ${d.config.baseline} samples is ${r.plan.baseline_stationary ? "approximately stationary" : "non-stationary"} by the planner's rule.`,
        `Маршрут ${r.plan.route}: вход ${r.profile.kind}, ${r.profile.n_time} отсчётов × ${r.profile.n_channels} канал(ов); регулярная ось времени, пропусков нет. Фон из ${d.config.baseline} отсчётов по правилу планировщика ${r.plan.baseline_stationary ? "приблизительно стационарен" : "нестационарен"}.`) },
    { node: "evt", who: L("EVT agent", "Агент EVT"), sys: "Ты агент EVT-детекции.",
      reply: d.detected
        ? L(`Persistent POT/GPD exceedance at t = ${d.timeIndex} (threshold ${fmt(d.alarmThreshold, 3)}, indicator ${fmt(ev.detection.indicator_min, 2)}–${fmt(ev.detection.indicator_max, 2)}); ${tr.onset == null ? "no event was injected, so this is a false alarm" : `${delay >= 0 ? `${delay} samples after` : `${-delay} samples before`} the labelled onset ${tr.onset}`}. This is statistical rarity relative to the baseline, not evidence of a cause.`,
          `Устойчивое превышение POT/GPD в t = ${d.timeIndex} (порог ${fmt(d.alarmThreshold, 3)}, индикатор ${fmt(ev.detection.indicator_min, 2)}–${fmt(ev.detection.indicator_max, 2)}); ${tr.onset == null ? "событие не впрыскивалось, значит, это ложная тревога" : `${delay >= 0 ? `через ${delay} отсчётов после` : `за ${-delay} отсчётов до`} размеченного начала ${tr.onset}`}. Это статистическая редкость относительно фона, а не доказательство причины.`)
        : L(`No persistent exceedance of ${fmt(d.alarmThreshold, 3)} after the baseline; the route stops with no_evt_detected.`, `Нет устойчивого превышения ${fmt(d.alarmThreshold, 3)} после фона; маршрут останавливается: no_evt_detected.`) },
    { node: "graph", who: L("Graph agent", "Агент графа"), sys: "Ты агент графовых зависимостей.", skip: skipReason,
      reply: r.graph ? L(`${ev.graph.method}: ${ev.graph.undirected_edge_count} undirected edges among ${ev.graph.node_count} nodes, dependence ${fmt(ev.graph.dependence_min, 2)}–${fmt(ev.graph.dependence_max, 2)}. Built on the baseline only and frozen before t_EVT, so the event cannot leak into the edges. The graph is a dependence hypothesis, not a causal map.`,
        `${ev.graph.method}: ${ev.graph.undirected_edge_count} неориентированных рёбер на ${ev.graph.node_count} узлах, зависимость ${fmt(ev.graph.dependence_min, 2)}–${fmt(ev.graph.dependence_max, 2)}. Построен только по фону и зафиксирован до t_EVT, поэтому событие не утекает в рёбра. Граф — гипотеза зависимости, а не причинная карта.`) : "" },
    { node: "loc", who: L("Localization agent", "Агент локализации"), sys: "Ты агент локализации.", skip: skipReason,
      reply: r.ranking ? L(`Top candidates: ${top}; margin ${fmt(sorted[0] - sorted[1], 2)}.${r.process ? ` Learned models: GNN → ${short(r.process.gnnSource)}, GAT → ${short(r.process.gatSource)}.` : ""} The ranking rewards early, strong responses supported by graph neighbours; it is a hypothesis about the initial source, not proof of a physical cause.`,
        `Лидеры: ${top}; отрыв ${fmt(sorted[0] - sorted[1], 2)}.${r.process ? ` Обученные модели: GNN → ${short(r.process.gnnSource)}, GAT → ${short(r.process.gatSource)}.` : ""} Ранжирование поощряет ранний сильный отклик, поддержанный соседями по графу; это гипотеза о начальном источнике, а не доказательство физической причины.`) : "" },
  ];
  const est = r.ranking ? r.ranking.source : null;
  msgs.push({ node: "coord", who: L("Coordinator", "Координатор"), sys: "Ты координатор команды.", final: true,
    reply: !d.detected ? L("No extreme event relative to the chosen baseline. Nothing to localize; consider the baseline and spec before concluding there is no event.", "Экстремального события относительно выбранного фона нет. Локализовать нечего; прежде чем утверждать отсутствие события, проверьте фон и спецификацию.")
      : !isNd ? L(`Extreme event detected at t = ${d.timeIndex}. With one channel the source is not identifiable; report detection time and tail probability, or use a temporal graph to localize the onset window.`, `Экстремальное событие обнаружено в t = ${d.timeIndex}. С одним каналом источник не идентифицируем; сообщите момент детекции и хвостовую вероятность или используйте временной граф для локализации окна начала.`)
      : L(`Event at t = ${d.timeIndex}; estimated initial source ${ch(est)} (P = ${fmt(r.ranking.probabilities[est], 2)})${tr.source != null ? `, which ${est === tr.source ? "matches" : "does not match"} the synthetic label ${ch(tr.source)}` : ""}. Check robustness to persistence, tail quantile and graph method before any conclusion; one run is not a benchmark.`,
        `Событие в t = ${d.timeIndex}; оценка начального источника — ${ch(est)} (P = ${fmt(r.ranking.probabilities[est], 2)})${tr.source != null ? `, что ${est === tr.source ? "совпадает" : "не совпадает"} с синтетической меткой ${ch(tr.source)}` : ""}. Перед выводами проверьте устойчивость к persistence, квантилю хвоста и методу графа; один запуск — не benchmark.`) });
  return msgs;
}
function renderTranscript(msgs, upto = Infinity) {
  const box = $("s6-transcript");
  box.replaceChildren();
  msgs.forEach((m, i) => {
    const card = el("div", `msg${m.skip ? " skip" : ""}${m.final ? " final" : ""}${i >= upto ? " hidden" : ""}`);
    const who = el("div", "who");
    who.append(el("b", null, m.who), el("span", "sys", `system: "${m.sys} ${GUARD.slice(0, 38)}…"`));
    const reply = el("div", "reply");
    reply.textContent = m.skip ? `Пропущено: ${m.skip[0]}.${lang === "en" ? ` — skipped: ${m.skip[1]}.` : ""}` : m.reply;
    card.append(who, reply);
    box.append(card);
  });
}
function paintTeam(msgs, upto = Infinity) {
  const svg = $("team"), nodes = flowNodes(svg), edges = flowEdges(svg);
  const order = ["pipeline", "evidence", "input", "evt", "graph", "loc", "coord", "report"];
  const skipped = new Set(msgs.filter((m) => m.skip).map((m) => m.node));
  order.forEach((name, i) => {
    const g = nodes[name];
    g.classList.remove("done", "active", "skipped");
    if (skipped.has(name)) g.classList.add("skipped");
    else if (i < upto) g.classList.add("done");
    else if (i === upto) g.classList.add("active");
  });
  const edgeStep = { "pipe-evidence": 1, "evidence-input": 2, "evidence-evt": 3, "evidence-graph": 4, "evidence-loc": 5, "input-coord": 6, "evt-coord": 6, "graph-coord": 6, "loc-coord": 6, "coord-report": 7 };
  for (const [name, p] of Object.entries(edges)) {
    const target = name.split("-")[1], source = name.split("-")[0];
    const dead = skipped.has(target) || skipped.has(source);
    p.classList.toggle("on", !dead && edgeStep[name] <= upto);
  }
}
let teamAnimating = 0;
function renderTeam() {
  const r = state.case === "1d" ? res1() : resN(!!model);
  $("s6-evidence").textContent = JSON.stringify(evidence(r), null, 2);
  $("s6-evidence-case").textContent = state.case === "1d" ? "1-D" : "n-D";
  if (teamAnimating) return;
  const msgs = teamMessages(r);
  renderTranscript(msgs);
  paintTeam(msgs);
}
$("s6-team-run").addEventListener("click", () => {
  const r = state.case === "1d" ? res1() : resN(!!model);
  const msgs = teamMessages(r), token = ++teamAnimating;
  let step = 0;
  const tick = () => {
    if (token !== teamAnimating) return;
    paintTeam(msgs, step);
    renderTranscript(msgs, Math.max(0, step - 1));
    if (step < 8) { step++; setTimeout(tick, 520); } else { teamAnimating = 0; renderTeam(); }
  };
  tick();
});

const OUTCOME_STYLE = {
  hit: { icon: "✓", en: "hit", ru: "попадание" },
  correct_rejection: { icon: "○", en: "correct rejection", ru: "верный отказ" },
  miss: { icon: "✕", en: "miss", ru: "пропуск" },
  early_alarm: { icon: "«", en: "early alarm", ru: "ранняя тревога" },
  false_alarm: { icon: "!", en: "false alarm", ru: "ложная тревога" },
};
let evalDirty = true, evalCases = null;
function runEvaluation() {
  const b = state.evalb, rng = C.makeRng(b.seed, "batch");
  const cases = [];
  for (let i = 0; i < b.cases; i++) {
    const is1d = rng.uniform() < b.share1d, hasEvent = rng.uniform() >= b.shareNone, onset = rng.int(330, 450);
    // Oracle: the baseline ends at the (pseudo-)onset, as in notebooks 07/10.
    const baseline = b.oracle ? onset : 300;
    if (is1d) {
      const s = C.additiveSeries({ ...state.sig, n: 600, seed: b.seed * 1000 + i, onset, duration: 60, magnitude: hasEvent ? state.sig.magnitude : 0 });
      cases.push({ id: `case_${i}`, data: C.fromSeries(s.values), eventTime: hasEvent ? onset : null, source: null, opts: { evt: { ...evtConfig("1d"), baseline } } });
    } else {
      const source = rng.int(0, state.nd.p - 1);
      const d = C.propagationSeries({ ...state.nd, n: 600, seed: b.seed * 1000 + i, source, onset, duration: 30, magnitude: hasEvent ? Math.max(state.nd.magnitude, 0.5) : 0 });
      cases.push({ id: `case_${i}`, data: d, eventTime: hasEvent ? onset : null, source: hasEvent ? source : null, opts: { evt: { ...evtConfig("nd"), baseline }, graph: graphConfig(), processModel: model } });
    }
  }
  evalCases = cases.map((c) => C.evaluateCase(c, c.opts, b.earlyTolerance));
  evalDirty = false;
}
function renderEvaluation() {
  if (evalDirty || !evalCases) runEvaluation();
  const legend = $("s6-outcome-legend");
  legend.replaceChildren();
  for (const [key, s] of Object.entries(OUTCOME_STYLE)) {
    const k = el("span", "key");
    const sw = el("span", `case-chip ${key}`, s.icon);
    sw.style.cssText = "width:20px;height:20px;font-size:11px;border-width:2px";
    k.append(sw, el("span", null, lang === "ru" ? s.ru : s.en));
    legend.append(k);
  }
  const grid = $("s6-cases");
  grid.replaceChildren();
  for (const c of evalCases) {
    const chip = el("div", `case-chip ${c.outcome}`, OUTCOME_STYLE[c.outcome].icon);
    chip.append(el("span", "r", c.route === "1-d" ? "1" : "n"));
    const parts = [`${c.id} · ${c.route} · ${lang === "ru" ? OUTCOME_STYLE[c.outcome].ru : OUTCOME_STYLE[c.outcome].en}`,
      `${L("true onset", "истинное начало")}: ${c.eventTime ?? "—"} · ${L("alarm", "тревога")}: ${c.detected ?? "—"}${c.delay != null ? ` (+${c.delay})` : ""}`];
    if (c.source != null) parts.push(`${L("true source", "истинный источник")}: ${short(c.source)} · ${Object.entries(c.predicted).map(([m, v]) => `${m} → ${short(v)}${c.ranks[m] ? ` (rank ${c.ranks[m]})` : ""}`).join(", ")}`);
    if (c.stop_reason) parts.push(`stop_reason: ${c.stop_reason}`);
    chip.title = parts.join("\n");
    chip.tabIndex = 0;
    chip.setAttribute("aria-label", parts.join(". "));
    grid.append(chip);
  }
  const k = state.evalb.k, summary = C.summarize(evalCases, k);
  const routes = Object.keys(summary);
  const metrics = [
    ["cases", L("cases", "случаев")], ["hit", L("hit", "попадание")], ["miss", L("miss", "пропуск")], ["early_alarm", L("early alarm", "ранняя тревога")],
    ["false_alarm", L("false alarm", "ложная тревога")], ["correct_rejection", L("correct rejection", "верный отказ")],
    ["detection_rate", L("detection rate", "доля обнаружений")], ["false_alarm_rate", L("false-alarm rate", "доля ложных тревог")], ["mean_delay", L("mean delay", "средняя задержка")],
    ...["heuristic", "gnn", "gat"].flatMap((m) => [[`${m}_n`, `${m} n`], [`${m}_top1`, `${m} Top-1`], [`${m}_top${k}`, `${m} Top-${k}`], [`${m}_mrr`, `${m} MRR`]]),
  ];
  const show = (key, v) => (v == null || (typeof v === "number" && !Number.isFinite(v)) ? "–" : /rate|top|mrr/.test(key) ? fmt(v, 2) : key === "mean_delay" ? fmt(v, 1) : String(v));
  table("s6-summary", [L("summary()", "summary()"), ...routes],
    metrics.filter(([key]) => routes.some((r) => key in summary[r])).map(([key, name]) => [name, ...routes.map((r) => show(key, summary[r][key]))]));
}
$("s6-eval-run").addEventListener("click", () => { evalDirty = true; renderEvaluation(); });
renderers.s6 = () => { renderTeam(); evalDirty = true; renderEvaluation(); };

// ============================================================ 7 — graph hypotheses
const METHODS7 = ["ar", "dfa", "kuramoto", "ensemble"];
const p7 = Object.fromEntries(METHODS7.map((m) => [m, new RingGraph(`p7-${m}`, { height: 250 })]));
renderers.s7 = () => {
  const d = nd(), r = resN(), det = r.detection, B = det.config.baseline, base = C.slice(d, 0, B);
  const stationary = C.approximatelyStationary(base);
  const rows = [];
  const tiles = $("s7-tiles");
  tiles.replaceChildren(
    tile(L("Baseline stationarity", "Стационарность фона"), stationary ? L("≈ stationary", "≈ стационарен") : L("non-stationary", "нестационарен"), L("4 chunks: mean range < σ, variance ratio < 4", "4 куска: размах средних < σ, отношение дисперсий < 4")),
    tile(L("auto picks", "auto выбирает"), stationary ? "AR" : "DFA ∩ Kuramoto", L("planner rule from the method", "правило планировщика из метода")),
    tile(L("Alarm used for ranking", "Тревога для ранжирования"), det.detected ? `t = ${det.timeIndex}` : L("none", "нет"), L(`baseline B = ${B}`, `фон B = ${B}`)),
  );
  for (const method of METHODS7) {
    const g = C.buildGraph(base, { ...graphConfig(), method });
    const ranking = det.detected ? C.rankSources(d, det.timeIndex, g.adjacency, det.location, det.scale) : null;
    const prob = ranking ? ranking.probabilities : new Float64Array(d.p);
    const pmax = Math.max(1e-9, ...prob);
    const edges = [];
    let depSum = 0;
    for (let a = 0; a < d.p; a++) for (let b = a + 1; b < d.p; b++) {
      depSum += g.dependence[a * d.p + b];
      if (g.adjacency[a * d.p + b]) edges.push({ a, b, w: Math.max(0.05, Math.min(1, g.dependence[a * d.p + b])), color: "--muted" });
    }
    const pairs = (d.p * (d.p - 1)) / 2;
    p7[method].update({
      n: d.p, label: short,
      fill: (i) => seqColor(prob[i] / pmax), labelColor: (i) => seqInk(prob[i] / pmax),
      ring: (i) => (ranking && i === ranking.source ? { color: "--ink", width: 3 } : null),
      badge: (i) => (i === d.source ? "★" : ""),
      edges,
      tooltip: (i) => ({ title: ch(i) + (i === d.source ? " ★" : ""), rows: [
        { value: fmt(prob[i], 3), label: "P(source)" },
        { value: String(edges.filter((e) => e.a === i || e.b === i).length), label: L("edges", "рёбер") },
      ] }),
    });
    document.querySelector(`[data-g="${method}"]`).textContent = L(`${edges.length} edges`, `${edges.length} рёбер`);
    const rank = ranking ? 1 + Array.from(prob).filter((v) => v > prob[d.source]).length : null;
    rows.push([g.method, String(edges.length), fmt(edges.length / pairs, 2), fmt(depSum / pairs, 3), ranking ? ch(ranking.source) : "–", ranking ? fmt(prob[d.source], 3) : "–", rank == null ? "–" : { text: String(rank), cls: rank === 1 ? "best" : "" }]);
  }
  table("s7-table", [L("graph", "граф"), L("edges", "рёбра"), L("density", "плотность"), L("mean dependence", "средняя зависимость"), "ŝ", L("P(true source)", "P(ист. источник)"), L("rank of true", "ранг истинного")], rows);
};

// ============================================================ 8 — temporal graph
const p8 = {
  series: new LinePlot("p8-series", { height: 170, group: "s8" }),
  pred: new LinePlot("p8-pred", { height: 150, group: "s8" }),
  loss: new LinePlot("p8-loss", { height: 180 }),
  acc: new LinePlot("p8-acc", { height: 180 }),
};
const temporalConfig = () => ({ windowSize: state.tg.windowSize, stride: state.tg.stride, neighborhood: state.tg.neighborhood, baseline: 300, context: state.tg.context, arOrder: 8, horizons: [32, 64, 128] });
let temporalKey = null;
function drawNodeStrip(graph, target, preds) {
  const canvas = $("p8-nodes"), g = p8.series.g;
  if (!g) return;
  const dpr = window.devicePixelRatio || 1, width = canvas.parentElement.clientWidth, height = 76;
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const w = state.tg.windowSize, cx = (i) => g.px(graph.starts[i] + w / 2), y = 30;
  let pk = 1;
  for (let i = 0; i < graph.N; i++) pk = Math.max(pk, graph.features[i * graph.F]);
  ctx.strokeStyle = color("--axis"); ctx.lineWidth = 1;
  for (let i = 0; i < graph.N; i++) for (const j of graph.nbr[i].idx) {
    if (j <= i) continue;
    const a = cx(i), b = cx(j);
    ctx.beginPath(); ctx.moveTo(a, y); ctx.quadraticCurveTo((a + b) / 2, y - 6 - 5 * (j - i), b, y); ctx.stroke();
  }
  const r = Math.max(1.6, Math.min(4, (g.px(state.tg.stride) - g.px(0)) / 2.4));
  for (let i = 0; i < graph.N; i++) {
    ctx.fillStyle = seqColor(Math.log1p(graph.features[i * graph.F]) / Math.log1p(pk));
    ctx.beginPath(); ctx.arc(cx(i), y, r, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.strokeStyle = color("--series-2"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx(target), y, r + 3, 0, 2 * Math.PI); ctx.stroke();
  ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = "center"; ctx.textBaseline = "top";
  preds.forEach(([name, idx, tok], k) => {
    ctx.fillStyle = color(tok);
    ctx.fillText(`▲ ${name}`, cx(idx), y + r + 3 + k * 11);
  });
  ctx.fillStyle = color("--muted"); ctx.textAlign = "left";
  ctx.fillText(L(`${graph.N} nodes · ${graph.F} features`, `${graph.N} узлов · ${graph.F} признаков`), g.left, 2);
}
function renderTemporal() {
  const s = sig(), cfg = temporalConfig();
  const graph = C.temporalGraph(s.values, cfg);
  const target = C.temporalEpisode(graph, s.onset, cfg.windowSize).source;
  const w = cfg.windowSize, ts = graph.starts[target];
  p8.series.update({
    n: s.n, series: [{ y: s.values, color: "--series-1", width: 1.1 }],
    bands: [{ x0: 0, x1: cfg.baseline, color: "--band-neutral" }, { x0: ts, x1: ts + w, color: "--band" }],
    vlines: [{ x: s.onset, color: "--ink", dash: [4, 3], label: `t₀ = ${s.onset}` }], hideX: true,
    tooltip: (i) => ({ title: `t = ${i} · ${L("window", "окно")} #${Math.floor(i / cfg.stride)}`, rows: [{ value: fmt(s.values[i]), label: "x(t)", color: "--series-1" }] }),
  });
  const fresh = temporalModel && temporalKey === JSON.stringify([cfg.windowSize, cfg.stride, cfg.neighborhood, cfg.context]);
  const preds = [];
  const ep = C.temporalEpisode(graph, s.onset, w);
  const heur = C.heuristicProbabilities(graph.features, graph.F);
  preds.push([L("heuristic", "эвристика"), argmax(heur), "--series-1"]);
  if (fresh) {
    const q = temporalModel.predict(ep);
    preds.push(["GNN", q.gnnSource, "--series-2"], ["GAT", q.gatSource, "--series-3"]);
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of s.values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const span = hi - lo || 1, laneH = span / preds.length;
  p8.pred.update({
    n: s.n, series: [{ y: s.values, color: "--muted", width: 1 }], yDomain: [lo, hi],
    bands: preds.map(([, idx, tok], k) => ({ x0: graph.starts[idx], x1: graph.starts[idx] + w, y0: hi - (k + 1) * laneH, y1: hi - k * laneH, color: tok })),
    vlines: [{ x: s.onset, color: "--ink", dash: [4, 3] }], xLabel: L("sample t", "отсчёт t"),
    tooltip: (i) => ({ title: `t = ${i}`, rows: preds.map(([name, idx, tok]) => ({ value: `[${graph.starts[idx]}, ${graph.starts[idx] + w})`, label: name, color: tok })) }),
  });
  requestAnimationFrame(() => drawNodeStrip(graph, target, preds));
  const describe = ([name, idx]) => {
    const a = graph.starts[idx], hit = a <= s.onset && s.onset < a + w, inEvent = a < s.onset + s.duration && a + w > s.onset;
    return `${name} → [${a}, ${a + w}) ${hit ? L("contains t₀ ✓", "содержит t₀ ✓") : inEvent ? L("inside the event, not the onset", "внутри события, но не начало") : L("outside the event ✕", "вне события ✕")}`;
  };
  $("s8-pred-note").textContent = `${L("Target", "Цель")}: [${ts}, ${ts + w}) · ${preds.map(describe).join(" · ")}${fresh ? "" : L(" · train below to add GNN/GAT", " · обучите ниже, чтобы добавить GNN/GAT")}`;
  if (cfg.baseline >= s.onset) $("s8-pred-note").textContent += L(" · ⚠ onset is inside the 300-sample baseline", " · ⚠ начало внутри фона из 300 отсчётов");
}
let temporalTraining = 0;
function trainTemporal() {
  const cfg = temporalConfig(), t = state.tg;
  const crop = C.makeRng(42, "crop");
  const episodes = [];
  for (let r = 0; r < t.records; r++) {
    const s = C.additiveSeries({ ...state.sig, n: 1500, onset: null, duration: null, seed: 2000 + r });
    const c = crop.int(0, 499);
    const values = s.values.slice(c);
    const g = C.temporalGraph(values, cfg);
    episodes.push(C.temporalEpisode(g, s.onset - c, cfg.windowSize, `rec_${r}`));
  }
  const split = C.splitEpisodes(episodes, 0.25, 42);
  const m = new C.GraphProcessModel({ hiddenDim: t.hidden, epochs: t.epochs, learningRate: 0.02, randomState: 42 }).begin(split.train, split.validation);
  const token = ++temporalTraining;
  $("s8-train").disabled = true;
  const step = () => {
    if (token !== temporalTraining) return;
    const start = performance.now();
    let more = true;
    while (more && performance.now() - start < 24) more = m.epoch(2);
    renderCurves(m, p8, "s8-train-status", null);
    if (more) { requestAnimationFrame(step); return; }
    const sources = split.validation.map((e) => e.source);
    const preds = split.validation.map((e) => m.predict(e));
    const lists = { heuristic: split.validation.map((e) => C.heuristicProbabilities(e.x, e.F)), gnn: preds.map((q) => q.gnnProbabilities), gat: preds.map((q) => q.gatProbabilities) };
    temporalReport = Object.fromEntries(Object.entries(lists).map(([k, v]) => [k, { ...C.rankingMetrics(v, sources), within1: v.filter((w, e) => Math.abs(argmax(w) - sources[e]) <= 1).length / sources.length }]));
    temporalModel = m;
    temporalKey = JSON.stringify([cfg.windowSize, cfg.stride, cfg.neighborhood, cfg.context]);
    renderCurves(m, p8, "s8-train-status", () => temporalReport.heuristic.top1);
    metricsTable("s8-metrics", temporalReport, { head: [L("±1 window", "±1 окно")], row: (k) => [pct(temporalReport[k].within1)] });
    $("s8-train").disabled = false;
    renderTemporal();
  };
  requestAnimationFrame(step);
}
$("s8-train").addEventListener("click", trainTemporal);
let s8Started = false;
renderers.s8 = () => {
  renderTemporal();
  if (temporalModel) { renderCurves(temporalModel, p8, "s8-train-status", () => temporalReport.heuristic.top1); metricsTable("s8-metrics", temporalReport, { head: [L("±1 window", "±1 окно")], row: (k) => [pct(temporalReport[k].within1)] }); }
  if (!s8Started) { s8Started = true; trainTemporal(); }
};

// ------------------------------------------------------------ resets
document.querySelectorAll("[data-reset]").forEach((b) => b.addEventListener("click", () => {
  const what = b.dataset.reset;
  if (what === "sig") state.sig = structuredClone(DEFAULTS.sig);
  if (what === "nd") state.nd = structuredClone(DEFAULTS.nd);
  if (what === "evt") { state.evt1 = structuredClone(DEFAULTS.evt1); state.evtN = structuredClone(DEFAULTS.evtN); state.graph = structuredClone(DEFAULTS.graph); }
  clampState(); syncControls(); invalidate();
}));

// ------------------------------------------------------- language & theme
function setLang(next) {
  lang = next;
  root.dataset.lang = next;
  root.lang = next;
  document.title = L("Graph-EVT: from extreme to source", "Graph-EVT: от экстремума к источнику");
  try { localStorage.setItem("evt-lang", next); } catch (e) { /* storage unavailable */ }
  document.querySelectorAll("[data-set-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setLang === next)));
  syncControls();
  invalidate();
}
document.querySelectorAll("[data-set-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.setLang)));
$("theme-toggle").addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("evt-theme", root.dataset.theme); } catch (e) { /* storage unavailable */ }
  invalidate();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => invalidate());

// --------------------------------------------------------------- startup
document.querySelectorAll(".ctl").forEach(buildControl);
setLang(lang);
applyCase();
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const id = e.target.dataset.section;
    if (e.isIntersecting) visible.add(id); else visible.delete(id);
  }
  if (!frame) frame = requestAnimationFrame(flush);
}, { rootMargin: "400px 0px" });
document.querySelectorAll("[data-section]").forEach((s) => io.observe(s));
const tocLinks = [...document.querySelectorAll(".toc a")];
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) tocLinks.forEach((a) => a.classList.toggle("current", a.getAttribute("href") === `#${e.target.id}`));
}, { rootMargin: "-45% 0px -50% 0px" });
document.querySelectorAll(".sec").forEach((s) => spy.observe(s));
invalidate();
