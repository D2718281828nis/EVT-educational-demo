// Browser ports of the repository's numerical code:
//   evt_demo.data_generator  -> signals (additive 1-D, Kuramoto propagation)
//   graph_evt_agent          -> EVT detection, baseline graphs, source ranking,
//                               planner, GNN/GAT, temporal graphs, evaluation.
// The algorithms follow the Python line by line. Only the random generator
// differs (numpy's PCG64 is replaced by a seeded mulberry32), so the curves
// have the notebooks' statistics but not their exact samples.

export const TAU = 2 * Math.PI;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ------------------------------------------------------------ random numbers
export function hashSeed(...parts) {
  let h = 2166136261 >>> 0;
  for (const part of parts) {
    const s = String(part);
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    h ^= 0x9e3779b9; h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export function makeRng(...seedParts) {
  let a = hashSeed(...seedParts);
  let spare = null;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u === 0) u = uniform();
    const r = Math.sqrt(-2 * Math.log(u)), angle = TAU * uniform();
    spare = r * Math.sin(angle);
    return r * Math.cos(angle);
  };
  return {
    uniform, normal,
    int: (lo, hi) => lo + Math.floor(uniform() * (hi - lo + 1)), // inclusive, like integers(lo, hi + 1)
    permutation(n) {
      const order = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(uniform() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      return order;
    },
  };
}

// ------------------------------------------------------------------ statistics
export function quantileSorted(sorted, q) {
  if (!sorted.length) return NaN;
  const position = (sorted.length - 1) * q;
  const lo = Math.floor(position), hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo);
}
export const quantile = (values, q) => quantileSorted(Float64Array.from(values).sort(), q);
export const median = (values) => quantile(values, 0.5);
export function mean(values) {
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}
export function std(values) {
  const m = mean(values);
  let s = 0;
  for (const v of values) s += (v - m) ** 2;
  return Math.sqrt(s / values.length);
}
function pearson(a, b) {
  const n = a.length, ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  const r = sab / Math.sqrt(saa * sbb);
  return Number.isFinite(r) ? r : 0;
}
function linearFit(xs, ys) {
  const mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}
// Least squares via normal equations (the design matrices here are tiny).
function lstsq(rows, cols, design, target) {
  const g = new Float64Array(cols * cols), r = new Float64Array(cols);
  for (let i = 0; i < rows; i++) {
    for (let a = 0; a < cols; a++) {
      const xa = design[i * cols + a];
      r[a] += xa * target[i];
      for (let b = a; b < cols; b++) g[a * cols + b] += xa * design[i * cols + b];
    }
  }
  for (let a = 0; a < cols; a++) {
    g[a * cols + a] += 1e-10;
    for (let b = 0; b < a; b++) g[a * cols + b] = g[b * cols + a];
  }
  for (let c = 0; c < cols; c++) { // Gaussian elimination with partial pivoting
    let pivot = c;
    for (let k = c + 1; k < cols; k++) if (Math.abs(g[k * cols + c]) > Math.abs(g[pivot * cols + c])) pivot = k;
    if (pivot !== c) {
      for (let k = 0; k < cols; k++) [g[c * cols + k], g[pivot * cols + k]] = [g[pivot * cols + k], g[c * cols + k]];
      [r[c], r[pivot]] = [r[pivot], r[c]];
    }
    for (let k = c + 1; k < cols; k++) {
      const f = g[k * cols + c] / g[c * cols + c];
      for (let m = c; m < cols; m++) g[k * cols + m] -= f * g[c * cols + m];
      r[k] -= f * r[c];
    }
  }
  const beta = new Float64Array(cols);
  for (let c = cols - 1; c >= 0; c--) {
    let s = r[c];
    for (let k = c + 1; k < cols; k++) s -= g[c * cols + k] * beta[k];
    beta[c] = s / g[c * cols + c];
  }
  return beta;
}

// [time, channel] data is stored row-major: { n, p, v: Float64Array(n * p) }.
export const column = (data, j, start = 0, stop = data.n) => {
  const out = new Float64Array(stop - start);
  for (let t = start; t < stop; t++) out[t - start] = data.v[t * data.p + j];
  return out;
};
export const slice = (data, start, stop) => ({ n: stop - start, p: data.p, v: data.v.slice(start * data.p, stop * data.p) });
export const fromSeries = (values) => ({ n: values.length, p: 1, v: Float64Array.from(values) });
export const channelName = (j) => `channel_${String(j).padStart(2, "0")}`;

// ------------------------------------------------------------ signal generators
/** Unit-variance 1/f-style noise: amplitude f^-(H - 1/2), so PSD ∝ f^-(2H - 1). */
export function coloredNoise(n, hurst, rng) {
  const m = Math.floor(n / 2) + 1;
  const re = new Float64Array(m), im = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const amplitude = k === 0 ? 0 : (k / n) ** (-(hurst - 0.5));
    re[k] = rng.normal() * amplitude;
    im[k] = rng.normal() * amplitude;
  }
  const cos = new Float64Array(n), sin = new Float64Array(n);
  for (let j = 0; j < n; j++) { cos[j] = Math.cos(TAU * j / n); sin[j] = Math.sin(TAU * j / n); }
  const out = new Float64Array(n);
  const nyquist = n % 2 === 0 ? n / 2 : -1;
  for (let t = 0; t < n; t++) { // irfft
    let s = re[0], index = 0;
    for (let k = 1; k < m; k++) {
      index += t; if (index >= n) index %= n;
      const w = k === nyquist ? 1 : 2;
      s += w * (re[k] * cos[index] - (k === nyquist ? 0 : im[k] * sin[index]));
    }
    out[t] = s / n;
  }
  const mu = mean(out), sd = std(out) || 1;
  for (let t = 0; t < n; t++) out[t] = (out[t] - mu) / sd;
  return out;
}

export function envelope(duration, kind) {
  const out = new Float64Array(duration);
  for (let i = 0; i < duration; i++) {
    const x = duration > 1 ? i / (duration - 1) : 1;
    // 'exp': generate_additive_fractal_series; 'sin2': the Kuramoto propagation burst.
    out[i] = kind === "sin2" ? Math.sin(Math.PI * x) ** 2 : Math.expm1(5 * x) / Math.expm1(5);
  }
  return out;
}

/** generate_additive_fractal_series: periodic + stochastic + Hurst-coloured burst. */
export function additiveSeries(o) {
  const n = o.n;
  const periodic = new Float64Array(n);
  for (let k = 0; k < 3; k++) {
    for (let t = 0; t < n; t++) periodic[t] += o.amplitudes[k] * Math.sin(TAU * t / o.periods[k] + o.phases[k]);
  }
  const rng = makeRng(o.seed, "additive");
  const stochastic = new Float64Array(n);
  for (let t = 0; t < n; t++) stochastic[t] = o.sigma * rng.normal();
  const duration = clamp(Math.round(o.duration ?? 2 * Math.max(5, Math.floor(n / 20))), 5, n);
  const onset = clamp(Math.round(o.onset ?? Math.min(Math.round(2 * n / 3), n - duration)), 0, n - duration);
  const burst = coloredNoise(duration, o.hurst, rng);
  const taper = envelope(duration, o.envelope);
  const fractal = new Float64Array(n);
  for (let i = 0; i < duration; i++) fractal[onset + i] = o.magnitude * burst[i] * taper[i];
  const use = o.use || { periodic: true, stochastic: true, fractal: true };
  const values = new Float64Array(n);
  for (let t = 0; t < n; t++) {
    values[t] = (use.periodic ? periodic[t] : 0) + (use.stochastic ? stochastic[t] : 0) + (use.fractal ? fractal[t] : 0);
  }
  return { values, periodic, stochastic, fractal, onset, duration, n };
}

/** generate_kuramoto_time_series (dθᵢ/dt = ωᵢ + K·meanⱼ sin(θⱼ − θᵢ)), Euler steps. */
export function kuramoto({ n, p, source = 0, seed = 42, coupling = 0.7, dt = 0.02 }) {
  const rng = makeRng(seed, p, "kuramoto");
  const phases = new Float64Array(p), omega = new Float64Array(p);
  for (let j = 0; j < p; j++) phases[j] = -Math.PI + TAU * rng.uniform();
  for (let j = 0; j < p; j++) omega[j] = 1 + 0.12 * rng.normal();
  omega[source] += 0.2;
  const v = new Float64Array(n * p), coherence = new Float64Array(n);
  for (let t = 0; t < n; t++) {
    let c = 0, s = 0;
    for (let j = 0; j < p; j++) {
      const sj = Math.sin(phases[j]), cj = Math.cos(phases[j]);
      v[t * p + j] = sj; s += sj; c += cj;
    }
    c /= p; s /= p;
    coherence[t] = Math.hypot(c, s);
    // meanⱼ sin(θⱼ − θᵢ) = cos θᵢ · mean sin θⱼ − sin θᵢ · mean cos θⱼ
    for (let j = 0; j < p; j++) {
      const interaction = Math.cos(phases[j]) * s - Math.sin(phases[j]) * c;
      phases[j] += dt * (omega[j] + coupling * interaction);
    }
  }
  return { n, p, v, coherence, omega };
}

export const ringHops = (p, source) => Array.from({ length: p }, (_, j) => Math.min(Math.abs(j - source), p - Math.abs(j - source)));

/** generate_kuramoto_propagation_series: one burst at the source, delayed and damped per ring hop. */
export function propagationSeries(o) {
  const base = kuramoto(o);
  const { n, p } = base;
  const v = base.v.slice();
  const eventRng = makeRng(o.seed, p, o.source, "event");
  const duration = clamp(Math.round(o.duration ?? Math.max(5, Math.floor(n / 20))), 5, n);
  const hops = ringHops(p, o.source);
  const delays = hops.map((h) => h * o.delay);
  const maxDelay = Math.max(...delays);
  const low = Math.floor(n / 4), high = Math.max(low, n - duration - maxDelay);
  let onset = eventRng.int(low, high);
  if (o.onset != null) onset = clamp(Math.round(o.onset), 0, Math.max(0, n - duration - maxDelay));
  const burst = coloredNoise(duration, o.hurst, eventRng);
  const taper = envelope(duration, "sin2");
  const injected = new Float64Array(n * p);
  const mask = new Uint8Array(n * p);
  for (let j = 0; j < p; j++) {
    const start = onset + delays[j], gain = o.decay ** hops[j];
    for (let i = 0; i < duration && start + i < n; i++) {
      const add = o.magnitude * burst[i] * taper[i] * gain;
      v[(start + i) * p + j] += add;
      injected[(start + i) * p + j] = add;
      mask[(start + i) * p + j] = 1;
    }
  }
  return { n, p, v, background: base.v, injected, mask, coherence: base.coherence, source: o.source, onset, duration, delays, hops };
}

/** generate_kuramoto_propagation_recording: several events, each from a random source. */
export function propagationRecording(o) {
  const nEvents = o.nEvents ?? 4, baseline = o.baseline ?? 300, gap = o.gap ?? 150;
  const n = baseline + gap * nEvents + Math.floor(gap / 2);
  const base = kuramoto({ ...o, n, source: 0 });
  const p = base.p, v = base.v.slice();
  const rng = makeRng(o.seed, p, nEvents, "recording");
  const duration = Math.max(5, Math.min(Math.floor(gap / 3), Math.floor(n / 20)));
  const eventTimes = Array.from({ length: nEvents }, (_, i) => baseline + Math.floor(gap / 2) + gap * i);
  const sources = eventTimes.map(() => rng.int(0, p - 1));
  const taper = envelope(duration, "sin2");
  eventTimes.forEach((onset, e) => {
    const hops = ringHops(p, sources[e]);
    const burst = coloredNoise(duration, o.hurst, rng);
    for (let j = 0; j < p; j++) {
      const start = onset + hops[j] * o.delay, gain = o.decay ** hops[j];
      for (let i = 0; i < duration && start + i < n; i++) v[(start + i) * p + j] += o.magnitude * burst[i] * taper[i] * gain;
    }
  });
  return { n, p, v, eventTimes, sources, duration };
}

// ---------------------------------------------------- Hurst diagnostics (PSD, DFA)
export function fft(re, im) { // in place, radix-2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -TAU / len, wr = Math.cos(angle), wi = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
      }
    }
  }
}

/**
 * Log-binned periodogram and its log-log slope (expected −(2H − 1)).
 * Bins average log-power, so every bin carries the same log-bias whatever
 * its width, and the fitted slope stays unbiased.
 */
export function spectrum(x) {
  const n = x.length, re = Float64Array.from(x), im = new Float64Array(n);
  fft(re, im);
  const bins = [];
  const edges = [...new Set(Array.from({ length: 25 }, (_, i) => Math.round(2 ** (i * Math.log2(n / 2) / 24))))];
  for (let b = 0; b < edges.length - 1; b++) {
    let s = 0, f = 0, c = 0;
    for (let k = edges[b]; k < edges[b + 1] && k <= n / 2; k++) { s += Math.log10((re[k] ** 2 + im[k] ** 2) / n); f += Math.log10(k / n); c++; }
    if (c) bins.push({ f: 10 ** (f / c), power: 10 ** (s / c) });
  }
  const fit = linearFit(bins.map((b) => Math.log10(b.f)), bins.map((b) => Math.log10(b.power)));
  return { bins, slope: fit.slope, intercept: fit.intercept };
}

/** DFA-1 fluctuation function F(s) and exponent α (≈ H for fractional Gaussian noise). */
export function dfa(x, scales) {
  const n = x.length, mu = mean(x);
  const profile = new Float64Array(n);
  let c = 0;
  for (let t = 0; t < n; t++) { c += x[t] - mu; profile[t] = c; }
  const points = scales.map((s) => ({ s, F: Math.sqrt(meanSquareDetrended(profile, s)) }));
  const fit = linearFit(points.map((q) => Math.log10(q.s)), points.map((q) => Math.log10(q.F)));
  return { points, alpha: fit.slope, intercept: fit.intercept };
}
function detrendSegments(profile, scale) {
  const count = Math.floor(profile.length / scale);
  const out = new Float64Array(count * scale);
  const xm = (scale - 1) / 2;
  let sxx = 0;
  for (let i = 0; i < scale; i++) sxx += (i - xm) ** 2;
  for (let s = 0; s < count; s++) {
    let ym = 0;
    for (let i = 0; i < scale; i++) ym += profile[s * scale + i];
    ym /= scale;
    let sxy = 0;
    for (let i = 0; i < scale; i++) sxy += (i - xm) * (profile[s * scale + i] - ym);
    const slope = sxy / sxx;
    for (let i = 0; i < scale; i++) out[s * scale + i] = profile[s * scale + i] - (ym + slope * (i - xm));
  }
  return out;
}
function meanSquareDetrended(profile, scale) {
  const r = detrendSegments(profile, scale);
  let s = 0;
  for (const v of r) s += v * v;
  return s / r.length;
}

// ------------------------------------------------------------- EVT: POT / GPD
export function gpdPpf(prob, shape, scale) {
  return Math.abs(shape) < 1e-9 ? -scale * Math.log(1 - prob) : scale * ((1 - prob) ** (-shape) - 1) / shape;
}
export function gpdPdf(y, shape, scale) {
  if (y < 0) return 0;
  if (Math.abs(shape) < 1e-9) return Math.exp(-y / scale) / scale;
  const t = 1 + shape * y / scale;
  return t <= 0 ? 0 : t ** (-1 / shape - 1) / scale;
}
function gpdNegLogLik(shape, logScale, excess) {
  const scale = Math.exp(logScale);
  let s = excess.length * logScale;
  for (const y of excess) {
    if (Math.abs(shape) < 1e-9) { s += y / scale; continue; }
    const t = 1 + shape * y / scale;
    if (t <= 0) return Infinity;
    s += (1 + 1 / shape) * Math.log(t);
  }
  return s;
}
function nelderMead(f, start, step = 0.3, iterations = 400) {
  const dim = start.length;
  let simplex = [start.slice()];
  for (let i = 0; i < dim; i++) { const v = start.slice(); v[i] += step; simplex.push(v); }
  let values = simplex.map(f);
  for (let it = 0; it < iterations; it++) {
    const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
    simplex = order.map((i) => simplex[i]); values = order.map((i) => values[i]);
    if (Math.abs(values[dim] - values[0]) < 1e-10) break;
    const centroid = new Array(dim).fill(0);
    for (let i = 0; i < dim; i++) for (let d = 0; d < dim; d++) centroid[d] += simplex[i][d] / dim;
    const towards = (coef) => centroid.map((c, d) => c + coef * (simplex[dim][d] - c));
    const reflected = towards(-1), fr = f(reflected);
    if (fr < values[0]) {
      const expanded = towards(-2), fe = f(expanded);
      [simplex[dim], values[dim]] = fe < fr ? [expanded, fe] : [reflected, fr];
    } else if (fr < values[dim - 1]) {
      [simplex[dim], values[dim]] = [reflected, fr];
    } else {
      const contracted = towards(0.5), fc = f(contracted);
      if (fc < values[dim]) { [simplex[dim], values[dim]] = [contracted, fc]; } else {
        for (let i = 1; i <= dim; i++) {
          simplex[i] = simplex[i].map((v, d) => simplex[0][d] + 0.5 * (v - simplex[0][d]));
          values[i] = f(simplex[i]);
        }
      }
    }
  }
  const best = values.indexOf(Math.min(...values));
  return simplex[best];
}
/** Maximum-likelihood GPD fit with location 0 (scipy genpareto.fit(excess, floc=0)). */
export function gpdFit(excess) {
  const m = mean(excess), v = std(excess) ** 2 || m * m;
  const shape0 = clamp(0.5 * (1 - (m * m) / v), -0.45, 0.9);
  const scale0 = Math.max(1e-6, 0.5 * m * ((m * m) / v + 1));
  const [shape, logScale] = nelderMead(([s, ls]) => (s <= -1 || s > 5 ? Infinity : gpdNegLogLik(s, ls, excess)), [shape0, Math.log(scale0)]);
  return { shape, scale: Math.exp(logScale) };
}

export function validateEVT(c, n) {
  const baseline = clamp(Math.round(c.baseline), 20, n - 2);
  const tailQuantile = clamp(c.tailQuantile, 0.51, 0.995);
  const alarmProbability = clamp(Math.max(c.alarmProbability, tailQuantile + 0.001), 0.52, 0.9999);
  return { baseline, tailQuantile, alarmProbability, topK: Math.max(1, c.topK | 0), persistence: Math.max(1, c.persistence | 0), epsilon: 1e-9 };
}

/** graph_evt_agent.evt.detect: robust z → top-k indicator → POT/GPD alarm → first persistent exceedance. */
export function detect(data, config) {
  const c = validateEVT(config, data.n);
  const { n, p, v } = data, B = c.baseline;
  const location = new Float64Array(p), scale = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    const base = column(data, j, 0, B);
    location[j] = median(base);
    scale[j] = Math.max(1.4826 * median(base.map((x) => Math.abs(x - location[j]))), c.epsilon);
  }
  const z = new Float64Array(n * p), indicator = new Float64Array(n);
  const k = Math.min(c.topK, p), row = new Float64Array(p);
  for (let t = 0; t < n; t++) {
    for (let j = 0; j < p; j++) { row[j] = Math.abs(v[t * p + j] - location[j]) / scale[j]; z[t * p + j] = row[j]; }
    row.sort();
    let s = 0;
    for (let i = p - k; i < p; i++) s += row[i];
    indicator[t] = s / k;
  }
  const baseIndicator = indicator.slice(0, B);
  const sorted = Float64Array.from(baseIndicator).sort();
  const u = quantileSorted(sorted, c.tailQuantile);
  const excess = Array.from(baseIndicator).filter((x) => x > u).map((x) => x - u);
  let threshold, gpd = null;
  if (excess.length >= 5 && excess.some((x) => x > 0)) {
    gpd = gpdFit(excess);
    const conditional = (c.alarmProbability - c.tailQuantile) / (1 - c.tailQuantile);
    threshold = u + gpdPpf(conditional, gpd.shape, gpd.scale);
    gpd.conditional = conditional;
  } else {
    threshold = quantileSorted(sorted, c.alarmProbability);
  }
  let found = null;
  for (let start = B; start <= n - c.persistence && found === null; start++) {
    let ok = true;
    for (let i = 0; i < c.persistence && ok; i++) ok = indicator[start + i] > threshold;
    if (ok) found = start;
  }
  return { detected: found !== null, timeIndex: found, alarmThreshold: threshold, u, excess, gpd, indicator, location, scale, z, config: c };
}

/** graph_evt_agent.evt.find_events: every persistent alarm, declustered by a refractory period. */
export function findEvents(detection, refractory = 50) {
  const c = detection.config, ind = detection.indicator, thr = detection.alarmThreshold;
  const hits = (t) => ind[t] > thr;
  const onsets = [];
  let start = c.baseline;
  const last = ind.length - c.persistence;
  while (start <= last) {
    let ok = true;
    for (let i = 0; i < c.persistence && ok; i++) ok = hits(start + i);
    if (ok) {
      onsets.push(start);
      start += Math.max(refractory, c.persistence);
      while (start <= last && hits(start)) start++;
    } else start++;
  }
  return onsets;
}

// -------------------------------------------------- baseline graph hypotheses
function adjacencyFrom(dependence, p, threshold) {
  const adj = new Float64Array(p * p);
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < p; b++) if (dependence[a * p + b] >= threshold || dependence[b * p + a] >= threshold) adj[a * p + b] = 1;
    adj[a * p + a] = 1;
  }
  return adj;
}
function absCorrMatrix(columns) {
  const p = columns.length, out = new Float64Array(p * p);
  for (let a = 0; a < p; a++) {
    out[a * p + a] = 1;
    for (let b = a + 1; b < p; b++) out[a * p + b] = out[b * p + a] = Math.abs(pearson(columns[a], columns[b]));
  }
  return out;
}
export function arResiduals(series, order) {
  const rows = series.length - order, cols = order + 1;
  const design = new Float64Array(rows * cols), target = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    design[r * cols] = 1;
    for (let lag = 0; lag < order; lag++) design[r * cols + lag + 1] = series[order + r - lag - 1];
    target[r] = series[order + r];
  }
  const beta = lstsq(rows, cols, design, target);
  const residual = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let fit = 0;
    for (let c = 0; c < cols; c++) fit += design[r * cols + c] * beta[c];
    residual[r] = target[r] - fit;
  }
  return { residual, beta };
}

/** innovation_graph: |corr| between separate-channel AR(order) innovations. */
export function innovationGraph(baseline, cfg) {
  const p = baseline.p;
  const residuals = Array.from({ length: p }, (_, j) => arResiduals(column(baseline, j), cfg.arOrder).residual);
  const dependence = absCorrMatrix(residuals);
  return { adjacency: adjacencyFrom(dependence, p, cfg.edgeThreshold), dependence, method: "ar_innovation_correlation", p };
}

/** dfa_graph: ½·mean detrended cross-correlation + ½·exp(−|αᵢ − αⱼ|). */
export function dfaGraph(baseline, cfg) {
  const p = baseline.p, scales = [...new Set(cfg.dfaScales)].sort((a, b) => a - b);
  const profiles = Array.from({ length: p }, (_, j) => {
    const x = column(baseline, j), mu = mean(x), out = new Float64Array(x.length);
    let c = 0;
    for (let t = 0; t < x.length; t++) { c += x[t] - mu; out[t] = c; }
    return out;
  });
  const crossMean = new Float64Array(p * p);
  const fluct = scales.map(() => new Float64Array(p));
  scales.forEach((scale, si) => {
    const flat = profiles.map((pr) => detrendSegments(pr, scale));
    flat.forEach((r, j) => { let s = 0; for (const v of r) s += v * v; fluct[si][j] = Math.sqrt(s / r.length); });
    const cross = absCorrMatrix(flat);
    for (let i = 0; i < p * p; i++) crossMean[i] += cross[i] / scales.length;
  });
  const logS = scales.map(Math.log);
  const alpha = Array.from({ length: p }, (_, j) => linearFit(logS, scales.map((_, si) => Math.log(Math.max(fluct[si][j], 1e-12)))).slope);
  const dependence = new Float64Array(p * p);
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) dependence[a * p + b] = 0.5 * crossMean[a * p + b] + 0.5 * Math.exp(-Math.abs(alpha[a] - alpha[b]));
  return { adjacency: adjacencyFrom(dependence, p, cfg.edgeThreshold), dependence, method: "cross_dfa", p, alpha };
}

function morletPhases(baseline, periods) {
  const { n, p } = baseline;
  return periods.map((period) => {
    const radius = Math.min(Math.ceil(4 * period), Math.floor((n - 1) / 2));
    const L = 2 * radius + 1, kr = new Float64Array(L), ki = new Float64Array(L);
    let norm = 0;
    for (let k = 0; k < L; k++) {
      const t = k - radius, g = Math.exp(-(t * t) / (2 * period * period));
      kr[k] = Math.cos(TAU * t / period) * g; ki[k] = Math.sin(TAU * t / period) * g;
      norm += kr[k] ** 2 + ki[k] ** 2;
    }
    norm = Math.sqrt(norm);
    const cos = new Float64Array(n * p), sin = new Float64Array(n * p);
    for (let j = 0; j < p; j++) {
      for (let i = 0; i < n; i++) { // np.convolve(x, kernel, mode="same")
        let sr = 0, si = 0;
        const lo = Math.max(0, i + radius - L + 1), hi = Math.min(n - 1, i + radius);
        for (let m = lo; m <= hi; m++) {
          const x = baseline.v[m * p + j], k = i + radius - m;
          sr += x * kr[k]; si += x * ki[k];
        }
        const phase = Math.atan2(si / norm, sr / norm);
        cos[i * p + j] = Math.cos(phase); sin[i * p + j] = Math.sin(phase);
      }
    }
    return { cos, sin };
  });
}
function phaseLocking(bands, n, p, shifts = null) {
  const out = new Float64Array(p * p);
  for (let a = 0; a < p; a++) {
    out[a * p + a] = 1;
    for (let b = a + 1; b < p; b++) {
      let re = 0, im = 0;
      const sa = shifts ? shifts[a] : 0, sb = shifts ? shifts[b] : 0;
      for (const { cos, sin } of bands) {
        for (let t = 0; t < n; t++) {
          const ia = ((t - sa) % n + n) % n, ib = ((t - sb) % n + n) % n;
          const ca = cos[ia * p + a], sa_ = sin[ia * p + a], cb = cos[ib * p + b], sb_ = sin[ib * p + b];
          re += ca * cb + sa_ * sb_; im += sa_ * cb - ca * sb_;
        }
      }
      out[a * p + b] = out[b * p + a] = Math.hypot(re, im) / (bands.length * n);
    }
  }
  return out;
}

// Phase locking and its surrogates do not depend on edge_threshold, so a
// slider over the threshold reuses them instead of recomputing ~10⁷ terms.
const phaseCache = new Map();
function phaseLockingWithSurrogates(baseline, cfg) {
  const { n, p, v } = baseline;
  let fingerprint = 0;
  for (let i = 0; i < v.length; i += 7) fingerprint = (fingerprint * 31 + v[i] * 1e6) % 1e12;
  const key = [n, p, fingerprint, cfg.waveletPeriods, cfg.surrogateCount, cfg.surrogateQuantile, cfg.randomState].join("|");
  if (phaseCache.has(key)) return phaseCache.get(key);
  const bands = morletPhases(baseline, cfg.waveletPeriods);
  const dependence = phaseLocking(bands, n, p);
  const rng = makeRng(cfg.randomState, "surrogates");
  const minShift = Math.max(1, Math.floor(n / 10));
  const samples = Array.from({ length: p * p }, () => new Float64Array(cfg.surrogateCount));
  for (let s = 0; s < cfg.surrogateCount; s++) {
    const shifts = Array.from({ length: p }, () => rng.int(minShift, Math.max(minShift + 1, n - minShift) - 1));
    const plv = phaseLocking(bands, n, p, shifts);
    for (let i = 0; i < p * p; i++) samples[i][s] = plv[i];
  }
  const surrogateThreshold = samples.map((s) => quantile(s, cfg.surrogateQuantile));
  const result = { bands, dependence, surrogateThreshold };
  if (phaseCache.size > 6) phaseCache.delete(phaseCache.keys().next().value);
  phaseCache.set(key, result);
  return result;
}

/** kuramoto_wavelet_graph: Morlet phase locking, kept only above circular-shift surrogates. */
export function kuramotoWaveletGraph(baseline, cfg) {
  const { n, p } = baseline;
  const { bands, dependence, surrogateThreshold } = phaseLockingWithSurrogates(baseline, cfg);
  const adjacency = new Float64Array(p * p);
  for (let a = 0; a < p; a++) {
    for (let b = 0; b < p; b++) {
      const keep = (x, y) => dependence[x * p + y] >= cfg.edgeThreshold && dependence[x * p + y] > surrogateThreshold[x * p + y];
      if (a === b || keep(a, b) || keep(b, a)) adjacency[a * p + b] = 1;
    }
  }
  let order = 0;
  for (const { cos, sin } of bands) for (let t = 0; t < n; t++) {
    let c = 0, s = 0;
    for (let j = 0; j < p; j++) { c += cos[t * p + j]; s += sin[t * p + j]; }
    order += Math.hypot(c, s) / p / (n * bands.length);
  }
  return { adjacency, dependence, method: "kuramoto_wavelet_phase_locking", p, surrogateThreshold, orderParameter: order };
}

/** _approximately_stationary: 4 chunks, mean range < std and variance ratio < 4 on every channel. */
export function approximatelyStationary(data) {
  const { n, p } = data;
  const sizes = Array.from({ length: 4 }, (_, i) => Math.floor(n / 4) + (i < n % 4 ? 1 : 0));
  for (let j = 0; j < p; j++) {
    const x = column(data, j);
    const means = [], vars = [];
    let start = 0;
    for (const size of sizes) { const chunk = x.subarray(start, start + size); means.push(mean(chunk)); vars.push(std(chunk) ** 2); start += size; }
    const full = Math.max(std(x), 1e-12);
    const locationStable = Math.max(...means) - Math.min(...means) < full;
    const ratio = Math.max(...vars) / Math.max(Math.min(...vars), 1e-12);
    if (!(locationStable && ratio < 4)) return false;
  }
  return true;
}

export const GRAPH_DEFAULTS = {
  method: "ar", arOrder: 2, edgeThreshold: 0.35, dfaScales: [8, 16, 32, 64],
  waveletPeriods: [8, 16, 32], surrogateCount: 99, surrogateQuantile: 0.95, randomState: 0,
};

/** build_graph: the configured hypothesis, or the DFA ∩ Kuramoto ensemble. */
export function buildGraph(baseline, config) {
  const cfg = { ...GRAPH_DEFAULTS, ...config };
  let method = cfg.method;
  if (method === "auto") method = approximatelyStationary(baseline) ? "ar" : "ensemble";
  if (method === "ar") return innovationGraph(baseline, cfg);
  if (method === "dfa") return dfaGraph(baseline, cfg);
  if (method === "kuramoto") return kuramotoWaveletGraph(baseline, cfg);
  const d = dfaGraph(baseline, cfg), k = kuramotoWaveletGraph(baseline, cfg), p = baseline.p;
  const adjacency = new Float64Array(p * p), dependence = new Float64Array(p * p);
  for (let i = 0; i < p * p; i++) { adjacency[i] = d.adjacency[i] && k.adjacency[i] ? 1 : 0; dependence[i] = Math.sqrt(d.dependence[i] * k.dependence[i]); }
  return { adjacency, dependence, method: "dfa_kuramoto_intersection", p, parts: { dfa: d, kuramoto: k } };
}
export const edgeCount = (g) => {
  let e = 0;
  for (let a = 0; a < g.p; a++) for (let b = a + 1; b < g.p; b++) if (g.adjacency[a * g.p + b]) e++;
  return e;
};

// ---------------------------------------------------------- source ranking
export const FEATURE_NAMES = ["peak", "energy", "latency", "degree", "neighbor"];

/** heuristic_probabilities: softmax(peak + √energy + 0.15·neighbor − 0.5·latency). */
export function heuristicProbabilities(features, F) {
  const N = features.length / F, raw = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const f = features.subarray(i * F, i * F + 5);
    raw[i] = f[0] + Math.sqrt(f[1]) + 0.15 * f[4] - 0.5 * f[2];
  }
  return softmax(raw);
}

/** rank_sources: early, strong responses in the 12 samples after the alarm, plus neighbor support. */
export function rankSources(data, eventTime, adjacency, location, scale, window = 12) {
  const { n, p, v } = data, stop = Math.min(n, eventTime + window);
  const features = new Float64Array(p * 5);
  const peak = new Float64Array(p);
  for (let j = 0; j < p; j++) {
    let pk = 0, energy = 0;
    const z = [];
    for (let t = eventTime; t < stop; t++) { const zz = Math.abs(v[t * p + j] - location[j]) / scale[j]; z.push(zz); pk = Math.max(pk, zz); energy += zz * zz; }
    const cutoff = Math.max(3, 0.5 * pk);
    const latency = Math.max(0, z.findIndex((zz) => zz >= cutoff)); // np.argmax(all False) = 0
    peak[j] = pk;
    features.set([pk, energy / z.length, latency], j * 5);
  }
  for (let j = 0; j < p; j++) {
    let deg = 0, nb = 0;
    for (let k = 0; k < p; k++) { deg += adjacency[j * p + k]; nb += adjacency[j * p + k] * peak[k]; }
    features[j * 5 + 3] = deg - 1;
    features[j * 5 + 4] = nb / Math.max(deg, 1);
  }
  const probabilities = heuristicProbabilities(features, 5);
  const order = Array.from(probabilities.keys()).sort((a, b) => probabilities[b] - probabilities[a]);
  return { order, probabilities, features, source: order[0] };
}

// ------------------------------------------------------- planner + pipeline
export const ACTIONS = ["inspect_input", "check_stationarity", "detect_evt", "build_baseline_graph", "rank_source", "infer_gnn_gat_process"];

/** GraphEVTPipeline.run_detailed with the PipelinePlanner's plan/act/observe trace. */
export function runPipeline(data, { evt, graph, processModel = null, names = null }) {
  const n = data.n, p = data.p;
  names = names || Array.from({ length: p }, (_, j) => (p === 1 ? "value" : channelName(j)));
  const c = validateEVT(evt, n);
  const route = p === 1 ? "1-d" : "n-d";
  const stationary = approximatelyStationary(slice(data, 0, c.baseline));
  const initialActions = ["inspect_input", "check_stationarity", "detect_evt"];
  if (route === "n-d") {
    initialActions.push("build_baseline_graph", "rank_source");
    if (processModel) initialActions.push("infer_gnn_gat_process");
  }
  const profile = { kind: p === 1 ? "univariate" : "multivariate", n_time: n, n_channels: p, channel_names: names, missing: 0, regular_time: true };
  const trace = [
    { action: "inspect_input", message: `input ${profile.kind} (${n}, ${p})` },
    { action: "check_stationarity", message: stationary ? "stationary baseline" : "non-stationary baseline" },
  ];
  const detection = detect(data, c);
  trace.push({ action: "detect_evt", message: detection.detected ? `event at ${detection.timeIndex}` : "no event" });
  const completed = ["inspect_input", "check_stationarity", "detect_evt"];
  const plan = { route, baseline_stationary: stationary, initial_actions: initialActions, completed_actions: completed, stop_reason: null };
  if (p === 1 || !detection.detected) {
    plan.stop_reason = detection.detected ? "single_channel_cannot_localize" : "no_evt_detected";
    trace.push({ action: "stop", message: `stop: ${plan.stop_reason}` });
    return { profile, plan, detection, graph: null, ranking: null, process: null, trace, names };
  }
  const g = buildGraph(slice(data, 0, c.baseline), graph);
  trace.push({ action: "build_baseline_graph", message: `${g.method}: ${edgeCount(g)} edges` });
  const ranking = rankSources(data, detection.timeIndex, g.adjacency, detection.location, detection.scale);
  trace.push({ action: "rank_source", message: `source ${names[ranking.source]}` });
  completed.push("build_baseline_graph", "rank_source");
  let process = null;
  if (processModel) {
    process = processModel.predict(episodeFromDense(ranking.features, 5, g.adjacency, null));
    trace.push({ action: "infer_gnn_gat_process", message: `GNN ${names[process.gnnSource]} / GAT ${names[process.gatSource]}` });
    completed.push("infer_gnn_gat_process");
  }
  return { profile, plan, detection, graph: g, ranking, process, trace, names };
}

// ------------------------------------------------------------ GNN / GAT
export function softmax(values) {
  let mx = -Infinity;
  for (const v of values) mx = Math.max(mx, v);
  const out = new Float64Array(values.length);
  let s = 0;
  for (let i = 0; i < values.length; i++) { out[i] = Math.exp(values[i] - mx); s += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= s;
  return out;
}

/** A GraphEpisode as neighbor lists (self-loop weight ≥ 1, like learning._validate). */
export function episodeFromDense(features, F, adjacency, source, group = null) {
  const N = features.length / F, nbr = [];
  for (let i = 0; i < N; i++) {
    const idx = [], w = [];
    for (let j = 0; j < N; j++) {
      let a = 0.5 * (adjacency[i * N + j] + adjacency[j * N + i]);
      if (i === j) a = Math.max(a, 1);
      if (a > 0) { idx.push(j); w.push(a); }
    }
    nbr.push({ idx: Int32Array.from(idx), w: Float64Array.from(w) });
  }
  return { x: Float64Array.from(features), N, F, nbr, source, group };
}

export const LEARNING_DEFAULTS = { hiddenDim: 8, epochs: 200, learningRate: 0.02, l2: 1e-4, randomState: 0 };

/**
 * GraphProcessModel: a GNN on [x, Âx, Â²x] and a GAT whose attention
 * αᵢⱼ = softmaxⱼ(LeakyReLU(a₁·xᵢ + a₂·xⱼ) + log wᵢⱼ) aggregates neighbors.
 * Both end in tanh(·W + b)·o → softmax over nodes; trained by cross-entropy with Adam.
 */
export class GraphProcessModel {
  constructor(config = {}) { this.config = { ...LEARNING_DEFAULTS, ...config }; this.history = null; }

  begin(train, validation = []) {
    if (!train.length) throw new Error("at least one labelled episode is required");
    const F = train[0].F, H = this.config.hiddenDim;
    this.F = F;
    let count = 0;
    const m = new Float64Array(F), s = new Float64Array(F);
    for (const ep of train) for (let i = 0; i < ep.N; i++) { count++; for (let f = 0; f < F; f++) m[f] += ep.x[i * F + f]; }
    for (let f = 0; f < F; f++) m[f] /= count;
    for (const ep of train) for (let i = 0; i < ep.N; i++) for (let f = 0; f < F; f++) s[f] += (ep.x[i * F + f] - m[f]) ** 2;
    for (let f = 0; f < F; f++) s[f] = Math.max(Math.sqrt(s[f] / count), 1e-8);
    this.mean = m; this.scale = s;
    this.train = train.map((ep) => this.prepare(ep));
    this.validation = validation.map((ep) => this.prepare(ep));
    const rng = makeRng(this.config.randomState, "weights");
    const init = (inDim) => [
      Float64Array.from({ length: inDim * H }, () => 0.15 * rng.normal()), new Float64Array(H),
      Float64Array.from({ length: H }, () => 0.15 * rng.normal()), new Float64Array(1),
    ];
    this.gnn = init(3 * F);
    this.gat = init(2 * F);
    this.gat.push(Float64Array.from({ length: F * 2 }, () => 0.1 * rng.normal()));
    this.optimizer = {
      gnn: [this.gnn.map((a) => new Float64Array(a.length)), this.gnn.map((a) => new Float64Array(a.length))],
      gat: [this.gat.map((a) => new Float64Array(a.length)), this.gat.map((a) => new Float64Array(a.length))],
    };
    this.step = 0;
    this.history = { epoch: [], gnn_train_loss: [], gnn_train_accuracy: [], gnn_val_loss: [], gnn_val_accuracy: [], gat_train_loss: [], gat_train_accuracy: [], gat_val_loss: [], gat_val_accuracy: [] };
    return this;
  }

  prepare(ep) {
    const F = this.F, x = new Float64Array(ep.N * F);
    for (let i = 0; i < ep.N; i++) for (let f = 0; f < F; f++) x[i * F + f] = (ep.x[i * F + f] - this.mean[f]) / this.scale[f];
    // GNN input [x, Tx, T²x] with T the row-normalized weights; fixed for an episode.
    const hop = (src) => {
      const out = new Float64Array(ep.N * F);
      for (let i = 0; i < ep.N; i++) {
        const { idx, w } = ep.nbr[i];
        let total = 0;
        for (let k = 0; k < idx.length; k++) total += w[k];
        for (let k = 0; k < idx.length; k++) for (let f = 0; f < F; f++) out[i * F + f] += (w[k] / total) * src[idx[k] * F + f];
      }
      return out;
    };
    const one = hop(x), two = hop(one);
    const gnnInput = new Float64Array(ep.N * 3 * F);
    for (let i = 0; i < ep.N; i++) {
      gnnInput.set(x.subarray(i * F, (i + 1) * F), i * 3 * F);
      gnnInput.set(one.subarray(i * F, (i + 1) * F), i * 3 * F + F);
      gnnInput.set(two.subarray(i * F, (i + 1) * F), i * 3 * F + 2 * F);
    }
    const logW = ep.nbr.map(({ w }) => w.map((v) => Math.log(v)));
    return { ...ep, x, gnnInput, logW };
  }

  static dense(inputs, N, D, params) {
    const [W, b, o, ob] = params, H = b.length;
    const hidden = new Float64Array(N * H), logits = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let out = ob[0];
      for (let h = 0; h < H; h++) {
        let s = b[h];
        for (let d = 0; d < D; d++) s += inputs[i * D + d] * W[d * H + h];
        const a = Math.tanh(s);
        hidden[i * H + h] = a; out += a * o[h];
      }
      logits[i] = out;
    }
    return { hidden, logits };
  }

  attention(ep, A) {
    const F = this.F, N = ep.N;
    const s = new Float64Array(N), t = new Float64Array(N);
    for (let i = 0; i < N; i++) for (let f = 0; f < F; f++) { s[i] += ep.x[i * F + f] * A[f * 2]; t[i] += ep.x[i * F + f] * A[f * 2 + 1]; }
    const alpha = [], raw = [];
    for (let i = 0; i < N; i++) {
      const { idx } = ep.nbr[i];
      const r = new Float64Array(idx.length), score = new Float64Array(idx.length);
      for (let k = 0; k < idx.length; k++) {
        r[k] = s[i] + t[idx[k]];
        score[k] = (r[k] >= 0 ? r[k] : 0.2 * r[k]) + ep.logW[i][k];
      }
      raw.push(r); alpha.push(softmax(score));
    }
    return { alpha, raw };
  }

  gatInputs(ep, alpha) {
    const F = this.F, N = ep.N, inputs = new Float64Array(N * 2 * F);
    for (let i = 0; i < N; i++) {
      inputs.set(ep.x.subarray(i * F, (i + 1) * F), i * 2 * F);
      const { idx } = ep.nbr[i];
      for (let k = 0; k < idx.length; k++) for (let f = 0; f < F; f++) inputs[i * 2 * F + F + f] += alpha[i][k] * ep.x[idx[k] * F + f];
    }
    return inputs;
  }

  logits(ep, isGat) {
    if (!isGat) return GraphProcessModel.dense(ep.gnnInput, ep.N, 3 * this.F, this.gnn).logits;
    const { alpha } = this.attention(ep, this.gat[4]);
    return GraphProcessModel.dense(this.gatInputs(ep, alpha), ep.N, 2 * this.F, this.gat).logits;
  }

  score(episodes, isGat) {
    if (!episodes.length) return [NaN, NaN];
    let loss = 0, hits = 0;
    for (const ep of episodes) {
      const prob = softmax(this.logits(ep, isGat));
      loss += -Math.log(Math.max(prob[ep.source], 1e-12));
      let best = 0;
      for (let i = 1; i < prob.length; i++) if (prob[i] > prob[best]) best = i;
      hits += best === ep.source ? 1 : 0;
    }
    return [loss / episodes.length, hits / episodes.length];
  }

  epochOf(params, isGat, moments, variances) {
    const F = this.F, D = isGat ? 2 * F : 3 * F, H = this.config.hiddenDim, l2 = this.config.l2;
    const grads = params.map((a) => new Float64Array(a.length));
    const [W, , o] = params, scale = 1 / this.train.length;
    for (const ep of this.train) {
      const N = ep.N;
      let inputs, att = null;
      if (isGat) { att = this.attention(ep, params[4]); inputs = this.gatInputs(ep, att.alpha); } else inputs = ep.gnnInput;
      const { hidden, logits } = GraphProcessModel.dense(inputs, N, D, params);
      const delta = softmax(logits);
      delta[ep.source] -= 1;
      const hd = new Float64Array(N * H);
      for (let i = 0; i < N; i++) {
        for (let h = 0; h < H; h++) {
          const a = hidden[i * H + h], g = delta[i] * o[h] * (1 - a * a);
          hd[i * H + h] = g;
          grads[1][h] += g * scale;
          grads[2][h] += a * delta[i] * scale;
          for (let d = 0; d < D; d++) grads[0][d * H + h] += inputs[i * D + d] * g * scale;
        }
        grads[3][0] += delta[i] * scale;
      }
      if (isGat) { // gradient into the attention vectors (input-side features stay fixed)
        const rowSum = new Float64Array(N), colSum = new Float64Array(N);
        const nd = new Float64Array(F);
        for (let i = 0; i < N; i++) {
          nd.fill(0);
          for (let f = 0; f < F; f++) for (let h = 0; h < H; h++) nd[f] += hd[i * H + h] * W[(F + f) * H + h];
          const { idx } = ep.nbr[i], alpha = att.alpha[i], raw = att.raw[i];
          const ad = new Float64Array(idx.length);
          let weighted = 0;
          for (let k = 0; k < idx.length; k++) {
            let s = 0;
            for (let f = 0; f < F; f++) s += nd[f] * ep.x[idx[k] * F + f];
            ad[k] = s; weighted += s * alpha[k];
          }
          for (let k = 0; k < idx.length; k++) {
            const rd = alpha[k] * (ad[k] - weighted) * (raw[k] >= 0 ? 1 : 0.2);
            rowSum[i] += rd; colSum[idx[k]] += rd;
          }
        }
        for (let i = 0; i < N; i++) for (let f = 0; f < F; f++) {
          grads[4][f * 2] += ep.x[i * F + f] * rowSum[i] * scale;
          grads[4][f * 2 + 1] += ep.x[i * F + f] * colSum[i] * scale;
        }
      }
    }
    for (const [k, param] of [[0, params[0]], [2, params[2]], ...(isGat ? [[4, params[4]]] : [])]) for (let i = 0; i < param.length; i++) grads[k][i] += l2 * param[i];
    const step = this.step, rate = this.config.learningRate;
    params.forEach((param, k) => {
      const m = moments[k], v = variances[k], g = grads[k];
      for (let i = 0; i < param.length; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
        param[i] -= rate * (m[i] / (1 - 0.9 ** step)) / (Math.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8);
      }
    });
  }

  /** One epoch for both models; returns false once config.epochs are done. */
  epoch(recordEvery = 1) {
    if (this.step >= this.config.epochs) return false;
    this.step += 1;
    this.epochOf(this.gnn, false, ...this.optimizer.gnn);
    this.epochOf(this.gat, true, ...this.optimizer.gat);
    if (this.step % recordEvery === 0 || this.step === this.config.epochs || this.step === 1) {
      const h = this.history;
      h.epoch.push(this.step);
      for (const [name, isGat] of [["gnn", false], ["gat", true]]) {
        const [tl, ta] = this.score(this.train, isGat), [vl, va] = this.score(this.validation, isGat);
        h[`${name}_train_loss`].push(tl); h[`${name}_train_accuracy`].push(ta);
        h[`${name}_val_loss`].push(vl); h[`${name}_val_accuracy`].push(va);
      }
    }
    return this.step < this.config.epochs;
  }

  fit(train, validation = []) {
    this.begin(train, validation);
    while (this.epoch()) { /* run all epochs */ }
    return this;
  }

  predict(ep) {
    const prepared = this.prepare(ep);
    const gnn = softmax(this.logits(prepared, false)), gat = softmax(this.logits(prepared, true));
    const { alpha } = this.attention(prepared, this.gat[4]);
    const argmax = (a) => a.reduce((best, v, i) => (v > a[best] ? i : best), 0);
    return { gnnProbabilities: gnn, gatProbabilities: gat, attention: alpha, nbr: ep.nbr, gnnSource: argmax(gnn), gatSource: argmax(gat) };
  }
}

/** ranking_metrics: Top-1, Top-k, MRR, NLL of the true source. */
export function rankingMetrics(probabilityList, sources, k = 3) {
  if (!sources.length) return { n: 0, top1: NaN, topk: NaN, mrr: NaN, nll: NaN };
  let top1 = 0, topk = 0, mrr = 0, nll = 0;
  probabilityList.forEach((w, e) => {
    const s = sources[e];
    let rank = 1;
    for (const v of w) if (v > w[s]) rank++;
    top1 += rank === 1; topk += rank <= k; mrr += 1 / rank; nll += -Math.log(Math.max(w[s], 1e-12));
  });
  const n = sources.length;
  return { n, top1: top1 / n, topk: topk / n, mrr: mrr / n, nll: nll / n };
}

/** split_episodes: hold out whole groups (recordings) for validation. */
export function splitEpisodes(episodes, validationFraction = 0.25, randomState = 0) {
  const groups = [...new Set(episodes.map((e, i) => e.group ?? String(i)))].sort();
  const rng = makeRng(randomState, "split");
  let held = Math.round(validationFraction * groups.length);
  if (validationFraction > 0) held = Math.min(Math.max(held, 1), groups.length - 1);
  const order = rng.permutation(groups.length);
  const validationGroups = new Set(order.slice(0, held).map((i) => groups[i]));
  const train = [], validation = [];
  episodes.forEach((e, i) => (validationGroups.has(e.group ?? String(i)) ? validation : train).push(e));
  return { train, validation, unit: "group" };
}

/** EpisodeLabeler.label_series with known (manual) sources; abstains on weak pseudo-labels. */
export function labelRecording(rec, evtConfig, graphConfig, id, { refractory = 50, matchTolerance = 25, minConfidence = 0.5, minMargin = 0.2 } = {}) {
  const detection = detect(rec, evtConfig);
  const onsets = findEvents(detection, refractory);
  if (!onsets.length) return [];
  const graph = buildGraph(slice(rec, 0, detection.config.baseline), graphConfig);
  const episodes = [];
  for (const onset of onsets) {
    const ranking = rankSources(rec, onset, graph.adjacency, detection.location, detection.scale);
    const sorted = Array.from(ranking.probabilities).sort((a, b) => b - a);
    const match = rec.eventTimes.findIndex((t) => Math.abs(t - onset) <= matchTolerance);
    let source = null, origin = "abstained";
    if (match >= 0) { source = rec.sources[match]; origin = "manual"; } else if (sorted[0] >= minConfidence && sorted[0] - sorted[1] >= minMargin) { source = ranking.source; origin = "pseudo"; }
    episodes.push({ onset, source, origin, features: ranking.features, adjacency: graph.adjacency, group: id });
  }
  return episodes;
}

// ------------------------------------------------- temporal graph (1-D, time as nodes)
/** build_temporal_graph with the neighborhood topology and optional AR onset-context columns. */
export function temporalGraph(values, cfg) {
  const n = values.length, w = cfg.windowSize, B = Math.min(cfg.baseline, n - 1);
  const base = values.subarray(0, B);
  const location = median(base);
  const scale = Math.max(1.4826 * median(base.map((x) => Math.abs(x - location))), 1e-8);
  const starts = [];
  for (let s = 0; s < n; s += cfg.stride) starts.push(s);
  const N = starts.length;
  const extra = cfg.context ? 2 * cfg.horizons.length : 0, F = 5 + extra;
  const features = new Float64Array(N * F), peak = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let pk = 0, energy = 0, latency = 0;
    const stop = Math.min(starts[i] + w, n);
    for (let t = starts[i]; t < stop; t++) {
      const z = Math.abs(values[t] - location) / scale;
      if (z > pk) { pk = z; latency = t - starts[i]; }
      energy += z * z;
    }
    peak[i] = pk;
    features.set([pk, energy / (stop - starts[i]), latency], i * F);
  }
  const nbr = [];
  for (let i = 0; i < N; i++) {
    const idx = [];
    for (let j = Math.max(0, i - cfg.neighborhood); j <= Math.min(N - 1, i + cfg.neighborhood); j++) idx.push(j);
    nbr.push({ idx: Int32Array.from(idx), w: new Float64Array(idx.length).fill(1) });
    let nb = 0;
    for (const j of idx) nb += peak[j];
    features[i * F + 3] = idx.length - 1;
    features[i * F + 4] = nb / idx.length;
  }
  if (cfg.context) { // "ar" context: log energy of whitened baseline-AR innovations after / before each start
    const order = cfg.arOrder;
    const fit = arResiduals(base, order).beta;
    const white = new Float64Array(n);
    for (let t = order; t < n; t++) {
      let pred = fit[0];
      for (let lag = 0; lag < order; lag++) pred += fit[lag + 1] * values[t - lag - 1];
      white[t] = values[t] - pred;
    }
    const sd = Math.max(std(white.subarray(order, B)), 1e-12);
    const cumulative = new Float64Array(n + 1);
    for (let t = 0; t < n; t++) cumulative[t + 1] = cumulative[t] + (t < order ? 0 : (white[t] / sd) ** 2);
    const level = (a, b) => (b > a ? Math.log((cumulative[b] - cumulative[a]) / (b - a) + 1e-3) : 0);
    cfg.horizons.forEach((hz, h) => {
      for (let i = 0; i < N; i++) {
        const s = starts[i];
        const after = level(s, Math.min(s + hz, n)), before = level(Math.max(s - hz, 0), s);
        features[i * F + 5 + 2 * h] = after;
        features[i * F + 6 + 2 * h] = s > 0 ? after - before : after;
      }
    });
  }
  return { features, F, N, nbr, starts, location, scale };
}
/** univariate_graph_episode: the target node is the window containing the onset (closest centre). */
export function temporalEpisode(graph, eventTime, windowSize, group = null) {
  let best = 0, bestDistance = Infinity;
  graph.starts.forEach((s, i) => {
    const contains = s <= eventTime && eventTime < s + windowSize;
    const distance = Math.abs(s + windowSize / 2 - eventTime);
    if (contains && distance < bestDistance) { best = i; bestDistance = distance; }
  });
  return { x: graph.features, N: graph.N, F: graph.F, nbr: graph.nbr, source: best, group };
}

// ------------------------------------------------------- evaluation orchestrator
export const OUTCOMES = ["hit", "miss", "early_alarm", "false_alarm", "correct_rejection"];

/** EvaluationOrchestrator.evaluate_case: route, outcome, delay and source ranks. */
export function evaluateCase(c, pipelineOptions, earlyTolerance = 0) {
  const result = runPipeline(c.data, pipelineOptions);
  const d = result.detection, detected = d.detected ? d.timeIndex : null;
  let outcome, delay = null;
  if (c.eventTime == null) outcome = detected !== null ? "false_alarm" : "correct_rejection";
  else if (detected === null) outcome = "miss";
  else if (detected < c.eventTime - earlyTolerance) outcome = "early_alarm";
  else { outcome = "hit"; delay = detected - c.eventTime; }
  const ranks = {}, predicted = {};
  if (result.ranking) {
    const weights = { heuristic: result.ranking.probabilities };
    if (result.process) { weights.gnn = result.process.gnnProbabilities; weights.gat = result.process.gatProbabilities; }
    for (const [method, w] of Object.entries(weights)) {
      predicted[method] = w.reduce((best, v, i) => (v > w[best] ? i : best), 0);
      if (outcome === "hit" && c.source != null) {
        let rank = 1;
        for (const v of w) if (v > w[c.source]) rank++;
        ranks[method] = rank;
      }
    }
  }
  return { id: c.id, route: result.plan.route, outcome, stop_reason: result.plan.stop_reason, detected, eventTime: c.eventTime, delay, source: c.source, ranks, predicted };
}
export function summarize(cases, k = 3) {
  const out = {};
  for (const route of [...new Set(cases.map((c) => c.route))].sort()) {
    const rows = cases.filter((c) => c.route === route);
    const count = Object.fromEntries(OUTCOMES.map((o) => [o, rows.filter((c) => c.outcome === o).length]));
    const withEvent = count.hit + count.miss + count.early_alarm, without = count.false_alarm + count.correct_rejection;
    const delays = rows.filter((c) => c.delay !== null).map((c) => c.delay);
    const row = {
      cases: rows.length, ...count,
      detection_rate: withEvent ? count.hit / withEvent : NaN,
      false_alarm_rate: without ? count.false_alarm / without : NaN,
      mean_delay: delays.length ? mean(delays) : NaN,
    };
    if (route === "n-d") {
      for (const method of ["heuristic", "gnn", "gat"]) {
        const ranks = rows.filter((c) => method in c.ranks).map((c) => c.ranks[method]);
        row[`${method}_n`] = ranks.length;
        row[`${method}_top1`] = ranks.length ? ranks.filter((r) => r === 1).length / ranks.length : NaN;
        row[`${method}_top${k}`] = ranks.length ? ranks.filter((r) => r <= k).length / ranks.length : NaN;
        row[`${method}_mrr`] = ranks.length ? mean(ranks.map((r) => 1 / r)) : NaN;
      }
    }
    out[route] = row;
  }
  return out;
}
