// Graph methods of Graph-EVT-agent (graph_evt_agent.graph, method = corr | partial | xcorr | pwxcorr | granger),
// copied verbatim from Educational-Live-Demo js/onedim-core.js, where tests/port_check.js compares them with the
// Python library (9 configurations, max relative difference 5e-13). Self-contained: no imports.
// tests/graph_methods_port_check.mjs repeats that comparison against the same Python reference.
//
// functionalGraph(chans, from, to, opt): dependence D between channels on [from, to) by opt.method
//   corr     |corr| at lag 0
//   partial  |partial corr| from the inverse correlation matrix
//   xcorr    max over |lag| <= maxLag of corr(x_i(t), x_j(t + lag))
//   pwxcorr  the same on AR(order) innovations (prewhitening)
//   granger  conditional Granger strength ln(RSS without i / RSS with all), lags lagStep..maxLag, ridge
// Edges: D >= threshold (rule "threshold") or the k strongest partners per node (rule "topk").

const TAU = 2 * Math.PI;

function hashSeed() {
  var h = 2166136261 >>> 0;
  for (var a = 0; a < arguments.length; a++) {
    var s = String(arguments[a]);
    for (var i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    h ^= 0x9e3779b9; h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}
function makeRng() {
  var a = hashSeed.apply(null, arguments), spare = null;
  function uniform() {
    a = (a + 0x6d2b79f5) >>> 0;
    var t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function normal() {
    if (spare !== null) { var s = spare; spare = null; return s; }
    var u = 0;
    while (u === 0) u = uniform();
    var r = Math.sqrt(-2 * Math.log(u)), ang = TAU * uniform();
    spare = r * Math.sin(ang);
    return r * Math.cos(ang);
  }
  return { uniform: uniform, normal: normal };
}
function mean(x) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i]; return s / x.length; }
function std(x) { var m = mean(x), s = 0; for (var i = 0; i < x.length; i++) s += (x[i] - m) * (x[i] - m); return Math.sqrt(s / x.length); }
function quantile(x, q) {
  var s = Float64Array.from(x).sort(), pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
function solve(A, b) { // Gaussian elimination, small dense systems
  var n = b.length, M = A.map(function (r) { return r.slice(); }), y = b.slice();
  for (var c = 0; c < n; c++) {
    var p = c;
    for (var k = c + 1; k < n; k++) if (Math.abs(M[k][c]) > Math.abs(M[p][c])) p = k;
    var tmp = M[c]; M[c] = M[p]; M[p] = tmp; var ty = y[c]; y[c] = y[p]; y[p] = ty;
    for (k = c + 1; k < n; k++) {
      var f = M[k][c] / M[c][c];
      for (var m = c; m < n; m++) M[k][m] -= f * M[c][m];
      y[k] -= f * y[c];
    }
  }
  var x = new Array(n).fill(0);
  for (c = n - 1; c >= 0; c--) { var s = y[c]; for (k = c + 1; k < n; k++) s -= M[c][k] * x[k]; x[c] = s / M[c][c]; }
  return x;
}
function arResiduals(x, order) {
  var rows = x.length - order, cols = order + 1, G = [], b = new Array(cols).fill(0);
  for (var i = 0; i < cols; i++) G.push(new Array(cols).fill(0));
  var row = function (r) { var v = [1]; for (var l = 0; l < order; l++) v.push(x[order + r - l - 1]); return v; };
  for (var r = 0; r < rows; r++) { var v = row(r); for (i = 0; i < cols; i++) { b[i] += v[i] * x[order + r]; for (var j = 0; j < cols; j++) G[i][j] += v[i] * v[j]; } }
  for (i = 0; i < cols; i++) G[i][i] += 1e-10;
  var beta = solve(G, b), res = new Float64Array(rows);
  for (r = 0; r < rows; r++) { var w = row(r), f = 0; for (i = 0; i < cols; i++) f += w[i] * beta[i]; res[r] = x[order + r] - f; }
  return res;
}
function pearsonLag(a, b, lag, from, to) { // corr(a(t), b(t + lag)) over t with both inside [from, to)
  var lo = Math.max(from, from - lag), hi = Math.min(to, to - lag), n = hi - lo;
  if (n < 20) return 0;
  var ma = 0, mb = 0;
  for (var t = lo; t < hi; t++) { ma += a[t]; mb += b[t + lag]; }
  ma /= n; mb /= n;
  var sab = 0, saa = 0, sbb = 0;
  for (t = lo; t < hi; t++) { var da = a[t] - ma, db = b[t + lag] - mb; sab += da * db; saa += da * da; sbb += db * db; }
  return sab / Math.sqrt(saa * sbb || 1);
}
function invert(M) { // Gauss–Jordan, small dense
  var n = M.length, A = M.map(function (r, i) { return r.concat(M.map(function (_, j) { return i === j ? 1 : 0; })); });
  for (var c = 0; c < n; c++) {
    var p = c; for (var k = c + 1; k < n; k++) if (Math.abs(A[k][c]) > Math.abs(A[p][c])) p = k;
    var tmp = A[c]; A[c] = A[p]; A[p] = tmp;
    var d = A[c][c] || 1e-12; for (k = 0; k < 2 * n; k++) A[c][k] /= d;
    for (var r = 0; r < n; r++) if (r !== c) { var f = A[r][c]; for (k = 0; k < 2 * n; k++) A[r][k] -= f * A[c][k]; }
  }
  return A.map(function (r) { return r.slice(n); });
}
function lagRSS(y, cols, ridge) {
  var k = cols.length + 1, n = y.length, G = [], b = new Array(k).fill(0), i, j, t;
  for (i = 0; i < k; i++) G.push(new Array(k).fill(0));
  var val = function (c, t2) { return c < 0 ? 1 : cols[c][t2]; };
  for (t = 0; t < n; t++) for (i = 0; i < k; i++) { var vi = val(i - 1, t); b[i] += vi * y[t]; for (j = i; j < k; j++) G[i][j] += vi * val(j - 1, t); }
  for (i = 0; i < k; i++) { for (j = 0; j < i; j++) G[i][j] = G[j][i]; if (i) G[i][i] += ridge; }
  var beta = solve(G, b), rss = 0;
  for (t = 0; t < n; t++) { var f = 0; for (i = 0; i < k; i++) f += beta[i] * val(i - 1, t); rss += (y[t] - f) * (y[t] - f); }
  return rss;
}
function grangerMatrix(X, maxLag, lagStep, ridge) {
  X = X.map(function (x) { var m = mean(x), s = std(x) || 1e-12; return Array.prototype.map.call(x, function (v) { return (v - m) / Math.max(s, 1e-12); }); });
  var p = X.length, n = X[0].length, lags = [];
  for (var l = lagStep; l <= maxLag; l += lagStep) lags.push(l);
  var T0 = maxLag, cols = [];
  for (var c = 0; c < p; c++) cols.push(lags.map(function (lg) { return X[c].slice(T0 - lg, n - lg); }));
  var D = [];
  for (var i = 0; i < p; i++) D.push(new Array(p).fill(0));
  for (var m = 0; m < p; m++) {
    var y = X[m].slice(T0), all = [];
    for (c = 0; c < p; c++) all = all.concat(cols[c]);
    var full = lagRSS(y, all, ridge);
    for (i = 0; i < p; i++) {
      if (i === m) continue;
      var red = [];
      for (c = 0; c < p; c++) if (c !== i) red = red.concat(cols[c]);
      D[i][m] = Math.max(0, Math.log(lagRSS(y, red, ridge) / Math.max(full, 1e-300)));
    }
  }
  for (i = 0; i < p; i++) for (m = i + 1; m < p; m++) D[i][m] = D[m][i] = Math.max(D[i][m], D[m][i]);
  return D;
}
function functionalGraph(chans, from, to, opt) {
  var p = chans.length, D = [], lag = [], i, j;
  for (i = 0; i < p; i++) { D.push(new Array(p).fill(0)); lag.push(new Array(p).fill(0)); }
  var method = opt.method || "ar", L = opt.maxLag || 40;
  var dep = function (a, b) {
    if (method === "xcorr" || method === "pwxcorr") { var best = 0, at = 0; for (var l = -L; l <= L; l++) { var c = pearsonLag(a, b, l, 0, a.length); if (c > best) { best = c; at = l; } } return { d: best, lag: at }; }
    return { d: Math.abs(pearsonLag(a, b, 0, 0, a.length)), lag: 0 };
  };
  var X = chans.map(function (x) { return Array.prototype.slice.call(x, from, to); });
  // "ar" and "pwxcorr" work on AR(order) innovations: prewhitening sharpens the lagged correlation peaks
  if (method === "ar" || method === "pwxcorr") X = X.map(function (x) { return Array.prototype.slice.call(arResiduals(x, opt.order || 2)); });
  if (method === "granger") {
    var G = grangerMatrix(X, L, opt.lagStep || 5, opt.ridge == null ? 1e-3 : opt.ridge);
    for (i = 0; i < p; i++) for (j = 0; j < p; j++) D[i][j] = G[i][j];
  } else if (method === "partial") {
    var R = X.map(function (a) { return X.map(function (b) { return pearsonLag(a, b, 0, 0, a.length); }); }), P = invert(R);
    for (i = 0; i < p; i++) for (j = 0; j < p; j++) if (i !== j) D[i][j] = Math.abs(-P[i][j] / Math.sqrt(P[i][i] * P[j][j]));
  } else {
    for (i = 0; i < p; i++) for (j = i + 1; j < p; j++) { var r = dep(X[i], X[j]); D[i][j] = D[j][i] = r.d; lag[i][j] = r.lag; lag[j][i] = -r.lag; }
  }
  var thr = null;
  if (opt.surrogates && method !== "partial" && method !== "granger") {
    var rng = makeRng(opt.seed || 0, "surr"), S = 39; thr = [];
    for (i = 0; i < p; i++) thr.push(new Array(p).fill(0));
    for (i = 0; i < p; i++) for (j = i + 1; j < p; j++) {
      var sv = [], n = X[j].length;
      for (var s = 0; s < S; s++) {
        var k = Math.floor(n * (0.1 + 0.8 * rng.uniform())), sh = X[j].slice(k).concat(X[j].slice(0, k));
        sv.push(dep(X[i], sh).d);
      }
      thr[i][j] = thr[j][i] = quantile(sv, 0.95);
    }
  }
  var keep = [];
  for (i = 0; i < p; i++) { keep.push(new Array(p).fill(false)); }
  if (opt.rule === "topk") {
    for (i = 0; i < p; i++) {
      var order = D[i].map(function (v, jj) { return jj; }).filter(function (jj) { return jj !== i; }).sort(function (a, b) { return D[i][b] - D[i][a]; });
      order.slice(0, opt.k || 2).forEach(function (jj) { keep[i][jj] = keep[jj][i] = true; });
    }
  } else for (i = 0; i < p; i++) for (j = 0; j < p; j++) if (i !== j && D[i][j] >= (opt.threshold == null ? 0.35 : opt.threshold)) keep[i][j] = true;
  if (thr) for (i = 0; i < p; i++) for (j = 0; j < p; j++) if (i !== j && D[i][j] <= thr[i][j]) keep[i][j] = false;
  var pruned = [];
  if (opt.prune && (method === "xcorr" || method === "pwxcorr")) {
    var sig = keep.map(function (r) { return r.slice(); }), tol = opt.lagTol || 6;
    for (i = 0; i < p; i++) for (j = i + 1; j < p; j++) {
      if (!sig[i][j]) continue;
      for (var m = 0; m < p; m++) {
        if (m === i || m === j || !sig[i][m] || !sig[m][j]) continue;
        if (D[i][m] > D[i][j] && D[m][j] > D[i][j] && Math.abs(Math.abs(lag[i][m]) + Math.abs(lag[m][j]) - Math.abs(lag[i][j])) <= tol) {
          keep[i][j] = keep[j][i] = false; pruned.push([i, j, m]); break;
        }
      }
    }
  }
  var adjacency = new Float64Array(p * p), dependence = new Float64Array(p * p);
  for (i = 0; i < p; i++) for (j = 0; j < p; j++) { dependence[i * p + j] = i === j ? 1 : D[i][j]; adjacency[i * p + j] = i === j || keep[i][j] ? 1 : 0; }
  return { adjacency: adjacency, dependence: dependence, lag: lag, threshold: thr, pruned: pruned, p: p, method: method };
}

export { functionalGraph };
