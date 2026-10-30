// Small canvas chart kit: line plots with a snapping crosshair, bars, heatmaps
// and ring graphs. Colors come from CSS custom properties, so every chart
// follows the light/dark theme; call redraw() after a theme change.

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const color = (token) => (token && token.startsWith("--") ? css(token) : token);

const FONT = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
const groups = new Map();

function niceTicks(lo, hi, count = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo];
  const step0 = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
export const fmt = (v, digits = 2) => {
  if (v == null || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
  return (+v.toFixed(digits)).toString();
};

function setupCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.parentElement.clientWidth || 600;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

class Base {
  constructor(host, options = {}) {
    this.host = typeof host === "string" ? document.getElementById(host) : host;
    this.host.classList.add("plot-wrap");
    this.canvas = document.createElement("canvas");
    this.tip = document.createElement("div");
    this.tip.className = "tip";
    this.tip.hidden = true;
    this.host.replaceChildren(this.canvas, this.tip);
    this.options = options;
    this.height = options.height || 200;
    this.spec = null;
    this.canvas.addEventListener("pointermove", (e) => this.pointer(e));
    this.canvas.addEventListener("pointerleave", () => this.leave());
    this.canvas.addEventListener("click", (e) => this.click && this.click(e));
    new ResizeObserver(() => this.spec && this.draw()).observe(this.host);
  }
  update(spec) { this.spec = spec; this.draw(); }
  redraw() { if (this.spec) this.draw(); }
  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  showTip(rows, x, y, title) {
    const tip = this.tip;
    tip.replaceChildren();
    if (title) { const h = document.createElement("div"); h.className = "tip-title"; h.textContent = title; tip.append(h); }
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "tip-row";
      if (row.color) { const key = document.createElement("span"); key.className = "tip-key"; key.style.background = color(row.color); line.append(key); }
      const value = document.createElement("strong");
      value.textContent = row.value;
      const label = document.createElement("span");
      label.textContent = row.label ?? "";
      line.append(value, label);
      tip.append(line);
    }
    tip.hidden = false;
    const w = this.host.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    const left = x + 14 + tw > w - 4 ? x - tw - 14 : x + 14;
    tip.style.left = `${Math.max(4, left)}px`;
    tip.style.top = `${Math.max(4, Math.min(y - th / 2, this.height - th - 4))}px`;
  }
  leave() { this.tip.hidden = true; this.hover = null; if (this.spec) this.draw(); }
}

// ------------------------------------------------------------------ line plot
/**
 * spec: { n, x?(i), series: [{ y, color, width, dash, alpha, label, offset }],
 *   bands: [{ x0, x1, y0?, y1?, color }], vlines: [{ x, color, dash, label }],
 *   hlines: [{ y, color, dash, label }], yDomain, xDomain, xLabel, yLabel,
 *   yTicks: [{ v, label }], logX, logY, points: [{ x, y, color, r }],
 *   tooltip(i) -> { title, rows } }
 */
export class LinePlot extends Base {
  constructor(host, options = {}) {
    super(host, options);
    if (options.group) {
      if (!groups.has(options.group)) groups.set(options.group, new Set());
      groups.get(options.group).add(this);
    }
  }
  geometry() {
    const s = this.spec, o = this.options;
    const left = o.left ?? 46, right = o.right ?? 12, top = o.top ?? 10, bottom = o.hideX ? 8 : (o.bottom ?? 26);
    const X = (i) => (s.x ? s.x(i) : i);
    const tx = (v) => (s.logX ? Math.log10(v) : v), ty = (v) => (s.logY ? Math.log10(v) : v);
    let [x0, x1] = s.xDomain || [X(0), X(s.n - 1)];
    x0 = tx(x0); x1 = tx(x1);
    let y0, y1;
    if (s.yDomain) [y0, y1] = s.yDomain.map(ty);
    else {
      y0 = Infinity; y1 = -Infinity;
      for (const ser of s.series) {
        const off = ser.offset || 0;
        for (let i = 0; i < s.n; i++) { const v = ty(ser.y[i] + off); if (Number.isFinite(v)) { y0 = Math.min(y0, v); y1 = Math.max(y1, v); } }
      }
      for (const h of s.hlines || []) if (Number.isFinite(h.y)) { y0 = Math.min(y0, ty(h.y)); y1 = Math.max(y1, ty(h.y)); }
      for (const p of s.points || []) { y0 = Math.min(y0, ty(p.y)); y1 = Math.max(y1, ty(p.y)); }
      const pad = (y1 - y0) * 0.06 || 1;
      y0 -= pad; y1 += pad;
    }
    const { width, height } = this.size;
    const px = (v) => left + ((tx(v) - x0) / (x1 - x0 || 1)) * (width - left - right);
    const py = (v) => top + (1 - (ty(v) - y0) / (y1 - y0 || 1)) * (height - top - bottom);
    return { left, right, top, bottom, x0, x1, y0, y1, px, py, X, width, height };
  }
  draw() {
    const s = this.spec;
    const { ctx, width, height } = setupCanvas(this.canvas, this.height);
    this.size = { width, height };
    const g = (this.g = this.geometry());
    const { left, right, top, bottom, px, py } = g;
    ctx.clearRect(0, 0, width, height);
    ctx.font = FONT;
    // grid + y ticks
    ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const yTicks = s.yTicks || (s.logY
      ? niceTicks(g.y0, g.y1, 4).filter((v) => Number.isInteger(v)).map((v) => ({ v: 10 ** v, label: `10${superscript(v)}` }))
      : niceTicks(g.y0, g.y1, this.options.yTickCount || 4).map((v) => ({ v, label: fmt(v) })));
    for (const t of yTicks) {
      const y = Math.round(py(t.v)) + 0.5;
      if (y < top - 1 || y > height - bottom + 1) continue;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
      ctx.fillText(t.label, left - 6, y);
    }
    // x axis
    ctx.strokeStyle = css("--axis");
    ctx.beginPath(); ctx.moveTo(left, height - bottom + 0.5); ctx.lineTo(width - right, height - bottom + 0.5); ctx.stroke();
    if (!this.options.hideX) {
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      const xt = s.logX
        ? niceTicks(g.x0, g.x1, 4).filter((v) => Number.isInteger(v)).map((v) => ({ v: 10 ** v, label: `10${superscript(v)}` }))
        : niceTicks(g.x0, g.x1, Math.max(3, Math.floor((width - left) / 90))).map((v) => ({ v, label: s.xFormat ? s.xFormat(v) : fmt(v, 1) }));
      for (const t of xt) {
        const x = px(t.v);
        if (x < left - 1 || x > width - right + 1) continue;
        ctx.fillText(t.label, x, height - bottom + 5);
      }
      if (s.xLabel) { ctx.textAlign = "right"; ctx.fillText(s.xLabel, width - right, height - 12 + 1); }
    }
    if (s.yLabel) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted"); ctx.fillText(s.yLabel, left + 4, top); }
    ctx.save();
    ctx.beginPath(); ctx.rect(left, top - 2, width - left - right, height - top - bottom + 4); ctx.clip();
    for (const b of s.bands || []) {
      ctx.fillStyle = color(b.color);
      const ya = b.y1 != null ? py(b.y1) : top, yb = b.y0 != null ? py(b.y0) : height - bottom;
      ctx.fillRect(px(b.x0), ya, Math.max(1, px(b.x1) - px(b.x0)), yb - ya);
    }
    for (const h of s.hlines || []) {
      if (!Number.isFinite(h.y)) continue;
      ctx.strokeStyle = color(h.color); ctx.lineWidth = h.width || 1.25; ctx.setLineDash(h.dash || []);
      const y = py(h.y);
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const ser of s.series) {
      ctx.strokeStyle = color(ser.color); ctx.lineWidth = ser.width || 1.5; ctx.globalAlpha = ser.alpha ?? 1;
      ctx.setLineDash(ser.dash || []); ctx.lineJoin = "round";
      ctx.beginPath();
      let started = false;
      const off = ser.offset || 0;
      for (let i = 0; i < s.n; i++) {
        const v = ser.y[i];
        if (!Number.isFinite(v) || (s.logY && v <= 0)) { started = false; continue; }
        const x = px(g.X(i)), y = py(v + off);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.setLineDash([]);
    for (const p of s.points || []) {
      ctx.fillStyle = color(p.color); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px(p.x), py(p.y), p.r || 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
    for (const v of s.vlines || []) {
      if (v.x == null) continue;
      const x = Math.round(px(v.x)) + 0.5;
      if (x < left || x > width - right) continue;
      ctx.strokeStyle = color(v.color); ctx.lineWidth = v.width || 1.5; ctx.setLineDash(v.dash || []);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, height - bottom); ctx.stroke();
      ctx.setLineDash([]);
      if (v.label) {
        ctx.font = FONT; ctx.textBaseline = "top";
        const w = ctx.measureText(v.label).width;
        const flip = x + 4 + w > width - right;
        ctx.textAlign = flip ? "right" : "left";
        ctx.fillStyle = css("--surface"); ctx.globalAlpha = 0.85;
        ctx.fillRect(flip ? x - 4 - w - 2 : x + 2, top + (v.row || 0) * 14, w + 4, 13);
        ctx.globalAlpha = 1;
        ctx.fillStyle = css("--ink-2");
        ctx.fillText(v.label, flip ? x - 4 : x + 4, top + 1 + (v.row || 0) * 14);
      }
    }
    for (const h of s.hlines || []) {
      if (!h.label || !Number.isFinite(h.y)) continue;
      const leftSide = h.side === "left";
      ctx.font = FONT; ctx.textAlign = leftSide ? "left" : "right"; ctx.textBaseline = h.below ? "top" : "bottom";
      const y = h.below ? py(h.y) + 2 : py(h.y) - 2, w = ctx.measureText(h.label).width;
      const x0 = leftSide ? left + 4 : width - right - w - 6;
      ctx.fillStyle = css("--surface"); ctx.globalAlpha = 0.85; ctx.fillRect(x0, h.below ? y : y - 13, w + 4, 13); ctx.globalAlpha = 1;
      ctx.fillStyle = css("--ink-2"); ctx.fillText(h.label, leftSide ? left + 6 : width - right - 4, y);
    }
    if (this.hoverIndex != null) this.drawCrosshair(ctx, this.hoverIndex);
  }
  drawCrosshair(ctx, i) {
    const g = this.g, x = Math.round(g.px(g.X(i))) + 0.5;
    ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 1; ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(x, g.top); ctx.lineTo(x, g.height - g.bottom); ctx.stroke();
    ctx.globalAlpha = 1;
    for (const ser of this.spec.series) {
      if (ser.noDot) continue;
      const v = ser.y[i];
      if (!Number.isFinite(v)) continue;
      ctx.fillStyle = color(ser.color); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, g.py(v + (ser.offset || 0)), 3.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
  }
  indexAt(px) {
    const g = this.g, s = this.spec;
    if (!g) return null;
    const frac = (px - g.left) / (g.width - g.left - g.right);
    if (frac < -0.02 || frac > 1.02) return null;
    const xv = g.x0 + frac * (g.x1 - g.x0);
    if (!s.x) return Math.max(0, Math.min(s.n - 1, Math.round(xv)));
    let best = 0, bd = Infinity;
    for (let i = 0; i < s.n; i++) { const d = Math.abs((s.logX ? Math.log10(s.x(i)) : s.x(i)) - xv); if (d < bd) { bd = d; best = i; } }
    return best;
  }
  pointer(e) {
    if (!this.spec) return;
    const { x, y } = this.local(e);
    const i = this.indexAt(x);
    if (i == null) return this.leave();
    this.setHover(i, x, y, true);
  }
  setHover(i, x, y, primary) {
    this.hoverIndex = i;
    this.draw();
    if (primary && this.spec.tooltip) {
      const t = this.spec.tooltip(i, y);
      if (t) this.showTip(t.rows, x, y, t.title);
    }
    if (primary && this.options.group) for (const other of groups.get(this.options.group)) if (other !== this && other.spec) other.setHover(Math.min(i, other.spec.n - 1), 0, 0, false);
    if (primary && this.options.onHover) this.options.onHover(i);
  }
  leave() {
    this.hoverIndex = null;
    this.tip.hidden = true;
    if (this.spec) this.draw();
    if (this.options.group) for (const other of groups.get(this.options.group)) if (other !== this && other.hoverIndex != null) { other.hoverIndex = null; other.redraw(); }
  }
}
const superscript = (v) => String(v).split("").map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(c)] ?? (c === "-" ? "⁻" : c)).join("");

// ----------------------------------------------------------------- bar chart
/** spec: { labels, groups: [{ values, color, label }], emphasis(i)->alpha, marks(i)->text, valueLabels(i)->bool, yMax, tooltip(i) } */
export class BarPlot extends Base {
  draw() {
    const s = this.spec;
    const { ctx, width, height } = setupCanvas(this.canvas, this.height);
    this.size = { width, height };
    const left = 40, right = 8, top = 16, bottom = s.marks ? 34 : 22;
    const n = s.labels.length, G = s.groups.length;
    const yMax = s.yMax ?? Math.max(1e-9, ...s.groups.flatMap((g) => Array.from(g.values))) * 1.12;
    const py = (v) => top + (1 - v / yMax) * (height - top - bottom);
    const slot = (width - left - right) / n;
    const barW = Math.max(3, Math.min(28, (slot - 6) / G - 2));
    this.layout = { left, slot, n };
    ctx.clearRect(0, 0, width, height);
    ctx.font = FONT; ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--grid");
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of niceTicks(0, yMax, 3)) {
      const y = Math.round(py(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
      ctx.fillText(fmt(t), left - 6, y);
    }
    ctx.strokeStyle = css("--axis");
    ctx.beginPath(); ctx.moveTo(left, py(0) + 0.5); ctx.lineTo(width - right, py(0) + 0.5); ctx.stroke();
    for (let i = 0; i < n; i++) {
      const cx = left + slot * (i + 0.5);
      const start = cx - (G * (barW + 2) - 2) / 2;
      s.groups.forEach((grp, k) => {
        const v = grp.values[i];
        if (!Number.isFinite(v)) return;
        const x = start + k * (barW + 2), y = py(v), h = py(0) - y;
        ctx.globalAlpha = (s.emphasis ? s.emphasis(i, k) : 1) * (this.hover === i ? 1 : 0.95);
        ctx.fillStyle = color(grp.color);
        roundTop(ctx, x, y, barW, Math.max(h, 0.5), Math.min(4, barW / 2));
        ctx.globalAlpha = 1;
        if (s.valueLabels && s.valueLabels(i, k)) {
          ctx.fillStyle = css("--ink"); ctx.textAlign = "center"; ctx.textBaseline = "bottom";
          ctx.fillText(fmt(v), x + barW / 2, y - 2);
        }
      });
      if (this.hover === i) { ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 1; ctx.strokeRect(cx - slot / 2 + 1.5, top - 6, slot - 3, py(0) - top + 6); }
      ctx.fillStyle = css("--ink-2"); ctx.textAlign = "center"; ctx.textBaseline = "top";
      if (slot > 20 || i % 2 === 0) ctx.fillText(s.labels[i], cx, py(0) + 5);
      if (s.marks) { const m = s.marks(i); if (m) { ctx.fillStyle = css("--ink"); ctx.fillText(m, cx, py(0) + 19); } }
    }
  }
  pointer(e) {
    if (!this.layout) return;
    const { x, y } = this.local(e);
    const i = Math.floor((x - this.layout.left) / this.layout.slot);
    if (i < 0 || i >= this.layout.n) return this.leave();
    if (this.hover !== i) { this.hover = i; this.draw(); }
    if (this.spec.tooltip) { const t = this.spec.tooltip(i); this.showTip(t.rows, x, y, t.title); }
  }
}
function roundTop(ctx, x, y, w, h, r) {
  r = Math.min(r, h);
  ctx.beginPath();
  ctx.moveTo(x, y + h); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r); ctx.lineTo(x + w, y + h);
  ctx.closePath(); ctx.fill();
}

// ------------------------------------------------------------------- heatmap
/** spec: { rows, cols, value(r, c), vmax, rowLabel(r), colLabel(c) for ticks, xTicks: [c], tooltip(r, c), marks: [{ c, label }] } */
export class Heatmap extends Base {
  draw() {
    const s = this.spec;
    const { ctx, width, height } = setupCanvas(this.canvas, this.height);
    const left = this.options.left ?? 70, right = 8, top = 8, bottom = 24;
    const cw = (width - left - right) / s.cols, ch = (height - top - bottom) / s.rows;
    this.layout = { left, top, cw, ch };
    ctx.clearRect(0, 0, width, height);
    const lo = parseRgb(css("--seq-lo")), hi = parseRgb(css("--seq-hi"));
    for (let r = 0; r < s.rows; r++) {
      for (let c = 0; c < s.cols; c++) {
        const f = Math.max(0, Math.min(1, s.value(r, c) / s.vmax));
        ctx.fillStyle = `rgb(${lo.map((v, k) => Math.round(v + (hi[k] - v) * f)).join(",")})`;
        ctx.fillRect(left + c * cw, top + r * ch, cw + 0.6, ch - (ch > 6 ? 1 : 0));
      }
    }
    ctx.font = FONT; ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let r = 0; r < s.rows; r++) ctx.fillText(s.rowLabel(r), left - 6, top + (r + 0.5) * ch);
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = css("--muted");
    for (const c of s.xTicks || []) ctx.fillText(s.colLabel(c), left + (c + 0.5) * cw, height - bottom + 5);
    for (const m of s.marks || []) {
      const x = Math.round(left + (m.c + 0.5) * cw) + 0.5;
      ctx.strokeStyle = color(m.color || "--critical"); ctx.lineWidth = 1.5; ctx.setLineDash(m.dash || []);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, height - bottom); ctx.stroke(); ctx.setLineDash([]);
    }
    if (this.hover) {
      ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
      ctx.strokeRect(left + this.hover.c * cw, top + this.hover.r * ch, cw, ch);
    }
  }
  pointer(e) {
    if (!this.layout) return;
    const { x, y } = this.local(e), L = this.layout;
    const c = Math.floor((x - L.left) / L.cw), r = Math.floor((y - L.top) / L.ch);
    if (c < 0 || r < 0 || c >= this.spec.cols || r >= this.spec.rows) return this.leave();
    this.hover = { r, c };
    this.draw();
    const t = this.spec.tooltip(r, c);
    this.showTip(t.rows, x, y, t.title);
  }
}
function parseRgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}

// ---------------------------------------------------------------- ring graph
/**
 * spec: { n, label(i), fill(i) (token or css color), ring(i) -> { color, width } | null,
 *   edges: [{ a, b, w (0..1), color, dashed }], badge(i) -> text outside the node,
 *   tooltip(i) -> { title, rows }, onClick(i), selected }
 */
export class RingGraph extends Base {
  draw() {
    const s = this.spec;
    const { ctx, width, height } = setupCanvas(this.canvas, this.height);
    const cx = width / 2, cy = height / 2;
    const R = Math.min(width, height) / 2 - (s.badge ? 34 : 22);
    const nodeR = Math.max(9, Math.min(17, (Math.PI * R) / s.n - 3));
    const pos = Array.from({ length: s.n }, (_, i) => {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / s.n;
      return { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), a };
    });
    this.layout = { pos, nodeR };
    ctx.clearRect(0, 0, width, height);
    for (const e of s.edges || []) {
      const focus = this.hover != null && (e.a === this.hover || e.b === this.hover);
      ctx.strokeStyle = color(e.color || "--muted");
      ctx.globalAlpha = this.hover != null && !focus ? 0.12 : 0.25 + 0.7 * (e.w ?? 1);
      ctx.lineWidth = 0.6 + 4 * (e.w ?? 0.5);
      ctx.setLineDash(e.dashed ? [4, 3] : []);
      ctx.beginPath(); ctx.moveTo(pos[e.a].x, pos[e.a].y);
      if (e.curve) {
        ctx.quadraticCurveTo(cx + (pos[e.a].x + pos[e.b].x - 2 * cx) * 0.25, cy + (pos[e.a].y + pos[e.b].y - 2 * cy) * 0.25, pos[e.b].x, pos[e.b].y);
      } else ctx.lineTo(pos[e.b].x, pos[e.b].y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.setLineDash([]);
    ctx.font = `600 ${nodeR > 12 ? 11 : 9}px system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < s.n; i++) {
      const { x, y } = pos[i];
      ctx.fillStyle = color(s.fill ? s.fill(i) : "--surface");
      ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, nodeR, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      const ring = s.ring && s.ring(i);
      ctx.strokeStyle = ring ? color(ring.color) : css("--axis"); ctx.lineWidth = ring ? ring.width || 2.5 : 1;
      ctx.beginPath(); ctx.arc(x, y, nodeR + (ring ? 1.5 : 0), 0, 2 * Math.PI); ctx.stroke();
      if (this.hover === i || s.selected === i) { ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, nodeR + 5, 0, 2 * Math.PI); ctx.stroke(); }
      ctx.fillStyle = s.labelColor ? color(s.labelColor(i)) : css("--ink");
      ctx.fillText(s.label(i), x, y + 0.5);
      if (s.badge) {
        const b = s.badge(i);
        if (b) {
          const a = pos[i].a, bx = cx + (R + nodeR + 12) * Math.cos(a), by = cy + (R + nodeR + 12) * Math.sin(a);
          ctx.font = FONT; ctx.fillStyle = css("--ink-2"); ctx.fillText(b, bx, by);
          ctx.font = `600 ${nodeR > 12 ? 11 : 9}px system-ui, sans-serif`;
        }
      }
    }
    if (s.center) {
      ctx.font = FONT; ctx.fillStyle = css("--muted");
      s.center.forEach((line, k) => ctx.fillText(line, cx, cy + (k - (s.center.length - 1) / 2) * 15));
    }
  }
  nodeAt(e) {
    if (!this.layout) return null;
    const { x, y } = this.local(e);
    let best = null, bd = Infinity;
    this.layout.pos.forEach((p, i) => { const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = i; } });
    return bd <= Math.max(this.layout.nodeR + 8, 14) ? best : null;
  }
  pointer(e) {
    const i = this.nodeAt(e);
    if (i == null) { if (this.hover != null) this.leave(); return; }
    this.canvas.style.cursor = this.spec.onClick ? "pointer" : "default";
    if (this.hover !== i) { this.hover = i; this.draw(); }
    if (this.spec.tooltip) { const { x, y } = this.local(e); const t = this.spec.tooltip(i); this.showTip(t.rows, x, y, t.title); }
  }
  click(e) {
    const i = this.nodeAt(e);
    if (i != null && this.spec.onClick) this.spec.onClick(i);
  }
}

export function redrawAll(charts) { for (const c of charts) c.redraw(); }
