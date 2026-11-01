// site/js/core.js buildGraph (methods corr, partial, xcorr, pwxcorr, granger) must reproduce
// graph_evt_agent.graph.build_graph (Python) on a fixed input. Reference: tests/fixtures/graph_methods_reference.json,
// made by tests/make_graph_reference.py. Run from the repository root: node tests/graph_methods_port_check.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as C from "../site/js/core.js";

const ref = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/graph_methods_reference.json", import.meta.url)), "utf8"));
const n = ref.data.length, p = ref.data[0].length;
const baseline = { n, p, v: Float64Array.from(ref.data.flat()) };

let failures = 0, checked = 0;
for (const c of ref.cases) {
  const cfg = c.config;
  if (!(cfg.method in C.DEPENDENCE_METHODS)) continue;           // ar is ported separately (innovationGraph)
  if (cfg.prune_indirect) continue;                               // pruning is not exposed by the guide
  const g = C.buildGraph(baseline, {
    method: cfg.method, edgeThreshold: cfg.edge_threshold ?? 0.35, edgeRule: cfg.edge_rule ?? "threshold", topK: cfg.top_k_edges ?? 2,
    maxLag: cfg.max_lag ?? 40, lagStep: cfg.lag_step ?? 5, ridge: cfg.ridge ?? 1e-3, arOrder: cfg.ar_order ?? 2,
  });
  let worst = 0, adjOk = g.method === c.method;
  for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
    if (a === b) continue;
    const want = c.dependence[a][b], got = g.dependence[a * p + b];
    worst = Math.max(worst, Math.abs(got - want) / Math.max(1e-9, Math.abs(want)));
    if (!!g.adjacency[a * p + b] !== !!c.adjacency[a][b]) adjOk = false;
  }
  const ok = worst < 1e-6 && adjOk;
  checked++; if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${JSON.stringify(cfg)}  max rel diff ${worst.toExponential(2)}${adjOk ? "" : "  adjacency/method differs"}`);
}
console.log(failures ? `${failures} of ${checked} cases differ` : `all ${checked} cases match the Python reference`);
process.exit(failures || !checked ? 1 : 0);
