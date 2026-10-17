// Runs the repository's evt_demo/data_generator.py in Pyodide, off the UI thread.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

// The Pages workflow copies the generator to py/; the second path works when
// the repository root is served locally (python -m http.server).
const GENERATOR_PATHS = [
  "py/evt_demo/data_generator.py",
  "../demo/evt-educational-demo/src/evt_demo/data_generator.py",
];

// data_generator.py imports TimeSeriesInput only for the multi-event recording,
// which this page never builds; a minimal stand-in keeps the import working.
const GRAPH_EVT_AGENT_STUB = `
class TimeSeriesInput:
    def __init__(self, values, channel_names=None):
        self.values = values
        self.channel_names = channel_names
`;

let bridge = null;

async function fetchText(path) {
  const response = await fetch(new URL(path, self.location.href));
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.text();
}

async function init() {
  const pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy");

  let generator = null, generatorPath = null;
  for (const path of GENERATOR_PATHS) {
    try { generator = await fetchText(path); generatorPath = path; break; } catch (_) { /* try next */ }
  }
  if (!generator) throw new Error("Could not load data_generator.py from " + GENERATOR_PATHS.join(" or "));

  const fs = pyodide.FS;
  fs.mkdirTree("/lib/evt_demo");
  fs.mkdirTree("/lib/graph_evt_agent");
  fs.writeFile("/lib/evt_demo/__init__.py", "");
  fs.writeFile("/lib/evt_demo/data_generator.py", generator);
  fs.writeFile("/lib/graph_evt_agent/__init__.py", "");
  fs.writeFile("/lib/graph_evt_agent/input.py", GRAPH_EVT_AGENT_STUB);
  fs.writeFile("/lib/bridge.py", await fetchText("bridge.py"));
  pyodide.runPython("import sys; sys.path.insert(0, '/lib')");
  bridge = pyodide.pyimport("bridge");
  return { generatorPath, pyodideVersion: pyodide.version, numpyVersion: pyodide.runPython("import numpy; numpy.__version__") };
}

function toFloat64(bytes) {
  return new Float64Array(bytes.slice().buffer);
}

const ready = init();
ready.then(
  (info) => postMessage({ type: "ready", ...info }),
  (error) => postMessage({ type: "fatal", message: String(error && error.message || error) }),
);

onmessage = async ({ data }) => {
  await ready;
  const { id, kind } = data;
  try {
    if (kind === "defaults") {
      const result = JSON.parse(bridge.defaults(data.n_channels, data.source_channel, data.seed));
      postMessage({ type: "result", id, result });
    } else if (kind === "run") {
      const started = performance.now();
      const proxy = bridge.run(JSON.stringify(data.params));
      const raw = proxy.toJs({ dict_converter: Object.fromEntries });
      proxy.destroy();
      const result = { meta: JSON.parse(raw.meta), elapsed: performance.now() - started };
      const transfer = [];
      for (const key of ["values", "background", "coherence", "channel_event_mask", "arrival"]) {
        result[key] = toFloat64(raw[key]);
        transfer.push(result[key].buffer);
      }
      postMessage({ type: "result", id, result }, transfer);
    }
  } catch (error) {
    const message = String(error && error.message || error);
    // Python tracebacks are long; the last line carries the ValueError text.
    postMessage({ type: "error", id, message: message.trim().split("\n").pop() });
  }
};
