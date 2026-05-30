// Project Vimaan dashboard — vanilla JS, single-file.

const $ = (id) => document.getElementById(id);

const els = {
  modelSelect: $("model-select"),
  modelMeta: $("model-meta"),
  refreshModels: $("refresh-models"),
  predictInput: $("predict-input"),
  predictBtn: $("predict-btn"),
  predictResult: $("predict-result"),

  datasetSelect: $("dataset-select"),
  refreshDatasets: $("refresh-datasets"),
  datasetUpload: $("dataset-upload"),
  uploadMsg: $("upload-msg"),

  hp: {
    base: $("hp-base"), epochs: $("hp-epochs"), lr: $("hp-lr"),
    bs: $("hp-bs"), ml: $("hp-ml"), pat: $("hp-pat"),
  },

  trainStart: $("train-start"),
  trainStop: $("train-stop"),
  trainStatusPill: $("train-status-pill"),
  trainLog: $("train-log"),
  trainSummary: $("train-summary"),
  trainPid: $("train-pid"),

  chartCanvas: $("loss-chart"),
};

let chart = null;
let pollTimer = null;
let modelCache = [];

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${url}: ${r.status}`);
  return data;
}

// ------- model + dataset catalogues -----------------------------------------

async function refreshModels() {
  const { models } = await jget("/api/models");
  modelCache = models;
  els.modelSelect.innerHTML = "";
  if (!models.length) {
    const opt = document.createElement("option");
    opt.textContent = "— no trained models found —";
    opt.disabled = true;
    els.modelSelect.appendChild(opt);
    els.modelMeta.textContent = "train a model first";
    return;
  }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.path;
    opt.textContent = `${m.version} ${m.has_manifest ? "" : "(no manifest)"}`;
    els.modelSelect.appendChild(opt);
  }
  updateModelMeta();
}

function updateModelMeta() {
  const m = modelCache.find((x) => x.path === els.modelSelect.value);
  if (!m) { els.modelMeta.textContent = ""; return; }
  if (m.manifest) {
    const hp = m.manifest.hyperparams || {};
    const rc = m.manifest.row_count ?? "?";
    els.modelMeta.textContent =
      `${m.manifest.framework_versions?.transformers || "?"} · ` +
      `rows=${rc} · epochs=${hp.epochs ?? "?"} · base=${hp.base_model ?? "?"}`;
  } else {
    els.modelMeta.textContent = `${m.version} · no manifest`;
  }
}

async function refreshDatasets() {
  const { datasets } = await jget("/api/datasets");
  els.datasetSelect.innerHTML = "";
  if (!datasets.length) {
    const opt = document.createElement("option");
    opt.textContent = "— no datasets found —";
    opt.disabled = true;
    els.datasetSelect.appendChild(opt);
    return;
  }
  for (const d of datasets) {
    const opt = document.createElement("option");
    opt.value = d.path;
    opt.textContent = `[${d.source}] ${d.name} (${fmtBytes(d.size_bytes)})`;
    els.datasetSelect.appendChild(opt);
  }
}

// ------- prediction ---------------------------------------------------------

function renderPrediction(r) {
  const slots = r.slots || {};
  const chips = Object.entries(slots)
    .map(([k, v]) => `<span class="chip"><span class="k">${k}</span>${v}</span>`).join("");
  els.predictResult.classList.remove("empty");
  els.predictResult.innerHTML = `
    <div><span class="intent">${r.intent}</span>
         <span class="conf">conf ${(r.confidence * 100).toFixed(1)}%</span></div>
    <div class="slots">${chips || '<span class="subtle">no slots</span>'}</div>
  `;
}

async function onPredict() {
  const text = els.predictInput.value.trim();
  const modelPath = els.modelSelect.value;
  if (!text || !modelPath) return;
  els.predictBtn.disabled = true;
  els.predictResult.classList.add("empty");
  els.predictResult.textContent = "predicting…";
  try {
    const r = await jpost("/api/predict", { model_path: modelPath, text });
    renderPrediction(r);
  } catch (e) {
    els.predictResult.classList.add("empty");
    els.predictResult.textContent = `error: ${e.message}`;
  } finally {
    els.predictBtn.disabled = false;
  }
}

// ------- upload -------------------------------------------------------------

async function onUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  els.uploadMsg.textContent = `uploading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/datasets/upload", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    els.uploadMsg.textContent = `uploaded ${data.name} (${data.rows} rows)`;
    await refreshDatasets();
    els.datasetSelect.value = data.path;
  } catch (err) {
    els.uploadMsg.textContent = `upload failed: ${err.message}`;
  } finally {
    e.target.value = "";
  }
}

// ------- training -----------------------------------------------------------

function ensureChart() {
  if (chart) return chart;
  chart = new Chart(els.chartCanvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "train loss", data: [], borderColor: "#38bdf8", backgroundColor: "rgba(56,189,248,.15)", tension: .25, fill: true },
        { label: "val loss",   data: [], borderColor: "#818cf8", backgroundColor: "rgba(129,140,248,.15)", tension: .25, fill: true },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      plugins: { legend: { labels: { color: "#cbd5e1" } } },
      scales: {
        x: { title: { display: true, text: "epoch", color: "#94a3b8" }, ticks: { color: "#cbd5e1" }, grid: { color: "rgba(148,163,184,.1)" } },
        y: { title: { display: true, text: "loss", color: "#94a3b8" }, ticks: { color: "#cbd5e1" }, grid: { color: "rgba(148,163,184,.1)" } },
      },
    },
  });
  return chart;
}

function applyState(s) {
  els.trainStatusPill.textContent = s.status;
  els.trainStatusPill.dataset.status = s.status;
  els.trainPid.textContent = s.pid ? `pid ${s.pid}` : "";
  els.trainStop.disabled = s.status !== "running";
  els.trainStart.disabled = s.status === "running";
  els.trainLog.textContent = (s.log_tail || []).join("\n");
  els.trainLog.scrollTop = els.trainLog.scrollHeight;

  const c = ensureChart();
  c.data.labels = s.metrics.map((m) => m.epoch);
  c.data.datasets[0].data = s.metrics.map((m) => m.train_loss ?? null);
  c.data.datasets[1].data = s.metrics.map((m) => m.val_loss ?? null);
  c.update("none");

  const last = s.metrics[s.metrics.length - 1];
  const ckptCount = (s.checkpoints || []).length;
  els.trainSummary.textContent = last
    ? `epoch ${last.epoch} · train=${(last.train_loss ?? NaN).toFixed?.(4) || "—"} · val=${(last.val_loss ?? NaN).toFixed?.(4) || "—"} · saved ${ckptCount}×`
    : "";
}

async function pollStatus() {
  try {
    const s = await jget("/api/train/status");
    applyState(s);
    if (s.status === "running") {
      pollTimer = setTimeout(pollStatus, 1500);
    } else {
      pollTimer = null;
      // a fresh checkpoint may have landed — refresh the model dropdown
      await refreshModels();
    }
  } catch {
    pollTimer = setTimeout(pollStatus, 4000);
  }
}

async function onTrainStart() {
  const body = {
    dataset: els.datasetSelect.value,
    epochs: +els.hp.epochs.value,
    lr: +els.hp.lr.value,
    batch_size: +els.hp.bs.value,
    base_model: els.hp.base.value.trim() || "distilbert-base-uncased",
    max_length: +els.hp.ml.value,
    patience: +els.hp.pat.value,
  };
  if (!body.dataset) { alert("Pick a dataset first."); return; }
  try {
    const s = await jpost("/api/train/start", body);
    applyState(s);
    if (!pollTimer) pollStatus();
  } catch (e) {
    alert(`Could not start: ${e.message}`);
  }
}

async function onTrainStop() {
  if (!confirm("Stop training? Latest saved checkpoint is preserved.")) return;
  try {
    applyState(await jpost("/api/train/stop", {}));
  } catch (e) { alert(e.message); }
}

// ------- wire-up ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  els.refreshModels.addEventListener("click", refreshModels);
  els.refreshDatasets.addEventListener("click", refreshDatasets);
  els.modelSelect.addEventListener("change", updateModelMeta);
  els.predictBtn.addEventListener("click", onPredict);
  els.predictInput.addEventListener("keydown", (e) => { if (e.key === "Enter") onPredict(); });
  els.datasetUpload.addEventListener("change", onUpload);
  els.trainStart.addEventListener("click", onTrainStart);
  els.trainStop.addEventListener("click", onTrainStop);

  await Promise.all([refreshModels(), refreshDatasets()]);
  pollStatus();
});
