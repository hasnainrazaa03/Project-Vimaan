// Project Vimaan dashboard — vanilla JS, single-file.

const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  "set heading to two seven zero",
  "tune comm one to one two one decimal five",
  "climb to flight level three five zero",
  "lower the landing gear",
  "set altitude one zero thousand",
  "engage autopilot one",
];

const els = {
  // nav / chrome
  tabs: document.querySelectorAll(".tab"),
  panels: { inference: $("panel-inference"), training: $("panel-training") },
  themeToggle: $("theme-toggle"),
  connState: $("conn-state"),

  // stats
  statModels: $("stat-models"),
  statLatest: $("stat-latest"),
  statIntents: $("stat-intents"),
  statRows: $("stat-rows"),

  // inference
  modelSelect: $("model-select"),
  modelMeta: $("model-meta"),
  refreshModels: $("refresh-models"),
  predictInput: $("predict-input"),
  predictBtn: $("predict-btn"),
  predictResult: $("predict-result"),
  exampleChips: $("example-chips"),
  modelDetail: $("model-detail"),
  modelDetailVersion: $("model-detail-version"),

  // training
  datasetSelect: $("dataset-select"),
  refreshDatasets: $("refresh-datasets"),
  datasetUpload: $("dataset-upload"),
  dropzone: $("dropzone"),
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
  copyLog: $("copy-log"),
  chartCanvas: $("loss-chart"),
  progressWrap: $("progress-wrap"),
  progressBar: $("progress-bar"),
  progressText: $("progress-text"),
  elapsed: $("elapsed"),

  toasts: $("toasts"),
};

let chart = null;
let pollTimer = null;
let modelCache = [];
let trainStartedAt = null;
let elapsedTimer = null;
let lastResult = null;

// ------- helpers ------------------------------------------------------------

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function fmtInt(n) {
  return typeof n === "number" ? n.toLocaleString() : n;
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

function toast(msg, kind = "info") {
  const icons = { info: "ℹ", ok: "✓", err: "✕", warn: "⚠" };
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<span class="t-ic">${icons[kind] || icons.info}</span><span>${msg}</span>`;
  els.toasts.appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    el.addEventListener("animationend", () => el.remove(), { once: true });
  }, 3600);
}

// ------- tabs + theme -------------------------------------------------------

function setTab(name) {
  els.tabs.forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
  Object.entries(els.panels).forEach(([k, p]) => p.classList.toggle("is-active", k === name));
}

function initTheme() {
  const saved = localStorage.getItem("vimaan-theme") || "dark";
  document.documentElement.dataset.theme = saved;
}
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("vimaan-theme", next);
}

// ------- model + dataset catalogues -----------------------------------------

async function refreshModels() {
  let models = [];
  try {
    ({ models } = await jget("/api/models"));
    setConn(true);
  } catch (e) {
    setConn(false);
    toast("Could not load models", "err");
    return;
  }
  modelCache = models;
  els.modelSelect.innerHTML = "";
  if (!models.length) {
    const opt = document.createElement("option");
    opt.textContent = "— no trained models found —";
    opt.disabled = true;
    els.modelSelect.appendChild(opt);
    els.modelMeta.textContent = "train a model first";
    updateStats();
    renderModelDetail();
    return;
  }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.path;
    opt.textContent = `${m.version}${m.has_manifest ? "" : "  (no manifest)"}`;
    els.modelSelect.appendChild(opt);
  }
  updateModelMeta();
  updateStats();
  renderModelDetail();
}

function currentModel() {
  return modelCache.find((x) => x.path === els.modelSelect.value);
}

function updateModelMeta() {
  const m = currentModel();
  if (!m) { els.modelMeta.textContent = ""; return; }
  if (m.manifest) {
    const hp = m.manifest.hyperparams || {};
    const rc = m.manifest.row_count ?? "?";
    els.modelMeta.textContent =
      `tf ${m.manifest.framework_versions?.transformers || "?"} · ` +
      `${fmtInt(rc)} rows · ${hp.epochs ?? "?"} epochs`;
  } else {
    els.modelMeta.textContent = `${m.version} · no manifest`;
  }
}

function renderModelDetail() {
  const m = currentModel();
  els.modelDetailVersion.textContent = m ? m.version : "—";
  if (!m) { els.modelDetail.innerHTML = `<p class="muted">No model selected.</p>`; return; }
  if (!m.manifest) {
    els.modelDetail.innerHTML = `<p class="muted">No <code>train_manifest.json</code> for ${m.version}. Run the backfill script to add provenance.</p>`;
    return;
  }
  const mf = m.manifest;
  const hp = mf.hyperparams || {};
  const fw = mf.framework_versions || {};
  const counts = mf.intent_counts || {};
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = entries.length ? entries[0][1] : 1;
  const bars = entries.slice(0, 8).map(([name, n]) => `
    <div class="ib-row">
      <span class="ib-name" title="${name}">${name}</span>
      <span class="ib-track"><span class="ib-fill" style="width:${(n / max * 100).toFixed(1)}%"></span></span>
      <span class="ib-val">${fmtInt(n)}</span>
    </div>`).join("");

  els.modelDetail.innerHTML = `
    <dl class="kv">
      <dt>Dataset</dt><dd>${(mf.dataset_path || "?").split("/").pop()}</dd>
      <dt>Rows</dt><dd>${fmtInt(mf.row_count ?? "?")}</dd>
      <dt>Intents</dt><dd>${Object.keys(counts).length || "?"}</dd>
      <dt>Base</dt><dd>${hp.base_model ?? "?"}</dd>
      <dt>Epochs · LR</dt><dd>${hp.epochs ?? "?"} · ${hp.lr ?? "?"}</dd>
      <dt>Stack</dt><dd>torch ${fw.torch ?? "?"} · tf ${fw.transformers ?? "?"}</dd>
      <dt>Git</dt><dd>${mf.git_sha ?? "?"}${hp.backfilled ? " <span class=\"muted\">(backfilled)</span>" : ""}</dd>
    </dl>
    ${entries.length ? `<div class="section-label">Intent distribution (top 8)</div><div class="intent-bars">${bars}</div>` : ""}
  `;
}

function updateStats() {
  els.statModels.textContent = modelCache.length || "0";
  const latest = modelCache[0];
  els.statLatest.textContent = latest ? latest.version : "—";
  const mf = latest && latest.manifest;
  els.statIntents.textContent = mf && mf.intent_counts ? Object.keys(mf.intent_counts).length : "—";
  els.statRows.textContent = mf && mf.row_count != null ? fmtInt(mf.row_count) : "—";
}

async function refreshDatasets() {
  let datasets = [];
  try { ({ datasets } = await jget("/api/datasets")); }
  catch { toast("Could not load datasets", "err"); return; }
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

// ------- example chips ------------------------------------------------------

function buildExampleChips() {
  for (const ex of EXAMPLES) {
    const b = document.createElement("button");
    b.className = "ex-chip";
    b.type = "button";
    b.textContent = ex;
    b.addEventListener("click", () => {
      els.predictInput.value = ex;
      els.predictInput.focus();
      onPredict();
    });
    els.exampleChips.appendChild(b);
  }
}

// ------- prediction ---------------------------------------------------------

function renderPrediction(r) {
  lastResult = r;
  const slots = r.slots || {};
  const chips = Object.entries(slots)
    .map(([k, v]) => `<span class="chip"><span class="k">${k}</span><span class="v">${v}</span></span>`)
    .join("");
  const conf = (r.confidence * 100);
  els.predictResult.classList.remove("empty");
  els.predictResult.innerHTML = `
    <div class="intent-row">
      <span class="intent">${r.intent}</span>
      <span class="conf-num">${conf.toFixed(1)}%</span>
    </div>
    <div class="conf-bar"><div class="conf-fill" id="conf-fill"></div></div>
    <div class="slots">${chips || '<span class="no-slots">no slots detected</span>'}</div>
    <div class="result-foot">
      <button class="linklike" id="toggle-json">show raw JSON</button>
    </div>
    <pre class="raw-json" id="raw-json" hidden></pre>
  `;
  // animate the confidence bar after paint
  requestAnimationFrame(() => { $("conf-fill").style.width = `${conf}%`; });
  $("toggle-json").addEventListener("click", () => {
    const pre = $("raw-json");
    const btn = $("toggle-json");
    if (pre.hidden) { pre.textContent = JSON.stringify(r, null, 2); pre.hidden = false; btn.textContent = "hide raw JSON"; }
    else { pre.hidden = true; btn.textContent = "show raw JSON"; }
  });
}

async function onPredict() {
  const text = els.predictInput.value.trim();
  const modelPath = els.modelSelect.value;
  if (!text) { toast("Type a command first", "warn"); return; }
  if (!modelPath) { toast("No model selected", "warn"); return; }
  els.predictBtn.classList.add("is-loading");
  els.predictBtn.disabled = true;
  try {
    const r = await jpost("/api/predict", { model_path: modelPath, text });
    renderPrediction(r);
  } catch (e) {
    toast(`Predict failed: ${e.message}`, "err");
    els.predictResult.classList.add("empty");
    els.predictResult.innerHTML = `<span class="result-hint">error: ${e.message}</span>`;
  } finally {
    els.predictBtn.classList.remove("is-loading");
    els.predictBtn.disabled = false;
  }
}

// ------- upload (button + drag/drop) ----------------------------------------

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.endsWith(".jsonl")) { toast("Only .jsonl files are accepted", "warn"); return; }
  els.uploadMsg.textContent = `uploading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/datasets/upload", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    els.uploadMsg.textContent = `✓ ${data.name} (${fmtInt(data.rows)} rows)`;
    toast(`Uploaded ${data.name} (${fmtInt(data.rows)} rows)`, "ok");
    await refreshDatasets();
    els.datasetSelect.value = data.path;
  } catch (err) {
    els.uploadMsg.textContent = "";
    toast(`Upload failed: ${err.message}`, "err");
  }
}

function wireDropzone() {
  const dz = els.dropzone;
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    uploadFile(file);
  });
  els.datasetUpload.addEventListener("change", (e) => {
    uploadFile(e.target.files[0]);
    e.target.value = "";
  });
}

// ------- training -----------------------------------------------------------

function ensureChart() {
  if (chart) return chart;
  chart = new Chart(els.chartCanvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "train loss", data: [], borderColor: "#38bdf8", backgroundColor: "rgba(56,189,248,.14)", tension: .3, fill: true, pointRadius: 3, pointHoverRadius: 5 },
        { label: "val loss", data: [], borderColor: "#818cf8", backgroundColor: "rgba(129,140,248,.14)", tension: .3, fill: true, pointRadius: 3, pointHoverRadius: 5 },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 350 },
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { color: "#9fb0cf", usePointStyle: true, boxWidth: 8 } } },
      scales: {
        x: { title: { display: true, text: "epoch", color: "#7d8aa6" }, ticks: { color: "#9fb0cf" }, grid: { color: "rgba(148,163,184,.08)" } },
        y: { title: { display: true, text: "loss", color: "#7d8aa6" }, ticks: { color: "#9fb0cf" }, grid: { color: "rgba(148,163,184,.08)" } },
      },
    },
  });
  return chart;
}

function fmtElapsed(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function startElapsed() {
  stopElapsed();
  trainStartedAt = trainStartedAt || Date.now();
  elapsedTimer = setInterval(() => {
    els.elapsed.textContent = fmtElapsed(Date.now() - trainStartedAt);
  }, 1000);
}
function stopElapsed() { if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; } }

function applyState(s) {
  // status pill
  els.trainStatusPill.dataset.status = s.status;
  els.trainStatusPill.querySelector(".label").textContent = s.status;
  els.trainPid.textContent = s.pid ? `pid ${s.pid}` : "";
  els.trainStop.disabled = s.status !== "running";
  els.trainStart.disabled = s.status === "running";

  // log
  els.trainLog.textContent = (s.log_tail || []).join("\n");
  els.trainLog.scrollTop = els.trainLog.scrollHeight;

  // chart
  const c = ensureChart();
  c.data.labels = s.metrics.map((m) => m.epoch);
  c.data.datasets[0].data = s.metrics.map((m) => m.train_loss ?? null);
  c.data.datasets[1].data = s.metrics.map((m) => m.val_loss ?? null);
  c.update("none");

  // summary + progress
  const last = s.metrics[s.metrics.length - 1];
  const ckptCount = (s.checkpoints || []).length;
  els.trainSummary.textContent = last
    ? `epoch ${last.epoch} · train=${last.train_loss?.toFixed?.(4) ?? "—"} · val=${last.val_loss?.toFixed?.(4) ?? "—"} · saved ${ckptCount}×`
    : "no data yet";

  const totalEpochs = +els.hp.epochs.value || 10;
  if (s.status === "running") {
    els.progressWrap.hidden = false;
    const done = last ? last.epoch : 0;
    const pct = Math.min(100, (done / totalEpochs) * 100);
    els.progressBar.style.width = `${pct}%`;
    els.progressText.textContent = `epoch ${done} / ${totalEpochs}`;
    startElapsed();
  } else {
    stopElapsed();
    if (s.status === "finished") { els.progressBar.style.width = "100%"; els.progressText.textContent = "complete"; }
    else if (s.status === "failed") { els.progressText.textContent = "failed"; }
  }
}

async function pollStatus() {
  try {
    const s = await jget("/api/train/status");
    setConn(true);
    const prev = els.trainStatusPill.dataset.status;
    applyState(s);
    if (s.status === "running") {
      pollTimer = setTimeout(pollStatus, 1500);
    } else {
      pollTimer = null;
      if (prev === "running" && s.status === "finished") { toast("Training finished — new checkpoint saved", "ok"); }
      if (prev === "running" && s.status === "failed") { toast("Training failed — check the log", "err"); }
      if (prev === "running") { trainStartedAt = null; await refreshModels(); }
    }
  } catch {
    setConn(false);
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
  if (!body.dataset) { toast("Pick a dataset first", "warn"); return; }
  try {
    trainStartedAt = Date.now();
    const s = await jpost("/api/train/start", body);
    toast("Training started", "ok");
    applyState(s);
    if (!pollTimer) pollStatus();
  } catch (e) {
    trainStartedAt = null;
    toast(`Could not start: ${e.message}`, "err");
  }
}

async function onTrainStop() {
  if (!confirm("Stop training? The latest saved checkpoint is preserved.")) return;
  try {
    applyState(await jpost("/api/train/stop", {}));
    toast("Training stopped", "warn");
  } catch (e) { toast(e.message, "err"); }
}

// ------- connection indicator ----------------------------------------------

function setConn(ok) {
  els.connState.textContent = ok ? "● connected" : "● disconnected";
  els.connState.classList.toggle("down", !ok);
}

// ------- wire-up ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  buildExampleChips();
  wireDropzone();

  els.tabs.forEach((t) => t.addEventListener("click", () => setTab(t.dataset.tab)));
  els.themeToggle.addEventListener("click", toggleTheme);
  els.refreshModels.addEventListener("click", () => { refreshModels(); toast("Models reloaded", "ok"); });
  els.refreshDatasets.addEventListener("click", () => { refreshDatasets(); toast("Datasets reloaded", "ok"); });
  els.modelSelect.addEventListener("change", () => { updateModelMeta(); renderModelDetail(); });
  els.predictBtn.addEventListener("click", onPredict);
  els.predictInput.addEventListener("keydown", (e) => { if (e.key === "Enter") onPredict(); });
  els.trainStart.addEventListener("click", onTrainStart);
  els.trainStop.addEventListener("click", onTrainStop);
  els.copyLog.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(els.trainLog.textContent || ""); toast("Log copied", "ok"); }
    catch { toast("Copy failed", "err"); }
  });

  await Promise.all([refreshModels(), refreshDatasets()]);
  pollStatus();
});
