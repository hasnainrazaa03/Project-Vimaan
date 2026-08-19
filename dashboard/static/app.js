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
  valCanvas: $("val-chart"),
  runBadge: $("run-badge"),
  splitBar: $("split-bar"),
  splitLegend: $("split-legend"),
  runStats: $("run-stats"),
  intentDistLabel: $("intent-dist-label"),
  runIntents: $("run-intents"),
  valmetBadge: $("valmet-badge"),
  progressWrap: $("progress-wrap"),
  progressBar: $("progress-bar"),
  progressText: $("progress-text"),
  elapsed: $("elapsed"),

  toasts: $("toasts"),
};

let chart = null;
let valChart = null;
let lastMetrics = null;
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

function quantile(arr, q) {
  if (!arr.length) return 0;
  const a = arr.slice().sort((x, y) => x - y);
  const i = (a.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

// theme-aware chart ink
function ink() {
  const cs = getComputedStyle(document.documentElement);
  return {
    grid: "rgba(148,163,184,.12)",
    tick: cs.getPropertyValue("--fg-dim").trim() || "#93a1bd",
    accent: cs.getPropertyValue("--accent").trim() || "#38bdf8",
    accent2: cs.getPropertyValue("--accent-2").trim() || "#818cf8",
    ok: cs.getPropertyValue("--ok").trim() || "#34d399",
    warn: cs.getPropertyValue("--warn").trim() || "#fbbf24",
  };
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
  // rebuild charts so their axis/legend ink picks up the new theme
  if (chart) { chart.destroy(); chart = null; }
  if (valChart) { valChart.destroy(); valChart = null; }
  if (lastMetrics) renderMetrics(lastMetrics);
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

// Loss: per-step train loss (dense) + per-epoch train/val markers, adaptive Y.
function ensureChart() {
  if (chart) return chart;
  const k = ink();
  chart = new Chart(els.chartCanvas, {
    type: "line",
    data: {
      datasets: [
        { label: "train (step)", data: [], parsing: false, borderColor: k.accent, borderWidth: 1.3, pointRadius: 0, tension: .2 },
        { label: "train (epoch)", data: [], parsing: false, borderColor: k.accent2, borderWidth: 2, pointRadius: 3, showLine: true, tension: .2 },
        { label: "val (epoch)", data: [], parsing: false, borderColor: k.warn, borderWidth: 2, pointRadius: 3, showLine: true, tension: .2 },
      ],
    },
    options: {
      responsive: true, animation: false, interaction: { mode: "nearest", intersect: false },
      plugins: { legend: { labels: { color: k.tick, usePointStyle: true, boxWidth: 8 } } },
      scales: {
        x: { type: "linear", title: { display: true, text: "step", color: k.tick }, ticks: { color: k.tick, maxTicksLimit: 6, callback: (v) => Number(v).toLocaleString() }, grid: { color: k.grid } },
        y: { min: 0, title: { display: true, text: "loss", color: k.tick }, ticks: { color: k.tick }, grid: { color: k.grid } },
      },
    },
  });
  return chart;
}

// Validation: per-epoch intent accuracy + slot F1, Y zoomed to the data.
function ensureValChart() {
  if (valChart) return valChart;
  const k = ink();
  valChart = new Chart(els.valCanvas, {
    type: "line",
    data: {
      datasets: [
        { label: "intent accuracy", data: [], parsing: false, borderColor: k.accent, borderWidth: 2, pointRadius: 3, tension: .2 },
        { label: "slot F1 (macro)", data: [], parsing: false, borderColor: k.ok, borderWidth: 2, pointRadius: 3, tension: .2 },
      ],
    },
    options: {
      responsive: true, animation: false, interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { labels: { color: k.tick, usePointStyle: true, boxWidth: 8 } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${(c.parsed.y * 100).toFixed(2)}%` } },
      },
      scales: {
        x: { type: "linear", title: { display: true, text: "epoch", color: k.tick }, ticks: { color: k.tick, stepSize: 1, precision: 0 }, grid: { color: k.grid } },
        y: { ticks: { color: k.tick, callback: (v) => (v * 100).toFixed(0) + "%" }, grid: { color: k.grid } },
      },
    },
  });
  return valChart;
}

const SPLIT_COLORS = ["#38bdf8", "#818cf8", "#c084fc"];

function renderMetrics(m) {
  lastMetrics = m;
  if (!m || m.waiting || !m.meta) {
    els.runBadge.textContent = "no run yet";
    return;
  }
  const meta = m.meta, steps = m.steps || [], epochs = m.epochs || [], done = m.done;
  const lastStep = steps.length ? steps[steps.length - 1] : null;
  const lastEp = epochs.length ? epochs[epochs.length - 1] : null;
  const best = epochs.reduce((a, e) => Math.min(a, e.val_loss), Infinity);

  els.runBadge.textContent = `${m.run} · ${done ? "done" : meta.device}`;

  // dataset split
  const sp = meta.split, tot = sp.train + sp.val + sp.test;
  els.splitBar.hidden = false;
  const segs = [["train", sp.train], ["val", sp.val], ["test", sp.test]];
  els.splitBar.innerHTML = segs.map((s, i) => {
    const pct = 100 * s[1] / tot;
    return `<div style="width:${pct}%;background:${SPLIT_COLORS[i]}">${pct > 8 ? Math.round(pct) + "%" : ""}</div>`;
  }).join("");
  els.splitLegend.innerHTML = segs.map((s, i) =>
    `<span><span style="color:${SPLIT_COLORS[i]}">■</span> ${s[0]} <b>${s[1].toLocaleString()}</b></span>`).join("");

  // run stats
  const rows = [
    ["device", meta.device],
    ["epoch", `${lastEp ? lastEp.epoch : 0} / ${meta.epochs}`],
    ["step", (lastStep ? lastStep.step : 0).toLocaleString()],
    ["best val", isFinite(best) ? best.toFixed(4) : "—"],
    ["intent acc", lastEp ? (lastEp.val_intent_acc * 100).toFixed(2) + "%" : "—"],
    ["slot F1", lastEp ? lastEp.val_slot_f1.toFixed(3) : "—"],
  ];
  els.runStats.innerHTML = rows.map(([k, v]) => `<div class="rs"><small>${k}</small><b>${v}</b></div>`).join("");

  // intent distribution
  const dist = Object.entries(meta.intent_dist || {}).slice(0, 10);
  els.intentDistLabel.hidden = !dist.length;
  const mx = dist.length ? dist[0][1] : 1;
  els.runIntents.innerHTML = dist.map(([k, v]) => `
    <div class="ib-row">
      <span class="ib-name" title="${k}">${k}</span>
      <span class="ib-track"><span class="ib-fill" style="width:${(v / mx * 100).toFixed(1)}%"></span></span>
      <span class="ib-val">${v.toLocaleString()}</span>
    </div>`).join("");

  // loss chart
  const spe = meta.steps_per_epoch || 1;
  const c = ensureChart();
  c.data.datasets[0].data = steps.map((s) => ({ x: s.step, y: s.loss }));
  c.data.datasets[1].data = epochs.map((e) => ({ x: e.epoch * spe, y: e.train_loss }));
  c.data.datasets[2].data = epochs.map((e) => ({ x: e.epoch * spe, y: e.val_loss }));
  const ys = c.data.datasets.flatMap((d) => d.data.map((p) => p.y)).filter((v) => isFinite(v));
  c.options.scales.y.suggestedMax = ys.length ? quantile(ys, 0.97) * 1.25 : undefined;
  c.update("none");

  // validation-metrics chart, Y zoomed to data
  const vc = ensureValChart();
  vc.data.datasets[0].data = epochs.map((e) => ({ x: e.epoch, y: e.val_intent_acc }));
  vc.data.datasets[1].data = epochs.map((e) => ({ x: e.epoch, y: e.val_slot_f1 }));
  const vy = epochs.flatMap((e) => [e.val_intent_acc, e.val_slot_f1]).filter((v) => isFinite(v));
  if (vy.length) {
    const lo = Math.min(...vy), hi = Math.max(...vy), pad = Math.max((hi - lo) * 0.6, 0.02);
    vc.options.scales.y.min = Math.max(0, lo - pad);
    vc.options.scales.y.max = Math.min(1, hi + pad);
  }
  vc.update("none");

  // badges
  els.trainSummary.textContent = lastEp
    ? `epoch ${lastEp.epoch} · train ${lastEp.train_loss.toFixed(4)} · val ${lastEp.val_loss.toFixed(4)}`
    : "no data yet";
  els.valmetBadge.textContent = lastEp
    ? `acc ${(lastEp.val_intent_acc * 100).toFixed(1)}% · F1 ${lastEp.val_slot_f1.toFixed(3)}`
    : "—";
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

  // progress (charts + summary come from renderMetrics via the live stream)
  const last = s.metrics[s.metrics.length - 1];
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
    const [s, m] = await Promise.all([
      jget("/api/train/status"),
      jget("/api/train/metrics").catch(() => ({ waiting: true })),
    ]);
    setConn(true);
    const prev = els.trainStatusPill.dataset.status;
    applyState(s);
    renderMetrics(m);
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
