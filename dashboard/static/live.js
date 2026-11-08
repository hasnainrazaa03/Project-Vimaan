// Live training view. One run's metrics stream (ML/training_runs/<v>/metrics.jsonl),
// drawn as it arrives over server-sent events, with an earlier run to compare.
// Uses app.js globals: $, jget, toast, fmtInt.

(() => {
  // ------- shared chart look (also used by evals.js) -------------------------

  function tokens() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
    return {
      s1: v("--series-1", "#3987e5"),
      s2: v("--series-2", "#d95926"),
      s3: v("--series-3", "#199e70"),
      base: v("--series-base", "#8b93a7"),
      grid: v("--grid", "rgba(148,163,184,.14)"),
      text: v("--fg", "#e8eefc"),
      text2: v("--fg-dim", "#93a1bd"),
      surface: v("--surface-solid", "#131a2c"),
    };
  }

  function alpha(hex, a) {
    const h = hex.replace("#", "");
    if (h.length !== 6) return hex;
    const n = parseInt(h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function logTick(v) {
    // Only label powers of ten on a log axis; the rest is clutter.
    const e = Math.log10(v);
    if (Math.abs(e - Math.round(e)) > 1e-9) return "";
    return v >= 0.01 ? String(+v.toPrecision(2)) : `1e${Math.round(e)}`;
  }

  function pctTick(v, i, ticks) {
    const step = ticks.length > 1 ? Math.abs(ticks[1].value - ticks[0].value) * 100 : 1;
    const d = Math.min(3, Math.max(0, Math.ceil(-Math.log10(step) + 0.001)));
    return `${(v * 100).toFixed(d)}%`;
  }

  function axes(k, { xTitle, yTitle, log = false, percent = false, xInt = false }) {
    return {
      x: {
        type: "linear",
        title: { display: true, text: xTitle, color: k.text2 },
        ticks: { color: k.text2, maxTicksLimit: 11, precision: 0, stepSize: xInt ? 1 : undefined },
        grid: { color: k.grid },
        border: { color: k.grid },
      },
      y: {
        type: log ? "logarithmic" : "linear",
        title: { display: !!yTitle, text: yTitle, color: k.text2 },
        ticks: {
          color: k.text2,
          maxTicksLimit: 7,
          callback: percent ? pctTick : log ? logTick : undefined,
        },
        grid: { color: k.grid },
        border: { color: k.grid },
      },
    };
  }

  function chartOptions(k, scales, { tooltipMode = "index", fmt = (v) => v } = {}) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      normalized: true,
      interaction: { mode: tooltipMode, axis: "x", intersect: false },
      plugins: {
        legend: {
          display: true,
          position: "top",
          align: "start",
          labels: {
            color: k.text2,
            boxWidth: 16,
            boxHeight: 3,
            padding: 14,
            // A comparison series with nothing to compare is left out, not struck through.
            filter: (item, data) => !data.datasets[item.datasetIndex]._absent,
          },
        },
        tooltip: {
          backgroundColor: k.surface,
          borderColor: k.grid,
          borderWidth: 1,
          titleColor: k.text,
          bodyColor: k.text,
          callbacks: { label: (c) => ` ${c.dataset.label}: ${fmt(c.parsed.y)}` },
        },
        decimation: { enabled: false },
      },
      scales,
    };
  }

  function line(label, color, { width = 2, points = 0, dash, fill } = {}) {
    return {
      label,
      data: [],
      borderColor: color,
      backgroundColor: fill || color,
      borderWidth: width,
      pointRadius: points,
      pointHoverRadius: points ? points + 2 : 4,
      pointBackgroundColor: color,
      pointBorderColor: tokens().surface,
      pointBorderWidth: points ? 2 : 0,
      borderDash: dash,
      tension: 0.25,
      spanGaps: true,
    };
  }

  window.vimaanViz = { tokens, alpha, axes, chartOptions, line, logTick, pctTick };

  // ------- formatting ------------------------------------------------------

  const fmtLoss = (v) =>
    v == null || !isFinite(v) ? "—" : v < 0.001 ? v.toExponential(2) : v.toFixed(4);
  const pct = (v, d = 2) => (v == null || !isFinite(v) ? "—" : `${(v * 100).toFixed(d)}%`);

  function fmtDur(sec) {
    if (sec == null || !isFinite(sec)) return "—";
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
    if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
    return `${s}s`;
  }

  const STATUS_LABEL = {
    running: "live",
    finished: "finished",
    failed: "failed",
    stopped: "stopped",
    interrupted: "interrupted",
  };

  // ------- state ------------------------------------------------------------

  const st = {
    runs: [],
    follow: true, // "Latest" picked: switch to a newer run when it appears
    current: null, // run name
    baseline: null, // run name, or "" for none
    baselineAuto: true,
    es: null,
    spe: 1,
    summary: null,
    meta: null,
    epochs: [],
    raw: [], // {x, y} per step (x = epochs, fractional)
    ema: [],
    grads: [],
    base: null, // {steps, epochs, spe}
    serverLog: [],
    managerLog: null,
    drawQueued: false,
  };
  const charts = {};

  const el = (id) => document.getElementById(id);

  // ------- charts -------------------------------------------------------------

  function logScale() {
    return el("log-scale")?.checked ?? true;
  }

  function buildCharts() {
    Object.values(charts).forEach((c) => c.destroy());
    const k = tokens();
    const baseName = st.baseline ? ` (${st.baseline})` : "";

    charts.step = new Chart(el("step-chart"), {
      type: "line",
      data: {
        datasets: [
          { ...line("each step", alpha(k.s1, 0.16), { width: 0.75 }), tension: 0 },
          line("smoothed", k.s1, { width: 2.25 }),
          line(`smoothed${baseName}`, k.base, { width: 1.5 }),
        ],
      },
      options: chartOptions(k, axes(k, { xTitle: "epoch", yTitle: "loss", log: logScale() }), {
        tooltipMode: "nearest",
        fmt: fmtLoss,
      }),
    });

    charts.epoch = new Chart(el("epoch-chart"), {
      type: "line",
      data: {
        datasets: [
          line("train", k.s1, { points: 4 }),
          line("validation", k.s2, { points: 4 }),
          line(`validation${baseName}`, k.base, { width: 1.5, points: 3 }),
        ],
      },
      options: chartOptions(k, axes(k, { xTitle: "epoch", yTitle: "loss", log: logScale(), xInt: true }), {
        fmt: fmtLoss,
      }),
    });

    charts.val = new Chart(el("val-chart"), {
      type: "line",
      data: {
        datasets: [
          line("intent accuracy", k.s1, { points: 4 }),
          line("slot F1", k.s2, { points: 4 }),
          line(`intent accuracy${baseName}`, k.base, { width: 1.5, points: 3 }),
          line(`slot F1${baseName}`, k.base, { width: 1.5, points: 3, dash: [4, 3] }),
        ],
      },
      options: chartOptions(k, axes(k, { xTitle: "epoch", percent: true, xInt: true }), {
        fmt: (v) => pct(v, 3),
      }),
    });

    charts.grad = new Chart(el("grad-chart"), {
      type: "line",
      data: { datasets: [line("gradient norm", k.s1, { width: 1.5 })] },
      options: chartOptions(k, axes(k, { xTitle: "epoch", yTitle: "L2 norm", log: logScale() }), {
        tooltipMode: "nearest",
        fmt: (v) => (v == null ? "—" : v.toPrecision(3)),
      }),
    });
    charts.grad.options.plugins.legend.display = false; // one series: the title names it

    fillCharts();
  }

  function fillCharts() {
    if (!charts.step) return;
    charts.step.data.datasets[0].data = st.raw;
    charts.step.data.datasets[1].data = st.ema;
    charts.grad.data.datasets[0].data = st.grads;
    const ep = st.epochs;
    const best = bestEpoch(ep);
    charts.epoch.data.datasets[0].data = ep.map((e) => ({ x: e.epoch, y: e.train_loss }));
    charts.epoch.data.datasets[1].data = ep.map((e) => ({ x: e.epoch, y: e.val_loss }));
    charts.epoch.data.datasets[1].pointRadius = ep.map((e) => (e.epoch === best ? 7 : 4));
    charts.val.data.datasets[0].data = ep.map((e) => ({ x: e.epoch, y: e.val_intent_acc }));
    charts.val.data.datasets[1].data = ep.map((e) => ({ x: e.epoch, y: e.val_slot_f1 }));

    const b = st.base;
    charts.step.data.datasets[2].data = b ? b.steps.map((s) => ({ x: s.step / b.spe, y: s.ema ?? s.loss })) : [];
    charts.epoch.data.datasets[2].data = b ? b.epochs.map((e) => ({ x: e.epoch, y: e.val_loss })) : [];
    charts.val.data.datasets[2].data = b ? b.epochs.map((e) => ({ x: e.epoch, y: e.val_intent_acc })) : [];
    charts.val.data.datasets[3].data = b ? b.epochs.map((e) => ({ x: e.epoch, y: e.val_slot_f1 })) : [];
    for (const ds of [charts.step.data.datasets[2], charts.epoch.data.datasets[2], charts.val.data.datasets[2], charts.val.data.datasets[3]]) {
      ds._absent = !b;
      ds.hidden = !b;
    }

    // Validation chart: zoom the y axis to the data (it lives near 100%).
    const vals = [...ep, ...(b ? b.epochs : [])]
      .flatMap((e) => [e.val_intent_acc, e.val_slot_f1])
      .filter((v) => v != null && isFinite(v));
    if (vals.length) {
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const pad = Math.max((hi - lo) * 0.15, 0.002);
      charts.val.options.scales.y.min = Math.max(0, lo - pad);
      charts.val.options.scales.y.max = Math.min(1, hi + pad);
    }
    const planned = st.summary?.planned_epochs || 0;
    for (const c of [charts.step, charts.epoch, charts.val, charts.grad]) {
      c.options.scales.x.min = c === charts.epoch || c === charts.val ? 1 : 0;
      c.options.scales.x.suggestedMax = Math.max(planned, b ? b.epochs.length : 0, 1);
    }
    queueDraw();
  }

  function queueDraw() {
    if (st.drawQueued) return;
    st.drawQueued = true;
    requestAnimationFrame(() => {
      st.drawQueued = false;
      Object.values(charts).forEach((c) => c.update("none"));
    });
  }

  function bestEpoch(epochs) {
    let best = null;
    for (const e of epochs) if (best == null || e.val_loss < best.val_loss) best = e;
    return best ? best.epoch : null;
  }

  // Thin the oldest half when a live curve gets long, so the chart stays fast.
  function cap(arr, limit = 4000) {
    if (arr.length <= limit) return arr;
    const half = Math.floor(arr.length / 2);
    return arr.slice(0, half).filter((_, i) => i % 2 === 0).concat(arr.slice(half));
  }

  // ------- data ---------------------------------------------------------------

  function loadSnapshot(snap) {
    st.summary = snap.summary;
    st.meta = snap.meta || {};
    st.spe = snap.summary.steps_per_epoch || st.meta.steps_per_epoch || 1;
    st.epochs = snap.epochs || [];
    st.raw = (snap.steps || []).map((s) => ({ x: s.step / st.spe, y: s.loss }));
    st.ema = (snap.steps || []).filter((s) => s.ema != null).map((s) => ({ x: s.step / st.spe, y: s.ema }));
    st.grads = (snap.grad_norms || []).map((s) => ({ x: s.step / st.spe, y: s.grad_norm }));
    st.serverLog = snap.log_tail || [];
    fillCharts();
    renderAll();
  }

  function applyRecords(records) {
    let epochChanged = false;
    for (const r of records) {
      if (r.type === "step") {
        const x = r.step / st.spe;
        st.raw.push({ x, y: r.loss });
        if (r.ema != null) st.ema.push({ x, y: r.ema });
        if (r.grad_norm != null) st.grads.push({ x, y: r.grad_norm });
      } else if (r.type === "epoch") {
        st.epochs.push(r);
        epochChanged = true;
      } else if (r.type === "meta") {
        st.meta = r;
      }
    }
    st.raw = cap(st.raw);
    st.ema = cap(st.ema);
    if (epochChanged) {
      fillCharts();
      renderEpochDetail();
    } else {
      charts.step.data.datasets[0].data = st.raw;
      charts.step.data.datasets[1].data = st.ema;
      charts.grad.data.datasets[0].data = st.grads;
      queueDraw();
    }
  }

  // ------- rendering ----------------------------------------------------------

  function renderAll() {
    renderSummary();
    renderEpochDetail();
    renderRunData();
    renderLog();
  }

  function renderSummary() {
    const s = st.summary;
    const chip = el("live-chip");
    if (!s) {
      chip.dataset.status = "none";
      chip.querySelector(".label").textContent = "no run";
      el("kpis").innerHTML = "";
      el("live-note").textContent = "No training runs yet. Start one below, or from a terminal.";
      return;
    }
    chip.dataset.status = s.status;
    chip.querySelector(".label").textContent = `${s.run} · ${STATUS_LABEL[s.status] || s.status}`;
    el("live-phase").textContent = s.phase ? `${s.phase}…` : "";

    const running = s.status === "running";
    const sinceUpdate = Math.max(0, Date.now() / 1000 - (s.updated || 0));
    const elapsed = running ? s.elapsed + sinceUpdate : s.elapsed;
    const spe = s.steps_per_epoch || 0;
    const planned = s.planned_epochs || 0;
    const epochShown = running ? s.epoch : s.epochs_done;
    const partial = running && spe && s.epochs_done < s.epoch ? Math.min(s.step_in_epoch, spe) / spe : 0;
    const frac = planned ? Math.min(1, (s.epochs_done + partial) / planned) : 0;

    el("rp-total").style.width = `${(frac * 100).toFixed(1)}%`;
    el("rp-total-text").textContent = planned ? `${s.epochs_done} of ${planned} epochs` : "—";
    const inEpoch = running && spe ? Math.min(1, s.step_in_epoch / spe) : s.status === "finished" ? 1 : 0;
    el("rp-epoch").style.width = `${(inEpoch * 100).toFixed(1)}%`;
    el("rp-epoch-label").textContent = running ? `Epoch ${s.epoch}` : "Last epoch";
    el("rp-epoch-text").textContent = spe
      ? running ? `${fmtInt(Math.min(s.step_in_epoch, spe))} / ${fmtInt(spe)} steps` : `${fmtInt(spe)} steps each`
      : "—";

    const tiles = [
      ["Epoch", planned ? `${epochShown} / ${planned}` : String(epochShown), s.phase || (running ? "training" : STATUS_LABEL[s.status])],
      ["Elapsed", fmtDur(elapsed), running ? "since the run started" : "total"],
      ["Time left", running && s.eta != null ? `~${fmtDur(Math.max(0, s.eta - sinceUpdate))}` : "—", running ? "if every epoch runs" : "not running"],
      ["Speed", s.rate ? `${s.rate.toFixed(1)}` : "—", s.rate ? `steps/s · ${fmtInt(Math.round(s.rate * (s.batch_size || 16)))} examples/s` : "steps per second"],
      ["Best val loss", fmtLoss(s.best_val_loss), s.best_epoch ? `epoch ${s.best_epoch}` : "no epoch yet"],
      ["Since best", s.best_epoch ? `${s.epochs_since_best}` : "—", s.patience != null ? `of ${s.patience} epochs patience` : "epochs without improvement"],
      ["Intent accuracy", pct(s.val_intent_acc), "validation, latest epoch"],
      ["Slot F1", pct(s.val_slot_f1), "validation, latest epoch"],
    ];
    el("kpis").innerHTML = tiles
      .map(([k, v, sub]) => `<div class="kpi"><small>${k}</small><strong>${v}</strong><span>${sub}</span></div>`)
      .join("");

    let note = "";
    if (s.status === "interrupted") {
      note = `No new metrics for ${fmtDur(sinceUpdate)}, and no end record. The run stopped without finishing (the process was killed or the machine slept).`;
    } else if (s.status === "failed") {
      note = `The run failed: ${s.error || "see the log"}.`;
    } else if (s.status === "stopped") {
      note = "Stopped by hand. The best checkpoint so far is kept.";
    } else if (s.status === "finished") {
      note = `Finished in ${fmtDur(s.elapsed)}: best epoch ${s.best_epoch ?? "—"} of ${s.epochs_done}${s.stopped_early ? ", stopped early (no improvement within patience)" : ""}.`;
    } else if (running) {
      note = `Training on ${s.device || "?"} · ${s.dataset || ""}. Updates arrive as the trainer writes them.`;
    }
    el("live-note").textContent = note;

    const pill = document.getElementById("train-status-pill");
    if (pill) {
      pill.dataset.status = running ? "running" : s.status === "finished" ? "finished" : s.status === "failed" ? "failed" : "idle";
      pill.querySelector(".label").textContent = running ? `${s.run} training` : "idle";
    }
  }

  // Near 100% every accuracy bar is full, so the bar shows the error rate
  // instead (zero-based; the longest is the worst); the text keeps the score.
  function bars(container, entries, { emptyText, missed = "missed" }) {
    if (!entries.length) {
      container.innerHTML = `<p class="muted">${emptyText}</p>`;
      return;
    }
    const worst = Math.max(...entries.map((e) => 1 - e.value), 1e-9);
    container.innerHTML = entries
      .map(({ name, value, n, prev }) => {
        const err = 1 - value;
        const delta = prev == null ? null : value - prev;
        const deltaText = delta == null || Math.abs(delta) < 0.00005 ? "" :
          `<span class="bl-delta" title="change since the previous epoch">${delta > 0 ? "▲" : "▼"}${(Math.abs(delta) * 100).toFixed(2)}</span>`;
        return `<div class="bl-row" title="${name}: ${pct(value, 3)} (${pct(err, 3)} ${missed}) of ${fmtInt(n)}">
          <span class="bl-name">${name}</span>
          <span class="bl-track"><span class="bl-fill err" style="width:${err > 0 ? Math.max(1.5, (err / worst) * 100).toFixed(2) : 0}%"></span></span>
          <span class="bl-val">${pct(value, value >= 0.9995 ? 1 : 2)}${deltaText}<span class="bl-sub">${err > 0 ? `${pct(err, 2)} ${missed}` : `no ${missed === "error" ? "errors" : "misses"}`}</span></span>
        </div>`;
      })
      .join("");
  }

  function renderEpochDetail() {
    const ep = st.epochs;
    const last = ep[ep.length - 1];
    const prev = ep[ep.length - 2];
    const best = bestEpoch(ep);

    el("epoch-badge").textContent = last ? `epoch ${last.epoch}: train ${fmtLoss(last.train_loss)} · val ${fmtLoss(last.val_loss)}` : "—";
    el("val-badge").textContent = last ? `intent ${pct(last.val_intent_acc)} · slot F1 ${pct(last.val_slot_f1)}` : "—";

    const intents = last && last.per_intent
      ? Object.entries(last.per_intent)
          .map(([name, r]) => ({ name, value: r.acc, n: r.n, prev: prev?.per_intent?.[name]?.acc }))
          .sort((a, b) => a.value - b.value || a.name.localeCompare(b.name))
      : [];
    bars(el("intent-bars"), intents, { emptyText: last ? "This run's trainer didn't log per-intent scores." : "Appears after the first epoch." });
    el("intent-badge").textContent = last ? `epoch ${last.epoch} · bar = error rate, worst first` : "latest epoch";

    const slots = last && last.per_slot
      ? Object.entries(last.per_slot)
          .map(([name, r]) => ({ name, value: r.f1, n: r.n, prev: prev?.per_slot?.[name]?.f1 }))
          .sort((a, b) => a.value - b.value || a.name.localeCompare(b.name))
      : [];
    bars(el("slot-bars"), slots, { emptyText: last ? "This run's trainer didn't log per-slot scores." : "Appears after the first epoch.", missed: "error" });
    el("slot-badge").textContent = last ? `epoch ${last.epoch} · bar = error rate, worst first` : "latest epoch";

    // Table view of every epoch (the charts' readable twin).
    el("epochs-badge").textContent = ep.length ? `${ep.length} epochs · best ${best}` : "—";
    let prevT = 0;
    const rows = ep.map((e) => {
      const took = e.elapsed != null ? e.elapsed - prevT : null;
      prevT = e.elapsed ?? prevT;
      return `<tr class="${e.epoch === best ? "is-best" : ""}">
        <td>${e.epoch}${e.epoch === best ? ' <span class="best-tag">best</span>' : ""}</td>
        <td>${fmtLoss(e.train_loss)}</td><td>${fmtLoss(e.val_loss)}</td>
        <td>${pct(e.val_intent_acc, 3)}</td><td>${pct(e.val_slot_f1, 3)}</td>
        <td>${fmtDur(took)}</td><td>${e.val_seconds != null ? fmtDur(e.val_seconds) : "—"}</td>
      </tr>`;
    });
    el("epoch-table").innerHTML = ep.length
      ? `<thead><tr><th>Epoch</th><th>Train loss</th><th>Val loss</th><th>Intent acc</th><th>Slot F1</th><th>Took</th><th>Validation</th></tr></thead><tbody>${rows.join("")}</tbody>`
      : `<tbody><tr><td class="muted">No epochs yet.</td></tr></tbody>`;
  }

  const SPLIT_COLORS = () => { const k = tokens(); return [k.s1, k.s2, k.s3]; };

  function renderRunData() {
    const m = st.meta || {};
    el("run-badge").textContent = st.current ? `${st.current} · ${m.dataset || ""}` : "no run yet";
    const sp = m.split;
    const bar = el("split-bar");
    if (sp) {
      const tot = sp.train + sp.val + sp.test;
      const segs = [["train", sp.train], ["validation", sp.val], ["held-out test", sp.test]];
      const colors = SPLIT_COLORS();
      bar.hidden = false;
      bar.innerHTML = segs
        .map((s, i) => `<div style="width:${(100 * s[1]) / tot}%;background:${colors[i]}"></div>`)
        .join("");
      el("split-legend").innerHTML = segs
        .map((s, i) => `<span><span class="sw" style="background:${colors[i]}"></span>${s[0]} <b>${fmtInt(s[1])}</b></span>`)
        .join("");
    } else {
      bar.hidden = true;
      el("split-legend").innerHTML = "";
    }
    const stats = [
      ["device", m.device], ["batch", m.batch_size], ["learning rate", m.lr],
      ["max length", m.max_length], ["patience", m.patience], ["intents", m.num_intents],
    ].filter(([, v]) => v != null);
    el("run-stats").innerHTML = stats.map(([k, v]) => `<div class="rs"><small>${k}</small><b>${v}</b></div>`).join("");
    const dist = Object.entries(m.intent_dist || {});
    el("intent-dist-label").hidden = !dist.length;
    const mx = dist.length ? Math.max(...dist.map((d) => d[1])) : 1;
    el("run-intents").innerHTML = dist
      .map(([k, v]) => `<div class="ib-row"><span class="ib-name" title="${k}">${k}</span><span class="ib-track"><span class="ib-fill" style="width:${((v / mx) * 100).toFixed(1)}%"></span></span><span class="ib-val">${fmtInt(v)}</span></div>`)
      .join("");
  }

  function renderLog() {
    const lines = st.managerLog && st.managerLog.length ? st.managerLog : st.serverLog;
    const pre = el("train-log");
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    pre.textContent = lines.length ? lines.join("\n") : "No log for this run (a terminal run's output goes to its train.log).";
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }

  // ------- runs and streams ---------------------------------------------------

  function connect(name) {
    if (st.es) { st.es.close(); st.es = null; }
    st.current = name;
    if (!name) { st.summary = null; renderAll(); return; }
    const es = new EventSource(`/api/runs/${encodeURIComponent(name)}/stream`);
    st.es = es;
    es.addEventListener("snapshot", (ev) => { loadSnapshot(JSON.parse(ev.data)); pickBaseline(); });
    es.addEventListener("records", (ev) => {
      const d = JSON.parse(ev.data);
      applyRecords(d.records);
      if (d.summary) { st.summary = d.summary; renderSummary(); }
    });
    es.addEventListener("summary", (ev) => {
      const before = st.summary?.status;
      st.summary = JSON.parse(ev.data);
      renderSummary();
      if (before === "running" && st.summary.status === "finished") toast(`${name} finished training`, "ok");
      if (before === "running" && st.summary.status === "failed") toast(`${name} failed`, "err");
    });
    es.onerror = () => { el("live-phase").textContent = "reconnecting…"; };
  }

  function fillSelects() {
    const sel = el("run-select");
    const label = (r) => `${r.run} · ${STATUS_LABEL[r.status] || r.status}`;
    sel.innerHTML = `<option value="">Latest${st.runs[0] ? ` (${st.runs[0].run})` : ""}</option>` +
      st.runs.map((r) => `<option value="${r.run}">${label(r)}</option>`).join("");
    sel.value = st.follow ? "" : st.current || "";

    const bsel = el("baseline-select");
    bsel.innerHTML = `<option value="">None</option>` +
      st.runs.filter((r) => r.run !== st.current).map((r) => `<option value="${r.run}">${label(r)}</option>`).join("");
    bsel.value = st.baseline || "";
  }

  async function refreshRuns() {
    let runs;
    try { ({ runs } = await jget("/api/runs")); } catch { return; }
    st.runs = runs;
    const newest = runs[0]?.run || null;
    if (st.follow && newest !== st.current) {
      connect(newest);
      if (st.baselineAuto) st.baseline = null;
    }
    fillSelects();
    if (!runs.length) renderSummary();
  }

  function pickBaseline() {
    if (!st.baselineAuto || st.baseline) return;
    // Default: the newest finished run before this one.
    const cur = st.runs.findIndex((r) => r.run === st.current);
    const prev = st.runs.slice(cur + 1).find((r) => r.status === "finished");
    if (prev) setBaseline(prev.run, true);
  }

  async function setBaseline(name, auto = false) {
    st.baseline = name || null;
    st.baselineAuto = auto;
    st.base = null;
    if (name) {
      try {
        const snap = await jget(`/api/runs/${encodeURIComponent(name)}?points=1200`);
        st.base = { steps: snap.steps, epochs: snap.epochs, spe: snap.summary.steps_per_epoch || 1 };
      } catch { toast(`Could not load ${name}`, "err"); }
    }
    buildCharts(); // the legend names the baseline
    fillSelects();
  }

  // ------- wire-up --------------------------------------------------------------

  window.vimaanLive = {
    setManagerLog(lines) { st.managerLog = lines; renderLog(); },
    followLatest() { st.follow = true; setTimeout(refreshRuns, 1500); },
  };

  document.addEventListener("DOMContentLoaded", () => {
    // ?run=v18 pins a run; ?compare=v17 picks the comparison.
    const qp = new URLSearchParams(location.search);
    if (qp.get("run")) { st.follow = false; connect(qp.get("run")); }
    if (qp.get("compare")) { st.baselineAuto = false; setBaseline(qp.get("compare")); }
    buildCharts();
    el("run-select").addEventListener("change", (e) => {
      const v = e.target.value;
      st.follow = !v;
      st.baselineAuto = true;
      st.baseline = null;
      st.base = null;
      connect(v || st.runs[0]?.run || null);
      fillSelects();
    });
    el("baseline-select").addEventListener("change", (e) => setBaseline(e.target.value, false));
    el("log-scale").addEventListener("change", buildCharts);
    document.addEventListener("vimaan:theme", buildCharts);
    refreshRuns();
    setInterval(refreshRuns, 5000);
    setInterval(() => { if (st.summary?.status === "running") renderSummary(); }, 1000);
  });
})();
