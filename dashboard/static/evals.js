// Evals tab: every model version's gate result (scripts/model_gate.py ->
// <version>/gate.json), the trend across versions, and one version in detail.
// Uses app.js globals ($, jget, toast, fmtInt) and live.js's window.vimaanViz.

(() => {
  const el = (id) => document.getElementById(id);
  const pct = (v, d = 2) => (v == null || !isFinite(v) ? "—" : `${(v * 100).toFixed(d)}%`);
  const num = (v) => (v == null ? "—" : fmtInt(v));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  const st = { versions: [], selected: null, detail: null, chart: null, loaded: false };

  const GATE_SHORT = {
    "1 new phrasings": "New phrasings",
    "2 old data": "Old data",
    "3 gold": "Gold set",
    "4 real voice": "Real voice",
    "5 negatives": "Negatives",
    "6 latency": "Latency",
  };

  // ------- trend chart --------------------------------------------------------

  function buildTrend() {
    const V = window.vimaanViz;
    if (!V) return;
    if (st.chart) st.chart.destroy();
    const k = V.tokens();
    const gated = st.versions.filter((v) => v.gate).slice().reverse();
    const pt = (f) => gated.map((v) => ({ x: v.version_num, y: f(v) })).filter((p) => p.y != null);
    const scales = V.axes(k, { xTitle: "model version", percent: true, xInt: true });
    scales.x.ticks.callback = (val) => `v${val}`;
    st.chart = new Chart(el("trend-chart"), {
      type: "line",
      data: {
        datasets: [
          { ...V.line("new phrasings: intent accuracy", k.s1, { points: 4 }), data: pt((v) => v.gate.new_phrasings_intent) },
          { ...V.line("new phrasings: slot F1", k.s2, { points: 4 }), data: pt((v) => v.gate.new_phrasings_slot_f1) },
        ],
      },
      options: V.chartOptions(k, scales, { fmt: (v) => pct(v, 2) }),
    });
    st.chart.options.onClick = (_, elements) => {
      if (!elements.length) return;
      const p = st.chart.data.datasets[elements[0].datasetIndex].data[elements[0].index];
      select(`v${p.x}`);
    };
    st.chart.update("none");
  }

  // ------- versions table -------------------------------------------------------

  function gatePills(g) {
    if (!g) return '<span class="muted">not gated</span>';
    return `<span class="gate-pills">${Object.entries(g.gates)
      .map(([name, ok]) => `<span class="gp ${ok ? "ok" : "bad"}" title="${esc(name)}: ${ok ? "pass" : "fail"}">${ok ? "✓" : "✕"}</span>`)
      .join("")}</span> <b>${g.passed}/${g.total}</b>`;
  }

  function renderTable() {
    const rows = st.versions.filter((v) => v.gate || v.run);
    el("evals-badge").textContent = `${rows.filter((v) => v.gate).length} gated`;
    el("evals-table").innerHTML = `
      <thead><tr>
        <th>Version</th><th>Trained</th><th>Rows</th><th>Intents</th><th>Gates</th>
        <th>New phrasings<br><small>intent · slot F1</small></th><th>Old data<br><small>right · wrong</small></th>
        <th>Real voice<br><small>right · rejected · wrong</small></th><th>Negatives<br><small>rejected</small></th><th>p95</th><th>vs</th>
      </tr></thead>
      <tbody>${rows
        .map((v) => {
          const g = v.gate;
          const rv = g?.real_voice || {};
          return `<tr data-v="${v.version}" class="${v.version === st.selected ? "is-selected" : ""}" tabindex="0">
            <td><b>${v.version}</b></td>
            <td>${v.created ? v.created.slice(0, 10) : "—"}</td>
            <td>${v.rows != null ? fmtInt(v.rows) : "—"}</td>
            <td>${v.intents ?? "—"}</td>
            <td>${gatePills(g)}</td>
            <td>${g ? `${pct(g.new_phrasings_intent)} · ${pct(g.new_phrasings_slot_f1)}` : "—"}</td>
            <td>${g && g.old_data_right_rate != null ? `${pct(g.old_data_right_rate)} · ${num(g.old_data_wrong)}` : "—"}</td>
            <td>${g && rv.n ? `${num(rv.right)} · ${num(rv.rejected)} · ${num(rv.wrong)}` : "—"}</td>
            <td>${g ? pct(g.negatives_rejected, 1) : "—"}</td>
            <td>${g && g.latency_p95_ms != null ? `${g.latency_p95_ms.toFixed(1)} ms` : "—"}</td>
            <td>${g?.baseline ?? "—"}</td>
          </tr>`;
        })
        .join("")}</tbody>`;
    el("evals-table").querySelectorAll("tbody tr").forEach((tr) => {
      tr.addEventListener("click", () => select(tr.dataset.v));
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") select(tr.dataset.v); });
    });
  }

  // ------- one version ----------------------------------------------------------

  async function select(version) {
    st.selected = version;
    renderTable();
    try {
      st.detail = await jget(`/api/evals/${encodeURIComponent(version)}`);
    } catch {
      toast(`Could not load ${version}`, "err");
      return;
    }
    renderDetail();
  }

  function renderDetail() {
    const d = st.detail;
    const g = d?.gate;
    const cand = d?.version, base = g?.baseline;
    if (!g) {
      el("gate-badge").textContent = `${cand}: not gated`;
      el("gate-list").innerHTML = `<p class="muted">${cand} has no gate.json. Run <code>python scripts/model_gate.py --baseline vN ${cand}</code>.</p>`;
      el("eval-intents").innerHTML = "";
      el("voice-table").innerHTML = "";
      el("eval-diffs").innerHTML = "";
      el("pi-badge").textContent = el("voice-badge").textContent = el("diff-badge").textContent = "—";
      return;
    }
    const sum = d.gate_summary;
    el("gate-badge").textContent = `${cand} vs ${base}: ${sum.passed}/${sum.total} pass`;
    el("gate-list").innerHTML = Object.entries(g.gates)
      .map(([name, r]) => `<div class="gate-row ${r.pass ? "ok" : "bad"}">
          <span class="gate-ic" aria-hidden="true">${r.pass ? "✓" : "✕"}</span>
          <div><b>${esc(GATE_SHORT[name] || name)}</b> <span class="gate-word">${r.pass ? "pass" : "fail"}</span><p>${esc(r.detail || "")}</p></div>
        </div>`)
      .join("") + (d.parity ? `<div class="gate-row ${d.parity.pass ? "ok" : "bad"}"><span class="gate-ic">${d.parity.pass ? "✓" : "✕"}</span><div><b>Backend parity</b> <span class="gate-word">${d.parity.pass ? "pass" : "fail"}</span><p>ONNX and int8 decisions against fp32 on ${fmtInt(d.parity.n_texts || 0)} texts</p></div></div>` : "");

    // Per intent on new phrasings: candidate bar, baseline as a tick.
    const c = g.new_phrasings?.candidate || {}, b = g.new_phrasings?.baseline || {};
    const intents = Object.keys(c).filter((k) => k !== "_all")
      .map((name) => ({ name, v: c[name].intent_acc, s: c[name].slot_f1, n: c[name].n, bv: b[name]?.intent_acc }))
      .sort((x, y) => x.v - y.v || x.name.localeCompare(y.name));
    el("pi-badge").textContent = `${cand} intent accuracy · tick = ${base}`;
    el("eval-intents").innerHTML = `<div class="bl-legend"><span><span class="sw s1"></span>${cand}</span><span><span class="sw tick"></span>${base}</span></div>` +
      intents.map((r) => `<div class="bl-row" title="${esc(r.name)}: ${cand} ${pct(r.v, 3)}, ${base} ${pct(r.bv, 3)}, slot F1 ${pct(r.s, 3)}, n ${fmtInt(r.n)}">
        <span class="bl-name">${esc(r.name)}</span>
        <span class="bl-track"><span class="bl-fill" style="width:${(r.v * 100).toFixed(2)}%"></span>${r.bv != null ? `<span class="bl-tick" style="left:${(r.bv * 100).toFixed(2)}%"></span>` : ""}</span>
        <span class="bl-val">${pct(r.v, r.v >= 0.9995 ? 1 : 2)}<span class="bl-sub">${base} ${pct(r.bv, 1)}</span></span>
      </div>`).join("");

    // Real voice per flight.
    const rc = g.real_voice?.candidate?.by_session || {}, rb = g.real_voice?.baseline?.by_session || {};
    const sessions = [...new Set([...Object.keys(rc), ...Object.keys(rb)])].sort();
    const trip = (r) => (r ? `${r.right || 0} · ${r.rejected || 0} · ${r.wrong || 0}` : "—");
    el("voice-badge").textContent = `${g.real_voice?.candidate?.n ?? 0} recordings · right · rejected · wrong`;
    el("voice-table").innerHTML = `<thead><tr><th>Flight</th><th>${cand}</th><th>${base}</th></tr></thead><tbody>${sessions
      .map((s) => `<tr><td>${s.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5")}</td><td>${trip(rc[s])}</td><td>${trip(rb[s])}</td></tr>`)
      .join("")}</tbody>`;

    // Where the candidate differs from the baseline.
    const od = g.old_data_actions || {};
    const groups = [
      ["Baseline right, candidate wrong", od.baseline_right_candidate_wrong, "bad"],
      ["Baseline right, candidate rejected", od.baseline_right_candidate_rejected, "warn"],
      ["Candidate wrong (old data)", od.candidate_wrong, "bad"],
      ["Gold commands failed", g.gold?.failed, "bad"],
      ["Fragments acted on", g.gold?.fragments_acted, "bad"],
      ["Negatives acted on", g.negatives?.candidate?.acted, "bad"],
    ];
    const total = groups.reduce((a, [, list]) => a + (list?.length || 0), 0);
    el("diff-badge").textContent = `${total} examples`;
    el("eval-diffs").innerHTML = groups
      .map(([title, list, tone]) => {
        const n = list?.length || 0;
        return `<details class="diff ${n ? tone : "none"}" ${n && n <= 5 ? "open" : ""}>
          <summary><span class="diff-n">${n}</span> ${title}</summary>
          ${n ? `<ul>${list.slice(0, 40).map((x) => `<li><code>${esc(typeof x === "string" ? x : JSON.stringify(x))}</code></li>`).join("")}${n > 40 ? `<li class="muted">…and ${n - 40} more in gate.json</li>` : ""}</ul>` : ""}
        </details>`;
      })
      .join("");
  }

  // ------- load -----------------------------------------------------------------

  async function load() {
    try {
      ({ versions: st.versions } = await jget("/api/evals"));
    } catch {
      toast("Could not load evals", "err");
      return;
    }
    st.loaded = true;
    renderTable();
    buildTrend();
    if (!st.selected) {
      const first = st.versions.find((v) => v.gate);
      if (first) select(first.version);
    } else {
      select(st.selected);
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    el("refresh-evals").addEventListener("click", load);
    document.addEventListener("vimaan:tab", (e) => { if (e.detail === "evals") load(); });
    document.addEventListener("vimaan:theme", () => { if (st.loaded) buildTrend(); });
    if (document.getElementById("panel-evals").classList.contains("is-active")) load();
  });
})();
