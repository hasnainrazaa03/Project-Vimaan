// Training tab: queued rounds of runs (ML/training_plans/*.json, read by
// dashboard/plans.py): each candidate's runs, which are done, which is
// training, and about when the round ends. A run opens in the live view.
// Uses app.js globals (jget) and live.js's window.vimaanLive.

(() => {
  const el = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const LABEL = { queued: "queued", running: "training", finished: "done", failed: "failed", stopped: "stopped", interrupted: "interrupted" };

  function dur(sec) {
    if (sec == null || !isFinite(sec)) return "—";
    const m = Math.round(sec / 60);
    if (m < 60) return `${Math.max(m, 1)} min`;
    return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
  }

  const clock = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  function runChip(r) {
    const bits = [];
    if (r.status === "running") bits.push(`epoch ${r.epoch ?? "—"}`, `${dur(r.left)} left`);
    if (r.status === "finished" && r.elapsed) bits.push(dur(r.elapsed));
    if (r.best_score != null) bits.push(`dev ${Number(r.best_score).toFixed(0)}`);
    return `<button class="plan-run" type="button" data-status="${esc(r.status)}" data-run="${esc(r.run)}"
      title="${esc(r.run)}: ${esc(LABEL[r.status] || r.status)}${r.status === "queued" ? "" : " (open in the charts below)"}">
      <span class="dot"></span><b>${esc(r.run)}</b>${bits.length ? ` <span class="muted">${esc(bits.join(" · "))}</span>` : ""}</button>`;
  }

  function renderPlan(p, open) {
    if (p.error) return `<details class="plan"><summary><b>${esc(p.name)}</b> <span class="muted">cannot be read: ${esc(p.error)}</span></summary></details>`;
    const pct = p.total ? Math.round((100 * p.done) / p.total) : 0;
    const left = p.left_seconds == null ? "time left not known yet" : p.left_seconds ? `about ${dur(p.left_seconds)} left, ends ≈ ${clock(p.ends_at)}` : "all runs done";
    const now = p.running.length ? `training <b>${p.running.map(esc).join(", ")}</b>` : "";
    const failed = p.failed.length ? ` · <span class="plan-bad">${p.failed.length} failed: ${p.failed.map(esc).join(", ")}</span>` : "";
    const rows = p.groups.map((g) => `<tr>
        <td><b>${esc(g.name)}</b>${g.note ? `<div class="muted plan-note">${esc(g.note)}</div>` : ""}${g.result ? `<div class="plan-result">${esc(g.result)}</div>` : ""}</td>
        <td class="plan-count">${g.done}/${g.runs.length}<div class="progress thin"><div class="progress-bar" style="width:${g.runs.length ? (100 * g.done) / g.runs.length : 0}%"></div></div></td>
        <td><div class="plan-runs">${g.runs.map(runChip).join("")}</div></td>
        <td class="muted">${g.each_seconds == null ? "—" : `~${dur(g.each_seconds)}`}</td>
      </tr>`).join("");
    return `<details class="plan" ${open ? "open" : ""}>
      <summary><b>${esc(p.title || p.name)}</b> · ${p.done} of ${p.total} runs done · ${left}${now ? ` · ${now}` : ""}${failed}</summary>
      <div class="progress plan-total"><div class="progress-bar" style="width:${pct}%"></div></div>
      ${p.note ? `<p class="muted plan-note">${esc(p.note)}</p>` : ""}
      <div class="table-wrap"><table class="data-table plan-table">
        <thead><tr><th>Candidate</th><th>Runs done</th><th>Runs</th><th>Each run</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </details>`;
  }

  const openState = new Map(); // plan name -> open, kept across refreshes

  async function refresh() {
    let plans;
    try { ({ plans } = await jget("/api/plans")); } catch { return; }
    const card = el("plans-card");
    card.hidden = !plans.length;
    if (!plans.length) return;
    const box = el("plans");
    box.querySelectorAll("details.plan").forEach((d) => openState.set(d.dataset.name, d.open));
    box.innerHTML = plans.map((p, i) => renderPlan(p, openState.get(p.name) ?? i === 0)).join("");
    box.querySelectorAll("details.plan").forEach((d, i) => { d.dataset.name = plans[i].name; });
    const running = plans.reduce((n, p) => n + (p.running?.length || 0), 0);
    el("plans-badge").textContent = running ? `${running} training` : `${plans.length} plan${plans.length > 1 ? "s" : ""}`;
  }

  document.addEventListener("DOMContentLoaded", () => {
    el("plans").addEventListener("click", (e) => {
      const b = e.target.closest(".plan-run");
      if (!b || b.dataset.status === "queued") return;
      window.vimaanLive?.show(b.dataset.run);
      el("live-chip")?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    refresh();
    setInterval(refresh, 10000);
  });
})();
