// Flights tab: each X-Plane session Vimaan logged (~/Vimaan_Logs), as a
// timeline of commands and autopilot changes. A flight in progress updates
// live over /api/flights/<id>/stream. Uses app.js globals (jget, toast, fmtInt).

(() => {
  const el = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pct = (n, d) => (d ? `${Math.round((n / d) * 100)}%` : "—");
  const ms = (v) => (v == null ? "—" : v >= 10000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
  const mmss = (t) => {
    if (t == null) return "";
    const s = Math.max(0, Math.round(t));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
  };
  const when = (session) => {
    const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(session || "");
    if (!m) return session;
    const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
    return `${d.toLocaleDateString(undefined, { day: "numeric", month: "short" })} ${m[4]}:${m[5]}`;
  };
  const pref = (k, fallback) => { try { return localStorage.getItem(k) ?? fallback; } catch { return fallback; } };
  const save = (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } };

  // Result groups: icon + label always shown, colour only adds to them.
  const GROUP = {
    done: { ic: "✓", cls: "g-done" },
    asked: { ic: "?", cls: "g-asked" },
    refused: { ic: "⊘", cls: "g-warn" },
    talk: { ic: "·", cls: "g-asked" },
    missed: { ic: "?", cls: "g-warn" },
    failed: { ic: "✕", cls: "g-bad" },
    other: { ic: "·", cls: "g-asked" },
  };
  const ISSUE_GROUPS = new Set(["refused", "missed", "failed", "asked", "other"]);

  const st = {
    flights: [], selected: null, auto: true, detail: null, source: null,
    open: new Set(), loaded: false, listTimer: null, audio: null, playing: null,
  };

  // ------- flights list --------------------------------------------------------

  function statusChip(s) {
    if (s === "live") return '<span class="fl-status live"><span class="dot"></span>live</span>';
    if (s === "ended") return '<span class="fl-status">ended</span>';
    return '<span class="fl-status warn" title="No stop record: X-Plane quit or crashed before the plugin stopped">no end</span>';
  }

  function renderList() {
    el("flights-badge").textContent = `${st.flights.length} flights`;
    if (!st.flights.length) {
      el("flights-table").innerHTML = "";
      el("flights-note").textContent = "No flights logged yet. Fly with Vimaan and they appear here, live.";
      return;
    }
    el("flights-note").textContent = "";
    el("flights-table").innerHTML = `
      <thead><tr>
        <th>When</th><th>Length</th><th>Aircraft</th><th>Model</th><th>Commands</th><th>Done</th>
        <th>Not done</th><th>Not understood</th><th>Autopilot drops</th><th>Problems</th><th>Status</th>
      </tr></thead>
      <tbody>${st.flights.map((f) => {
        const g = f.groups || {};
        return `<tr data-s="${esc(f.session)}" class="${f.session === st.selected ? "is-selected" : ""}" tabindex="0">
          <td><b>${esc(when(f.session))}</b></td>
          <td>${mmss(f.duration_sec)}</td>
          <td>${esc((f.aircraft || []).join(", ") || "—")}</td>
          <td>${esc(f.model || "—")}</td>
          <td>${fmtInt(f.commands)}</td>
          <td>${f.commands ? `${fmtInt(g.done || 0)} <small class="muted">${pct(g.done || 0, f.commands)}</small>` : "—"}</td>
          <td>${fmtInt(g.refused || 0)}</td>
          <td>${fmtInt(g.missed || 0)}</td>
          <td>${f.autopilot_drops ? `<span class="fl-flag">⚠ ${f.autopilot_drops}</span>` : "0"}</td>
          <td>${f.problems ? `<span class="fl-flag">⚠ ${f.problems}</span>` : "0"}</td>
          <td>${statusChip(f.status)}</td>
        </tr>`;
      }).join("")}</tbody>`;
    el("flights-table").querySelectorAll("tbody tr").forEach((tr) => {
      const pick = () => { st.auto = false; select(tr.dataset.s); };
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") pick(); });
    });
  }

  async function loadList() {
    try {
      ({ flights: st.flights } = await jget("/api/flights"));
    } catch {
      toast("Could not load flights", "err");
      return;
    }
    st.loaded = true;
    // Follow the newest flight until the pilot picks one; a new live session
    // takes over the view.
    const newest = st.flights[0];
    // ?flight=<session> opens that one (and stops following the newest).
    const asked = new URLSearchParams(location.search).get("flight");
    if (asked && !st.selected && st.flights.some((f) => f.session === asked)) {
      st.auto = false;
      st.selected = asked;
      st.detail = null;
    }
    const target = st.auto || !st.selected ? newest?.session : st.selected;
    const stale = !st.detail || (!st.source && st.detail.summary.status === "live");
    if (target && (target !== st.selected || stale)) select(target);
    renderList();
  }

  // ------- one flight ----------------------------------------------------------

  function closeStream() {
    if (st.source) { st.source.close(); st.source = null; }
  }

  async function select(session) {
    closeStream();
    if (session !== st.selected) st.open.clear();
    st.selected = session;
    renderList();
    let detail;
    try {
      detail = await jget(`/api/flights/${encodeURIComponent(session)}`);
    } catch {
      toast(`Could not load ${session}`, "err");
      return;
    }
    if (st.selected !== session) return;
    st.detail = detail;
    renderFlight();
    if (detail.summary.status === "live") follow(session);
  }

  function follow(session) {
    const src = new EventSource(`/api/flights/${encodeURIComponent(session)}/stream`);
    st.source = src;
    src.addEventListener("flight", (e) => {
      if (st.selected !== session) return;
      st.detail = JSON.parse(e.data);
      renderFlight();
      const row = st.flights.find((f) => f.session === session);
      if (row) { Object.assign(row, st.detail.summary); renderList(); }
      if (st.detail.summary.status !== "live") closeStream();
    });
    src.onerror = () => { /* EventSource reconnects by itself */ };
  }

  function kpi(label, value, sub, flag) {
    return `<div class="kpi${flag ? " flag" : ""}"><small>${label}</small><strong>${value}</strong><span>${sub}</span></div>`;
  }

  function renderKpis(s) {
    const g = s.groups || {}, n = s.commands;
    el("flight-kpis").innerHTML = [
      kpi("Commands", fmtInt(n), `${fmtInt(s.recordings)} recordings`),
      kpi("Done", fmtInt(g.done || 0), `${pct(g.done || 0, n)} of commands`),
      kpi("Not done", fmtInt(g.refused || 0), "Vimaan said why"),
      kpi("Not understood", fmtInt(g.missed || 0), "heard nothing usable"),
      kpi("Failed", fmtInt(g.failed || 0), "errors and timeouts", g.failed),
      kpi("Speech to action", ms(s.latency_p50_ms), `median · p95 ${ms(s.latency_p95_ms)}`),
      kpi("Autopilot drops", fmtInt(s.autopilot_drops), "not made by Vimaan", s.autopilot_drops),
      kpi("Problems", fmtInt(s.problems), "errors and freezes", s.problems),
    ].join("");
  }

  function renderHead(s) {
    el("flight-title").textContent = `${when(s.session)} · ${(s.aircraft || []).join(", ") || "no aircraft"}`;
    const chip = el("flight-chip");
    chip.dataset.status = s.status === "live" ? "running" : s.status === "ended" ? "finished" : "interrupted";
    chip.querySelector(".label").textContent = s.status === "live" ? "live" : s.status === "ended" ? "ended" : "no end record";
    const bits = [
      s.model && `model ${s.model}`,
      s.stt && `${s.stt} speech`,
      s.duration_sec != null && mmss(s.duration_sec),
      s.self_test && `self test ${s.self_test.passed}/${s.self_test.total}`,
      s.joystick && s.joystick.nullzone != null && `joystick null zone ${Math.round(s.joystick.nullzone * 100)}%`,
    ].filter(Boolean);
    el("flight-meta").textContent = bits.join(" · ");
  }

  function slotsText(slots) {
    return Object.entries(slots || {}).map(([k, v]) => `<span class="fl-slot">${esc(k.replace(/_/g, " "))} <b>${esc(v)}</b></span>`).join(" ");
  }

  function commandRow(c) {
    const g = GROUP[c.group] || GROUP.other;
    const open = st.open.has(c.cmd);
    const conf = c.confidence != null ? `<small class="muted">${(c.confidence * 100).toFixed(1)}%</small>` : "";
    const play = c.audio
      ? `<button class="fl-play${st.playing === c.cmd ? " is-playing" : ""}" data-play="${c.cmd}" aria-label="Play the recording of ${c.cmd}" title="Play the recording">${st.playing === c.cmd ? "■" : "▶"}</button>`
      : '<span class="fl-play none" title="No recording">–</span>';
    const main = `<tr class="fl-cmd${open ? " is-open" : ""}" data-cmd="${c.cmd}" tabindex="0">
      <td class="fl-time"><b>${esc(c.clock || "")}</b><small class="muted">+${mmss(c.t)}</small></td>
      <td class="fl-said">${play}<span>${c.heard ? `“${esc(c.heard)}”` : '<span class="muted">nothing heard</span>'}${c.held_ms != null ? `<small class="muted">held ${(c.held_ms / 1000).toFixed(1)} s · ${c.cmd}</small>` : `<small class="muted">${c.cmd}</small>`}</span></td>
      <td>${c.intent_name ? `<b>${esc(c.intent_name)}</b> ${slotsText(c.slots)} ${conf}` : '<span class="muted">—</span>'}</td>
      <td>${c.replies.length ? c.replies.map((r) => `<span class="fl-reply">${esc(r)}</span>`).join("") : '<span class="muted">(silent)</span>'}</td>
      <td><span class="fl-result ${g.cls}"><span aria-hidden="true">${g.ic}</span>${esc(c.label)}</span>${c.notes.some((n) => n.tone === "warn") ? ' <span class="fl-flag" title="See details">⚠</span>' : ""}</td>
      <td class="num">${ms(c.latency_ms)}${c.slow ? ' <span class="fl-flag" title="Slower than the budget">slow</span>' : ""}</td>
    </tr>`;
    if (!open) return main;
    const acts = c.actions.map((a) => `<li class="${a.ok === false ? "bad" : ""}"><code>${esc(a.kind === "write" ? `${a.name} = ${a.value}${a.readback !== a.value ? ` (read back ${a.readback})` : ""}` : a.name)}</code>${a.ok === false ? " failed" : ""}</li>`).join("");
    const checks = c.checks.map((v) => `<li class="${v.ok ? "" : "bad"}">${v.ok ? "✓" : "✕"} ${esc(v.label)}: expected <code>${esc(v.expected)}</code>, got <code>${esc(v.actual)}</code></li>`).join("");
    const notes = c.notes.map((n) => `<li class="${n.tone === "warn" ? "warn" : ""}">${esc(n.text)}</li>`).join("");
    const detail = `<tr class="fl-detail"><td></td><td colspan="5"><div class="fl-detail-grid">
      <div><h4>Understood</h4><p><code>${esc(c.normalized || "—")}</code></p><p class="muted">${esc(c.intent || "no intent")} · total ${ms(c.total_ms)}${c.stt_ms != null ? ` · recognition ${ms(c.stt_ms)}` : ""}${c.audio_sec != null ? ` · ${c.audio_sec} s of audio` : ""}</p></div>
      <div><h4>Sent to X-Plane</h4>${acts ? `<ul>${acts}</ul>` : '<p class="muted">nothing</p>'}</div>
      <div><h4>Checked afterwards</h4>${checks ? `<ul>${checks}</ul>` : '<p class="muted">nothing to check</p>'}</div>
      ${notes ? `<div><h4>Notes</h4><ul>${notes}</ul></div>` : ""}
    </div></td></tr>`;
    return main + detail;
  }

  function eventRow(e) {
    let body = esc(e.text);
    if (e.changes && e.changes.length) {
      body += ` <span class="fl-who ${e.tone === "warn" ? "warn" : ""}">${esc(e.who)}</span><div class="fl-changes">${e.changes.map((c) => `<span>${esc(c.name)} <b>${esc(c.from)} → ${esc(c.to)}</b></span>`).join("")}</div>`;
    }
    if (e.kind === "autopilot_disconnect") {
      const sp = e.stick_peak || {}, f = e.flying || {};
      const stick = Object.keys(sp).length ? Object.entries(sp).map(([a, v]) => `${a} ${v >= 0 ? "+" : ""}${Number(v).toFixed(2)}`).join(", ") : "no reading";
      body += `<div class="fl-changes"><span>Stick, last 4 s <b>${esc(stick)}</b></span>${e.nullzone != null ? `<span>Null zone <b>${Math.round(e.nullzone * 100)}%</b></span>` : ""}<span>Flying <b>${esc(f.altitude_actual)} ft, ${esc(f.vs_actual)} ft/min, ${esc(f.ias_actual)} kt</b></span>${e.said ? `<span>Said <b>“${esc(e.said)}”</b></span>` : ""}</div>`;
    }
    const ic = e.tone === "bad" ? "✕" : e.tone === "warn" ? "⚠" : e.tone === "good" ? "✓" : "";
    return `<tr class="fl-event tone-${e.tone}${e.level ? ` lvl-${e.level}` : ""}">
      <td class="fl-time"><b>${esc(e.clock || "")}</b><small class="muted">+${mmss(e.t)}</small></td>
      <td colspan="5">${ic ? `<span class="fl-ev-ic" aria-hidden="true">${ic}</span>` : ""}${body}</td>
    </tr>`;
  }

  function visible(items) {
    const f = el("flight-filter").value;
    const q = el("flight-search").value.trim().toLowerCase();
    const showAp = el("flight-autopilot").checked, showSys = el("flight-system").checked;
    return items.filter((it) => {
      if (it.type === "event") {
        if (it.level === "system" && !showSys) return false;
        if (it.level === "autopilot" && !showAp) return false;
        if (f === "done") return false;
        if (f === "issues" && !["warn", "bad"].includes(it.tone)) return false;
        if (f === "autopilot" && !(it.level === "autopilot" || it.kind === "autopilot_disconnect")) return false;
        return !q || it.text.toLowerCase().includes(q);
      }
      if (f === "autopilot" && !/autopilot|mode|throttle|director/i.test(`${it.intent || ""} ${it.intent_name || ""}`)) return false;
      if (f === "done" && it.group !== "done") return false;
      if (f === "issues" && !ISSUE_GROUPS.has(it.group) && !it.notes.some((n) => n.tone === "warn")) return false;
      if (!q) return true;
      return [it.heard, it.intent_name, it.intent, ...it.replies, JSON.stringify(it.slots)].join(" ").toLowerCase().includes(q);
    });
  }

  function renderTimeline() {
    const d = st.detail;
    let items = visible(d.items);
    if (el("flight-newest").checked) items = items.slice().reverse();
    const total = d.items.length;
    el("flight-count").textContent = items.length === total ? `${total} rows` : `${items.length} of ${total} rows`;
    el("flight-timeline").innerHTML = items.length
      ? `<thead><tr><th>Time</th><th>You said</th><th>Understood</th><th>Vimaan said</th><th>Result</th><th class="num">Speech to action</th></tr></thead>
         <tbody>${items.map((it) => (it.type === "command" ? commandRow(it) : eventRow(it))).join("")}</tbody>`
      : '<tbody><tr><td class="muted">Nothing matches.</td></tr></tbody>';
  }

  function renderFlight() {
    const s = st.detail.summary;
    renderHead(s);
    renderKpis(s);
    renderTimeline();
  }

  // ------- interaction ---------------------------------------------------------

  function toggleRow(cmd) {
    if (st.open.has(cmd)) st.open.delete(cmd); else st.open.add(cmd);
    renderTimeline();
  }

  function play(cmd) {
    if (st.audio) { st.audio.pause(); st.audio = null; }
    if (st.playing === cmd) { st.playing = null; renderTimeline(); return; }
    const a = new Audio(`/api/flights/${encodeURIComponent(st.selected)}/audio/${encodeURIComponent(cmd)}`);
    st.audio = a;
    st.playing = cmd;
    const done = () => { if (st.playing === cmd) { st.playing = null; st.audio = null; renderTimeline(); } };
    a.addEventListener("ended", done);
    a.addEventListener("error", () => { toast("Could not play the recording", "err"); done(); });
    a.play().catch(() => done());
    renderTimeline();
  }

  function startListTimer() {
    stopListTimer();
    st.listTimer = setInterval(loadList, 15000);  // a new session appears on its own
  }
  function stopListTimer() {
    if (st.listTimer) { clearInterval(st.listTimer); st.listTimer = null; }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const tl = el("flight-timeline");
    tl.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-play]");
      if (btn) { e.stopPropagation(); play(btn.dataset.play); return; }
      const row = e.target.closest("tr.fl-cmd");
      if (row) toggleRow(row.dataset.cmd);
    });
    tl.addEventListener("keydown", (e) => {
      const row = e.target.closest("tr.fl-cmd");
      if (row && e.key === "Enter") toggleRow(row.dataset.cmd);
    });
    for (const [id, key] of [["flight-autopilot", "vimaan-fl-ap"], ["flight-system", "vimaan-fl-sys"], ["flight-newest", "vimaan-fl-newest"]]) {
      const box = el(id);
      box.checked = pref(key, box.checked ? "1" : "0") === "1";
      box.addEventListener("change", () => { save(key, box.checked ? "1" : "0"); if (st.detail) renderTimeline(); });
    }
    el("flight-filter").value = pref("vimaan-fl-filter", "all");
    el("flight-filter").addEventListener("change", () => { save("vimaan-fl-filter", el("flight-filter").value); if (st.detail) renderTimeline(); });
    el("flight-search").addEventListener("input", () => { if (st.detail) renderTimeline(); });
    el("refresh-flights").addEventListener("click", () => { st.auto = true; loadList(); });
    document.addEventListener("vimaan:tab", (e) => {
      if (e.detail === "flights") { loadList(); startListTimer(); }
      else { stopListTimer(); closeStream(); }
    });
    if (el("panel-flights").classList.contains("is-active")) { loadList(); startListTimer(); }
  });
})();
