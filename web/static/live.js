const POLL_MS = 2000;   // rapido: a sessao pode estar a avancar a cada segundo
let accuracyChart = null;

function fmtTime(iso) {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("pt-PT", {
    day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function pretty(name) {
  return (name || "").replace(/_/g, " ").toLowerCase();
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function renderStrip(state) {
  const el = document.getElementById("live-strip");
  const s = state.session;
  if (!s) {
    el.innerHTML = `<span class="warn">nenhuma sessao ainda &mdash; corre `
      + `<code>py src/patterns/realtime.py --mode replay</code></span>`;
    return;
  }
  const running = s.status === "running";
  const modeLabel = s.mode === "replay"
    ? "REPLAY (barras historicas, uma a uma)"
    : "AO VIVO (dados atrasados 15-60min)";
  const market = state.market_open
    ? `<span class="ok">mercado aberto</span>`
    : `<span class="warn">mercado fechado</span>`;
  el.innerHTML = `
    <span class="${running ? "ok" : "dim"}">${running ? "&#9679; a correr" : "&#9675; parada"} &middot; ${modeLabel}</span>
    <span>${market}</span>
    <span>${s.timeframe}, horizonte ${s.horizon_bars} barras (${s.horizon_bars * 5} min)</span>
    <span>${s.online_updates} actualizacoes online do modelo</span>
    ${s.cursor_ts ? `<span class="dim">cursor: ${fmtTime(s.cursor_ts)}</span>` : ""}`;
}

function renderCards(state) {
  const el = document.getElementById("live-cards");
  const s = state.session;
  if (!s) { el.innerHTML = ""; return; }

  const accuracy = state.accuracy;
  const low = state.wilson_low, high = state.wilson_high;
  // Um intervalo largo significa que ainda nao se sabe nada. E' deliberado
  // mostra-lo do mesmo tamanho que o numero principal.
  const wide = low !== null && high !== null && (high - low) > 0.20;

  el.innerHTML = `
    <div class="card">
      <h3>Margem de acerto</h3>
      <div class="total-value">${accuracy === null ? "--" : (accuracy * 100).toFixed(1) + "%"}</div>
      <div class="pnl ${wide ? "" : (accuracy > 0.5 ? "positive" : "negative")}">
        ${low === null ? "sem resolucoes ainda"
          : `intervalo ${(low * 100).toFixed(1)}% &ndash; ${(high * 100).toFixed(1)}%`}
      </div>
      <div class="meta">
        <span>${s.n_correct}/${s.n_resolved} certas</span>
        ${wide ? `<span class="warn">intervalo largo &mdash; ainda nao diz nada</span>`
                : `<span class="ok">intervalo apertado</span>`}
      </div>
    </div>
    <div class="card">
      <h3>Ultimas 20</h3>
      <div class="total-value">${state.rolling_accuracy === null || state.rolling_n === 0
        ? "--" : (state.rolling_accuracy * 100).toFixed(1) + "%"}</div>
      <div class="pnl">${state.rolling_n} resolvidas recentes</div>
      <div class="meta"><span>mostra se esta' a melhorar face ao acumulado</span></div>
    </div>
    <div class="card">
      <h3>Previsoes</h3>
      <div class="total-value">${s.n_predictions}</div>
      <div class="pnl">${s.n_predictions - s.n_resolved} a aguardar resultado</div>
      <div class="meta"><span>desde ${fmtTime(s.started_at)}</span></div>
    </div>`;
}

function renderTables(state) {
  const openBody = document.querySelector("#open-table tbody");
  const open = state.open_predictions || [];
  openBody.innerHTML = open.length === 0
    ? `<tr><td colspan="6" class="empty-row">nenhuma previsao a aguardar</td></tr>`
    : open.map((p) => `<tr>
        <td><strong>${p.ticker}</strong></td>
        <td>${pretty(p.pattern_type)}</td>
        <td><span class="badge ${p.predicted_direction ? "up" : "down"}">
          ${p.predicted_direction ? "&#9650; sobe" : "&#9660; desce"}</span></td>
        <td>${(p.confidence * 100).toFixed(0)}%</td>
        <td>$${p.entry_price.toFixed(2)}</td>
        <td>${fmtTime(p.resolve_at_ts)}</td>
      </tr>`).join("");

  const resolvedBody = document.querySelector("#resolved-table tbody");
  const resolved = state.recent_resolved || [];
  resolvedBody.innerHTML = resolved.length === 0
    ? `<tr><td colspan="5" class="empty-row">ainda nenhuma resolvida</td></tr>`
    : resolved.map((p) => `<tr>
        <td><strong>${p.ticker}</strong></td>
        <td><span class="badge ${p.predicted_direction ? "up" : "down"}">
          ${p.predicted_direction ? "sobe" : "desce"}</span></td>
        <td><span class="badge ${p.actual_direction ? "up" : "down"}">
          ${p.actual_direction ? "subiu" : "desceu"}</span></td>
        <td><span class="badge ${p.correct ? "up" : "down"}">
          ${p.correct ? "&#10003; acertou" : "&#10007; falhou"}</span></td>
        <td class="${p.return_pct > 0 ? "positive" : p.return_pct < 0 ? "negative" : ""}">
          ${(p.return_pct * 100).toFixed(2)}%</td>
      </tr>`).join("");
}

function renderCurve(points) {
  const ctx = document.getElementById("accuracy-chart");
  const datasets = [
    { label: "acerto acumulado", data: points.map((p) => p.accuracy * 100),
      borderColor: "#3ddc97", borderWidth: 2.5, pointRadius: 0, tension: 0.15 },
    { label: "limite superior", data: points.map((p) => p.high * 100),
      borderColor: "rgba(91,157,255,0.35)", borderWidth: 1, pointRadius: 0,
      fill: "+1", backgroundColor: "rgba(91,157,255,0.08)" },
    { label: "limite inferior", data: points.map((p) => p.low * 100),
      borderColor: "rgba(91,157,255,0.35)", borderWidth: 1, pointRadius: 0 },
    { label: "acaso (50%)", data: points.map(() => 50),
      borderColor: "#8f9bb8", borderWidth: 1.5, borderDash: [5, 5], pointRadius: 0 },
  ];
  const labels = points.map((p) => p.n);

  if (accuracyChart) {
    accuracyChart.data.labels = labels;
    accuracyChart.data.datasets = datasets;
    accuracyChart.update("none");
    return;
  }
  accuracyChart = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "previsoes resolvidas", color: "#8f9bb8" },
             ticks: { color: "#8f9bb8", maxTicksLimit: 12 }, grid: { color: "#2a3450" } },
        y: { min: 0, max: 100, ticks: { color: "#8f9bb8", callback: (v) => v + "%" },
             grid: { color: "#2a3450" } },
      },
      plugins: { legend: { labels: { color: "#e6e9f2", filter: (i) => !i.text.startsWith("limite") } } },
    },
  });
}

async function refresh() {
  try {
    const [state, curve] = await Promise.all([
      fetchJSON("/api/live/state"),
      fetchJSON("/api/live/accuracy_curve"),
    ]);
    renderStrip(state);
    renderCards(state);
    renderTables(state);
    renderCurve(curve.points || []);
    document.getElementById("header-updated").textContent =
      "actualizado " + new Date().toLocaleTimeString("pt-PT");
  } catch (err) {
    document.getElementById("live-strip").innerHTML =
      `<span class="err">falha a carregar: ${err.message}</span>`;
  }
}

refresh();
setInterval(refresh, POLL_MS);
