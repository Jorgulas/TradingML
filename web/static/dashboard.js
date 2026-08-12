const CFG = window.APP_CONFIG;
const HORIZON_LABEL = { SHORT: "Curto Prazo", LONG: "Longo Prazo" };
const POLL_MS = 45000;
let equityChart = null;

function fmtMoney(v) {
  if (v === null || v === undefined) return "--";
  const sign = v < 0 ? "-" : "";
  return sign + CFG.currencySymbol + Math.abs(v).toLocaleString("pt-PT", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(v) {
  if (v === null || v === undefined) return "--";
  const pct = v * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

function pnlClass(v) {
  if (v === null || v === undefined) return "";
  return v > 0 ? "positive" : v < 0 ? "negative" : "";
}

function fmtDateTime(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString("pt-PT", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function renderSummaryCards(summary) {
  const container = document.getElementById("summary-cards");
  container.innerHTML = "";
  for (const horizon of CFG.horizons) {
    const h = summary.horizons[horizon];
    const card = document.createElement("div");
    card.className = "card";
    const model = h.model;
    const modelLine = model
      ? `modelo ${model.algorithm || "logistic_regression"} · treinado ${fmtDateTime(model.trained_at)} · ` +
        `CV ${(model.cv_accuracy * 100).toFixed(1)}% (baseline maioria ${(model.baseline_majority_accuracy * 100).toFixed(1)}%)`
      : "modelo ainda nao treinado (dados insuficientes)";
    card.innerHTML = `
      <h3>${HORIZON_LABEL[horizon]}</h3>
      <div class="total-value">${fmtMoney(h.total_value)}</div>
      <div class="pnl ${pnlClass(h.pnl_abs)}">${fmtMoney(h.pnl_abs)} (${fmtPct(h.pnl_pct)})</div>
      <div class="meta">
        <span>caixa: ${fmtMoney(h.cash)}</span>
        <span>posicoes: ${h.num_positions}</span>
        <span>atualizado: ${h.as_of || "--"}</span>
      </div>
      <div class="meta">${modelLine}</div>
    `;
    container.appendChild(card);
  }
}

function renderHealthStrip(summary) {
  const el = document.getElementById("health-strip");
  const parts = [];
  if (summary.last_run) {
    const cls = summary.last_run.status === "ERROR" ? "err" : summary.last_run.status === "WARN" ? "warn" : "ok";
    parts.push(`<span class="${cls}">ultima execucao: ${summary.last_run.stage} (${summary.last_run.status}) em ${fmtDateTime(summary.last_run.created_at)}</span>`);
  } else {
    parts.push(`<span class="warn">ainda sem nenhuma execucao registada -- corre scripts/bootstrap.py e src/run_daily.py</span>`);
  }
  if (summary.tickers_missing_news_today && summary.tickers_missing_news_today.length > 0) {
    parts.push(`<span class="warn">sem avaliacao de noticias hoje: ${summary.tickers_missing_news_today.join(", ")}</span>`);
  } else if (summary.latest_date) {
    parts.push(`<span class="ok">noticias do dia completas para todos os tickers</span>`);
  }
  if (summary.recent_errors && summary.recent_errors.length > 0) {
    parts.push(`<span class="err">${summary.recent_errors.length} erro(s) recente(s) em run_log</span>`);
  }
  el.innerHTML = parts.join(" &middot; ");
  document.getElementById("header-updated").textContent = "pagina atualizada " + new Date().toLocaleTimeString("pt-PT");
}

function renderEquityChart(series) {
  const ctx = document.getElementById("equity-chart");
  const colors = { SHORT: "#5b9dff", LONG: "#b98cff" };
  const datasets = CFG.horizons.map((h) => ({
    label: HORIZON_LABEL[h],
    data: (series[h] || []).map((p) => ({ x: p.date, y: p.total_value })),
    borderColor: colors[h] || "#3ddc97",
    backgroundColor: "transparent",
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.15,
  }));

  if (equityChart) {
    equityChart.data.datasets = datasets;
    equityChart.update();
    return;
  }

  equityChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { type: "category", ticks: { color: "#8f9bb8", maxTicksLimit: 12 }, grid: { color: "#2a3450" } },
        y: { ticks: { color: "#8f9bb8", callback: (v) => CFG.currencySymbol + v.toLocaleString("pt-PT") }, grid: { color: "#2a3450" } },
      },
      plugins: {
        legend: { labels: { color: "#e6e9f2" } },
        tooltip: { callbacks: { label: (item) => `${item.dataset.label}: ${fmtMoney(item.parsed.y)}` } },
      },
    },
  });
}

function directionBadge(signal) {
  if (!signal) return `<span class="badge none">sem modelo</span>`;
  const dir = signal.predicted_direction === 1 ? "up" : "down";
  const arrow = dir === "up" ? "&#9650;" : "&#9660;";
  return `<span class="badge ${dir}">${arrow} ${(signal.confidence * 100).toFixed(0)}%</span>`;
}

function newsChips(news, booleanFeatures) {
  if (!news) return `<span class="chip">sem avaliacao</span>`;
  return `<div class="chip-row">` + booleanFeatures.map((f) => {
    const active = news[f] === 1;
    return `<span class="chip ${active ? "active" : ""}" title="${f}">${f.replace(/_/g, " ")}</span>`;
  }).join("") + `</div>`;
}

function renderSignals(data) {
  const tbody = document.querySelector("#signals-table tbody");
  tbody.innerHTML = "";
  if (!data.date) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty-row">sem previsoes ainda -- corre src/run_daily.py</td></tr>`;
    return;
  }
  for (const entry of data.tickers) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${entry.ticker}</strong></td>
      <td class="signal-cell">${directionBadge(entry.signals.SHORT)}</td>
      <td class="signal-cell">${directionBadge(entry.signals.LONG)}</td>
      <td>${newsChips(entry.news, data.boolean_features || [])}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderPositions(data) {
  for (const horizon of CFG.horizons) {
    const table = document.querySelector(`.positions-table[data-horizon="${horizon}"] tbody`);
    const items = data[horizon] || [];
    if (items.length === 0) {
      table.innerHTML = `<tr><td colspan="5" class="empty-row">sem posicoes abertas</td></tr>`;
      continue;
    }
    table.innerHTML = items.map((p) => `
      <tr>
        <td><strong>${p.ticker}</strong></td>
        <td>${p.qty.toFixed(4)}</td>
        <td>${fmtMoney(p.avg_price)}</td>
        <td>${fmtMoney(p.latest_price)}</td>
        <td class="${pnlClass(p.unrealized_pnl)}">${fmtMoney(p.unrealized_pnl)} (${fmtPct(p.unrealized_pnl_pct)})</td>
      </tr>
    `).join("");
  }
}

function renderTrades(trades) {
  const tbody = document.querySelector("#trades-table tbody");
  if (!trades || trades.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-row">ainda sem trades simulados</td></tr>`;
    return;
  }
  tbody.innerHTML = trades.map((t) => `
    <tr>
      <td>${t.date}</td>
      <td><span class="badge ${t.horizon.toLowerCase()}">${HORIZON_LABEL[t.horizon] || t.horizon}</span></td>
      <td><strong>${t.ticker}</strong></td>
      <td><span class="badge ${t.side.toLowerCase()}">${t.side}</span></td>
      <td>${t.qty.toFixed(4)}</td>
      <td>${fmtMoney(t.price)}</td>
      <td>${t.reason}</td>
    </tr>
  `).join("");
}

async function refreshAll() {
  try {
    const [summary, equity, positions, signals, trades] = await Promise.all([
      fetchJSON("/api/summary"),
      fetchJSON("/api/equity_curve"),
      fetchJSON("/api/positions"),
      fetchJSON("/api/signals"),
      fetchJSON("/api/trades"),
    ]);
    renderSummaryCards(summary);
    renderHealthStrip(summary);
    renderEquityChart(equity);
    renderPositions(positions);
    renderSignals(signals);
    renderTrades(trades);
  } catch (err) {
    document.getElementById("health-strip").innerHTML = `<span class="err">falha a carregar dados: ${err.message}</span>`;
    console.error(err);
  }
}

refreshAll();
setInterval(refreshAll, POLL_MS);
