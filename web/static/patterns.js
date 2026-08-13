const CFG = window.PATTERN_CONFIG;
const POLL_MS = 15000;   // recalculo automatico; o backend responde em <100ms
let chart = null;
let pollTimer = null;

const BIAS_LABEL = { bullish: "alta", bearish: "baixa", neutral: "neutro" };

function prettyPattern(name) {
  return name.replace(/_/g, " ").toLowerCase();
}

function fmtTime(iso) {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("pt-PT", {
    day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function fmtDuration(minutes) {
  if (minutes < 60) return `${Math.round(minutes)}min`;
  const hours = minutes / 60;
  if (hours < 24) return `${hours.toFixed(1)}h`;
  return `${(hours / 6.5).toFixed(1)} sessoes`;   // ~6.5h de mercado por dia
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function selection() {
  return {
    ticker: document.getElementById("ticker-select").value,
    timeframe: document.getElementById("timeframe-select").value,
  };
}

function renderChart(bars, analysis) {
  const labels = bars.map((b) => b.ts);
  const closes = bars.map((b) => b.close);

  // Marca a zona do padrao actual pintando so' esse troco da serie
  const highlight = new Array(closes.length).fill(null);
  const current = analysis.current_pattern;
  if (current) {
    const startPos = labels.indexOf(current.start_ts);
    const endPos = labels.indexOf(current.end_ts);
    if (startPos !== -1 && endPos !== -1) {
      for (let i = startPos; i <= endPos; i++) highlight[i] = closes[i];
    }
  }

  const datasets = [
    {
      label: "preco", data: closes, borderColor: "#5b9dff", borderWidth: 1.5,
      pointRadius: 0, tension: 0.1,
    },
    {
      label: current ? prettyPattern(current.pattern_type) : "padrao actual",
      data: highlight, borderColor: "#f5b64c", borderWidth: 3.5,
      pointRadius: 0, tension: 0.1, spanGaps: false,
    },
  ];

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets = datasets;
    chart.update("none");
    return;
  }

  chart = new Chart(document.getElementById("pattern-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          ticks: {
            color: "#8f9bb8", maxTicksLimit: 10,
            callback(value) { return fmtTime(this.getLabelForValue(value)); },
          },
          grid: { color: "#2a3450" },
        },
        y: { ticks: { color: "#8f9bb8" }, grid: { color: "#2a3450" } },
      },
      plugins: { legend: { labels: { color: "#e6e9f2" } } },
    },
  });
}

function renderCurrent(analysis) {
  const el = document.getElementById("current-pattern");
  const current = analysis.current_pattern;
  if (!current) {
    el.innerHTML = `<p class="empty-row">${analysis.note || "sem padrao reconhecivel na janela recente"}</p>`;
    return;
  }
  el.innerHTML = `
    <div class="current-grid">
      <div>
        <div class="current-name">${prettyPattern(current.pattern_type)}</div>
        <span class="badge ${current.bias === "bullish" ? "up" : current.bias === "bearish" ? "down" : "none"}">
          vies ${BIAS_LABEL[current.bias] || current.bias}
        </span>
      </div>
      <div class="meta">
        <span>qualidade da geometria: <strong>${(current.quality * 100).toFixed(0)}%</strong></span>
        <span>formado entre ${fmtTime(current.start_ts)} e ${fmtTime(current.end_ts)}</span>
        <span>ha' ${current.bars_since_completion} barras que completou</span>
        <span>ultimo preco: $${analysis.last_price.toFixed(2)}</span>
      </div>
    </div>`;
}

function renderChain(analysis) {
  const el = document.getElementById("forecast-chain");
  if (!analysis.forecast || analysis.forecast.length === 0) {
    el.innerHTML = `<p class="empty-row">sem cadeia -- e' preciso um padrao actual como ponto de partida</p>`;
    return;
  }
  el.innerHTML = analysis.forecast.map((step) => {
    const alts = step.alternatives.map(
      (a) => `<span class="chip">${prettyPattern(a.pattern_type)} ${(a.probability * 100).toFixed(0)}% <em>n=${a.support}</em></span>`
    ).join("");
    const biasClass = step.bias === "bullish" ? "up" : step.bias === "bearish" ? "down" : "none";
    return `
      <div class="chain-step ${step.low_support ? "weak" : ""}">
        <div class="step-num">${step.step}</div>
        <div class="step-body">
          <div class="step-head">
            <span class="step-name">${prettyPattern(step.pattern_type)}</span>
            <span class="badge ${biasClass}">${BIAS_LABEL[step.bias] || step.bias}</span>
            ${step.low_support ? '<span class="badge warn-badge">suporte baixo</span>' : ""}
          </div>
          <div class="step-stats">
            <span>condicional <strong>${(step.step_confidence * 100).toFixed(1)}%</strong></span>
            <span>acumulada <strong>${(step.cumulative_confidence * 100).toFixed(2)}%</strong></span>
            <span>~${fmtDuration(step.expected_minutes)}</span>
            <span>n=${step.support}</span>
          </div>
          <div class="chip-row">${alts}</div>
        </div>
      </div>`;
  }).join("");
}

function renderRecent(analysis) {
  const tbody = document.querySelector("#recent-patterns tbody");
  const rows = (analysis.recent_patterns || []).slice().reverse();
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-row">nenhum padrao detectado na janela recente</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((p) => {
    const change = (p.end_price / p.start_price - 1) * 100;
    const cls = change > 0 ? "positive" : change < 0 ? "negative" : "";
    return `<tr>
      <td><strong>${prettyPattern(p.pattern_type)}</strong></td>
      <td>${BIAS_LABEL[p.bias] || p.bias}</td>
      <td>${(p.quality * 100).toFixed(0)}%</td>
      <td>${fmtTime(p.start_ts)}</td>
      <td>${fmtTime(p.end_ts)}</td>
      <td class="${cls}">${change >= 0 ? "+" : ""}${change.toFixed(2)}%</td>
    </tr>`;
  }).join("");
}

async function renderModelStrip() {
  const el = document.getElementById("model-strip");
  try {
    const model = await fetchJSON("/api/patterns/model");
    el.innerHTML = Object.entries(model).map(([tf, m]) => {
      if (m.markov_top1_accuracy === null || m.markov_top1_accuracy === undefined) {
        return `<span class="warn">${tf}: modelo por treinar</span>`;
      }
      const markov = m.markov_top1_accuracy * 100;
      const baseline = m.baseline_frequency_accuracy * 100;
      const lift = (m.lift ?? 0) * 100;
      const se = (m.standard_error ?? 0) * 100;
      // O lift so' e' mostrado como bom quando passa 2 erros-padrao. Sem esta
      // barra, um lift de ruido le-se como resultado.
      const cls = m.significant ? (lift > 0 ? "ok" : "err") : "dim";
      const verdict = m.significant ? "" : " &mdash; dentro do ruido";
      let text = `<span class="${cls}">${tf}: ${m.n_patterns} padroes &middot; `
        + `cadeia ${markov.toFixed(1)}% vs baseline ${baseline.toFixed(1)}% `
        + `(${lift >= 0 ? "+" : ""}${lift.toFixed(1)}pp &plusmn;${se.toFixed(1)}pp${verdict}) &middot; `
        + `matriz ${m.n_transition_cells}/${m.total_cells}</span>`;

      // Resultado da experiencia do classificador contextual, medido no
      // conjunto de teste que nunca foi usado para escolher nada.
      const c = m.classifier;
      if (c) {
        const delta = (c.accuracy - c.markov_accuracy) * 100;
        const se = c.standard_error * 100;
        const significant = Math.abs(delta) > 2 * se;
        text += ` <span class="${significant ? "ok" : "dim"}">| classificador ${(c.accuracy * 100).toFixed(1)}% `
          + `(${delta >= 0 ? "+" : ""}${delta.toFixed(1)}pp, &plusmn;${se.toFixed(1)}pp) `
          + `${significant ? "" : "&mdash; dentro do ruido, nao usado"}</span>`;
      }
      return text;
    }).join(" ");
  } catch (err) {
    el.innerHTML = `<span class="err">falha a carregar estado do modelo: ${err.message}</span>`;
  }
}

async function refresh() {
  const { ticker, timeframe } = selection();
  const started = performance.now();
  try {
    const [bars, analysis] = await Promise.all([
      fetchJSON(`/api/patterns/${ticker}/bars?timeframe=${timeframe}&limit=300`),
      fetchJSON(`/api/patterns/${ticker}?timeframe=${timeframe}`),
    ]);
    if (analysis.error) throw new Error(analysis.error);

    document.getElementById("chart-subtitle").textContent = `${ticker} - ${timeframe}`;
    renderChart(bars.bars, analysis);
    renderCurrent(analysis);
    renderChain(analysis);
    renderRecent(analysis);

    const elapsed = performance.now() - started;
    document.getElementById("calc-time").textContent = `recalculado em ${elapsed.toFixed(0)}ms`;
    document.getElementById("header-updated").textContent =
      "actualizado " + new Date().toLocaleTimeString("pt-PT");
  } catch (err) {
    document.getElementById("current-pattern").innerHTML =
      `<p class="empty-row">falha: ${err.message}</p>`;
    console.error(err);
  }
}

function restartPolling() {
  if (pollTimer) clearInterval(pollTimer);
  if (document.getElementById("autorefresh").checked) {
    pollTimer = setInterval(refresh, POLL_MS);
  }
}

document.getElementById("ticker-select").addEventListener("change", refresh);
document.getElementById("timeframe-select").addEventListener("change", () => {
  if (chart) { chart.destroy(); chart = null; }  // eixo muda de escala temporal
  refresh();
});
document.getElementById("refresh-now").addEventListener("click", refresh);
document.getElementById("autorefresh").addEventListener("change", restartPolling);

document.getElementById("timeframe-select").value = CFG.defaultTimeframe;
refresh();
renderModelStrip();
restartPolling();
