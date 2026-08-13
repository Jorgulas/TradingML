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

function renderDirectionPanel(dir) {
  const el = document.getElementById("direction-panel");
  if (!el) return;
  if (!dir || !dir.selected) {
    el.innerHTML = `<p class="empty-row">ainda sem modelo de direccao treinado</p>`;
    return;
  }
  const s = dir.selected;
  const lift = (s.accuracy - s.baseline_majority) * 100;
  const se = s.standard_error * 100;
  const rows = dir.horizons.map((h) => {
    const l = (h.accuracy - h.baseline_majority) * 100;
    return `<tr>
      <td>${h.horizon_bars}</td>
      <td>${h.market_neutral ? "neutro vs SPY" : "bruto"}</td>
      <td>${(h.accuracy * 100).toFixed(1)}%</td>
      <td>${(h.baseline_majority * 100).toFixed(1)}%</td>
      <td class="${l > 0 ? "positive" : l < 0 ? "negative" : ""}">${l >= 0 ? "+" : ""}${l.toFixed(1)}pp</td>
      <td>${h.auc ? h.auc.toFixed(3) : "--"}</td>
      <td>${h.random_control_accuracy ? (h.random_control_accuracy * 100).toFixed(1) + "%" : "--"}</td>
      <td>${h.significant ? "significativo" : "ruido"}</td>
    </tr>`;
  }).join("");

  el.innerHTML = `
    <p class="explainer">
      Alvo diferente do da cadeia: em vez de <em>que padrao vem a seguir</em>, pergunta-se
      <em>o preco sobe ou desce nas H barras seguintes</em>. Entrada no preco da CONFIRMACAO do
      padrao (nao no seu fim &mdash; so' se sabe que existe algumas barras depois).
      O <strong>controlo</strong> e' a mesma medicao em instantes aleatorios que nao sao fim de
      padrao: se for igual, o padrao nao esta' a dizer nada.
    </p>
    <div class="card" style="margin-bottom:14px">
      <h3>Escolhido pela validacao</h3>
      <div class="total-value">${(s.accuracy * 100).toFixed(1)}%</div>
      <div class="pnl ${s.significant ? (lift > 0 ? "positive" : "negative") : ""}">
        ${lift >= 0 ? "+" : ""}${lift.toFixed(1)}pp vs baseline &plusmn;${se.toFixed(1)}pp
        &mdash; ${s.significant ? "significativo" : "dentro do ruido"}
      </div>
      <div class="meta">
        <span>H=${s.horizon_bars} barras</span>
        <span>${s.market_neutral ? "neutro ao mercado" : "retorno bruto"}</span>
        <span>AUC ${s.auc ? s.auc.toFixed(3) : "--"}</span>
        <span>${s.n_test} padroes mas so' ${s.n_effective_days} dias distintos</span>
      </div>
      <div class="meta">
        efeito minimo detectavel: ${(2 * se).toFixed(1)}pp &middot;
        retorno medio previsto em alta ${(s.mean_return_when_up * 100).toFixed(3)}%,
        em baixa ${(s.mean_return_when_down * 100).toFixed(3)}%
      </div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>H</th><th>rotulo</th><th>acerto</th><th>baseline</th><th>lift</th>
        <th>AUC</th><th>controlo</th><th>veredicto</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}

let portfolioChart = null;

function renderPortfolioPanel(pf) {
  const el = document.getElementById("portfolio-panel");
  if (!el) return;
  if (!pf || !pf.run) {
    el.innerHTML = `<p class="empty-row">carteira de padroes ainda nao corrida</p>`;
    return;
  }
  const r = pf.run;
  const significant = r.t_statistic_days && Math.abs(r.t_statistic_days) > 2;
  el.innerHTML = `
    <p class="explainer">
      Negoceia o sinal de direccao: entra na confirmacao do padrao, sai exactamente
      ${r.horizon_bars} barras depois. Corre <strong>so' no periodo de teste</strong>,
      que nunca serviu para escolher nada. Sem comissoes nem slippage.
      A <strong>exposicao media e' ${(r.mean_exposure * 100).toFixed(0)}%</strong> &mdash; com um
      horizonte de poucas horas a carteira esta' em caixa quase sempre, por isso comparar o
      retorno total com buy &amp; hold nao e' comparacao justa.
    </p>
    <div class="cards">
      <div class="card">
        <h3>Carteira de padroes</h3>
        <div class="total-value">$${r.final_value.toLocaleString("pt-PT", {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
        <div class="pnl ${r.total_return > 0 ? "positive" : r.total_return < 0 ? "negative" : ""}">
          ${(r.total_return * 100).toFixed(2)}% em ${r.n_trades} trades
        </div>
        <div class="meta">
          <span>acerto ${(r.win_rate * 100).toFixed(1)}%</span>
          <span>${r.period_start.slice(0, 10)} a ${r.period_end.slice(0, 10)}</span>
        </div>
      </div>
      <div class="card">
        <h3>Retorno por trade</h3>
        <div class="total-value">${(r.mean_trade_return * 100).toFixed(3)}%</div>
        <div class="pnl ${significant ? "positive" : ""}">
          t=${r.t_statistic_days.toFixed(2)} sobre ${r.n_trade_days} dias
          &mdash; ${significant ? "significativo" : "dentro do ruido"}
        </div>
        <div class="meta">
          <span>entradas ao acaso: ${(r.random_entry_return * 100).toFixed(4)}%</span>
          <span>buy &amp; hold: ${(r.buy_hold_return * 100).toFixed(2)}% (100% investido)</span>
        </div>
      </div>
    </div>
    <canvas id="portfolio-chart" height="70"></canvas>`;

  const points = (pf.equity_curve || []).map((p) => ({ x: p.ts, y: p.total_value }));
  if (portfolioChart) portfolioChart.destroy();
  portfolioChart = new Chart(document.getElementById("portfolio-chart"), {
    type: "line",
    data: { labels: points.map((p) => p.x), datasets: [{
      label: "carteira de padroes", data: points.map((p) => p.y),
      borderColor: "#3ddc97", borderWidth: 2, pointRadius: 0, tension: 0.1,
    }] },
    options: {
      responsive: true, animation: false,
      scales: {
        x: { ticks: { color: "#8f9bb8", maxTicksLimit: 8, callback(v) { return fmtTime(this.getLabelForValue(v)); } },
             grid: { color: "#2a3450" } },
        y: { ticks: { color: "#8f9bb8" }, grid: { color: "#2a3450" } },
      },
      plugins: { legend: { labels: { color: "#e6e9f2" } } },
    },
  });
}

async function renderModelStrip() {
  const el = document.getElementById("model-strip");
  try {
    const payload = await fetchJSON("/api/patterns/model");
    const model = payload.timeframes || payload;
    renderDirectionPanel(payload.direction);
    renderPortfolioPanel(payload.portfolio);
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
