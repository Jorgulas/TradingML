const POLL_MS = 20000;
let depthChart = null;

const KIND_LABEL = { indicators: "Indicadores", microstructure: "Forma da janela" };
const DECISION_CLASS = { SOBE: "up", DESCE: "down", ESPERAR: "none" };

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

function badge(name) {
  const arrow = name === "SOBE" ? "&#9650;" : name === "DESCE" ? "&#9660;" : "&#9679;";
  return `<span class="badge ${DECISION_CLASS[name] || "none"}">${arrow} ${name.toLowerCase()}</span>`;
}

function probBar(probabilities) {
  const order = ["DESCE", "ESPERAR", "SOBE"];
  return order.map((k) => `${((probabilities[k] || 0) * 100).toFixed(0)}%`).join(" / ");
}

function renderStrip(models) {
  const el = document.getElementById("trees-strip");
  if (!models.length) {
    el.innerHTML = `<span class="warn">nenhuma arvore treinada &mdash; corre `
      + `<code>py src/trees/train.py</code></span>`;
    return;
  }
  el.innerHTML = models.map((m) => {
    const lift = (m.accuracy - m.baseline_majority) * 100;
    const se = m.standard_error * 100;
    const cls = m.significant ? (lift > 0 ? "ok" : "err") : "dim";
    return `<span class="${cls}">${KIND_LABEL[m.model_kind]} @ ${m.timeframe}: `
      + `decide ${(m.action_rate * 100).toFixed(0)}% das barras, acerta `
      + `${m.action_precision === null ? "--" : (m.action_precision * 100).toFixed(1) + "%"} `
      + `&middot; ${m.effective_days} dias efectivos</span>`;
  }).join(" ");
}

function renderCards(models) {
  const el = document.getElementById("model-cards");
  el.innerHTML = models.map((m) => {
    const precision = m.action_precision;
    // A precisao accionavel e' o numero grande. A accuracy vai em letra
    // pequena de proposito: e' a que engana.
    const beatsChance = precision !== null && precision > 0.5;
    return `
      <div class="card">
        <h3>${KIND_LABEL[m.model_kind]} &middot; ${m.timeframe}</h3>
        <div class="total-value">${precision === null ? "--" : (precision * 100).toFixed(1) + "%"}</div>
        <div class="pnl ${beatsChance ? "positive" : ""}">
          precisao quando decide &middot; decide em ${(m.action_rate * 100).toFixed(1)}% das barras
        </div>
        <div class="meta">
          <span>accuracy 3 classes: ${(m.accuracy * 100).toFixed(1)}%</span>
          <span>baseline (sempre ${m.baseline_class}): ${(m.baseline_majority * 100).toFixed(1)}%</span>
        </div>
        <div class="meta">
          <span>sobe ${m.up_precision === null ? "--" : (m.up_precision * 100).toFixed(1) + "%"}</span>
          <span>desce ${m.down_precision === null ? "--" : (m.down_precision * 100).toFixed(1) + "%"}</span>
          <span>inversoes graves ${(m.reversal_rate * 100).toFixed(1)}%</span>
        </div>
        <div class="meta">
          <span>profundidade ${m.max_depth}</span>
          <span>${m.latency_ms.toFixed(4)} ms/linha</span>
          <span>${m.n_train.toLocaleString("pt-PT")} linhas de treino</span>
          <span>${m.effective_days} dias efectivos</span>
        </div>
      </div>`;
  }).join("");
}

function renderDepthChart(models) {
  const palette = ["#5b9dff", "#b98cff", "#3ddc97", "#f5b64c"];
  const datasets = [];
  let labels = [];

  models.forEach((m, i) => {
    const sweep = m.sweep || [];
    if (!sweep.length) return;
    labels = sweep.map((s) => s.max_depth);
    datasets.push({
      label: `${KIND_LABEL[m.model_kind]} ${m.timeframe} — precisao`,
      data: sweep.map((s) => (s.action_precision || 0) * 100),
      borderColor: palette[i % palette.length], borderWidth: 2.5,
      pointRadius: 3, tension: 0.15, yAxisID: "y",
    });
    datasets.push({
      label: `${KIND_LABEL[m.model_kind]} ${m.timeframe} — treino (s)`,
      data: sweep.map((s) => s.train_seconds),
      borderColor: palette[i % palette.length], borderWidth: 1,
      borderDash: [5, 4], pointRadius: 0, tension: 0.15, yAxisID: "y1",
    });
  });

  if (!datasets.length) return;
  if (depthChart) depthChart.destroy();
  depthChart = new Chart(document.getElementById("depth-chart"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "profundidade maxima", color: "#8f9bb8" },
             ticks: { color: "#8f9bb8" }, grid: { color: "#2a3450" } },
        y: { position: "left", title: { display: true, text: "precisao quando decide (%)", color: "#8f9bb8" },
             ticks: { color: "#8f9bb8" }, grid: { color: "#2a3450" } },
        y1: { position: "right", title: { display: true, text: "tempo de treino (s)", color: "#8f9bb8" },
              ticks: { color: "#8f9bb8" }, grid: { drawOnChartArea: false } },
      },
      plugins: { legend: { labels: { color: "#e6e9f2", boxWidth: 12, font: { size: 10 } } } },
    },
  });
}

function renderImportances(models) {
  const el = document.getElementById("importances");
  el.innerHTML = models.map((m) => {
    const top = (m.top_features || []).slice(0, 8);
    if (!top.length) return "";
    const max = Math.max(...top.map(([, v]) => v)) || 1;
    return `
      <h3 style="margin:12px 0 8px;font-size:.9rem;color:var(--text-dim)">
        ${KIND_LABEL[m.model_kind]} &middot; ${m.timeframe}</h3>
      ${top.map(([name, value]) => `
        <div style="display:flex;align-items:center;gap:10px;margin:3px 0">
          <span style="flex:0 0 190px;font-size:.8rem">${name.replace(/_/g, " ")}</span>
          <div style="flex:1;background:var(--bg-card);border-radius:4px;height:14px">
            <div style="width:${(value / max) * 100}%;background:var(--blue);height:100%;border-radius:4px"></div>
          </div>
          <span style="flex:0 0 52px;text-align:right;font-size:.76rem;color:var(--text-dim)">
            ${(value * 100).toFixed(1)}%</span>
        </div>`).join("")}`;
  }).join("");
}

function renderDecisions(payload) {
  const tbody = document.querySelector("#decisions-table tbody");
  const rows = payload.decisions || [];
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-row">sem decisoes</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((d) => {
    const a = d.indicators, b = d.microstructure;
    if (!a || !b) {
      return `<tr><td><strong>${d.ticker}</strong></td>
        <td colspan="6" class="empty-row">modelo por treinar ou dados insuficientes</td></tr>`;
    }
    return `<tr class="${d.agree && a.decision_name !== "ESPERAR" ? "agree-row" : ""}">
      <td><strong>${d.ticker}</strong></td>
      <td>$${a.last_price.toFixed(2)}</td>
      <td>${badge(a.decision_name)} <span class="confidence">${(a.confidence * 100).toFixed(0)}%</span></td>
      <td class="confidence">${probBar(a.probabilities)}</td>
      <td>${badge(b.decision_name)} <span class="confidence">${(b.confidence * 100).toFixed(0)}%</span></td>
      <td class="confidence">${probBar(b.probabilities)}</td>
      <td>${badge(d.consensus)}${d.agree ? ' <span class="confidence">concordam</span>' : ""}</td>
    </tr>`;
  }).join("");
}

async function refresh() {
  const started = performance.now();
  try {
    const timeframe = document.getElementById("timeframe-select").value;
    const [models, decisions] = await Promise.all([
      fetchJSON("/api/trees/models"),
      fetchJSON(`/api/trees/decisions?timeframe=${timeframe}`),
    ]);
    renderStrip(models.models);
    renderCards(models.models);
    renderDepthChart(models.models);
    renderImportances(models.models);
    renderDecisions(decisions);
    document.getElementById("calc-time").textContent =
      `actualizado em ${(performance.now() - started).toFixed(0)}ms`;
    document.getElementById("header-updated").textContent =
      "actualizado " + new Date().toLocaleTimeString("pt-PT");
  } catch (err) {
    document.getElementById("trees-strip").innerHTML =
      `<span class="err">falha: ${err.message}</span>`;
  }
}

let timer = null;
function restartPolling() {
  if (timer) clearInterval(timer);
  if (document.getElementById("autorefresh").checked) timer = setInterval(refresh, POLL_MS);
}

document.getElementById("timeframe-select").addEventListener("change", refresh);
document.getElementById("refresh-now").addEventListener("click", refresh);
document.getElementById("autorefresh").addEventListener("change", restartPolling);

refresh();
restartPolling();
