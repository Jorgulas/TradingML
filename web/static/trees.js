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
        <div class="meta decomposition">
          <span><strong>decomposicao:</strong> deteccao de movimento
            ${(m.move_detection * 100).toFixed(1)}% (base ${(m.move_base_rate * 100).toFixed(1)}%)
            &times; direccao <strong>${(m.direction_accuracy * 100).toFixed(1)}%</strong> (acaso 50%)</span>
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

function renderPrecisionCurve(models) {
  const el = document.getElementById("precision-curve");
  if (!el) return;
  const rows = models.flatMap((m) => (m.precision_curve || [])
    .filter((p) => p.precision !== null)
    .map((p) => ({ ...p, kind: KIND_LABEL[m.model_kind], tf: m.timeframe })));
  if (!rows.length) { el.innerHTML = `<p class="empty-row">sem curva</p>`; return; }

  el.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>Modelo</th><th>TF</th><th>Limiar</th><th>Cobertura</th>
      <th>Precisao</th><th>Lift</th><th>n</th></tr></thead>
    <tbody>${rows.map((p) => `<tr>
      <td>${p.kind}</td><td>${p.tf}</td><td>${p.threshold.toFixed(2)}</td>
      <td>${(p.coverage * 100).toFixed(1)}%</td>
      <td><strong>${(p.precision * 100).toFixed(1)}%</strong></td>
      <td class="${p.lift > 0 ? "positive" : "negative"}">${p.lift >= 0 ? "+" : ""}${(p.lift * 100).toFixed(1)}pp</td>
      <td>${p.n.toLocaleString("pt-PT")}</td>
    </tr>`).join("")}</tbody></table></div>`;
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

function renderMagnitude(models) {
  const el = document.getElementById("magnitude-panel");
  if (!el) return;
  if (!models.length) {
    el.innerHTML = `<p class="empty-row">corre <code>py src/trees/magnitude.py</code></p>`;
    return;
  }
  const best = models[0];
  const deciles = best.deciles || [];
  const maxRealised = Math.max(...deciles.map((d) => d.mean_realised), 1);

  el.innerHTML = `
    <div class="cards">
      ${models.map((m) => `
        <div class="card">
          <h3>${KIND_LABEL[m.model_kind]} &middot; ${m.timeframe}</h3>
          <div class="total-value">${(m.accuracy * 100).toFixed(1)}%</div>
          <div class="pnl ${m.significant ? "positive" : ""}">
            vs ${(m.majority_baseline * 100).toFixed(1)}% de base
            &plusmn;${(m.standard_error * 100).toFixed(1)}pp
            &mdash; ${m.significant ? "significativo" : "dentro do ruido"}
          </div>
          <div class="meta">
            <span>AUC <strong>${m.auc.toFixed(3)}</strong></span>
            <span>baseline volatilidade ${m.baseline_vol_auc.toFixed(3)}</span>
          </div>
          <div class="meta">
            <span>Spearman ${m.spearman.toFixed(3)}</span>
            <span>decil topo/fundo <strong>${m.top_bottom_ratio.toFixed(2)}&times;</strong></span>
            <span>${m.effective_days} dias efectivos</span>
          </div>
        </div>`).join("")}
    </div>
    <h3 style="font-size:.92rem;margin:16px 0 8px">
      Calibracao por decil &mdash; ${KIND_LABEL[best.model_kind]} @ ${best.timeframe}
      <span class="subtitle">previsto vs realmente acontecido, em desvios-padrao</span>
    </h3>
    ${deciles.map((d) => `
      <div style="display:flex;align-items:center;gap:10px;margin:3px 0">
        <span style="flex:0 0 60px;font-size:.78rem;color:var(--text-dim)">decil ${d.decile}</span>
        <span style="flex:0 0 78px;font-size:.78rem">prev ${d.mean_predicted.toFixed(2)}</span>
        <div style="flex:1;background:var(--bg-card);border-radius:4px;height:15px">
          <div style="width:${(d.mean_realised / maxRealised) * 100}%;background:var(--green);height:100%;border-radius:4px"></div>
        </div>
        <span style="flex:0 0 74px;text-align:right;font-size:.78rem">real ${d.mean_realised.toFixed(2)}</span>
      </div>`).join("")}
    <div class="table-wrap" style="margin-top:14px"><table>
      <thead><tr><th>Limiar</th><th>Cobertura</th><th>Precisao</th><th>Lift</th><th>n</th></tr></thead>
      <tbody>${(best.coverage_curve || []).filter((c) => c.precision !== null).map((c) => `
        <tr><td>${c.threshold.toFixed(2)}</td><td>${(c.coverage * 100).toFixed(1)}%</td>
        <td><strong>${(c.precision * 100).toFixed(1)}%</strong></td>
        <td class="positive">+${(c.lift * 100).toFixed(1)}pp</td>
        <td>${c.n.toLocaleString("pt-PT")}</td></tr>`).join("")}</tbody>
    </table></div>`;
}

async function refresh() {
  const started = performance.now();
  try {
    const timeframe = document.getElementById("timeframe-select").value;
    const [models, decisions, magnitudes] = await Promise.all([
      fetchJSON("/api/trees/models"),
      fetchJSON(`/api/trees/decisions?timeframe=${timeframe}`),
      fetchJSON("/api/trees/magnitude"),
    ]);
    renderMagnitude(magnitudes.models);
    renderStrip(models.models);
    renderCards(models.models);
    renderDepthChart(models.models);
    renderImportances(models.models);
    renderPrecisionCurve(models.models);
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
