// Pagina da tabela de verdade. Nao calcula nada: le a ultima execucao gravada
// por src/decision/run_decision.py. O que se mostra tem sempre proveniencia --
// cada celula vem com quantos DIAS a suportam e com o peso que isso lhe deu.

const pct = (v, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "--" : `${(v * 100).toFixed(digits)}%`;
const signed = (v, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "--" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;

const ACTION_CLASS = { COMPRAR: "buy", VENDER: "sell", ESPERAR: "none" };
const SIGNAL_CLASS = {
  SOBE: "up", ALTA: "up", DESCE: "down", BAIXA: "down",
  NEUTRO: "none", NENHUM: "none", CALMO: "none", NORMAL: "none",
  AGITADO: "sell", SIM: "sell", NAO: "none",
};

const badge = (value, map = SIGNAL_CLASS) =>
  `<span class="badge ${map[value] || "none"}">${value}</span>`;

let state = { cells: [] };

async function load() {
  const response = await fetch("/api/decision");
  const data = await response.json();

  if (data.error) {
    document.getElementById("decision-strip").innerHTML =
      `<span class="err">${data.error}</span>`;
    return;
  }

  state.cells = data.cells;
  renderStrip(data);
  renderToday(data);
  renderSchemes(data);
  renderWalkforward(data);
  renderAblation(data);
  renderCells();

  document.getElementById("header-updated").textContent =
    `execucao #${data.run.id} de ${new Date(data.run.created_at).toLocaleString("pt-PT")}`;
}

function renderStrip(data) {
  const run = data.run;
  document.getElementById("decision-strip").innerHTML = [
    `<span>${run.n_cells} celulas</span>`,
    `<span class="${run.n_cells_supported / run.n_cells > 0.7 ? "ok" : "warn"}">` +
      `${run.n_cells_supported} bem observadas</span>`,
    `<span class="dim">peso medio na propria celula ${run.mean_weight.toFixed(2)}</span>`,
    `<span class="dim">limiar ${pct(run.threshold)}</span>`,
    `<span class="dim">treino ${run.train_days}d / validacao ${run.validation_days}d / ` +
      `teste ${run.test_days}d</span>`,
  ].join("");
}

function renderToday(data) {
  const rows = data.today;
  document.getElementById("today-date").textContent =
    rows.length ? `-- ${rows[0].ts.slice(0, 10)}, ${rows.length} accoes` : "";

  const counts = rows.reduce((acc, r) => {
    acc[r.accao] = (acc[r.accao] || 0) + 1;
    return acc;
  }, {});
  document.getElementById("today-cards").innerHTML =
    ["COMPRAR", "VENDER", "ESPERAR"].map((action) => `
      <div class="card">
        <h3>${action}</h3>
        <div class="total-value">${counts[action] || 0}</div>
        <div class="meta"><span>de ${rows.length} accoes</span></div>
      </div>`).join("");

  // Primeiro o que age, e dentro disso pela estimativa: a lista comeca no que
  // a tabela mais defende e acaba no que ela mais recusa.
  const ordered = [...rows].sort((a, b) => {
    if ((a.accao === "ESPERAR") !== (b.accao === "ESPERAR")) {
      return a.accao === "ESPERAR" ? 1 : -1;
    }
    return (b.estimate ?? 0) - (a.estimate ?? 0);
  });

  document.querySelector("#today-table tbody").innerHTML = ordered.map((r) => `
    <tr>
      <td><strong>${r.ticker}</strong></td>
      <td>${badge(r.curto)}</td>
      <td>${badge(r.rapido)}</td>
      <td>${badge(r.padrao)}</td>
      <td>${badge(r.regime)}</td>
      <td>${badge(r.alarme)}</td>
      <td>${signed(r.estimate)}</td>
      <td>${badge(r.accao, ACTION_CLASS)}</td>
    </tr>`).join("");
}

function ciCell(low, high, significant) {
  const text = `[${signed(low, 3)}, ${signed(high, 3)}]`;
  return significant ? `<strong class="ok">${text}</strong>` : `<span class="dim">${text}</span>`;
}

function renderSchemes(data) {
  document.querySelector("#schemes-table tbody").innerHTML = data.results.map((r) => `
    <tr>
      <td>${r.name}</td>
      <td>${signed(r.mean)}</td>
      <td>${ciCell(r.ci[0], r.ci[1], r.significant)}</td>
      <td>${pct(r.hit_days, 1)}</td>
      <td>${pct(r.coverage, 1)}</td>
      <td>${signed(r.net_mean)}</td>
    </tr>`).join("");

  const run = data.run;
  const proven = run.interaction_low > 0;
  document.getElementById("interaction-verdict").innerHTML =
    `Tabela menos votacao simples: <strong>${signed(run.interaction_diff)}</strong>, ` +
    `IC95% [${signed(run.interaction_low, 3)}, ${signed(run.interaction_high, 3)}]. ` +
    (proven
      ? "O intervalo esta' inteiramente acima de zero: as interaccoes acrescentam valor mensuravel."
      : "O intervalo inclui zero: <strong>as interaccoes nao se demonstram</strong>. " +
        (run.interaction_diff > 0
          ? "A tabela tem o valor mais alto, mas nao se distingue de somar os sinais."
          : "Neste teste a tabela nem sequer supera a soma dos sinais."));
}

function renderWalkforward(data) {
  if (!data.walkforward) return;
  const walk = data.walkforward;
  document.getElementById("walkforward-panel").hidden = false;
  document.querySelector("#walkforward-table tbody").innerHTML = walk.results.map((r) => `
    <tr>
      <td>${r.name}</td>
      <td>${signed(r.mean)}</td>
      <td>${ciCell(r.ci[0], r.ci[1], r.significant)}</td>
      <td>${pct(r.hit_days, 1)}</td>
    </tr>`).join("");

  document.getElementById("walkforward-verdict").innerHTML =
    `${walk.n_days} dias fora da amostra, cerca de <strong>${walk.independent_windows} janelas ` +
    `independentes</strong>. Tabela menos votacao: ${signed(walk.interaction_difference)}, ` +
    `IC95% [${signed(walk.interaction_ci[0], 3)}, ${signed(walk.interaction_ci[1], 3)}].`;
}

function renderAblation(data) {
  const ordered = [...data.ablation].sort((a, b) => b.delta - a.delta);
  document.querySelector("#ablation-table tbody").innerHTML = ordered.map((r) => `
    <tr>
      <td>${badge(r.dropped.toUpperCase(), {})}</td>
      <td>${signed(r.mean)}</td>
      <td class="${r.delta > 0 ? "" : "dim"}">${signed(r.delta)}</td>
    </tr>`).join("");
}

function renderCells() {
  const mode = document.getElementById("cells-filter").value;
  const sorted = [...state.cells].sort((a, b) => b.estimate - a.estimate);

  let shown = sorted;
  let note = "";
  if (mode === "extremes") {
    shown = [...sorted.slice(0, 10), ...sorted.slice(-10)];
    note = "as 10 mais optimistas e as 10 mais pessimistas";
  } else if (mode === "acted") {
    shown = sorted.filter((c) => c.accao !== "ESPERAR");
    note = "celulas onde a tabela decide agir";
  } else if (mode === "supported") {
    shown = sorted.filter((c) => c.weight >= 0.6);
    note = "celulas com peso proprio >= 0.60";
  }

  document.getElementById("cells-count").textContent =
    `-- ${shown.length} de ${state.cells.length} celulas`;
  document.getElementById("cells-note").textContent = note;

  document.querySelector("#cells-table tbody").innerHTML = shown.map((c) => `
    <tr>
      <td>${badge(c.curto)}</td>
      <td>${badge(c.rapido)}</td>
      <td>${badge(c.padrao)}</td>
      <td>${badge(c.regime)}</td>
      <td>${badge(c.alarme)}</td>
      <td>${c.n_days}</td>
      <td>${c.n_rows}</td>
      <td>${signed(c.cell_mean)}</td>
      <td>${signed(c.additive)}</td>
      <td>${c.weight.toFixed(2)}</td>
      <td><strong>${signed(c.estimate)}</strong></td>
      <td>${badge(c.accao, ACTION_CLASS)}</td>
    </tr>`).join("");
}

document.getElementById("cells-filter").addEventListener("change", renderCells);
load();
