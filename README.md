# TradingML

Simulação de trading (paper trading) orientada a machine learning tabular.
**Não há dinheiro real em lado nenhum deste projeto.** É uma simulação para
aprender ML aplicado a séries temporais financeiras.

## A ideia

1. Todos os dias, uma tarefa agendada do Claude Code pesquisa notícias sobre
   cada empresa da watchlist e responde a 5 perguntas booleanas (`good_company_news`,
   `bad_company_news`, `peer_impact_news`, `sector_momentum_positive`,
   `macro_event_today`) — ver `config.py`.
2. Essas respostas juntam-se a indicadores técnicos (médias móveis, RSI,
   volatilidade, retornos) num vetor de features tabular.
3. **Dois modelos de machine learning separados** — um de curto prazo (previsão
   da sessão seguinte) e um de longo prazo (previsão a ~20 sessões, ~1 mês) —
   aprendem a prever se o preço sobe ou desce.
4. Um simulador decide compra/venda com dinheiro fictício ($100.000 por
   horizonte) com base nessa previsão.
5. Um dashboard local mostra o estado das duas carteiras em tempo real.

Watchlist por omissão: `AAPL, MSFT, NVDA, AMZN, GOOGL, JPM, XOM, JNJ`
(editável em `config.py`).

## Arrancar do zero

```bash
py -m pip install -r requirements.txt
py scripts/bootstrap.py      # ~2 anos de historico, features, outcomes, treino inicial dos 2 modelos
py src/run_daily.py          # 1 ciclo diario: precos -> features -> previsao -> simulacao
py web/app.py                # abre http://localhost:5000
```

`scripts/bootstrap.py` é idempotente (podes correr outra vez sem duplicar
nada), mas só precisas de o correr uma vez a sério.

## Uso diário manual (antes de ativares a tarefa agendada)

A tarefa agendada vai escrever um ficheiro `data/incoming/<data>.json` com as
respostas do dia e depois correr estes dois comandos. Para testar isto à mão:

```bash
py src/record_daily_features.py --schema                        # ve o formato esperado
py src/record_daily_features.py --file data/incoming/2026-08-12.json
py src/run_daily.py
```

`record_daily_features.py` valida tudo (tickers, campos, tipos, datas) antes
de escrever na base de dados, numa única transação — um payload inválido não
escreve nada e explica exatamente o que corrigir.

## Os dois horizontes

| | Curto Prazo (SHORT) | Longo Prazo (LONG) |
|---|---|---|
| Prevê | sessão seguinte | ~20 sessões (~1 mês) |
| Features técnicas | SMA5/20, RSI14, retornos 1d/5d | SMA50/200, RSI50, retornos 20d/60d |
| Janela de notícias | só o dia | média dos últimos ~10 dias úteis |
| Stop-loss | -10% | -20% |
| Retreino | semanal | mensal |
| Carteira | $100.000 fictícios, independente | $100.000 fictícios, independente |

As duas carteiras são completamente independentes (podes estar "comprado" a
longo prazo e "de fora" a curto prazo na mesma ação ao mesmo tempo).

## Simplificações assumidas (deliberadas, não são bugs)

- **Sem comissões nem slippage.** Cada trade executa exatamente ao preço de
  fecho do dia.
- **Fills a close-to-close.** Não há feed intradiário — a decisão de um dia
  usa o último fecho conhecido como preço de referência/entrada. "Tempo real"
  aqui significa "atualizado uma vez por dia útil", não tick a tick (isso
  precisaria de um feed pago).
- **Preços sempre `auto_adjust=True`** (dividendos refletidos no preço como
  total return). Splits durante a simulação são raros a esta escala; se
  acontecerem, corrige manualmente re-ingerindo o histórico completo desse
  ticker.
- **É normal os modelos começarem perto de "coin flip" (~50%).** Os
  coeficientes das features de notícias vão estar perto de zero durante
  semanas/meses até haver dados suficientes — isso é o resultado honesto
  esperado, não uma falha a corrigir apertando o modelo.

## Testes

```bash
py -m pytest tests/ -v
```

Cobrem, entre outras coisas: que as features nunca usam o preço do próprio
dia que estão a prever (regressão anti-leakage), a matemática do simulador
(sizing, stop-loss exato, ledger de caixa) para os dois horizontes, e todos
os casos de validação do CLI de notícias diárias.

## Estrutura

```
config.py              watchlist, features booleanas, parametros por horizonte
db/                     schema.sql + ligacao SQLite (WAL)
src/ingest_prices.py    precos diarios via yfinance
src/features.py         indicadores tecnicos + agregacao de noticias por horizonte
src/record_daily_features.py   CLI validado que a tarefa agendada usa
src/outcomes.py         resolve os labels (nunca antes da sessao-alvo existir)
src/model.py            treino/previsao (LogisticRegression + RandomForest de comparacao)
src/simulator.py        motor de paper trading
src/run_daily.py        orquestrador diario (idempotente por data)
scripts/bootstrap.py    setup inicial do zero
web/                    dashboard Flask (so-leitura)
tests/                  pytest
```

## Tarefa agendada (automação diária real)

Ainda não está ativa. Quando quiseres ligar a recorrência automática (que
passa a consumir uso do Claude todos os dias úteis), pede para a configurar
via `/schedule` -- o prompt vai: pesquisar noticias recentes por ticker ->
escrever `data/incoming/<data>.json` -> `record_daily_features.py --file ...`
-> `run_daily.py` -> resumo pass/fail.
