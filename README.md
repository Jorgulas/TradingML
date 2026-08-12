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

## Segundo sistema: padrões gráficos e previsão encadeada

Independente do de cima. Em vez de prever direção do preço, **deteta padrões
clássicos de análise técnica** e aprende **que padrão costuma seguir-se a
qual**, prevendo os 4 seguintes em cadeia.

16 padrões detetados: triângulos (ascendente/descendente/simétrico), bandeiras
(bull/bear), pennant, retângulos (topo/fundo), duplos topos/fundos, diamantes
(topo/fundo), head & shoulders (normal/invertido), cup with handle (normal/
invertido).

```bash
py src/patterns/ingest_intraday.py    # barras 1h (730d) e 5m (60d)
py src/patterns/backfill.py           # deteta padroes + treina matriz de transicoes
py src/patterns/run_patterns.py       # ciclo completo (corre no sync diario)
py web/app.py                         # depois abre http://localhost:5000/patterns
```

### Como funciona a cadeia de 4 passos

É uma **cadeia de Markov de 1ª ordem** sobre o alfabeto de padrões. O passo 1
sai de P(·|padrão atual); o passo 2 é condicionado no padrão **previsto** no
passo 1, e assim sucessivamente. Cada passo mostra dois números:

- **Condicional** — P(este padrão | o previsto no passo anterior)
- **Acumulada** — o produto de todas as condicionais até ali, ou seja a
  probabilidade da cadeia **inteira** acontecer

A acumulada cai depressa (ex.: 21% → 4,5% → 1,3% → 0,3%) e isso não é um
defeito: acertar quatro eventos incertos seguidos é mesmo improvável. Mostrar
só a condicional daria uma falsa sensação de certeza no passo 4.

Cada número vem acompanhado do **suporte (n)** — quantas transições reais o
sustentam. 60% assente em 2 observações não vale nada, e a interface marca
esses passos como "suporte baixo".

### Parâmetros e como foram escolhidos

Não são valores por omissão arbitrários — foram calibrados a olhar para
quantos padrões cada combinação produz no histórico real e para a métrica de
avaliação. Estão todos em `config.py` (`PATTERN_DETECTION`, `PATTERN_SEQUENCE`).

| Parâmetro | Valor | Porquê |
|---|---|---|
| `pivot_window` | 5 (1h) / 3 (5m) | janela do fractal; menor = mais pivots e mais ruído |
| `flat_slope_max` / `min_slope` | 0,00035 / 0,0009 | **zona morta deliberada** entre eles: sem separação, declives ambíguos caíam sempre no primeiro ramo e nunca se detetava um triângulo simétrico |
| `level_tolerance` | 2% | dois topos "ao mesmo nível" |
| `min_r2` | 0,70 | qualidade do ajuste das trendlines |
| `cup_min_r2` | 0,55 | parábola sobre preços reais é mais ruidosa que uma reta |
| `min_pattern_bars` / `max` | 12 / 150 | abaixo é ruído, acima já não é uma formação única |
| `flagpole_min_move` | 3% | distingue uma bandeira de um retângulo: **a mesma geometria de canal é bandeira se houver um movimento forte antes, retângulo se não houver** |
| `backoff_k` | 10 | força do recuo para a distribuição marginal (ver abaixo) |
| `beam_width` | 5 | caminhos mantidos vivos na beam search |

**Suavização:** interpolação de Jelinek-Mercer, não Dirichlet-para-uniforme.
Com ~300 padrões espalhados por 256 células, muitos estados de partida têm 2-3
observações. Recuar para a uniforme aí seria afirmar que todos os padrões são
igualmente prováveis — o que é *pior* do que o que já se sabe. Recuar para a
marginal observada faz a cadeia degradar suavemente até ao baseline em vez de
para baixo dele.

**Critério de desempate entre leituras sobrepostas:** um head & shoulders
contém sempre, geometricamente, um duplo topo e vários triângulos dentro de
si. Testaram-se quatro critérios; o escolhido pondera a qualidade pelo número
de pivots que sustentam *cada deteção concreta* (não pelo tipo de padrão — um
duplo topo são sempre 3 pivots, mas um triângulo pode ter 8, e nesse caso é a
leitura mais rica). Os outros três estão documentados em
`src/patterns/scanner.py` com o motivo de terem sido rejeitados.

### Resultado honesto da avaliação

Walk-forward por ticker (70% treino / 30% teste), top-1 accuracy contra o
baseline de "prever sempre o padrão mais frequente":

| Timeframe | Cadeia de Markov | Baseline frequência | Diferença |
|---|---|---|---|
| 1h | 34,9% | 33,3% | **+1,6 pp** |
| 5m | 18,7% | 20,9% | **−2,2 pp** |

**Ou seja: saber o padrão atual quase não ajuda a prever o seguinte, para lá
de saber quais são os padrões mais comuns.** A 1h ganha por uma margem
pequena; a 5m perde. Isto está visível permanentemente na faixa de topo da
página `/patterns`, não escondido — é o mesmo tipo de resultado honesto que os
~50% do primeiro sistema. Com 300-630 padrões espalhados por 256 células da
matriz, não há dados suficientes para mais; acumular mais meses de histórico é
o que pode mudar o quadro.

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
scripts/bootstrap.py    setup inicial do zero (inclui o subsistema de padroes)
scripts/local_daily_sync.py   o que a tarefa do Windows corre todas as manhas
web/                    dashboard Flask (so-leitura): / carteiras, /patterns padroes
tests/                  pytest

src/patterns/           SUBSISTEMA DE PADROES GRAFICOS
  ingest_intraday.py    barras 1h e 5m via yfinance
  pivots.py             deteccao de swing highs/lows (fractais) + lag de confirmacao
  detectors.py          geometria dos 16 padroes, cada um com quality score
  scanner.py            varre o historico, resolve sobreposicoes, sequencia limpa
  sequence.py           cadeia de Markov + beam search 4 passos + avaliacao
  live.py               deteccao + previsao em tempo real (<100ms, matriz em cache)
  backfill.py           redeteccao completa do historico
  run_patterns.py       ciclo completo, corre a seguir ao pipeline diario
```

## Tarefa agendada (automação diária real) — ATIVA

Duas peças, porque a rotina cloud não tem acesso a este PC/BD local:

1. **Rotina cloud** (`TradingML - avaliacao diaria de noticias`,
   https://claude.ai/code/routines/trig_01BFYXPMEX3E7nbT2RRUM5Cy) — dias
   úteis às 21:30 UTC (depois do fecho de Wall Street + after-hours). Lê
   `config.py` do repositório, pesquisa notícias por ticker, escreve
   `data/incoming/<data>.json` e faz commit+push só desse ficheiro para
   `Jorgulas/TradingML`. Gerível em https://claude.ai/code/routines.
2. **Tarefa local** (`TradingML Daily Sync`, Agendador de Tarefas do
   Windows) — dias úteis às 08:30 hora local. Corre `scripts/local_daily_sync.py`:
   `git pull` → ingere o JSON do dia se já lá estiver → corre `run_daily.py`
   sempre (booleanos ficam neutros se as notícias ainda não tiverem chegado)
   → corre `run_patterns.py` (uma falha aqui não invalida as carteiras, que
   já ficaram atualizadas). Só corre com sessão Windows iniciada (não guarda
   a password do Windows).

Para veres o estado: `Get-ScheduledTask -TaskName "TradingML Daily Sync"` no
PowerShell, ou o painel de estado do sistema no topo do dashboard.
