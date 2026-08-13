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

20 padrões detetados: triângulos (ascendente/descendente/simétrico), bandeiras
(bull/bear), pennant, retângulos (topo/fundo), duplos topos/fundos, diamantes
(topo/fundo), head & shoulders (normal/invertido), cup with handle (normal/
invertido), **cunhas (ascendente/descendente)** e **broadening/megafone
(topo/fundo)**.

### Dois universos de tickers, e porquê

Os dois sistemas escalam ao contrário um do outro, por isso têm listas
separadas em `config.py`:

| | `WATCHLIST` (8) | `ALL_INSTRUMENTS` (43) |
|---|---|---|
| Usado por | notícias + carteiras simuladas | só deteção de padrões |
| Escala | mal — o agente Claude pesquisa notícias **empresa a empresa**, todos os dias úteis | bem — é geometria pura, só custa yfinance e CPU |
| Coluna `active` na BD | 1 | 0 nos 35 extra |

Se os 43 entrassem no sistema de notícias, o custo diário em uso do Claude
seria 5× maior sem qualquer benefício — o detetor de padrões não olha para
notícias. Mover um ticker entre universos é editar o `config.py`; a coluna
`active` é derivada daí a cada arranque.

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
baseline de "prever sempre o padrão mais frequente", **sempre com barra de
erro**:

| Timeframe | Cadeia de Markov | Baseline frequência | Lift | Veredicto |
|---|---|---|---|---|
| 1h | 18,6% | 19,4% | −0,9 pp ±1,2 pp | dentro do ruído |
| 5m | 16,5% | 17,2% | −0,7 pp ±1,5 pp | dentro do ruído |

**Conclusão honesta: saber o padrão atual não ajuda a prever o seguinte, para
lá de saber quais são os padrões mais comuns.** Isto está permanentemente
visível na faixa de topo da página `/patterns`, não escondido.

#### Retratação de um resultado anterior

Uma versão anterior deste README reportava **+2,7 pp de vantagem a 1h**. Esse
número **inverteu de sinal** assim que o detector de cunhas foi corrigido.
Estava dentro de ~2 erros-padrão desde o início e nunca tinha sido robusto —
foi reportado sem barra de erro, o que é precisamente como se vende ruído como
resultado. O `sequence.evaluate()` passou a devolver sempre `lift`,
`standard_error`, `significant` e `kappa`, e a interface só pinta o lift a
verde quando passa 2 erros-padrão.

#### O que mediu bem e o que não mediu

Alargar o universo de 8 para 43 tickers melhorou de forma real e verificável a
**qualidade estatística da matriz**, mesmo não tendo melhorado a previsão:

| | 8 tickers | 43 tickers |
|---|---|---|
| Padrões (1h / 5m) | 632 / 300 | 3.472 / 1.911 |
| Observações por célula (1h) | 1,69 | **9,20** |
| Suporte da transição principal | n=8 | **n=296** |

#### Porque é que "mais padrões" não deu mais acerto

Acrescentar cunhas e broadening foi **correcto em termos de detecção** — a
cunha descendente estava a ser classificada como `BULL_FLAG` (um bug real: o
teste de paralelismo por declives deixava passar canais convergentes), e a
cunha ascendente não era detectada de todo. Juntas são 18% das formações a 1h,
e o `BULL_FLAG` caiu de 965 para 630 quando deixaram de lhe ser atribuídas.

Mas em previsão **piorou**, porque 20 tipos são 400 células de matriz em vez de
256, com os mesmos dados. Testaram-se três alfabetos:

| Alfabeto | Células | 1h kappa | 5m kappa |
|---|---|---|---|
| 20 estados (actual) | 400 | −0,0095 | −0,0083 |
| 16 (cunhas fundidas nas bandeiras) | 256 | −0,0027 | −0,0041 |
| 3 (só o viés alta/baixa/neutro) | 9 | 0,0000 | +0,0107 |

**Nenhum bate o acaso.** E o alfabeto de 3 estados ilustra bem porque é que o
acerto bruto engana: dá 46,9% de acerto a 1h — muito melhor que os 18,6% — mas
o baseline também é 46,9%, logo kappa zero. Comparar acerto entre alfabetos de
tamanhos diferentes não quer dizer nada; **kappa** (acerto corrigido pelo
acaso) é o único número comparável, e é por isso que passou a ser reportado.

### O classificador multiclasse contextual — experiência feita, resultado negativo

Foi construído (`src/patterns/classifier.py` + `context.py`) e **medido**: um
classificador de 16 classes com 26 features — identidade do padrão em one-hot
mais qualidade da deteção, duração, amplitude, retorno do padrão, tendência
prévia a 20 e 60 barras, volatilidade prévia, rácio de volume, nº de pivots e
posição na sessão. Isto usa o volume, que até aí era ingerido e ignorado.

**Onde entraria na cadeia, e porque não em todo o lado:** só no **passo 1**. O
padrão atual já se formou, logo a sua qualidade e volume são factos medidos.
Nos passos 2–4 o padrão de partida é um padrão *previsto que ainda não
existe* — não tem qualidade nem volume. Atribuir-lhe valores "típicos" daria
números com ar mais informado sem informação nova nenhuma por trás. A cadeia de
Markov é exactamente a versão marginalizada sobre esse contexto, que é o que se
deve usar quando o contexto é desconhecido.

**Protocolo:** três partições cronológicas por ticker (60% treino, 20%
validação, 20% teste). O algoritmo e o peso do ensemble escolhem-se na
**validação**; o teste só é tocado para reportar.

Medido com os 20 tipos de padrão actuais:

| | 1h | 5m |
|---|---|---|
| Classificador | 16,8% | 14,4% |
| Cadeia de Markov | **22,1%** | **14,6%** |
| **Diferença vs Markov** | **−5,2 pp** | **−0,3 pp** |
| Erro-padrão | ±1,4 pp | ±1,8 pp |
| Veredicto | **significativamente PIOR** | dentro do ruído |

**O contexto não ajuda a prever o tipo do próximo padrão** — e a 1h chega a
prejudicar de forma estatisticamente significativa. Com 16 tipos a diferença
era +0,3 pp (ruído); ao passar para 20, o classificador degradou-se muito mais
que a cadeia de Markov, que tem o recuo para a marginal a protegê-la da
esparsidão. Por isso `PATTERN_SEQUENCE["use_classifier"]` está a `False` e a
produção continua na cadeia de Markov — a flag existe e funciona, é só voltar a
ligá-la se a medição alguma vez justificar.

Três coisas que valeu a pena descobrir pelo caminho:

- **A regressão logística deu 4,9%, abaixo do acaso (1/16 = 6,25%).** Bug meu:
  `class_weight="balanced"` com 16 classes muito desequilibradas amplifica
  ~100× as raras e o modelo passa a prevê-las quase sempre. Balancear serve
  para recall em minorias; aqui a métrica é accuracy.
- **O split a duas partições mentia.** Com o peso do ensemble escolhido no
  próprio teste, o ganho a 5m parecia **+2,9 pp**; com validação separada,
  sobraram **+0,3 pp**. Na validação o ensemble chegou a 23,4% e no teste
  intocado deu 16,2% — é assim que o *test-set tuning* se manifesta.
- **O modelo usa mesmo o contexto — 48% da massa dos coeficientes**, com
  `range_pct` a ser o maior coeficiente de todos, acima de qualquer feature de
  identidade. Ou seja: o contexto é preditivo de *alguma coisa*, só não do
  **tipo** do próximo padrão.

O `run_patterns.py` retreina e remede isto todos os dias, e o veredicto aparece
na faixa de topo da página `/patterns`. Quando o histórico acumulado tornar o
ganho superior a 2 erros-padrão, é uma flag a mudar.

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
  context.py            features de contexto por padrao (para o classificador)
  classifier.py         classificador multiclasse + protocolo de 3 particoes
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
