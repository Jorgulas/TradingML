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

## Terceiro alvo: direção do preço depois do padrão

`src/patterns/direction.py`. Alvo **binário** em vez de 20 classes: *o preço
sobe ou desce nas H barras a seguir a este padrão?* Usa as mesmas 26 features
de contexto e os mesmos ~3.400 padrões, mas deixa de os diluir por 20 classes.
O rótulo é quase equilibrado (51,6% de subidas a H=20), que é o cenário ideal
para classificação binária.

### A descoberta que enquadra tudo o resto: amostra efetiva

| | Padrões | **Dias distintos** |
|---|---|---|
| 1h | 3.447 | **714** |
| 5m | 1.911 | **61** |

Os 43 tickers movem-se todos com o mercado no mesmo dia, logo observações do
mesmo dia **não são independentes** — a amostra efetiva conta-se em dias.

A 5m isto é fatal: n efetivo ≈ 61 dá erro-padrão ~6,4 pp, seria preciso uma
vantagem de ~13 pp para detetar seja o que for. **Nenhum resultado a 5m é
mensurável**, o que explica retroativamente porque é que tudo o que medi a 5m
deu ruído. Por isso `PATTERN_DIRECTION["timeframes"] = ["1h"]`.

### Três armadilhas que este módulo evita

1. **Preço de entrada em `confirmed_ts`, não no fim do padrão.** O padrão acaba
   em `end_ts`, mas só se *sabe* que existe `pivot_window` barras depois. Usar
   o preço do fim é lookahead puro. Há um teste dedicado a isto.
2. **Embargo no split.** Um padrão no fim do treino tem o resultado a acontecer
   dentro da validação — partilhariam as mesmas barras futuras. Descartam-se as
   observações cuja janela atravessa a fronteira.
3. **Erro-padrão sobre dias, não sobre padrões.** Calculá-lo sobre 3.400
   padrões daria ±0,9 pp em vez dos ±4,0 pp reais.

### Resultados (protocolo de 3 partições, horizonte escolhido na validação)

Escolhido pela validação: **H=5 barras, neutro ao mercado, regressão logística**.

| | Valor |
|---|---|
| Acerto | 54,9% |
| Baseline (classe maioritária) | 51,1% |
| Lift | **+3,8 pp ±4,0 pp** |
| AUC | 0,536 |
| Controlo (instantes aleatórios) | 48,9% |
| Retorno médio previsto em alta / baixa | +0,201% / −0,379% |
| **Veredicto** | **dentro do ruído** |

Todas as 8 combinações (4 horizontes × bruto/neutro) ficam dentro do ruído. A
melhor AUC é 0,551 (H=5, bruto) — um cabelo acima do acaso.

### Mas desta vez o "sem sinal" tem uma leitura diferente

O **efeito mínimo detetável é 8,1 pp**. Para detetar uma vantagem de 2 pp — a
ordem de grandeza do que plausivelmente existiria — seriam precisos ~2.500 dias
de teste (≈10 anos). Temos 154.

Ou seja: isto **não é "não há sinal", é "não consigo ver sinal desta ordem"**.
É uma afirmação diferente e mais honesta do que a que o alvo de 20 classes
permitia. O sinal mais encorajador é o retorno condicional — prever alta dá em
média +0,20% e prever baixa −0,38%, um diferencial de 0,58 pp com o sinal certo
— mas não é distinguível de sorte com esta amostra.

### Terceira carteira: negociar o sinal

`src/patterns/portfolio.py`. **Não** reutiliza o `simulator.py` das outras duas
carteiras: aquele opera em barras diárias, e o horizonte selecionado é de 5
barras de 1h (~5 horas de mercado). Manter a posição um dia inteiro
ultrapassaria o que o modelo prevê, e o backtest deixaria de testar o sinal
para testar uma aproximação. A regra aqui é a do modelo: **entra na confirmação
do padrão, sai exatamente `horizon_bars` barras depois**.

Treina em treino+validação, negoceia **só no período de teste**. Sem comissões
nem slippage.

| | |
|---|---|
| Período | 2025-12-19 a 2026-08-13 |
| Trades | 145 (91 dias distintos) |
| Taxa de acerto | 57,9% |
| Retorno total | **+2,40%** |
| Retorno médio/trade | **+0,207%**, t=0,88 → dentro do ruído |
| Entradas ao acaso | +0,0004%/trade |
| Buy & hold do índice | +14,46% |

**Dois números sem os quais isto se lê mal:**

- **A exposição média é 6,6%.** Com um horizonte de poucas horas a carteira
  está em caixa quase sempre, portanto comparar o seu retorno total com um buy
  & hold 100% investido **não é comparação justa**. Os +2,40% vêm de capital
  parado, não de um sinal fraco.
- **O t-statistic é calculado sobre dias, não sobre trades.** Sobre 145 trades
  daria t=1,11; sobre os 91 dias distintos dá **t=0,88**. Vários trades do
  mesmo dia partilham o movimento do mercado e não são independentes.

O sinal aponta na direção certa — +0,207%/trade contra +0,0004% de entradas ao
acaso, e 57,9% de acerto contra ~51% de base — mas com t=0,88 não é
distinguível de sorte. Coerente com tudo o resto: o desenho está correto, o que
falta é histórico.

## Janela de tempo real — o algoritmo a tentar acertar, ao vivo

`src/patterns/realtime.py` + página **`/live`**.

```bash
py src/patterns/realtime.py --mode replay          # ver funcionar agora
py src/patterns/realtime.py --mode replay --bars 400 --delay 0.4
py src/patterns/realtime.py --mode live            # durante o mercado aberto
```

**Porque é que esta é a avaliação mais honesta do projeto:** cada previsão é
gravada **antes de a barra que a resolve existir**. Não há lookahead possível —
nem por engano, nem por um bug subtil no split. É o único número aqui que não
pode ser viciado.

### Dois modos

| | |
|---|---|
| **live** | Busca as barras mais recentes de minuto a minuto. Só com o mercado aberto, e os dados do yfinance vêm com **15–60 min de atraso** (medido). "Tempo real" aqui é tempo real *atrasado*, não tick a tick — está dito na própria página. |
| **replay** | Percorre barras históricas uma a uma como se estivessem a chegar. Serve para ver o mecanismo com o mercado fechado. Garantia crítica: em cada passo o detetor só vê `bars[:cursor+1]` — há testes dedicados a isso. |

### O que a página mostra

- **Margem de acerto com intervalo de confiança de Wilson.** É o elemento mais
  importante e está do mesmo tamanho que o número principal. Numa corrida real:
  25% de acerto com intervalo **4,6% – 69,9%** (n=4). Uma interface ingénua
  diria "25%, o modelo é péssimo"; a honesta diz "ainda não sabes nada". O
  intervalo estreitou de 79pp para 62pp ao fim de 6 resoluções.
- **Curva de aprendizagem** — acerto acumulado com a banda de confiança e a
  linha do acaso a 50%.
- **Previsões abertas** a aguardar resultado, e as já resolvidas com acerto/falha.
- **Contador de atualizações online** do modelo.

### Aprendizagem online

Cada resultado que chega atualiza o modelo por `partial_fit` (SGD logístico),
arrancando dos coeficientes do modelo treinado nos ~1.900 exemplos históricos.
É isto que faz a margem mexer enquanto corre. Timeframe de 5m com horizonte de
6 barras (30 min) — a 1h cada previsão demoraria 5 horas a resolver e não havia
nada para ver.

## Árvores de decisão (XGBoost) — três classes com "esperar"

`src/trees/` + página **`/trees`**.

```bash
py src/trees/train.py      # treina os 4 modelos com varrimento de profundidade
py src/trees/live.py       # decisões agora, por ticker
```

Duas árvores sobre a **mesma** janela de tempo, a descrevê-la de maneiras
diferentes:

- **Indicadores** (30 features) — médias móveis, MACD, RSI, Bollinger, ATR,
  volume, volatilidade. A leitura "de analista".
- **Forma da janela** (28 features) — sem indicador nenhum. Conta quantos dos
  últimos 10 movimentos foram positivos/negativos, mede a **inércia** (corrida
  atual no mesmo sentido), a média, o desvio **inicial** e **final**, o declive
  e a aceleração — e recebe o sinal de cada barra como feature própria
  (`sign_lag1..sign_lag10`), que é como se dá a sequência `+ + − + − −` a uma
  árvore para ela ramificar.

**"Esperar" é uma decisão a sério**, não um caixote de restos: quando o
movimento seguinte cabe dentro de uma banda escalada pela volatilidade
recente, não agir é a resposta certa. A banda é adaptativa e não fixa — 5
pontos base são muito num minuto calmo e nada num agitado. Distribuição
resultante: **DESCE 28,9% / ESPERAR 42,4% / SOBE 28,7%**.

### A métrica que decide não é a accuracy

Com ESPERAR a ser ~42% das barras, responder sempre ESPERAR dá 42% de accuracy
e **nunca actua**. O número que interessa é a **precisão quando o modelo se
atreve a decidir**, comparada com a taxa-base dessa classe:

| Modelo | TF | Decide | Precisão | Ao calhas | **Lift** | Dias efetivos |
|---|---|---|---|---|---|---|
| Indicadores | 5m | 15,2% | 39,3% | 27,2% | **+12,1 pp** | 14 |
| Indicadores | 1m | 19,4% | 35,3% | 28,8% | +6,4 pp | **2** |
| Forma da janela | 5m | 5,0% | 33,0% | 27,1% | +5,9 pp | 14 |
| Forma da janela | 1m | 3,2% | 32,7% | 28,8% | +3,9 pp | **2** |

**Os indicadores ganham à forma da janela** por cerca do dobro do lift, nos
dois timeframes. A forma da janela é bem mais conservadora — só se atreve em
3-5% das barras.

### "Mais detalhado = mais lento mas mais preciso"? Metade certo

Varrimento de profundidade, medido (indicadores @ 5m):

| Profundidade | Precisão | Decide | Treino | Nós |
|---|---|---|---|---|
| 2 | **38,8%** | 13,0% | 2,2s | 6.300 |
| 4 | 38,0% | 15,7% | 2,9s | 27.432 |
| 6 | 36,7% | 17,9% | 4,0s | 99.216 |
| 10 | 35,0% | 23,5% | 11,1s | **747.012** |

**A parte do "mais lento" confirma-se** — 5× mais tempo e **119× mais nós**.
**A parte do "mais preciso" não** — a precisão *desce* monotonicamente. O mesmo
padrão aparece nos quatro modelos.

O que a profundidade aumenta mesmo é a **taxa de decisão** (13% → 23,5%): as
árvores profundas atrevem-se mais e acertam menos. É a assinatura clássica de
overfitting — mais confiantes e mais erradas. Por isso a escolha da
profundidade é feita pela **precisão accionável na validação**, e escolheu 2 a
4 na maioria dos casos.

### Limite honesto: dados ao segundo não existem

O yfinance rejeita `interval=1s` explicitamente. O mais fino é 1m, e só dá
**7 dias** de calendário. Daí o contraste na tabela: 1m tem 117 mil linhas mas
**2 dias efectivos no teste** (erro-padrão ±35 pp — nada é mensurável); 5m tem
14 dias (±13 pp). São as **linhas** que alimentam a capacidade da árvore, mas
são os **dias** que dão poder estatístico. Nenhum destes lifts está
estatisticamente estabelecido.

### Consenso entre os dois modelos

A página mostra os dois lado a lado. O **consenso** só se atreve quando ambos
dizem o mesmo e nenhum diz ESPERAR — duas leituras diferentes do mesmo gráfico
a concordarem valem mais do que qualquer uma sozinha. Latência de inferência:
**0,0007–0,0015 ms por linha**, folgadíssimo para tempo real.

### O resultado mais forte do projeto — e o que ele realmente significa

Depois de alargar para **161 tickers**, acrescentar **contexto de mercado** (relação
com o índice: beta, correlação, retorno em excesso, força relativa) e treinar
também em **1h** (onde há 741 dias de calendário em vez de 7):

**Indicadores @ 1h — 473.097 linhas de treino, 146 dias efectivos no teste:**
decide em 14,3% das barras e acerta **43,8%**, contra uma taxa-base de 23,0%.
São **+20,8 pp com erro-padrão de ±4,1 pp — cinco erros-padrão.** O primeiro
resultado estatisticamente significativo de todo o projeto.

A curva de precisão/cobertura é monotónica, que é a assinatura de sinal real
(ruído daria curva plana):

| Limiar | Cobertura | Precisão | Lift |
|---|---|---|---|
| 0,34 | 14,2% | 43,9% | +20,8 pp |
| 0,40 | 10,6% | 45,6% | +22,5 pp |
| 0,45 | 2,8% | 48,7% | +25,6 pp |
| 0,50 | 0,1% | **59,8%** | +36,8 pp |

#### Mas a decomposição muda o que isto quer dizer

As features de topo são quase todas de **volatilidade e volume** (`vol_20`,
`atr14_norm`, `volume_z20`, `bb_width`), não direcionais. Isso levantou uma
hipótese que teve de ser testada, e confirmou-se:

| Componente | Valor | Base |
|---|---|---|
| P(houve mesmo movimento \| modelo decidiu) | **82,0%** | 46,0% |
| P(direção certa \| houve movimento) | **53,5%** | 50% |
| **Produto = precisão** | **43,8%** | |

**O modelo é excelente a prever _que_ vai haver movimento** — +36 pp sobre a
taxa-base, muito para lá do ruído. Isso é *clustering de volatilidade*, um
efeito bem documentado e genuinamente forte.

**Mas a direção fica em 53,5%** — +3,5 pp contra ±4,1 pp de erro-padrão, ou
seja **dentro do ruído**.

Os +20,8 pp que pareciam uma vitória são quase todos previsão de
**volatilidade**, não de **direção**. Ler a precisão sem esta separação seria
confundir as duas coisas — por isso a decomposição é agora uma métrica de
primeira classe, gravada na BD e destacada na página.

Isto não é um resultado inútil: saber que vem movimento é acionável para
dimensionamento de posição e gestão de risco. Só não é o mesmo que saber para
onde.

### Mudar o alvo: magnitude em vez de direção

`src/trees/magnitude.py`. Se o sinal é de magnitude, o alvo deve ser magnitude.
Dois alvos sobre exatamente as mesmas features:

- **Classificação** — `big_move`: o movimento sai da banda de indiferença?
- **Regressão** — `magnitude_ratio`: vale quantos desvios-padrão recentes?
  Diz **quanto**, não apenas **se**.

A baseline exigente aqui não é a classe maioritária: é a **persistência de
volatilidade** (uma única feature). A volatilidade é fortemente
autocorrelacionada, portanto qualquer modelo tem obrigação de bater isso —
senão as 40 features não acrescentam nada a uma regra de uma linha.

| Modelo | TF | Accuracy | Base | AUC | vs baseline vol | Spearman | Decil topo/fundo | Dias |
|---|---|---|---|---|---|---|---|---|
| **Indicadores** | **1h** | **66,7%** | 54,0% | **0,728** | 0,578 | **+0,466** | **9,04×** | 146 |
| Forma da janela | 1h | 59,0% | 54,0% | 0,624 | 0,578 | +0,191 | 2,37× | 146 |
| Indicadores | 5m | 60,6% | 54,3% | 0,646 | 0,555 | +0,326 | 5,90× | 14 |
| Indicadores | 1m | 59,7% | 57,7% | 0,596 | 0,457 | +0,214 | 2,51× | **2** |

**Indicadores @ 1h: +12,7 pp sobre a base com erro-padrão de ±4,1 pp — 3,1
desvios, estatisticamente significativo.** E bate a persistência de
volatilidade por 0,15 de AUC, o que prova que as features acrescentam mesmo
informação além de "esteve volátil, continua volátil".

#### A calibração é o mais notável

Ordenando as barras pela magnitude prevista, o previsto e o realizado
acompanham-se quase exatamente:

| Decil | Previsto | Realizado |
|---|---|---|
| 1 | 0,27 | 0,28 |
| 5 | 0,51 | 0,51 |
| 8 | 0,74 | 0,75 |
| 9 | 1,09 | 1,21 |
| 10 | 2,17 | 2,50 |

**O decil de topo move-se 9× mais que o de baixo.** Não é só ordenar bem — os
valores estão certos. (Ligeira subestimação nos extremos, o que é o
comportamento habitual de modelos de regressão em caudas.)

E a curva de cobertura permite escolher o compromisso:

| Limiar | Cobertura | Precisão | Lift |
|---|---|---|---|
| 0,50 | 35,5% | 67,8% | +21,9 pp |
| 0,70 | 13,0% | 83,0% | +37,0 pp |
| 0,80 | 7,0% | **85,9%** | +39,9 pp |

#### O que isto é e o que não é

É um bom preditor de **volatilidade a uma hora**, com utilidade real:
dimensionamento de posição, gestão de risco, e sobretudo opções, onde a
volatilidade é o que se negoceia diretamente.

**Não diz para que lado.** Nada aqui contradiz o resultado anterior — a
direção continua a ~53%, dentro do ruído. O que mudou foi medir a coisa certa
em vez de dar à direção o crédito da volatilidade.

## Camada de decisão — tabela de verdade sobre todos os algoritmos

Até aqui cada algoritmo vivia sozinho e era avaliado sozinho. Esta camada
põe-nos todos a votar no **mesmo instante** e responde a uma pergunta que
nenhum deles consegue responder por si: quando é que a combinação das opiniões
vale mais do que a melhor opinião isolada?

```bash
py src/decision/short_term.py          # o algoritmo de 5 dias, isolado
py src/decision/signals.py             # distribuicao dos cinco veredictos
py src/decision/truth_table.py         # a tabela + ablacao + teste de interaccao
py src/decision/walkforward.py         # avaliacao trimestral + estabilidade
py src/decision/run_decision.py --walkforward   # calcula e grava para a web
```

Depois, `/decisao` no dashboard.

### O horizonte que faltava

Os prazos existentes eram os extremos: as árvores decidem a barra seguinte
(1 minuto) e a carteira LONG decide 20 dias. O novo modelo prevê **5 dias de
bolsa**, em **excesso sobre o índice**, com três classes e `ESPERAR` como
decisão a sério.

O alvo é neutro ao mercado por uma razão medida: a 20 barras, 53,7% das janelas
deste universo são de subida apenas porque o mercado subiu. Um modelo treinado
no retorno bruto aprende essa deriva e mais nada — e parece bom a fazê-lo.

### Features transversais (a novidade que produziu o resultado)

Todo o resto do projecto olhou para cada acção **ao longo do tempo**, e a
direcção temporal mostrou-se imprevisível em seis rondas de investigação. Estas
features fazem a outra pergunta: dentro de cada dia, em que **percentil do
universo** está esta acção em retorno, volatilidade, força relativa e volume?

É market-neutral por construção (o factor comum desaparece ao ordenar), e foi o
que funcionou:

| Conjunto | n features | IC diário | long-short a 5 dias |
|---|---|---|---|
| temporal | 48 | +0,0094 | +0,42% |
| **transversal** | **11** | **+0,0232** | **+0,77%** |
| ambos | 59 | +0,0112 | +0,12% |

Juntar os dois é **pior** do que usar só o transversal — as 48 features
temporais são ruído em que a árvore se afoga.

### Os cinco sinais

Cada algoritmo reduz-se a um veredicto, tudo avaliado ao fecho de cada dia:

| Sinal | Origem | Valores |
|---|---|---|
| `curto` | modelo de 5 dias | SOBE / NEUTRO / DESCE |
| `rapido` | modelo de 1 dia | SOBE / NEUTRO / DESCE |
| `padrao` | padrão gráfico confirmado nos últimos 10 dias | ALTA / NENHUM / BAIXA |
| `regime` | volatilidade do mercado em tercis | CALMO / NORMAL / AGITADO |
| `alarme` | risco de cauda (calendário de resultados) | SIM / NÃO |

`curto` e `rapido` são **transversais**: "SOBE" quer dizer *"está no tercil de
topo do universo hoje"*, não *"a probabilidade passou de 0,6"*. Assim o
veredicto não muda de sentido quando o mercado inteiro fica optimista.

Tudo o que define um veredicto — modelos, tercis, limiares — é estimado **só na
porção de treino**. Um tercil calculado sobre a série inteira já contém o
futuro.

### As três defesas contra 162 células

3×3×3×3×2 = 162 combinações sobre 144 dias de teste dá menos de um dia por
célula. Sem defesas, algumas células teriam números espectaculares por acaso e
escolher a melhor seria garantir que se escolhe a mais sortuda.

1. **Suporte contado em DIAS, não em linhas.** Uma célula com 800 linhas
   concentradas em 6 dias tem 6 observações — as 158 acções movem-se juntas.
2. **Encolhimento para o modelo aditivo.** Cada célula usa
   `peso × média_da_célula + (1 − peso) × votação_simples`, com
   `peso = n_dias / (n_dias + 40)`. Uma célula rara diz exactamente o que a
   votação diria; só as bem observadas se afastam dela.
3. **Cada esquema escolhe o seu próprio limiar** na validação. Sem isto, um
   limiar afinado para a tabela põe todos os sinais isolados a `ESPERAR`
   sempre, e a tabela "ganha" contra adversários que nunca chegaram a jogar.

### O resultado

Avaliação em **walk-forward** (re-treino trimestral, cada trimestre pontuado
por modelos que nunca o viram): 349 dias fora da amostra, ~69 janelas
independentes contra as 28 de um teste único.

| Esquema | Retorno/decisão | IC95% (blocos) |
|---|---|---|
| tabela crua | +0,527% | [+0,125%, +0,974%] |
| tabela encolhida | +0,449% | [−0,007%, +0,892%] |
| votação simples | +0,415% | [+0,082%, +0,739%] |
| **só o sinal `curto`** | **+0,383%** | **[+0,104%, +0,665%]** |

**Tabela menos votação: +0,033%, IC95% [−0,235%, +0,317%].**

O sinal de curto prazo é o primeiro resultado deste projecto com intervalo de
confiança inteiramente acima de zero fora da amostra. A tabela **não** melhora
sobre ele de forma demonstrável: repetindo o walk-forward só com outra semente
do XGBoost, a diferença entre esquemas move-se 0,135% — mais do que a própria
diferença (0,089%).

> **Corrigido pela secção seguinte.** Cerca de metade deste número é viés de
> sobrevivência. Repondo as acções que caíram do índice, o efeito cai para
> perto de metade e o intervalo passa a incluir o zero.

### Bootstrap de blocos (e o erro que ele corrigiu)

Forma-se carteira todos os dias e mede-se o que ela rende nos 5 dias seguintes:
a observação de segunda e a de terça partilham 4 dos 5 dias. Reamostrar dias
soltos assume 144 observações independentes quando há ~29, e estreita o
intervalo por cerca de √5.

A primeira versão desta experiência deu `+0,77%, IC95% [+0,17%, +1,38%]` —
significativo. Com bootstrap de blocos: `[−0,16%, +1,78%]`, não significativo.
O ponto estimado não mudou; a honestidade do intervalo mudou.

### Ressalvas

- **Viés de sobrevivência**: medido na secção seguinte. Explica cerca de
  metade do efeito, e o resto não é distinguível de zero.
- **Lookahead residual** no calendário de resultados (afecta só o `alarme`).
- **Três sementes não são uma distribuição**: a amplitude reportada é um piso
  para a incerteza.

## Viés de sobrevivência — medido, não assumido

```bash
py src/research/fallen_universe.py    # busca as accoes que cairam do universo
py src/research/survivorship.py       # os quatro testes
```

O painel da camada de decisão tem 158 grandes capitalizações de **hoje**. Das
158, todas têm histórico completo e todas chegam vivas ao último dia: zero
atrito em três anos, quando um índice real perde 5 a 8% dos membros por ano. A
lista foi escolhida no fim, e isso tinha de ser medido.

### Duas descobertas sobre os dados

**O yfinance apaga as empresas delistadas.** Das 20 que sabidamente saíram do
S&P 500 neste período por falência ou aquisição — SIVB, FRC, ATVI, PXD, HES,
MRO, CTLT, JNPR, WBA, DISH, RAD, YELL, WE, FSR, NKLA, MPW, SAVE, TUP, BIG —
**nenhuma devolve dados**. Nenhuma reconstrução completa é possível.

**Um ticker corrompido quase produziu a conclusão errada.** A BYND entrou no
painel com um desdobramento inverso aplicado de forma inconsistente pela fonte:
0,55 → 17,85 → 0,56 dólares em dias seguidos, seis vezes. Retornos de +2900%
seguidos de −97%. Esse ticker sozinho gerou **99%** de um colapso de −8,9% num
trimestre, e quase levou à conclusão de que o modelo não generalizava.
`src/decision/panel.py` passou a rejeitar qualquer série com variação diária
acima de 100% e a reportar quem rejeitou — perder um ticker em silêncio é pior
do que o incluir.

### O resultado

Quatro universos, mesmo protocolo, média de três sementes:

| Universo | n | Retorno 5d | IC95% | Amplitude entre sementes |
|---|---|---|---|---|
| A. só sobreviventes | 158 | **+0,694%** | [+0,192%, +1,202%] | 0,070% |
| B. + caídas do índice | 201 | **+0,353%** | [+0,026%, +0,703%] | 0,082% |
| C. + especulativas | 171 | +0,403% | [+0,056%, +0,699%] | 0,125% |
| D. + ambas | 214 | **+0,241%** | [−0,048%, +0,613%] | 0,138% |

A ordem repete-se nas três sementes e os intervalos entre universos são 4 a 5×
maiores do que a dispersão entre sementes — **a diferença é real**.

**Cerca de metade do efeito era a lista de acções.** Repor os ex-membros do
índice remove 49%; o universo mais completo remove 65% e o intervalo passa a
incluir o zero. O que sobra continua positivo e estável, mas é pequeno de mais
para esta amostra o resolver. E é um limite superior: as 19 delistadas
continuam irrecuperáveis, e **todas** as adições que foi possível fazer
baixaram o efeito.

### Os outros três testes

- **De que perna vem o resultado:** comprada 52,1%, vendida 47,9%. Metade vem
  do lado que a ausência de falidos afecta.
- **O modelo evita os desastres?** Não: põe as quedas violentas em 0,6175 de
  posto de confiança, ou seja *acima* da mediana (p = 1,2e−15). Carrega em
  magnitude, com margem direccional fina.
- **Concentra-se nas acções seguras?** Não: o IC é maior no grupo de maior
  drawdown (+0,0516) do que no mais sólido (+0,0267).

## Ângulo da tangente à média móvel

```bash
py src/research/tangent_angle.py --timeframe 1h --horizons 1,3,7
py src/research/tangent_angle.py --timeframe 1d --horizons 1,3,5
```

Desenha-se a média móvel e, no último ponto, mede-se o **ângulo entre a recta do
troço anterior e a do troço recente**. Média plana que passa a subir dá um
"joelho" positivo; quando volta a ficar plana, negativo. Limiar com zona morta,
e acima dele uma **moeda enviesada** para o lado do joelho.

### O ângulo tem de ser normalizado

Um ângulo exige os dois eixos nas mesmas unidades, e preço e tempo não são. Uma
acção de $500 e uma de $5 com o mesmo movimento relativo dariam ângulos
diferentes, e um limiar de 25° dependeria da escala do gráfico.

O declive é medido em **volatilidade por barra**, sobre **log-preço**: 45°
significa "a média avança um movimento típico por barra", igual em qualquer
acção. A primeira versão media sobre o preço e dividia por `volatilidade ×
preço`; um teste apanhou que uma subida perfeitamente constante dava joelho de
16,7° em vez de zero — o denominador andava durante a tendência e fabricava um
joelho inexistente.

Note-se também que **`arctan` é monótona**: "ângulo > 25°" é idêntico a "declive
normalizado > 0,466". O ângulo absoluto é momentum com outro nome. O que é novo
é o ângulo *entre dois troços* — curvatura, não direcção.

### Duas propriedades do mecanismo

**O joelho só dispara na viragem.** É a diferença entre duas rectas separadas
por `span` barras; assim que a subida dura mais do que `span`, as duas rectas
voltam a ser a mesma e o joelho colapsa, mesmo com a média a subir a pique. É um
impulso, não um estado — daí exposições de 2 a 27%.

**O limiar de 25° quase nunca dispara.** Com ruído realista, uma tendência forte
(1% de deriva por barra contra 0,8% de ruído) dá um joelho de pico a rondar os
20°. O varrimento de 972 configurações escolheu sempre 5°.

### O resultado

| Timeframe | Regra | Retorno | IC95% | Exposição |
|---|---|---|---|---|
| 1h | sempre comprado | +0,045% | [−0,639%, +0,613%] | 100% |
| 1h | declive | −0,069% | [−0,260%, +0,082%] | 24,8% |
| 1h | **joelho** | +0,008% | [−0,255%, +0,207%] | 27,0% |
| 1d | −declive | −0,068% | [−0,215%, +0,092%] | 33,6% |
| 1d | **−joelho** | −0,031% | [−0,165%, +0,104%] | 40,8% |

Nada significativo, e ao prazo diário as regras escolhidas foram as
**invertidas** — o varrimento decidiu sozinho que o lado rentável era vender a
média a subir.

Medindo o sinal directamente, sem regra pelo meio:

| Sinal | 1h (h=1/3/7) | 1d (h=1/3/5) |
|---|---|---|
| **joelho** | −0,0005 / −0,0006 / −0,0004 | +0,0034 / +0,0054 / +0,0086 |
| ângulo | −0,0078 / −0,0162 / −0,0270 | +0,0172 / +0,0204 / +0,0274 |

**O joelho é zero** — Spearman entre −0,0006 e +0,0086 sobre 790 mil
observações. Nulo medido com poder, não falta de dados. O ângulo tem sinal
pequeno e de **sentido instável** (reversão a 1h, momentum ao dia).

### A moeda é neutra

Determinista vs sorteada: médias que diferem em centésimos de ponto base,
desvios-padrão em ~1%. Numa carteira de 158 acções sorteiam-se 158 moedas
independentes por barra e a média converge para a probabilidade — a
aleatoriedade desaparece na agregação. Numa posição única acrescentaria
variância sem acrescentar retorno esperado.

### O falso positivo, e a defesa que ficou

Uma variante: `|ângulo|` como *força da tendência*. No teste, ao prazo diário,
topo-menos-fundo de **+1,53% a 5 dias, IC95% [+0,70%, +2,54%]** — acima de zero.
No **treino**, com quatro vezes mais dados: **+0,04%, IC [−0,17%, +0,28%]**.

Um efeito que só existe nos últimos 20% da amostra é uma propriedade daquele
período. `consistency()` fixa a verificação: antes de acreditar num resultado de
teste, perguntar o que diz o treino — não para o escolher, para o descartar.

### Ao minuto (trading em tempo real)

```bash
py src/research/tangent_intraday.py
```

O enquadramento original da ideia. Ao minuto três coisas mudam: a média move-se
muito mais depressa, o joelho dispara mais vezes, e os custos passam a ser da
ordem de grandeza do movimento.

**Nada atravessa a noite.** Uma média de 20 barras na quinta barra do dia usaria
15 barras de ontem e apanharia o salto overnight. Medido: com um gap sistemático
de +2%, a versão ingénua dá **+6,73°** de ângulo médio na abertura contra
**−0,48°** na versão que respeita a sessão; a meio do dia coincidem exactamente.

**Não há divisão treino/teste ao minuto** (9 dias). Por isso reporta-se a grelha
inteira de 54 células por prazo em vez de escolher a melhor. **A entrada é
atrasada uma barra**, o que é realista e serve de teste ao salto bid-ask.

Ao minuto apareceu sinal — 17 de 27 células do joelho com intervalo fora do
zero, e o salto bid-ask descartado (sobrevive ao atraso). E o sinal do IC não era
aleatório: positivo em todas as configurações com troço ≤ média, negativo em
todas as outras, com a fronteira exactamente onde as duas rectas deixam de
partilhar barras.

Mas **não replica**:

| Prazo | Dias | troço ≤ média | troço > média |
|---|---|---|---|
| 1m | 9 | **+0,0146** (20/21 positivas) | −0,0252 (0/6) |
| 2m | 30 | **−0,0016** (9/24) | −0,0085 (0/3) |
| 5m | 71 | **−0,0001** (12/24) | −0,0009 (0/3) |

E os custos decidiriam de qualquer maneira: ao minuto o spread de ida-e-volta
(0,060%) come **65,5%** do movimento típico em excesso (0,092%).

## Fluxo assinado aproximado (CLV × volume)

```bash
py src/research/order_flow_proxy.py
```

Veio de uma pergunta sobre order flow: a literatura de microestrutura (Cont,
Kukanov e Stoikov, 2014) mostra que o desequilíbrio de fluxo de ordens prevê
variações de preço a prazos curtos. A ronda 10 não encontrou nada ao minuto — e a
diferença pode ser a categoria de dados que este projecto nunca teve.

O que sobra numa barra OHLCV é o **close location value** (+1 se fechou no topo do
próprio intervalo, −1 se no fundo), vezes o volume.

**A armadilha:** em bruto, `clv × volume` põe sempre a AAPL no topo por negociar
mais. O volume tem de ser normalizado pela média do próprio ticker, senão
ordena-se capitalização bolsista disfarçada de fluxo.

**O resultado principal** é o artefacto de cotação, medido entrando uma barra
depois:

| Prazo | IC (entrada imediata) | IC (uma barra depois) | Era artefacto |
|---|---|---|---|
| 1m | **−0,0406** | −0,0049 | **88%** |
| 2m | −0,0312 | −0,0060 | 81% |
| 5m | −0,0182 | −0,0002 | **99%** |

O −0,041 a 1 minuto é oito vezes maior do que qualquer coisa que o joelho
produziu. Lido sem o atraso, seria o melhor resultado do projecto. É quase todo
salto bid-ask.

**O que sobrevive** é o fluxo acumulado (`ofi_20`), com o mesmo sentido nos três
prazos (−0,0027 / −0,0072 / −0,0060) — antecede reversão. Com o sentido estimado
no treino e medido no resto, o bruto é positivo a 2m (+0,0036%) e 5m (+0,0016%),
mas o custo de execução é **5 a 13× maior**. Para pagar, o IC teria de ser
0,03–0,08 — a ordem de grandeza que a literatura reporta para fluxo de ordens a
sério, com lado agressor e profundidade do livro.

## A referência que faltava: comprar e ficar quieto

```bash
py src/research/benchmark_comparison.py
```

Onze rondas mediram sinais contra o acaso, contra o HAR-RV, contra votações
simples — nunca contra a alternativa que qualquer pessoa tem sem escrever código.

**Longo prazo (SPY, 34 anos):** +10,84% anualizado, volatilidade 18,5%, Sharpe
0,58, **drawdown máximo −55,2%**. Piores anos: 2008 −36,8%, 2002 −21,6%, 2022
−18,2%. É a metade da resposta que costuma ficar de fora — "seguro" não quer
dizer que não dói.

**Confronto directo** (2025-04 a 2026-08, 214 acções, rebalanceamento de 5 dias
sem sobreposição, custos cobrados, composto):

| Estratégia | Total | Anual | Sharpe | Drawdown |
|---|---|---|---|---|
| **Comprar o índice (SPY)** | **+35,9%** | 23,98% | **1,46** | −6,5% |
| Comprar as 214 acções, peso igual | +31,3% | 21,02% | 1,29 | −5,2% |
| Sinal transversal long-short (líquido) | −1,3% | −0,88% | −0,11 | −6,5% |

**Nenhum algoritmo deste projecto bate comprar o índice.** Com 70 períodos
nenhuma diferença é estatisticamente demonstrável — o IC do próprio SPY é
[−0,11%, +0,98%] — mas o ponto estimado é inequívoco.

Ressalvas, todas a favor dos algoritmos e nenhuma suficiente: o período é curto
e de alta forte; o long-short é neutro ao mercado e tem metade da volatilidade;
e o sinal existe, só é pequeno de mais para pagar a execução (IC ~0,006 contra
custos de 5 a 13× isso).

### Combinar o sinal com o índice

```bash
py src/research/blend_with_index.py
```

Pergunta diferente da anterior: um activo pode melhorar uma carteira **mesmo
rendendo menos do que ela**, se se mover de forma independente. A condição é
aritmética — juntar S a I melhora o Sharpe se e só se
`Sharpe(S) > correlação × Sharpe(I)`.

**Correlação medida: −0,0207** (p = 0,863). Sendo ligeiramente negativa, a
fasquia desce para **−0,23% ao ano** — o sinal podia perder até isso e ainda
ajudar. Fez **−0,69%**.

| No sinal | Anual | Vol | Sharpe | Drawdown | Sharpe c/ dinheiro |
|---|---|---|---|---|---|
| 0% | 23,98% | 16,47% | 1,46 | −6,5% | 1,46 |
| 10% | 21,45% | 14,83% | 1,45 | −5,9% | 1,45 |
| 30% | 16,43% | 11,72% | 1,40 | −4,9% | 1,44 |
| 50% | 11,45% | 9,03% | 1,27 | −4,5% | **1,43** |

A fasquia certa não é "melhora o retorno" — é **bater dinheiro parado**, que
deixa o Sharpe aritmético invariante em todos os pesos e é grátis. O sinal fica
abaixo dessa linha em todos os pesos, e a distância cresce.

Nos 15 piores períodos do índice (média −2,33%) o sinal rendeu −0,27%. Não é
hedge: não faz nada de útil quando faria falta.

### O sinal em regimes de queda

```bash
py src/research/regime_conditional.py
```

Um livro transversal vive de **dispersão** entre acções, e a dispersão dispara
quando o mercado cai. A hipótese é razoável.

A distinção que faz a experiência: a condição **contemporânea** ("nos períodos em
que o índice caiu") é diagnóstico e **não é negociável** — no momento de decidir
não se sabe. A condição **ex-ante** ("quando o mercado já vinha a cair") usa só
informação disponível na altura e é a única que dá uma regra.

| Regime ex-ante | Períodos | Dentro | Fora | Anual dentro | p placebo |
|---|---|---|---|---|---|
| Abaixo da MA200 | 11 | **+0,160%** | −0,050% | **+8,06%** | 0,469 |
| Abaixo da MA50 | 24 | +0,138% | −0,079% | +6,94% | 0,325 |
| Retorno 20d negativo | 31 | +0,136% | −0,101% | +6,85% | 0,219 |
| Em drawdown > 5% | 13 | −0,123% | −0,011% | −6,18% | 0,644 |

Três dos quatro apontam no sentido da hipótese. Nenhum chega perto de
significância — e desta vez **o nulo é diferente**: o mínimo detectável é 3 a 6×
maior do que o efeito, ou seja a amostra é **cega**, não está a medir ausência.
Faltariam 13 a 42 anos de histórico.

Testei também se o livro carregava **beta residual**, que explicaria a perda
contemporânea nas quedas. Não carrega: beta +0,031 ± 0,048 (p = 0,512), R²
0,0045. A hipótese era minha e estava errada.

Defesas contra fatiar até encontrar: regimes clássicos definidos à partida,
Bonferroni, e um **placebo por deslocamento circular** que preserva a estrutura
de blocos dos regimes em vez de assumir independência entre períodos.

## Histórico profundo e a forma da cauda

```bash
py src/research/deep_history.py   # 2 milhoes de barras, 1962-2026
py src/research/tail_shape.py     # a forma da cauda e' caracter da empresa?
```

A ronda 14 acabou em "seriam precisos 13 a 42 anos" e o painel tinha três. Em
barras diárias o yfinance dá desde **1962**: 2.022.301 barras, 222 acções, 65
anos — 20 a 60× mais dados.

### A pergunta

Volatilidade é um *nível*. A **forma** com que é entregue é outra coisa: duas
acções com a mesma volatilidade anual podem entregá-la em muitos movimentos
médios ou em poucos enormes. Três medidas sem escala: `q99(|r|)/q75(|r|)`, a
quota de variância do 1% de dias piores, e o estimador de Hill.

### O resultado

| Medida | Persistência bruta | Residual (sem liquidez, preço, dimensão, vol) | Transições positivas |
|---|---|---|---|
| volatilidade (referência) | +0,804 | +0,569 | 100% |
| tail_ratio | +0,182 | +0,161 | 85% |
| **concentração** | **+0,221** | **+0,255** | **98%** |

E a volatilidade passada prevê a forma futura a apenas **+0,02–0,03**, contra
**+0,18–0,22** da forma passada. É uma dimensão separada.

Persiste **20 anos** (+0,075), replica em **5/5** transições de década, e o
placebo por troca de identidade dá **p = 0,005**.

### Mas não é novo

Antes de publicar, fui à literatura: **já é conhecido**. Kelly & Jiang (NBER
w19375) reportam AR(1) mensal de 0,927 na série de cauda; a ortogonalidade à
volatilidade também é estabelecida. Redescoberta com método mais duro e amostra
mais longa — o que tem valor, mas não é descoberta.

### A peça menos certa

Nas mesmas 115 acções presentes em todas as décadas, a concentração de risco
subiu de 0,1974 (1980s) para 0,2302 (2020s) enquanto a volatilidade ficou parada
(32,0% → 32,5%). Tendência +0,00088/ano, p = 0,0003; sem 2020 fica p = 0,0008,
portanto **não é COVID**. A mesma volatilidade está a ser entregue em menos dias
e maiores. Não sei se é novo.

### E o mecanismo? As notícias agendadas não explicam

```bash
py src/research/scheduled_news_share.py
```

Baker et al. (NBER w28687) explicam a versão de **mercado** deste facto: os
saltos do índice passaram de notícias não agendadas (guerras) para agendadas
(política monetária). Ao nível da **empresa**, a notícia agendada dominante são
os **resultados trimestrais** — o que dá uma previsão testável.

**Não se confirma.** A intensidade dos dias de resultados (variância que lá cai,
a dividir pela fracção de dias que são de resultados) está **plana**: 3,55× em
2005, 3,45× em 2025, tendência p = 0,24. E retirando esses dias, a tendência
sobrevive: declive cai de +0,00230 para +0,00153, mas a **diferença** entre os
dois é +0,00077/ano com IC95% [−0,00076, +0,00247] — inclui zero.

**Um erro estatístico que o código chegou a cometer** e que está corrigido e
testado: concluir que os resultados explicavam a subida porque uma série era
significativa e a outra não. *A diferença entre significativo e não significativo
não é ela própria significativa.* O teste certo é sobre a diferença de declives.

## O bracket de pivots: 25 pontos de risco para ganhar 5

Hipótese trazida de fora: MNQ/USTEC ao minuto, pivots com 10 barras de cada
lado como suporte/resistência, stop 25 pontos, alvo 5 pontos, 89-91% de acerto
contra 83% de ponto morto. Código em `src/research/pivot_bracket.py`.

### O ponto morto É a moeda ao ar

Com stop S e alvo A, o ponto morto é S/(S+A) = 83,33%. E a probabilidade de uma
caminhada sem deriva tocar +A antes de −S é também S/(S+A) = 83,33%. O mesmo
número — é a definição de jogo justo. "83% de ponto morto" não é fasquia
nenhuma: é o que se tem sem estratégia.

Como o alvo é pequeno, cada ponto de custo vale 3,33pp de taxa de acerto. Ao
spread típico de um CFD (2,5 pontos) o ponto morto sobe a **91,67%** — acima da
taxa alegada. O mesmo sistema é lucrativo em MNQ e perdedor em USTEC.

### A alegação reproduz-se sem sinal nenhum

27 033 barras de MNQ, 20 sessões. Entradas **aleatórias** no mesmo bracket:

| | acerto |
|---|---|
| acaso, deixando a barra de entrada resolver | **89,07%** [87,4 – 90,7] |
| acaso, honesto (resolve a partir de t+1) | 83,60% (teoria: 83,33%) |
| pivot *fade* | 82,01% |
| pivot *rompimento* | 84,32% |

Os 89-91% alegados são o que o acaso produz com um artefacto de preenchimento.
Com um alvo de 5 pontos e amplitude mediana de 8,5 pontos por barra, **4 em
cada 5 negócios resolvem-se no primeiro minuto** — o minuto que um backtest
sobre barras não vê por dentro. Medidos honestamente, os pivots ficam dentro da
banda do acaso, e muito abaixo do ponto morto realista de 87%.

Líquido: **−1,276 pontos por negócio, IC95% [−2,257, −0,319]** (reamostrando
sessões). Das 128 células da grelha, **zero** são positivas — e todas as 60
grelhas de puro ruído tiveram uma melhor célula superior à melhor célula real.

### A alegação contradiz-se

Uma sequência máxima de 5 perdas em 300 negócios tem p = 0,26% a uma taxa de
90%. Essa sequência é típica de **61%** de acerto — onde a esperança é −7,80
pontos por negócio.

## O RSI(2) de Connors, com o mesmo protocolo

Segunda estratégia pública passada pelo protocolo da ronda 17. Regra do livro
de Connors e Alvarez (2008): comprar quando o fecho está acima da MA200,
abaixo da MA5 e o RSI(2) abaixo de 10; vender quando o fecho volta acima da
MA5. Alegações públicas: 71-75% de acerto. Código em
`src/research/connors_rsi2.py`. 222 acções, 1962-2026, 100 245 negócios.

### A taxa de acerto sai de ruído puro

Com ganho médio de 1,22% e perda média de 2,10%, o ponto morto é **63,25%** —
os 71% valem 8pp de margem, não 71. E a regra de saída fabrica o acerto: sair
quando o preço recupera selecciona vencedores por construção. Sobre
**caminhadas aleatórias sem deriva**, a mesma regra dá **62,3% de acerto com
média −0,077%**.

### Mas o efeito de média é real

| | média/negócio | IC95% |
|---|---|---|
| regra completa | **+0,51%** | [+0,46%, +0,56%] |
| acaso em qualquer recuo (sem RSI) | +0,27% | [+0,24%, +0,31%] |
| comprar e esperar emparelhado | +0,20% | [+0,17%, +0,22%] |

Intervalos que não se tocam, 48 de 48 parametrizações positivas, p(ruído ≥
real) = 0,0000, e **73% do efeito mantém-se nas 62 acções que caíram do
índice** — a sobrevivência explica ~27%, não tudo. Com custos da época, porém,
só 4 das 7 décadas são positivas: até 2001 os spreads comiam-no inteiro.

### E perde à mesma para comprar e ficar quieto

| | ret/ano | vol | Sharpe | pior queda |
|---|---|---|---|---|
| estratégia líquida 10 bps | 18,40% | 20,09% | **0,92** | −52,5% |
| comprar e esperar, peso igual | 17,70% | 16,79% | **1,05** | −48,4% |

56 rotações por unidade de capital por ano convertem 25,66% brutos em 18,40%
líquidos. A 5 bps ganha por pouco, a 20 bps é esmagada: **o veredito inverte-se
dentro da gama de custos que separa um intermediário de outro.**

## Simulação: 4 horas ao minuto com 3000 euros

`src/research/live_simulation.py`. MNQ ao minuto, **99 janelas de 4 horas que
não se sobrepõem**, conta de €3000 = $3366, custo de 1,10 pontos ($2,20) por
ida-e-volta.

Em quatro horas não cabem 240 negócios: uma conta tem uma posição de cada vez
e o negócio mediano dura 3,4 minutos, portanto cabem **71**.

| | negócios | acerto | bruto | custos | líquido |
|---|---|---|---|---|---|
| ao minuto, 1 contrato | 71,4 | **82,0%** | −$36,68 | $157,13 | **−$193,82** |
| ao minuto, 2 contratos | 71,4 | 82,0% | −$73,36 | $314,27 | −$387,63 |
| só no sinal, 1 contrato | 8,5 | 82,6% | +$0,29 | $18,69 | −$18,40 |

**82% de acerto com −5,76% de conta por dia**, e **81% da perda é custo puro**
(no modo "só no sinal", 101,6% — o bruto é zero e a perda é exactamente o
custo). Duplicar o tamanho duplica a perda.

Mediana de **46 dias de negociação** até ficar sem margem (~9 semanas); 100%
de probabilidade de ruína ao fim de um ano, 0% de probabilidade de lucro.

Para empatar seria preciso **87,0%** de acerto e obtém-se **82,0%** — faltam 5
pontos percentuais. Se os 89% alegados fossem verdade o plano daria +8,6% da
conta por dia: a aritmética do plano não é absurda, a premissa é que falha.

A única alavanca que melhora o resultado é negociar **menos** — de 71 para 8,5
negócios a perda cai de $193,82 para $18,40 — e o limite dessa alavanca é zero.

### Um ano a 8,5 negócios por sessão

A ronda 19 reportou $808 de mediana e 0% de ruína para este caso. **Estava
errado**: o composto multiplicava por (1+retorno), o que pressupõe ajustar o
tamanho ao saldo, e com MNQ o mínimo é um contrato inteiro. A perda em dólares
é fixa, portanto a conta desce em linha recta.

| sizing | mediana ao fim do ano | ruína | sessões até à ruína |
|---|---|---|---|
| **fixo, 1 contrato** (executável) | **$0** | **95,3%** | **165** |
| proporcional (não executável) | $793 | 0,0% | — |

A conta do ano: 2 141 negócios, **$4 710 de custos = 140% da conta**, contra
**+$73 de P&L bruto**. A conta aguenta 180 sessões só a pagar custos.

Correndo o mesmo ano **sem custos nenhuns**: mediana **$3 366** — exactamente
onde começou — e **50,0% de probabilidade de lucro**. Em bruto isto é uma moeda
ao ar perfeita.

Com 2 141 negócios o erro-padrão do acerto é 0,82pp: IC95% [81,0%, 84,2%],
contra os 87,0% necessários. Mesmo o extremo optimista fica 2,8pp curto.

### O RSI(2) ao minuto, em vez dos pivots

| | negócios/janela | acerto | bruto | custos | líquido |
|---|---|---|---|---|---|
| pivots + bracket 25/5 | 8,5 | 82,9% | +$0,29 | $18,69 | **−$18,40** |
| RSI(2) + bracket 25/5 | 6,5 | **86,7%** | +$16,03 | $14,22 | +$1,81 |
| RSI(2) + saída Connors | 5,5 | **69,8%** | +$21,62 | $12,18 | **+$9,44** |

A variante com **maior acerto** ganha $1,81 e a com **menor** ganha $9,44 —
cinco vezes mais dinheiro com 17pp menos de acerto.

Bruto por negócio: RSI(2) **+1,9530 pts** contra **+0,0293** de entradas ao
acaso emparelhadas na contagem, com custo de 1,10. É a primeira coisa deste
projecto ao minuto cujo bruto excede o atrito.

**Mas decai com o atraso de entrada** — a assinatura do salto bid-ask da ronda
11:

| atraso | bruto/negócio | líquido/janela |
|---|---|---|
| 0 | +1,9530 pts | +$9,44 [−3,78, +24,03] |
| 1 min | +1,3084 pts | +$2,62 [−7,27, +13,34] |
| 2 min | +0,7639 pts | −$4,96 |
| 5 min | +0,2560 pts | −$14,36 |

Metade do bruto desaparece no primeiro minuto e o resto cai abaixo do custo no
segundo. **Em nenhum atraso o intervalo exclui zero** — são 404 negócios em 20
dias, e seriam precisos ~59 dias para decidir.

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
web/                    dashboard Flask (so-leitura): / carteiras, /patterns, /live,
                        /trees, /decisao
tests/                  pytest

src/patterns/           SUBSISTEMA DE PADROES GRAFICOS
  ingest_intraday.py    barras 1h e 5m via yfinance
  pivots.py             deteccao de swing highs/lows (fractais) + lag de confirmacao
  detectors.py          geometria dos 16 padroes, cada um com quality score
  scanner.py            varre o historico, resolve sobreposicoes, sequencia limpa
  sequence.py           cadeia de Markov + beam search 4 passos + avaliacao
  context.py            features de contexto por padrao (para o classificador)
  classifier.py         classificador multiclasse + protocolo de 3 particoes
  direction.py          alvo binario (sobe/desce), embargo, controlo aleatorio
  portfolio.py          terceira carteira: negoceia o sinal de direccao
  realtime.py           forward test ao vivo (modo live + replay) + aprendizagem online

src/research/           EXPERIENCIAS (ver RESEARCH.md)
  fallen_universe.py    accoes que cairam do universo + filtro de liquidez
  survivorship.py       os quatro testes ao vies de sobrevivencia
  tangent_angle.py      angulo da tangente a media movel, normalizado em
                        volatilidade; varrimento, decis e treino-vs-teste
  tangent_intraday.py   o mesmo ao minuto: sessao fechada, atraso de entrada,
                        grelha completa e replica entre 1m/2m/5m
  order_flow_proxy.py   fluxo assinado aproximado (CLV x volume normalizado),
                        teste ao salto bid-ask e liquido de custos
  benchmark_comparison.py  tudo contra comprar o indice: 34 anos de SPY e o
                        confronto directo com custos e composicao
  blend_with_index.py   o sinal como diversificador: condicao de Sharpe, ponto
                        de equilibrio e comportamento nas quedas do indice
  regime_conditional.py  o sinal em regimes de queda: ex-ante contra
                        contemporaneo, placebo circular e poder estatistico
  deep_history.py       2 milhoes de barras diarias, 1962-2026, ajustadas
  tail_shape.py         a forma da cauda como caracter da empresa: medidas sem
                        escala, residualizacao, decadencia e placebo
  scheduled_news_share.py  a subida da concentracao vem dos resultados
                        trimestrais? teste da diferenca de declives

src/decision/           CAMADA DE DECISAO (tabela de verdade)
  panel.py              grelha diaria comum + alvo neutro ao mercado + percentis do dia
  short_term.py         algoritmo de 5 dias, bootstrap de blocos, custos medidos
  signals.py            cada algoritmo reduzido a um veredicto
  truth_table.py        162 celulas com encolhimento, ablacao, teste de interaccao
  walkforward.py        re-treino trimestral + estabilidade entre sementes
  run_decision.py       calcula e grava para a web (unico ponto de escrita)

src/trees/              ARVORES DE DECISAO (XGBoost, 3 classes com ESPERAR)
  features.py           indicadores tecnicos + forma da janela + rotulo de 3 classes
  train.py              treino, varrimento de profundidade, metricas accionaveis
  live.py               inferencia ao vivo dos dois modelos + consenso
  magnitude.py          alvo de magnitude (classificacao + regressao)
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
