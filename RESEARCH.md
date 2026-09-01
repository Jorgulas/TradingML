# Caderno de investigação

Registo das experiências, incluindo as que falharam e as que corrigiram
conclusões anteriores. Cada ronda: pesquisar → formular hipótese → medir →
tentar destruir o resultado → concluir.

---

## Ronda 1 — De onde vem o poder preditivo do modelo de magnitude?

**Ponto de partida.** O modelo de árvores previa a magnitude do próximo
movimento com Spearman 0,466 e lift de 9× entre decis, batendo a baseline de
"persistência de volatilidade" (uma feature) por larga margem. Parecia forte.

### O que a literatura diz

Duas pesquisas mudaram o desenho das experiências:

- **HAR-RV** ([Corsi, 2009](https://www.researchgate.net/publication/227464926_A_Simple_Long_Memory_Model_of_Realized_Volatility))
  é o benchmark padrão para previsão de volatilidade: três médias da
  volatilidade passada (curta, média, longa). Estudos comparativos concluem
  que ML **não o bate consistentemente** — "the classic HAR-RV model
  consistently outperform both machine learning approaches, especially when
  the information set is limited"
  ([estudo comparativo](https://www.diva-portal.org/smash/get/diva2:2031701/FULLTEXT01.pdf)).
  Isto tornou a minha baseline anterior (uma feature) insuficiente.
- **Meta-labeling** ([López de Prado](https://en.wikipedia.org/wiki/Meta-Labeling))
  separa a decisão de *lado* da de *tamanho* — desenhado exactamente para
  situações como esta, onde a direcção é fraca.

### Experiência A — HAR-RV vs XGBoost

**Primeira tentativa, e estava viciada a meu favor.** Corri a comparação e o
XGBoost ganhou por +0,31 de Spearman. Mas os coeficientes do HAR saíram
`0,45 / −12,13 / −67,09 / +51,79` — sinais alternados e magnitudes absurdas,
sintoma clássico de **multicolinearidade** (as componentes do HAR são médias
móveis sobrepostas da mesma série). E pior: testei-o no alvo
`magnitude_ratio`, que já está normalizado pela volatilidade — precisamente o
que o HAR prevê, dividido fora.

**Correcção:** RidgeCV em vez de OLS, e o alvo `abs_return` (o nível do
movimento), que é o terreno do HAR. Coeficientes passaram a
`0,016 / 0,265 / 0,279 / 0,281`, todos positivos, como a teoria manda.

| Modelo | Spearman | Decis |
|---|---|---|
| HAR-RV (4 coeficientes) | 0,2179 | 2,70× |
| XGBoost (40 features) | **0,4388** | 7,05× |
| HAR + XGBoost | 0,4387 | 7,06× |

O XGBoost ganha mesmo depois da correcção — e o blend dá ao HAR peso −0,02,
ou seja não acrescenta nada.

**Mas isto contradiz a literatura, o que é motivo para desconfiar de mim, não
dela.**

### Experiência B — a direcção é previsível em algum regime?

13 regimes testados a 1h (volatilidade alta/baixa, pico de volume, abertura,
fecho, depois de movimento grande, RSI extremo, bandas apertadas, beta alto,
baixa correlação…), com **correcção de Bonferroni** (|z| > 2,89) e regimes
definidos em treino, medidos no teste.

| | |
|---|---|
| Acerto global | 49,4% (z = −0,15) |
| Melhor regime | 50,4% (bandas apertadas, z = +0,10) |
| Regimes que sobrevivem a Bonferroni | **nenhum** |

**Conclusão: a direcção não é previsível em nenhum regime testado.** Fecha a
porta ao "e se procurássemos melhor". Consistente com os seis modelos
anteriores, todos entre 49,1% e 53,5%.

### Experiência C — de onde vem mesmo a vantagem sobre o HAR?

Suspeita: a feature nº1 do XGBoost é `session_position`. O HAR foi desenhado
para dados **diários**, onde não existe hora do dia. Aplicado a barras de 1h,
não tem como capturar o "U" de volatilidade intradiária.

| Modelo | Spearman | Fosso fechado |
|---|---|---|
| HAR (só volatilidade) | 0,2179 | 0% |
| **HAR + 8 dummies de hora** (linear) | **0,4206** | **91%** |
| XGBoost (só HAR + hora) | 0,4272 | 94% |
| XGBoost (40 features) | 0,4398 | 100% |

E isolando cada bloco:

| Bloco | Spearman | % do modelo completo |
|---|---|---|
| **Só a hora da sessão** (8 dummies) | 0,3256 | **74%** |
| Só o HAR (4 médias) | 0,2179 | 50% |
| Os dois (12 parâmetros lineares) | 0,4206 | 96% |

**Conclusão: 74% do poder preditivo vem de saber que horas são.** As outras 36
features valem 4%. A literatura tinha razão — o HAR é difícil de bater assim
que se lhe dá o relógio que ele estruturalmente não tem nesta frequência.

Um modelo linear de **12 parâmetros** captura 96% do que as 40 features e o
XGBoost fazem.

### Experiência D — sobra alguma coisa depois de tirar o calendário?

Construiu-se um alvo residualizado: estimou-se em treino a volatilidade típica
de cada `(ticker, faixa horária)` e dividiu-se o movimento realizado por essa
expectativa. O que resta é **surpresa**.

O "U" medido é na verdade um **J** — a última faixa da sessão (leilão de
fecho) move-se **3,43×** a mais calma:

```
faixa 0/7  0.00292  ###################################
faixa 3/7  0.00228  ###########################
faixa 5/7  0.00191  ######################
faixa 7/7  0.00657  ##############################################################################
```

| Alvo | Spearman | Decis |
|---|---|---|
| Bruto (com sazonalidade) | 0,4398 | 7,09× |
| **Surpresa (des-sazonalizado)** | **0,1698** | 1,90× |
| Surpresa, só com HAR | 0,1316 | 1,67× |

**Sobra 39% do sinal, e é estatisticamente significativo:** bootstrap por dia
(144 dias, 400 reamostragens) dá **IC95% [0,1528, 0,1847]** — não contém zero.

E aqui as features ricas finalmente justificam-se: 0,1698 contra 0,1316 do HAR
sozinho. Estavam mascaradas pelo efeito de calendário.

O que prevê a surpresa: `atr14_norm` (10,4%), `har_day` (9,9%), `ret_10`
(6,9%), `mkt_vol_20` (6,7%), `mkt_ret_1` (5,6%).

### Conclusões da ronda 1

1. **A direcção não é previsível** — nem na média, nem em nenhum dos 13
   regimes testados, com correcção para testes múltiplos.
2. **O número de topo do modelo de magnitude estava inflacionado por um efeito
   de calendário.** 74% vinha de saber a hora do dia. Isso é real e útil, mas
   não é prever o mercado — é ter relógio.
3. **Sobra sinal genuíno**, mais modesto: Spearman 0,17 na surpresa
   des-sazonalizada, significativo, com lift de 1,90× entre decis.
4. **Um modelo linear de 12 parâmetros faz 96% do trabalho** das 40 features
   com XGBoost. A complexidade não se paga.

### O que isto muda na prática

Se o objectivo fosse produção, a recomendação seria **substituir o XGBoost de
40 features por HAR + dummies horárias** — 96% do desempenho, uma fracção da
complexidade, e interpretável. O XGBoost só se justifica no alvo residual,
onde a vantagem sobre o HAR é real (+0,038 de Spearman).

### Perguntas em aberto para a ronda 2

- O sinal residual de 0,17 sobrevive a **custos de transacção**? Um lift de
  1,90× entre decis pode não pagar o spread.
- **Meta-labeling**: usar o modelo de magnitude para dimensionar posições de um
  sinal direccional fraco melhora o resultado, ou multiplica ruído por ruído?
- A sazonalidade é **estável ao longo do tempo** ou muda de mês para mês? Se
  mudar, o modelo linear precisa de reestimação frequente.
- O sinal residual é **transversal** (uma acção move-se mais que as outras) ou
  **temporal** (todas se movem mais ao mesmo tempo)? São explorações diferentes.


---

## Ronda 2 — O sinal residual sobrevive a custos?

**Pergunta.** O sinal residual (Spearman 0,17, significativo) existe. Mas um
sinal de **magnitude** não se negoceia direccionalmente — é preciso um
instrumento que pague pelo tamanho do movimento. Serve para alguma coisa
depois de custos?

### O que a literatura diz

- **Spreads em large-caps americanas**: ~2-3 pontos base por lado, e estreitam
  cerca de 4× da abertura para o fecho
  ([dados de execução](https://frec.com/resources/blog/direct-indexing-transaction-costs)).
- **Prémio de risco de volatilidade**: a volatilidade implícita excede a
  realizada **85-90% das vezes**, em média 2-4 pontos percentuais, ou ~**11% do
  prémio** ([Macrosynergy](https://macrosynergy.com/research/realistic-volatility-risk-premia/),
  [Alpha Architect](https://alphaarchitect.com/the-variance-risk-premium-is-pervasive/)).
  Quem **compra** volatilidade paga sistematicamente a mais.

Isto define o teste: o modelo tem de prever surpresa acima de **1,11** só para
empatar, antes de qualquer custo.

### Erro apanhado a meio (e é grave)

A primeira versão deu **+76% de lucro** no decil de topo. Era falso. A
verificação de calibração denunciou-o: a surpresa **média** saiu **1,26** em
vez de 1,00.

Causa: estimei o "movimento esperado" como a média estática do período de
treino e apliquei-a meses depois. A volatilidade no teste foi 26% mais alta —
**todo o lucro era mudança de regime**. Um vendedor de opções cota a
volatilidade *de hoje*.

**Correcção:** movimento esperado = nível corrente (média móvel de 200 barras,
só a olhar para trás) × forma sazonal (estimada em treino, porque é estrutural).
A verificação de calibração ficou no código como guarda permanente.

### Teste 1 — sobrevive ao prémio de risco?

| Decil previsto | Surpresa realizada |
|---|---|
| 1 (fundo) | 0,719 |
| 5 | 0,912 |
| 9 | 1,085 |
| **10 (topo)** | **1,241** |

Bootstrap por dia (137 dias, 500 reamostragens):

| | |
|---|---|
| Surpresa média no topo | **1,2410** |
| IC95% | **[1,2011, 1,2871]** |
| Ponto de equilíbrio (1 + prémio) | 1,1100 |
| **Margem** | **+0,131** (limite inferior do IC: **+0,091**) |

**O sinal sobrevive: o intervalo de confiança está inteiramente acima do
equilíbrio.** Sobram 9-18% do prémio para custos de transacção.

### Teste 2 — mas há instrumento para o explorar?

**Não a este horizonte.** Não existem opções a uma hora. O prazo mais curto
realista é diário, e a esta granularidade o sinal teria de ser remedido.

E na única aplicação implementável com acções — **cronometrar a execução para
reduzir slippage** — o modelo **perde para uma regra gratuita**:

| Estratégia | Movimento médio |
|---|---|
| Barra qualquer | 50,9 bps |
| **Grátis: negociar na faixa mais calma do dia** | **30,9 bps** |
| Modelo: decil previsto mais calmo | 38,5 bps |

**O modelo é 7,6 bps PIOR do que olhar para o relógio.**

A razão é instrutiva: o modelo prevê **surpresa** (movimento relativo ao
esperado), mas para execução o que interessa é o **nível absoluto**. Uma barra
com surpresa baixa no fecho volátil continua a ter volatilidade alta. O modelo
optimiza a quantidade errada para este uso.

### Conclusões da ronda 2

1. **O sinal é real e sobrevive ao prémio de risco de volatilidade** — margem
   de +13,1% do prémio, estatisticamente significativa.
2. **Mas não há instrumento acessível** para o explorar a 1 hora. Opções não
   existem nesse prazo.
3. **Para execução, uma regra de relógio bate o modelo.** O sinal é sobre
   surpresa; a execução depende do nível.
4. **A verificação de calibração é obrigatória.** Sem ela, uma mudança de
   regime aparecia como 76% de lucro.

Nem "funciona" nem "não vale nada": **existe um efeito real sem instrumento
acessível, e na aplicação acessível uma regra mais simples ganha.**

### Ronda 3 — a pergunta que se segue

O sinal só é negociável se sobreviver ao **horizonte diário**, onde existem
opções. É o teste directo de implementabilidade, e é o próximo passo óbvio.


---

## Ronda 3 — O horizonte diário, onde as opções existem

**Pergunta.** O sinal só é negociável se sobreviver ao prazo diário. Duas
razões para esperar pior: desaparece o relógio intradiário (74% do sinal a 1h),
e o HAR-RV joga em casa — foi desenhado para dados diários.

Dados: 741 dias de barras 1h dos 160 tickers agregadas em barras diárias.

### A vantagem do XGBoost evapora — e isso confirma a ronda 1

| Modelo | Spearman |
|---|---|
| HAR-RV (4 médias) | 0,2453 |
| HAR + dia da semana | 0,2494 |
| XGBoost (40 features) | 0,2533 |

**+0,008 — empate técnico.** Tirado o relógio, 40 features e XGBoost empatam
com **4 coeficientes lineares**. E o dia da semana acrescenta 0,004, contra os
+0,20 da hora do dia.

Isto **confirma definitivamente** a hipótese da ronda 1: a vantagem toda do ML
era sazonalidade intradiária. A literatura tinha razão — a prazo diário, o
HAR-RV não é batido por machine learning.

### O sinal de surpresa parecia sobreviver…

| | |
|---|---|
| Surpresa no decil de topo | 1,2995 |
| IC95% (bootstrap por dia, 115 dias) | [1,2054, 1,3843] |
| Equilíbrio (1 + prémio) | 1,1100 |
| Líquido a 5% de custo de opções | **+0,1395** |

Intervalo inteiramente acima do equilíbrio, lucrativo em todos os cenários de
custo. Parecia o primeiro resultado do projecto simultaneamente significativo
**e** com instrumento acessível.

### …mas a mediana desmonta-o

| | |
|---|---|
| **Média** do decil de topo | 1,2995 |
| **Mediana** do decil de topo | **0,9750** |

**O trade típico perde dinheiro.** A média só está acima do equilíbrio porque
é puxada por uma minoria de movimentos enormes:

| Aparar os maiores | Média resultante | |
|---|---|---|
| Top 1% removido | 1,2177 | ainda bate |
| **Top 5% removido** | **1,0809** | **já não bate** |
| Top 10% removido | 0,9737 | perde claramente |

E **nenhum dos dez decis** tem mediana acima do equilíbrio — a do topo é a
maior, e fica em 0,975.

### O modelo tem competência real, mas na coisa errada

Não é sorte. O modelo detecta genuinamente os eventos:

| Extremos | Capturados pelo decil de topo | Acaso | Lift |
|---|---|---|---|
| Top 1% de surpresas | **29,8%** | 10% | **2,98×** |
| Top 5% | 18,6% | 10% | 1,86× |
| Top 10% | 15,7% | 10% | 1,57× |

Quase **3× melhor que o acaso** a encontrar os movimentos do top 1%.

**O problema é que essa é exactamente a competência que o mercado de opções já
tem.** A volatilidade implícita real sabe quando há resultados trimestrais e
sobe antes deles. O meu "movimento esperado" é uma média móvel de 60 dias que
não sabe nada sobre calendários de eventos — logo subestima sistematicamente o
prémio nesses dias, criando um lucro fantasma.

Não está concentrado em poucos tickers (158 de 158 aparecem no decil, os 5 mais
frequentes são 7% das linhas), mas está concentrado **no tempo**: poucos dias
grandes por nome. É o padrão de eventos agendados.

### Conclusões da ronda 3

1. **A prazo diário o ML não bate o HAR-RV.** Confirma a ronda 1 e a
   literatura: a vantagem era o relógio intradiário.
2. **O sinal de surpresa não é negociável.** A média batia o equilíbrio, a
   mediana não. O lucro depende de ~5% de trades, e esses são precisamente os
   eventos que a volatilidade implícita real já cobra.
3. **Mas o modelo tem competência genuína**: 2,98× o acaso a identificar os
   movimentos extremos.

### O que isto realmente vale

A conclusão útil não é de trading, é de **gestão de risco**. Saber que uma
posição tem hoje risco de cauda elevado — 3× melhor que o acaso — permite
**reduzir exposição**, e isso não paga spread de opções nem prémio de risco de
volatilidade. É a aplicação onde a competência que o modelo tem é a competência
de que se precisa.

Como sinal de compra de volatilidade, falha. Como alarme de risco, funciona.

### Ronda 4 — perguntas em aberto

- O alarme de risco reduz mesmo a variância de uma carteira, na prática?
- A sazonalidade intradiária é estável mês a mês, ou precisa de reestimação?
- O sinal residual é transversal (esta acção move-se mais que as outras) ou
  temporal (todas ao mesmo tempo)?


---

## Ronda 4 — O alarme de risco reduz a variância da carteira?

**Pergunta.** O modelo detecta risco de cauda 2,98× melhor que o acaso. Como
sinal de compra de volatilidade falhou, mas dimensionar posições não paga
spread nem prémio de risco. Serve para isso?

**Desenho.** A comparação justa **não** é contra pesos iguais — é contra
dimensionar pelo inverso da volatilidade recente, que é prática corrente e não
custa nada. Exposição idêntica em todos os esquemas (pesos somam 1), senão
reduzir variância seria trivial: bastava investir menos.

### Resultado: não acrescenta nada

| Esquema | Vol anual | Pior dia | VaR 99% | Drawdown |
|---|---|---|---|---|
| Pesos iguais | 6,83% | −1,38% | −1,04% | −2,64% |
| **Inverso da vol recente (grátis)** | 6,91% | −1,05% | −0,95% | −3,82% |
| Inverso da vol prevista (modelo) | 6,86% | −1,01% | −0,97% | −3,30% |
| Combinação | 6,88% | −1,03% | −0,96% | −3,55% |

Diferença de volatilidade diária entre modelo e referência gratuita:
**−0,00003, IC95% [−0,00011, +0,00004]** — contém zero.

### Testei a minha própria desculpa, e estava errada

Hipótese óbvia: com 158 acções o risco idiossincrático está diversificado, e o
modelo prevê exactamente isso. Num portfólio concentrado devia ajudar.

| Nº de acções | Vol grátis | Vol modelo | Diferença |
|---|---|---|---|
| 5 | 15,32% | 15,35% | **+0,030%** (pior) |
| 10 | 11,89% | 11,87% | −0,027% |
| 20 | 9,50% | 9,46% | −0,040% |
| 50 | 7,81% | 7,76% | −0,053% |
| 158 | 6,91% | 6,86% | −0,048% |

**A hipótese está rejeitada.** A diferença é constante em todas as dimensões, e
a 5 acções o modelo é ligeiramente *pior*.

### Porquê — a explicação mecânica

| | |
|---|---|
| Correlação risco previsto vs volatilidade recente | **0,78** |
| Correlação entre os pesos resultantes | **0,86** |
| Diferença média de peso | 11,8% do peso médio |

A previsão do modelo é **dominada pelo nível de volatilidade recente**, que já
estava na referência gratuita. A informação incremental — a componente de
surpresa — é real, mas pequena demais em magnitude para mover os pesos de forma
útil.

### Conclusão da ronda 4

**O alarme de risco não reduz a variância da carteira**, em nenhuma dimensão de
portfólio, face ao que a volatilidade recente já dá de graça. Fecha a última
aplicação em aberto.


---

## Ronda 5 — A sazonalidade intradiária é estável mês a mês?

**Porque importa.** O "J" intradiário é a única coisa neste projecto que prevê
de forma fiável — sozinho vale 74% do poder preditivo do modelo de 40 features.
Se for estável, é a descoberta durável. Se derivar, até ela perde valor.

Dados: 36 meses (2023-09 a 2026-08), 160 tickers, barras de 1h.

### Um bug meu, apanhado antes de concluir

A primeira versão dividia a sessão em 8 faixas por posição normalizada. Mas há
**7 barras por dia**: as posições 0..6 mapeiam para as faixas 0,1,2,4,5,6,7 e a
**faixa 3 fica sempre vazia**. Só recebia dados dos raros dias curtos, o que
produzia 63% de dispersão nessa faixa — ruído de amostra minúscula, não um
fenómeno.

Corrigido para usar a **hora do dia** directamente (inequívoca em barras
horárias), o resultado ficou **mais forte**:

| | Com bug | Corrigido |
|---|---|---|
| Dispersão por faixa | 5-12%, uma a **63%** | **4,8% a 11,4%** |
| Amplitude (desvio-padrão) | 4,44× ± **2,01** | 4,09× ± **0,45** |
| Intervalo da amplitude | 3,08× a **15,86×** | 3,08× a **5,39×** |

O outlier de 15,86× era inteiramente o artefacto.

### Teste 1 — A forma repete-se?

Cada hora varia apenas **4,8% a 11,4%** de mês para mês. Correlação mediana
entre perfis mensais: **0,79**.

### Teste 2 — Um perfil velho ainda serve? *(o decisivo)*

| Idade do perfil | Spearman | Perda vs oráculo |
|---|---|---|
| Oráculo (o próprio mês, impossível) | 0,3078 | — |
| 1 mês antes | 0,2943 | −4,4% |
| 3 meses antes | 0,2905 | −5,6% |
| 6 meses antes | 0,2908 | −5,5% |
| **12 meses antes** | **0,2931** | **−4,8%** |

**Um perfil de há 12 meses funciona tão bem como o do mês passado.** Não há
decaimento nenhum com a idade — a perda fica plana em ~5% face a um oráculo
impossível.

### Teste 3 — Quanto histórico chega?

| Meses usados | Spearman |
|---|---|
| 1 | 0,2943 |
| 6 | 0,2983 |
| 12 | 0,3045 |

Um ano de histórico chega para desempenho praticamente de oráculo.

### Conclusão da ronda 5

**A sazonalidade intradiária é notavelmente estável.** Estima-se uma vez com um
ano de dados e serve indefinidamente: a forma varia menos de 12% por hora, a
amplitude é 4,09× ± 0,45, e um perfil com um ano de idade não perde nada face
a um recém-estimado.

É o **primeiro resultado inequivocamente positivo e durável** de todo o
programa. E, ao contrário de tudo o resto, corrigir o bug tornou-o mais forte.

*(Nota: as rondas C, D e E usaram o esquema de 8 faixas com a faixa 3 vazia.
Os resultados mantêm-se — as outras 7 faixas estavam correctas — mas as "8
dummies" eram na prática 7 úteis e uma inútil.)*


---

## Ronda 6 — Dados novos (resultados trimestrais) e modelar a cauda

Duas linhas em paralelo: um **dado novo** que faltava, e uma **ideia minha**
nascida de um diagnóstico.

**Sobre fóruns:** verifiquei e **não consigo aceder**. A API do Reddit exige
credenciais OAuth que não tenho e o StockTwits fechou a pública. Não vou
reportar o que não testei. Mas o yfinance dá **calendário de resultados**, que
é mais directamente relevante ao que a ronda 3 diagnosticou: 160 tickers,
**14.878 datas**, média de 93 por empresa.

### A hipótese da ronda 3 está confirmada

| | Taxa de eventos de cauda (\|r\| > 3× esperado) |
|---|---|
| Dias de resultados (8,4% dos dias) | **11,49%** |
| Dias normais | 3,01% |
| Rácio | **3,81×** |

Surpresa média: **1,502** em dias de resultados contra **0,994** em dias
normais — este último exactamente 1,0, como deveria. O "lucro" da ronda 3 era
o efeito dos resultados, que o mercado de opções obviamente cobra.

### O achado que reescreve a ronda 4

| Features | AUC a prever cauda |
|---|---|
| **Só volatilidade recente (1)** | **0,5071** |
| Só calendário de resultados (4) | 0,6290 |
| Modelo completo sem resultados (48) | 0,6168 |
| **Modelo completo + resultados (52)** | **0,6633** |

**A volatilidade recente é inútil para prever caudas — AUC 0,507, acaso puro.**

Isto explica a ronda 4. Lá dimensionei posições pelo inverso da volatilidade
recente e por um modelo que correlacionava 0,78 com ela. Nenhum tinha poder
sobre caudas, e as caudas é que produzem drawdowns. **Estava a medir a coisa
certa com o preditor errado.**

Note-se também: **4 features de calendário batem 48 features técnicas** (0,629
vs 0,617). Mas combinadas dão 0,663 — o modelo *sabe* algo além do calendário,
+0,034 de AUC.

### A minha ideia: modelar a cauda, não a escala

O diagnóstico: o decil de topo tinha mediana 0,975 (normal) mas 2,98× mais
extremos. Isso não é mudança de escala — é mudança na **espessura da cauda**. E
eu treinava com erro quadrático, que ajusta a **média condicional**.

| Alvo treinado | AUC na cauda | Lift no decil |
|---|---|---|
| Escala E[\|r\|] | 0,6543 | 2,88× |
| Quantil Q90 | 0,6450 | 3,00× |
| **Cauda P(\|r\|>3×)** | **0,6633** | **3,07×** |

**A hipótese estava certa mas o efeito é modesto**: +0,009 de AUC sobre a
regressão de escala. Não estava a subestimar dramaticamente o modelo, como
suspeitei — estava a subestimá-lo um pouco.

### Dimensionar pela cauda: o primeiro resultado *promissor mas não provado*

| Esquema | Vol anual | Pior dia | VaR 99% | 5 piores | Drawdown |
|---|---|---|---|---|---|
| Inverso da volatilidade (grátis) | 6,91% | −1,05% | −0,95% | −0,91% | −3,82% |
| **Inverso do risco de cauda** | **6,78%** | **−1,04%** | **−0,89%** | **−0,80%** | **−1,70%** |

**Todas as métricas favorecem o modelo de cauda.** O drawdown cai 55%.

Mas testado formalmente no CVaR 5% (mais estável que o VaR pontual):

| | |
|---|---|
| Melhoria | **+0,116 pp** |
| IC95% | [−0,085%, +0,270%] |
| **Efeito mínimo detectável com 114 dias** | **0,181 pp** |

**O observado é 64% do necessário para significância.**

### Porque é que esta ronda é diferente

Nas cinco rondas anteriores, sempre que um número parecia bom houve uma
explicação que o desmontou — o relógio, a mudança de regime, a mediana, a
correlação com a baseline. **Aqui não houve.** O efeito é consistente em
direcção nas cinco métricas de cauda e nenhuma inspecção o explicou.

O que falta é **poder estatístico**: seriam precisos **~278 dias de teste**
(1,1 anos) contra os 114 disponíveis — o que implica ~5,5 anos de histórico
total, e eu tenho 2,0.

É a diferença entre *"mostrou-se ausente"* e *"não se conseguiu provar"*. Todas
as conclusões negativas anteriores eram do primeiro tipo. Esta é do segundo.

---

# Ronda 7 — a tabela de verdade, e a primeira coisa que sobreviveu

Pergunta do Jorge: *"e se fizeres uma tabela de verdade com os algoritmos
todos quando for para tomar uma decisão? Inclusive um algoritmo de curto prazo
ao invés do super curto de decisões de 1 minuto."*

Duas peças: um horizonte que faltava, e uma forma de combinar tudo.

## O degrau que faltava

Os prazos existentes eram os extremos — árvores a decidir o minuto seguinte,
carteira LONG a decidir 20 dias. Construí o intermédio: **5 dias de bolsa,
retorno em excesso sobre o índice**, três classes (`src/decision/short_term.py`).

E aproveitei para testar uma hipótese que nunca tinha sido testada em seis
rondas. Toda a investigação anterior olhou para **cada acção ao longo do
tempo** — e a direcção temporal mostrou-se imprevisível, sempre. Nunca tinha
sido testada a informação **transversal**: não *"esta acção vai subir"*, mas
*"esta acção vai subir mais do que as outras"*. São perguntas diferentes, e a
segunda é neutra ao mercado por construção.

Três conjuntos de features, mesmos dados, mesmo alvo, 158 acções × 716 dias:

| Conjunto | n | IC diário | long-short a 5d (teste) |
|---|---|---|---|
| temporal (séries de cada acção) | 48 | +0,0094 | +0,42% |
| **transversal (percentis do dia)** | **11** | **+0,0232** | **+0,77%** |
| ambos | 59 | +0,0112 | +0,12% |

O transversal ganha nas duas métricas, em validação **e** em teste. E `ambos` é
pior do que o transversal sozinho — as 48 features temporais são ruído em que a
árvore se afoga.

**O que o modelo usa.** A feature dominante é o percentil de **volatilidade**
(26,5% do ganho), não de momento. E usa-a de forma não-linear: a regra ingénua
`-xs_vol_20` sozinha rende **−1,06%**. Também não é reversão a 1 dia
disfarçada (`-xs_ret_1` sozinha vale +0,29%). É um condicionamento.

## O erro que quase deixei passar

A primeira versão dava `+0,77%, IC95% [+0,17%, +1,38%]` — significativo. Estava
errado. Forma-se carteira todos os dias e mede-se o que rende nos **5 dias
seguintes**: a observação de segunda e a de terça partilham 4 dos 5 dias. São
quase a mesma medição feita duas vezes. Um bootstrap que reamostra dias soltos
assume 144 observações independentes quando há ~29, e estreita o intervalo por
cerca de √5.

Corrigido com bootstrap de **blocos** e confirmado sem sobreposição nenhuma, o
mesmo resultado passa a `+0,77%, IC95% [−0,16%, +1,78%]` — **não
significativo**. O ponto estimado não mudou; a honestidade do intervalo mudou.

## A tabela de verdade

Cinco algoritmos reduzidos a um veredicto cada, no mesmo instante
(`src/decision/signals.py`): `curto` (5 dias), `rapido` (1 dia), `padrao`
(padrão gráfico confirmado nos últimos 10 dias), `regime` (volatilidade do
mercado em tercis) e `alarme` (risco de cauda, herdado da ronda 6). Dá
3×3×3×3×2 = **162 células**, todas observadas.

O perigo é óbvio: 162 células sobre 144 dias de teste. Três defesas:
**suporte contado em dias e não em linhas**; **encolhimento para o modelo
aditivo** com peso `n_dias/(n_dias + 40)`, de modo a que uma célula rara diga
exactamente o que a votação simples diria; e **cada esquema escolhe o seu
próprio limiar** na validação — sem isso, um limiar afinado para a tabela põe
todos os sinais isolados a ESPERAR sempre, e a tabela "ganha" contra
adversários que nunca chegaram a jogar.

## O teste que decide

Não *"a tabela ganha?"* mas *"a tabela ganha **mais do que a soma dos
sinais**?"*. Para ter poder estatístico, avaliação em **walk-forward**:
re-treino trimestral, cada trimestre pontuado por modelos que nunca o viram.
349 dias fora da amostra, ~69 janelas independentes contra as 28 do teste único.

| Esquema | Retorno/decisão | IC95% (blocos) | Dias + |
|---|---|---|---|
| tabela crua | +0,527% | [+0,125%, +0,974%] | 55,3% |
| tabela encolhida | +0,449% | [−0,007%, +0,892%] | 58,2% |
| votação simples | +0,415% | [+0,082%, +0,739%] | 58,2% |
| **só o sinal `curto`** | **+0,383%** | **[+0,104%, +0,665%]** | 61,0% |

**Tabela menos votação: +0,033%, IC95% [−0,235%, +0,317%].**

## O que isto quer dizer

**1. O sinal transversal de curto prazo é real.** É o primeiro resultado deste
projecto com intervalo de confiança inteiramente acima de zero fora da amostra,
em 349 dias, e mantém-se em todas as configurações que experimentei. Líquido
dos custos medidos (rotação 48%, ~0,03% por ciclo) continua positivo.

> **Corrigido na ronda 8.** Cerca de metade deste número era viés de
> sobrevivência: o painel só contém as grandes capitalizações que chegaram
> vivas ao fim. Repondo as que caíram do índice, o efeito cai para perto de
> metade e o intervalo passa a incluir o zero. O sinal existe; a magnitude
> reportada aqui está inflacionada.

**2. A tabela de verdade não acrescenta nada de demonstrável.** Combinar os
cinco algoritmos rende +0,03% sobre somar os seus votos, com intervalo a
atravessar o zero. As interacções que só uma tabela consegue captar — *"o
padrão só funciona em regime calmo"* — não se demonstram nestes dados.

**3. E medi o quanto isto é frágil**, em vez de o descrever. Repetindo o
walk-forward inteiro só com outra semente do XGBoost — mesmos dados, mesmo
protocolo, muda apenas a amostragem interna das árvores:

| Esquema | Média | Amplitude entre sementes |
|---|---|---|
| tabela crua | +0,596% | 0,146% |
| tabela encolhida | +0,486% | 0,107% |
| votação | +0,397% | 0,029% |
| só o `curto` | +0,387% | 0,037% |
| *tabela − votação* | *+0,089%* | *0,135%* |

A amplitude da diferença entre tabela e votação (0,135%) é **maior do que a
própria diferença** (0,089%). E os esquemas simples são 3 a 4× mais estáveis do
que os da tabela: a vantagem aparente vem acompanhada de quatro vezes mais
variância de estimação. A ablação por sinal chegou a trocar de sinal entre
execuções — não é uma ordenação, é ruído.

## A resposta honesta à pergunta

A tabela de verdade é uma boa forma de **ver** como os algoritmos concordam, e é
isso que a página `/decisao` mostra. Como mecanismo de decisão, não bate somar
os votos, e somar os votos não bate usar só o melhor sinal. **Toda a evidência
está concentrada num único algoritmo — o transversal de 5 dias — e a estrutura
por cima dele não paga a variância que introduz.**

Que é, mais uma vez, o aparelho de medição a funcionar. Desta vez o número bom
sobreviveu, mas foi a coisa mais simples que o produziu.

## Ressalvas que ficam por resolver

- **Viés de sobrevivência.** As 158 acções são as grandes capitalizações de
  *hoje*. Uma empresa que colapsou entre 2023 e 2026 não está no painel, e num
  long-short transversal isso remove precisamente os piores nomes do lado
  vendido. Não sei o tamanho do efeito; sei o sentido.
- **Lookahead residual no calendário de resultados**, herdado da ronda 6:
  `days_to_earnings` com valores altos pode conter datas que não eram públicas
  na altura. Afecta o sinal `alarme`, que é o que menos pesa.
- **Três sementes não são uma distribuição.** A amplitude reportada é um piso
  para a incerteza, não o total dela.

---

# Ronda 8 — o viés de sobrevivência, medido

Pergunta do Jorge, e era a ressalva certa a atacar: *"testa se o viés de
sobrevivência explica o resultado."*

O painel da ronda 7 tem 158 grandes capitalizações de **hoje**. O primeiro
número diz tudo: das 158, **todas** têm histórico completo e **todas** chegam
vivas ao último dia. Zero atrito em três anos. Um índice real perde 5 a 8% dos
membros por ano — seriam 15 a 25 empresas. A lista foi escolhida no fim.

## Duas descobertas antes sequer de medir

**A fonte de dados também está enviesada.** Testei 20 empresas que saíram do
S&P 500 neste período por falência ou aquisição — SIVB, FRC, ATVI, PXD, HES,
MRO, CTLT, JNPR, WBA, DISH, RAD, YELL, WE, FSR, NKLA, MPW, SAVE, TUP, BIG.
**Nenhuma devolve dados.** O yfinance apaga as delistadas. Não é a minha lista
que está mal: é a fonte, e nenhuma reconstrução completa é possível aqui.

**Um único ticker corrompido quase produziu a conclusão errada.** Ao alargar o
universo, a BYND entrou com um desdobramento inverso aplicado de forma
inconsistente: oscilava entre 0,55 e 17,85 dólares em dias seguidos, seis
vezes, o que dá retornos de +2900% seguidos de −97%. Esse ticker sozinho
produziu **99%** de um colapso de −8,9% num trimestre, e eu estava a um passo
de concluir que o modelo não generalizava para fora das grandes capitalizações.
Verifiquei todo o universo: só BYND e MRNA estão contaminadas, **nenhuma delas
nas 158 originais** — o resultado da ronda 7 não vinha daí. O construtor do
painel passou a rejeitar qualquer série com variação diária acima de 100% e a
reportar quem rejeitou.

## Os quatro testes

**1. De que perna vem o resultado?** Comprada 52,1%, vendida 47,9%. Metade vem
do lado vendido, que é precisamente o que a ausência de falidos afecta. Não
absolve.

**2. O modelo evita os desastres que existem?** Não. O posto médio de confiança
das quedas violentas (2% piores) é **0,6175** — o modelo põe-nas *acima* da
mediana, tende a comprá-las (p = 1,2e−15). As subidas violentas estão em
0,6413. Ou seja: carrega em magnitude, com uma margem direccional fina. Isto
apontava para viés a inflacionar o resultado.

**3. O efeito concentra-se nas acções seguras?** Não. Estratificando por
drawdown máximo, o IC é +0,0516 no grupo mais frágil, +0,0194 no intermédio e
+0,0267 no mais sólido. Não é um artefacto de mega-caps tranquilas.

**4. Repor as que faltam.** O teste directo: 44 ex-membros do índice e 14 nomes
especulativos que o yfinance ainda serve, com filtro de liquidez idêntico para
todos. Média de três sementes, walk-forward trimestral:

| Universo | n | Retorno 5d | IC95% | Amplitude entre sementes |
|---|---|---|---|---|
| A. só sobreviventes (a ronda 7) | 158 | **+0,694%** | [+0,192%, +1,202%] | 0,070% |
| B. + caídas do índice | 201 | **+0,353%** | [+0,026%, +0,703%] | 0,082% |
| C. + especulativas | 171 | +0,403% | [+0,056%, +0,699%] | 0,125% |
| D. + ambas | 214 | **+0,241%** | [−0,048%, +0,613%] | 0,138% |

A ordem é idêntica nas três sementes, e os intervalos entre universos (0,34pp
de A para B) são **4 a 5× maiores** do que a dispersão entre sementes
(0,07-0,08pp). Ao contrário da diferença tabela-versus-votação da ronda 7,
**esta diferença é real**.

## A resposta

**O viés de sobrevivência não explica tudo, mas explica cerca de metade.**

- Repor só os ex-membros do índice — a correcção legítima do viés — corta o
  efeito de +0,694% para +0,353%: **49% desaparece**. E a redução é
  estatisticamente significativa.
- No universo mais completo que consigo montar, sobra +0,241%, uma queda de
  **65%**, e o intervalo de confiança passa a incluir o zero.
- O que sobra continua positivo e continua estável entre sementes. Não é zero.
  Mas é pequeno de mais para esta amostra o resolver.

Transposto para o número da ronda 7 (+0,383% por decisão no walk-forward
completo), a estimativa corrigida ronda **+0,13% a 5 dias** — que já não
sobreviveria ao seu próprio intervalo de confiança.

**E mesmo isto é um limite superior.** As 19 empresas verdadeiramente
delistadas continuam irrecuperáveis, e todas as adições que consegui fazer
baixaram o efeito. Não há razão para achar que as que faltam o fariam subir.

## O que isto obriga a corrigir na ronda 7

A frase *"o primeiro resultado deste projecto com intervalo de confiança
inteiramente acima de zero"* estava certa para o universo em que foi medida e
é enganadora fora dele. A versão honesta: **existe um sinal transversal de
curto prazo, mas cerca de metade do que lhe atribuí era a lista de acções e não
o modelo.** O sinal residual não é distinguível de zero com os dados que tenho.

Sétima ronda, sétima vez que a inspecção mais fina explicou o número. Desta vez
foi o Jorge a apontar onde olhar.

---

# Ronda 9 — o ângulo da tangente à média móvel

Proposta do Jorge: desenhar a média móvel sobre o gráfico e, no último ponto,
medir o **ângulo entre a recta do troço anterior e a do troço recente**. Média
plana que passa a subir dá um "joelho" de +30°; se voltar a ficar plana, −30°.
Limiar de 25° com zona morta, e acima dele uma **moeda enviesada** para o lado
do joelho em vez de uma decisão seca.

## Duas correcções que a ideia obriga

**O ângulo precisa de normalização.** Um ângulo exige os dois eixos nas mesmas
unidades, e preço e tempo não são. Uma acção de $500 e uma de $5 com o mesmo
movimento relativo dão ângulos diferentes, e o "25°" passaria a depender da
escala do gráfico. Aqui o declive é medido em **volatilidade por barra**: 45°
significa "a média avança um movimento típico por barra", igual na AAPL e na KO.

**O ângulo absoluto sozinho não acrescenta nada ao declive.** `arctan` é
monótona, portanto "ângulo > 25°" é idêntico a "declive normalizado > 0,466".
Testá-lo seria testar momentum com outro nome. Mas o **ângulo entre dois
troços** não é transformação monótona de nada — depende dos dois declives ao
mesmo tempo e mede **curvatura**. Isso é conteúdo genuinamente novo, e é o que
esta ronda mede.

## O que os testes de unidade encontraram antes dos dados

**A minha primeira normalização estava errada.** Media a média sobre o preço e
dividia por `volatilidade × preço`. Um teste verificou que uma subida
perfeitamente constante devia dar joelho zero — e dava **16,7°**. O denominador
andava durante a tendência e fabricava um joelho que não existia no gráfico. Em
**log-preço** o declive já é um retorno por barra e o denominador é a
volatilidade desse retorno: adimensional de verdade, e a tendência constante dá
zero, como tem de dar.

**O joelho só dispara no instante da viragem.** É a diferença entre duas rectas
separadas por `span` barras: assim que a subida dura mais do que `span`, as duas
rectas voltam a ser a mesma e o joelho colapsa — mesmo com a média a subir a
pique. É um impulso curto, não um estado. Daí a exposição de 2,2% a 20°.

**E o limiar de 25° quase nunca dispara.** Com ruído realista, uma tendência
*forte* (1% de deriva por barra contra 0,8% de ruído) dá um joelho de pico a
rondar os 20°. O varrimento escolheu sempre 5°.

## O resultado

Varrimento de **972 configurações** em cada timeframe (3 médias × 3 troços × 6
limiares × 3 horizontes × regra normal e invertida), escolha no treino, medição
no teste:

| Timeframe | Regra escolhida | Retorno | IC95% | Exposição |
|---|---|---|---|---|
| 1h | sempre comprado (referência) | +0,045% | [−0,639%, +0,613%] | 100% |
| 1h | declive | −0,069% | [−0,260%, +0,082%] | 24,8% |
| 1h | **joelho** | **+0,008%** | [−0,255%, +0,207%] | 27,0% |
| 1h | ambos | −0,031% | [−0,183%, +0,074%] | 14,9% |
| 1d | −declive | −0,068% | [−0,215%, +0,092%] | 33,6% |
| 1d | **−joelho** | **−0,031%** | [−0,165%, +0,104%] | 40,8% |
| 1d | −ambos | −0,033% | [−0,112%, +0,045%] | 18,4% |

Nada é significativo, e no prazo diário as três regras escolhidas foram as
**invertidas** — o varrimento decidiu sozinho que o lado rentável era vender a
média a subir, não comprá-la.

## A medição que decide

A conta de lucros mistura duas perguntas: *o sinal contém informação* e *a regra
sabe extraí-la*. Um limiar mal posto afunda uma regra sem que o sinal tenha
culpa. Medindo o sinal directamente, contra o retorno futuro:

| Timeframe | Horizonte | `joelho` (direcção) | `ângulo` (direcção) |
|---|---|---|---|
| 1h | 1 | **−0,0005** | −0,0078 |
| 1h | 3 | **−0,0006** | −0,0162 |
| 1h | 7 | **−0,0004** | −0,0270 |
| 1d | 1 | **+0,0034** | +0,0172 |
| 1d | 3 | **+0,0054** | +0,0204 |
| 1d | 5 | **+0,0086** | +0,0274 |

**O joelho é zero.** Spearman entre −0,0006 e +0,0086 em seis combinações, sobre
790 mil observações a 1h. Não é "pequeno de mais para medir" — é medido com
muito poder e é zero. Os decis não são monótonos: formam um U, que é a
assinatura de uma quantidade de magnitude, não de direcção.

**O ângulo tem um sinal pequeno e de sentido instável**: −0,027 a 1h (reversão)
e +0,027 ao dia (momentum). Sinais opostos no mesmo universo, e ao prazo diário
troca também entre treino e teste. Não é utilizável.

**A leitura de magnitude também falha.** `|joelho|` contra `|retorno|` dá +0,023
a 1h e −0,048 ao dia. Troca de sinal entre timeframes: ruído.

## O falso positivo que quase passou

Uma variante construtiva: usar `|ângulo|` como **força da tendência** — estar
numa acção em tendência forte, para qualquer lado, paga? No teste, ao prazo
diário: topo-menos-fundo **+1,53% a 5 dias, IC95% [+0,70%, +2,54%]**,
inteiramente acima de zero.

No **treino**, que tem quatro vezes mais dados: **+0,04%, IC [−0,17%, +0,28%]**.

Um efeito que só existe nos últimos 20% da amostra e desaparece nos 80% com mais
poder estatístico é uma propriedade daquele período. A regra que daqui fica:
*antes de acreditar num resultado de teste, perguntar o que diz o treino — não
para o escolher, para o descartar*. Está fixada em `consistency()`.

## E a moeda?

| Regra | Determinista | Com moeda | Desvio det. | Desvio moeda |
|---|---|---|---|---|
| declive | −0,0689% | −0,0726% | 1,5412% | 1,5423% |
| joelho | +0,0075% | +0,0180% | 1,6822% | 1,6939% |
| ambos | −0,0305% | −0,0405% | 1,1739% | 1,1899% |

**Exactamente neutra**, e por uma razão estrutural: numa carteira de 158 acções
sorteiam-se 158 moedas independentes por barra, e a média delas converge para a
probabilidade. A aleatoriedade desaparece na agregação. Numa posição única a
moeda acrescentaria variância sem acrescentar retorno esperado — não é um erro
da ideia, é o que uma decisão probabilística faz quando a esperança é a mesma.

## O que fica da ideia

Negativo no essencial: **a curvatura da média móvel não prevê direcção**, e a
regra proposta não bate estar sempre comprado em nenhum dos dois timeframes.

Mas três coisas ficam, e não são de consolação:

1. **A normalização em unidades de volatilidade** é o que torna qualquer limiar
   angular comparável entre acções. Está em `normalised_slope()` e serve
   qualquer sinal geométrico futuro.
2. **A verificação treino-versus-teste** (`consistency()`) nasceu aqui e apanhou
   um falso positivo com intervalo de confiança acima de zero.
3. **O varrimento foi dominado pelo joelho** — 19 das 30 melhores configurações
   no treino a 1h — e **nada sobreviveu**. É a demonstração mais limpa que este
   projecto tem do que 972 comparações fazem a quem só olha para o melhor
   resultado.

---

# Ronda 10 — o joelho ao minuto, que era o enquadramento original

A ronda 9 mediu o ângulo da tangente a 1 hora e ao dia. Mas o pedido do Jorge
era *"para o trading real-time"*, e ao minuto três coisas mudam mesmo: a média
move-se muito mais depressa, o joelho dispara muitas mais vezes, e os custos
passam a ser da ordem de grandeza do movimento.

## Três decisões que este prazo obriga

**Nada atravessa a noite.** Uma média de 20 barras na quinta barra do dia usaria
15 barras de ontem, e o `diff` apanharia o salto overnight — que pode ser 1% e
não é um movimento de um minuto. Medi a contaminação: com um gap sistemático de
+2%, a versão ingénua dá um ângulo médio de **+6,73°** nas primeiras barras de
cada sessão, contra **−0,48°** na versão que respeita a sessão. A meio do dia as
duas coincidem ao décimo-quinto decimal. A regra estaria a negociar o gap
disfarçado de joelho.

**Não há divisão treino/teste honesta ao minuto.** São 9 dias distintos. Cortar
80/20 daria 7 dias para escolher e 2 para medir. Por isso **não se escolhe**:
reporta-se a grelha inteira de 54 células por prazo. Uma grelha completa não
pode ser acusada de ter escolhido a melhor.

**A entrada é atrasada uma barra.** O sinal vê o fecho da barra e a posição só
abre na seguinte — é o que acontece na realidade, e é também o teste ao **salto
bid-ask**, o contaminante clássico deste prazo.

## O que apareceu ao minuto

Ao contrário de 1h e do prazo diário, ao minuto apareceu sinal: **17 das 27
células do joelho com intervalo de confiança fora do zero**. E o salto bid-ask
ficou descartado — o efeito *sobrevive* ao atraso de uma barra (16/27 → 17/27).

Melhor ainda: o sinal do IC **não era aleatório entre configurações**. Era
positivo em todas as que tinham troço ≤ média e negativo em todas as outras, sem
uma única excepção:

| Razão troço/média | 0,08 | 0,17 | 0,25 | 0,50 | 1,00 | 1,50 | 3,00 |
|---|---|---|---|---|---|---|---|
| IC médio | +0,005 | +0,009 | +0,021 | +0,016 | +0,020 | **−0,025** | **−0,026** |

A fronteira cai exactamente em troço = média, que é onde as duas rectas deixam
de partilhar barras e passam a comparar troços de preço disjuntos. Tinha uma
explicação mecânica plausível e **parecia estrutural**.

## O teste que decide

Se a fronteira é estrutural, tem de aparecer também a 2 e a 5 minutos, que têm
30 e 71 dias em vez de 9.

| Prazo | Dias | troço ≤ média | troço > média |
|---|---|---|---|
| 1m | 9 | **+0,0146** (20/21 positivas) | −0,0252 (0/6) |
| 2m | 30 | **−0,0016** (9/24) | −0,0085 (0/3) |
| 5m | 71 | **−0,0001** (12/24) | −0,0009 (0/3) |

**Não replica.** A região que a 1 minuto tinha 20 de 21 células positivas passa a
9 de 24 e a 12 de 24 — indistinguível de moedas ao ar — e o IC médio cai de
+0,0146 para −0,0016 e −0,0001.

O padrão de 1 minuto é uma propriedade daqueles 9 dias. Os prazos com três e
oito vezes mais dias dizem zero.

## E os custos decidiriam de qualquer maneira

| Prazo | Movimento típico | Ida-e-volta | Custo / movimento | Bruto / ciclo | Líquido / ciclo |
|---|---|---|---|---|---|
| 1m | 0,0915% | 0,0600% | **65,5%** | +0,0051% | +0,0013% |
| 2m | 0,1034% | 0,0600% | 58,0% | +0,0005% | −0,0018% |
| 5m | 0,1657% | 0,0600% | 36,2% | −0,0023% | −0,0062% |

Ao minuto, o spread de ida-e-volta come **dois terços do movimento típico**. Um
sinal teria de acertar de forma esmagadora só para empatar. Este número não
depende de o joelho funcionar ou não: é a aritmética do prazo.

## A conclusão

O joelho ao minuto não funciona, e a maneira como falha é mais instrutiva do que
a maneira como falhou a 1 hora. A 1 hora era um zero liso — Spearman −0,0005 e
nada mais a dizer. Ao minuto apareceu um padrão com intervalo de confiança fora
do zero em 17 células, com uma fronteira nítida e uma explicação mecânica
convincente — e evaporou-se assim que houve mais dias.

É a segunda vez em duas rondas que um resultado sobrevive ao intervalo de
confiança e morre na réplica. Na ronda 9 foi o treino a desmentir o teste; aqui
foi um prazo com mais dias a desmentir outro com menos. **O intervalo de
confiança responde "isto podia ser acaso?"; só a réplica responde "isto volta a
acontecer?"** — e as duas perguntas não são a mesma.

---

# Ronda 11 — fluxo assinado aproximado, e o artefacto de cotação medido

Veio da pesquisa sobre a estratégia de order flow que ganhou um contest mensal
da Robbins. A estratégia em si é menos interessante do que a pergunta que
levantou: a literatura de microestrutura (Cont, Kukanov e Stoikov, 2014) mostra
que o **desequilíbrio de fluxo de ordens** prevê variações de preço a prazos
curtos, de forma quase linear e estável entre acções. A ronda 10 não encontrou
nada ao minuto. A diferença pode ser exactamente a categoria de dados que este
projecto nunca teve.

O que sobra numa barra OHLCV é o **close location value**:

    clv = ((fecho − mínimo) − (máximo − fecho)) / (máximo − mínimo)

+1 se fechou no topo do próprio intervalo, −1 se no fundo. Multiplicado pelo
volume, é o bloco da linha de Acumulação/Distribuição e a aproximação clássica
de volume assinado a partir de OHLCV.

## A armadilha que invalidaria tudo

A métrica é a correlação **transversal** dentro de cada barra. Em bruto,
`clv × volume` põe sempre a AAPL no topo por ela negociar mais, não por estar a
comprar — ordenar-se-ia capitalização bolsista disfarçada de fluxo. O volume tem
de ser dividido pela média recente do **próprio** ticker. Há um teste que o fixa:
duas acções com padrões de fecho idênticos e volumes a diferir 100× têm de dar
exactamente o mesmo sinal.

## O resultado principal: o artefacto medido

Uma barra que fecha no topo do intervalo fechou provavelmente no preço de venda,
e a seguinte tende a voltar ao de compra. Isso produz previsibilidade que não se
pode negociar. Mede-se entrando uma barra depois:

| Prazo | Sinal | IC (entrada imediata) | IC (uma barra depois) | Quanto era artefacto |
|---|---|---|---|---|
| 1m | clv | **−0,0406** | −0,0049 | **88%** |
| 1m | ofi | −0,0365 | −0,0052 | 86% |
| 2m | clv | −0,0312 | −0,0060 | 81% |
| 5m | clv | −0,0182 | −0,0002 | **99%** |

O IC de −0,041 a 1 minuto é **oito vezes maior** do que qualquer coisa que o
joelho da ronda 10 produziu. Lido sem o atraso de entrada, seria o resultado mais
forte do projecto inteiro. É quase todo salto bid-ask — e a assinatura confirma-o:
o artefacto encolhe à medida que a barra engorda (88% → 81% → 99% do sinal, com o
IC bruto a cair de −0,041 para −0,018), que é exactamente como o spread se comporta
enquanto fracção do intervalo da barra.

## O que sobrevive

As acumulações longas guardam mais do seu (pequeno) sinal, porque somar 20 ou 60
barras faz a média do salto. E o `ofi_20` é o **primeiro sinal deste projecto cujo
sentido replica nos três prazos**:

| Sinal | 1m (11 dias) | 2m (32 dias) | 5m (73 dias) | Mesmo sentido? |
|---|---|---|---|---|
| clv | −0,0049 | −0,0060 | −0,0002 | sim |
| ofi | −0,0052 | −0,0056 | +0,0003 | não |
| **ofi_20** | **−0,0027** | **−0,0072** | **−0,0060** | **sim** |
| ofi_60 | −0,0092 | −0,0095 | — | sim |

Fluxo acumulado antecede **reversão**, não continuação.

## O erro que cometi, e que era do sinal contrário ao habitual

A primeira carteira ia sempre comprada no topo do ranking e vendida no fundo. Mas
o IC é negativo — logo ela negociava ao contrário, e reportava prejuízo naquilo
que era o sinal a funcionar. Um bruto de −0,06% lido como fracasso era +0,06%
lido ao direito.

Corrigido sem olhar para a resposta: o sentido é estimado na primeira fatia do
período e aplicado à segunda.

| Prazo | Treino | Teste | Sentido | Bruto/ciclo | Custo | Líquido |
|---|---|---|---|---|---|---|
| 1m | 8d | 4d | +1 | −0,0142% | 0,0225% | −0,0367% |
| 2m | 23d | 10d | **−1** | **+0,0036%** | 0,0166% | −0,0130% |
| 5m | 51d | 23d | **−1** | **+0,0016%** | 0,0211% | −0,0195% |

A 2 e 5 minutos o sentido escolhido no treino produz **bruto positivo no teste**.
O sinal é real. Mas o custo é **5 a 13 vezes** o ganho bruto.

## A conclusão, e o que ela diz sobre os dados que faltam

Para pagar a execução, o IC teria de ser 5 a 13 vezes maior — da ordem de 0,03 a
0,08 em vez de 0,006. É aproximadamente a magnitude que a literatura reporta para
**fluxo de ordens a sério**, com lado agressor e profundidade do livro.

Ou seja: o proxy retém talvez um décimo do que os dados verdadeiros dariam, o que
é coerente com ele ser uma aproximação muito lossy. E dá um número concreto ao que
os dados reais teriam de comprar, em vez de uma intuição.

**É também a primeira vez em onze rondas que um sinal replica em sentido e produz
bruto positivo fora da amostra.** Não chega — morre nos custos, e por uma margem
larga. Mas é uma falha de magnitude, não de existência, e essas são as únicas que
se resolvem comprando dados melhores.

---

# Ronda 12 — tudo contra comprar e ficar quieto

Pergunta do Jorge: *"dado os algoritmos todos que implementámos, qual seria o
mais seguro e com maior retorno? Não estou a falar só do trading real time mas
investimentos a curto e longo prazo."*

Onze rondas mediram sinais uns contra os outros, contra o acaso, contra o
HAR-RV, contra votações simples. **Nunca contra a alternativa que qualquer
pessoa tem sem escrever uma linha de código.** Era a referência que decidia se
alguma coisa aqui valeu a pena, e faltava.

## O longo prazo, com o histórico todo

SPY desde 1993, 34 anos:

| Métrica | Valor |
|---|---|
| Retorno total | **+3.059%** |
| Anualizado | **10,84%** |
| Volatilidade | 18,54% |
| Sharpe | 0,58 |
| **Drawdown máximo** | **−55,2%** |
| Pior dia | −10,9% |

Os cinco piores anos: 2008 **−36,8%**, 2002 −21,6%, 2022 −18,2%, 2001 −11,8%,
2000 −9,7%.

É a metade da resposta que costuma ficar de fora. "Seguro" não quer dizer que
não dói — quer dizer que a probabilidade de recuperar é alta se houver prazo
para esperar. Custa perder metade do capital uma vez por década.

## O confronto directo

Mesmas datas, mesmos custos, mesma aritmética. A estratégia é simulada **a
sério** desta vez — rebalanceamento de 5 em 5 dias sem sobreposição, modelo
re-treinado por trimestre só com o passado, spread cobrado sobre a rotação
efectiva, e composto. O que sai é uma curva de capital, não uma média de janelas
sobrepostas que não se pode compor.

Período 2025-04 a 2026-08, 214 acções (universo corrigido da ronda 8), ~70
períodos de rebalanceamento:

| Estratégia | Total | Anual | Vol | Sharpe | Drawdown |
|---|---|---|---|---|---|
| **Comprar o índice (SPY)** | **+35,9%** | **23,98%** | 16,47% | **1,46** | −6,5% |
| Comprar as 214 acções, peso igual | +31,3% | 21,02% | 16,24% | 1,29 | −5,2% |
| Sinal transversal long-short (líquido) | **−1,3%** | −0,88% | 7,72% | −0,11 | −6,5% |

O melhor sinal que este projecto encontrou, simulado com custos no universo
corrigido, **perdeu 1,3% enquanto o índice fazia 35,9%**.

## O erro que corrigi a meio, e que teria dado a resposta errada

A primeira versão punha a carteira de peso igual a **−4,2%** enquanto o índice
fazia +35,9%. Parecia uma descoberta — o universo a ser esmagado pelo índice.
Era uma subtracção esquecida: o `forward_return` do painel é o retorno em
**excesso** sobre o índice, porque é isso que os modelos prevêem. Somando o
índice de volta, a carteira de peso igual dá +31,3%.

O erro está no sentido perigoso: fazia os nossos algoritmos parecerem melhores
por comparação com um universo artificialmente afundado.

## A resposta

**Nenhum. Comprar o índice bate tudo o que construímos.**

E o intervalo de confiança nem sequer distingue as três linhas — com 70
períodos, o IC do próprio SPY é [−0,11%, +0,98%]. O que se pode afirmar é mais
fraco do que a tabela sugere: nenhuma diferença é estatisticamente
demonstrável, e o ponto estimado do nosso sinal é negativo.

**Três ressalvas honestas, todas a favor dos algoritmos, e nenhuma suficiente:**

1. O período são 1,4 anos de mercado em alta forte (SPY a 24% anualizado contra
   10,8% de média histórica). Favorece comprar e esperar de forma atípica.
2. O long-short é **neutro ao mercado** por construção — metade comprado, metade
   vendido. Tem metade da volatilidade e um perfil de risco diferente. Comparar
   Sharpe entre os dois é comparar coisas distintas.
3. O sinal existe: as rondas 8 e 11 mediram-no positivo fora da amostra. Só é
   pequeno de mais para pagar a execução.

Mesmo concedendo as três: o IC medido é ~0,006 e os custos comem 5 a 13 vezes
isso. Não é uma questão de afinação.

## O que isto fecha

O projecto começou a perguntar se se consegue prever o mercado com machine
learning. A resposta, medida ao longo de doze rondas com protocolo que apanhou
oito conclusões erradas minhas: **não com estes dados**, e a alternativa gratuita
— comprar o índice e não lhe tocar — venceu por uma margem que nenhuma das
ressalvas cobre.

O valor do que construímos não é a carteira. É o aparelho de medição.

---

# Ronda 13 — o sinal vale como diversificador?

Pergunta do Jorge, e é diferente da da ronda 12: *"testa se combinar o sinal com
comprar o índice melhora alguma coisa"*.

Lá mediu-se o sinal **sozinho** contra o índice sozinho, e ele perdeu. Mas um
activo pode melhorar uma carteira **mesmo rendendo menos do que ela**, desde que
se mova de forma independente: a volatilidade da mistura cai mais depressa do
que o retorno, e o rácio melhora. É o único almoço grátis que existe em finanças,
e o livro long-short tem exactamente o perfil de quem podia servi-lo — neutro ao
mercado por construção, com 7,8% de volatilidade contra 16,5% do índice.

## A condição, que é aritmética e não opinião

Juntar um activo S a uma carteira I melhora o Sharpe **se e só se**

    Sharpe(S) > correlação(S, I) × Sharpe(I)

Com correlação zero, reduz-se a "o retorno de S tem de ser positivo". Com
correlação **negativa**, a fasquia desce abaixo de zero e até um activo que perde
um pouco ainda ajuda. É por isso que a correlação decide, e não o retorno isolado.

## O que se mediu

**Correlação entre o sinal e o índice: −0,0207** (p = 0,863). Praticamente zero,
como se esperava de um livro neutro — e indistinguível de zero.

Com essa correlação ligeiramente negativa, a fasquia fica em **−0,23% ao ano**:
o sinal podia perder até isso e ainda assim melhorar a carteira. Perdeu **−0,69%**.

| | Valor |
|---|---|
| Mínimo para a mistura melhorar | **−0,23%** ao ano |
| Retorno que o sinal teve | **−0,69%** ao ano |
| Falta | −0,45pp |

## As misturas, contra a alternativa gratuita

A fasquia certa para um diversificador não é "melhora o retorno" — é **bater
dinheiro parado**. Misturar com dinheiro deixa o Sharpe aritmético *exactamente*
invariante em todos os pesos (a carteira desliza numa recta que passa pela
origem), e é grátis.

| No sinal | Anual | Vol | Sharpe | Drawdown | Sharpe c/ dinheiro | Diferença |
|---|---|---|---|---|---|---|
| 0% | 23,98% | 16,47% | 1,46 | −6,5% | 1,46 | — |
| 5% | 22,71% | 15,64% | 1,45 | −6,2% | 1,45 | −0,001 |
| 10% | 21,45% | 14,83% | 1,45 | −5,9% | 1,45 | −0,004 |
| 20% | 18,93% | 13,24% | 1,43 | −5,4% | 1,44 | −0,014 |
| 30% | 16,43% | 11,72% | 1,40 | −4,9% | 1,44 | −0,036 |
| 50% | 11,45% | 9,03% | 1,27 | −4,5% | 1,43 | **−0,158** |

**Fica abaixo da linha do dinheiro em todos os pesos, e a distância cresce.** A
mistura *reduz* mesmo o risco — volatilidade de 16,5% para 9,0%, drawdown de
−6,5% para −4,5% — mas reduz o retorno mais do que proporcionalmente. Guardar a
mesma fatia em dinheiro dá a mesma redução de risco, de graça, sem risco de
execução, sem custos de empréstimo de títulos e sem complexidade.

## E quando o índice cai?

O teste que separa um diversificador a sério de um que só parece. Uma correlação
média perto de zero pode esconder um activo que acompanha o índice exactamente
quando ele afunda.

Nos 15 piores períodos do índice (média **−2,33%**), o sinal rendeu **−0,27%**,
com 53% deles positivos. Nos melhores períodos rendeu +0,06%.

Não é hedge. Não faz nada de útil quando faria falta.

## A resposta

**Não melhora nada.** Nem em retorno, nem em rácio, nem como protecção nas
quedas. Fica abaixo de dinheiro parado em todos os pesos testados.

E a ressalva de sempre, que aqui é a favor do sinal: com 72 períodos, nenhuma das
diferenças de Sharpe é estatisticamente demonstrável — todos os intervalos
atravessam o zero. Mas a **condição de Sharpe** não depende da amostra curta da
mesma maneira: correlação e volatilidade estimam-se com muito mais precisão do
que uma média, e é a média do sinal que está abaixo da fasquia.

A conclusão da ronda 12 mantém-se e ganha uma segunda perna: o sinal não bate o
índice sozinho, **e também não o complementa**.

---

# Síntese do programa de investigação

Quatro rondas, e um padrão consistente: **sempre que um número pareceu bom, foi
a inspecção mais fina que o explicou.**

| Ronda | O que parecia | O que era |
|---|---|---|
| 1 | Magnitude prevista com Spearman 0,47 e lift de 9× | 74% era saber a hora do dia; 12 parâmetros lineares fazem 96% |
| 2 | +76% de lucro no decil de topo | Mudança de regime; corrigido, sobra +13% do prémio |
| 3 | Média 1,30 bate o equilíbrio de 1,11 | **Mediana 0,975** — o trade típico perde; o lucro está em 5% de eventos que as opções já cobram |
| 4 | Alarme de risco 2,98× melhor que o acaso | Não reduz variância nenhuma; a previsão correlaciona 0,78 com a volatilidade recente |
| 7 | Tabela de verdade a render +0,45% contra +0,38% do melhor sinal isolado | A diferença (0,03%) é menor do que o que ela se move ao trocar a semente (0,14%) |
| 8 | Sinal transversal a +0,69% com intervalo acima de zero | Metade era a lista de acções: repondo as que caíram do índice sobra +0,35%, e no universo completo +0,24% com o zero lá dentro |
| 9 | Joelho da média móvel a dominar 19 das 30 melhores configurações do treino | Spearman −0,0005 contra a direcção futura: zero medido com 790 mil observações, e nada sobreviveu ao teste |
| 10 | Ao minuto, 17 de 27 células com intervalo fora do zero e uma fronteira nítida em troço = média | Não replica: a 2m e 5m, com 3x e 8x mais dias, a mesma região dá 9/24 e 12/24 positivas — moedas ao ar |
| 11 | CLV com IC de −0,041 ao minuto, oito vezes qualquer coisa anterior | 88% era salto bid-ask: entra-se uma barra depois e sobra −0,005. O que resta replica nos três prazos mas os custos são 5 a 13× o ganho |
| 12 | Carteira de peso igual a −4,2% enquanto o índice fazia +35,9% | Subtracção esquecida: o painel prevê retorno em excesso. Somando o índice de volta dá +31,3% — e o erro fazia os nossos algoritmos parecerem melhores |
| 13 | Correlação −0,02 com o índice: o perfil perfeito para diversificar | A fasquia com essa correlação é −0,23% ao ano e o sinal fez −0,69%. Fica abaixo de dinheiro parado em todos os pesos |

## O que ficou estabelecido

**Negativo, com solidez:**

0. **Comprar o índice bate tudo o que construímos** (ronda 12). No período em
   que há dados para todos os algoritmos, o SPY fez +35,9% e o melhor sinal do
   projecto, simulado com custos no universo corrigido, fez −1,3%. Nenhuma
   diferença é estatisticamente demonstrável com 70 períodos, mas o ponto
   estimado é inequívoco e nenhuma das ressalvas a favor dos algoritmos cobre a
   margem. **E também não serve de diversificador** (ronda 13): a correlação com
   o índice é −0,02, o que baixaria a fasquia para −0,23% ao ano, mas o sinal fez
   −0,69% e fica abaixo de dinheiro parado em todos os pesos. Nos 15 piores
   períodos do índice rendeu −0,27%, portanto também não é protecção.

1. **A direcção não é previsível.** Seis modelos entre 49,1% e 53,5%; 13
   regimes testados com correcção de Bonferroni, nenhum sobrevive.
2. **O ML não bate o HAR-RV a prazo diário.** Empate técnico (+0,008) entre 40
   features com XGBoost e 4 coeficientes lineares.
3. **Nada aqui é negociável.** Nem por opções (o lucro está em eventos já
   cobrados), nem por execução (uma regra de relógio ganha), nem por
   dimensionamento (a volatilidade recente já dá tudo).
4. **A curvatura da média móvel não prevê direcção** (rondas 9 e 10). Spearman
   entre −0,0006 e +0,0086 a 1 hora e ao dia, sobre 790 mil observações. Ao
   minuto aparece um padrão forte e com fronteira nítida que **não replica** a 2
   e 5 minutos, onde há três e oito vezes mais dias. E ao minuto o spread de
   ida-e-volta come 65% do movimento típico, o que decidiria o prazo de qualquer
   maneira.

**Positivo, e real:**

0. **O fluxo acumulado antecede reversão** (ronda 11), e é o único sinal deste
   projecto cujo sentido replica em três prazos independentes e produz bruto
   positivo fora da amostra. Mas o custo de execução é 5 a 13× o ganho: para
   pagar, o IC teria de ser 0,03–0,08 em vez de 0,006 — que é a ordem de
   grandeza reportada para fluxo de ordens verdadeiro, com lado agressor e
   profundidade do livro. É uma falha de magnitude, não de existência.

5. **A volatilidade é previsível** por sazonalidade intradiária — e essa
   sazonalidade é **estável**: um perfil de há 12 meses funciona tão bem como
   um do mês passado (ronda 5). É a descoberta durável do projecto.
6. **Existe sinal residual genuíno** depois de tirar calendário e inércia:
   Spearman 0,17 a 1h, IC95% [0,153, 0,185].
7. **O modelo detecta eventos de cauda** 2,98× melhor que o acaso.
8. **Existe um sinal transversal de curto prazo** (ronda 7), mas **cerca de
   metade do que lhe atribuí era viés de sobrevivência** (ronda 8). Medido no
   universo de sobreviventes dava +0,694%; repondo as acções que caíram do
   índice sobra +0,353%, e no universo mais completo possível +0,241% com o
   zero dentro do intervalo. O efeito é positivo e estável entre sementes, mas
   já não é distinguível de zero com estes dados. Vem de comparar acções entre
   si — não de prever cada uma ao longo do tempo — e combinar algoritmos por
   cima dele não acrescenta nada demonstrável.

O ponto 7 é competência real que não se converteu em valor em nenhuma das três
aplicações testadas. Isso não o torna falso — torna-o insuficiente.

## O que este projecto realmente demonstra

Não é um sistema de trading. É um **aparelho de medição honesto**: protocolo de
três partições com embargo, erros-padrão sobre dias e não sobre linhas,
correcção para testes múltiplos, verificações de calibração, e comparação
sistemática contra a alternativa gratuita mais simples.

Foi esse aparelho que apanhou, uma a uma, quatro conclusões erradas que eu
próprio tinha produzido. O valor está aí.
