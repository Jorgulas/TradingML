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

# Síntese do programa de investigação

Quatro rondas, e um padrão consistente: **sempre que um número pareceu bom, foi
a inspecção mais fina que o explicou.**

| Ronda | O que parecia | O que era |
|---|---|---|
| 1 | Magnitude prevista com Spearman 0,47 e lift de 9× | 74% era saber a hora do dia; 12 parâmetros lineares fazem 96% |
| 2 | +76% de lucro no decil de topo | Mudança de regime; corrigido, sobra +13% do prémio |
| 3 | Média 1,30 bate o equilíbrio de 1,11 | **Mediana 0,975** — o trade típico perde; o lucro está em 5% de eventos que as opções já cobram |
| 4 | Alarme de risco 2,98× melhor que o acaso | Não reduz variância nenhuma; a previsão correlaciona 0,78 com a volatilidade recente |

## O que ficou estabelecido

**Negativo, com solidez:**

1. **A direcção não é previsível.** Seis modelos entre 49,1% e 53,5%; 13
   regimes testados com correcção de Bonferroni, nenhum sobrevive.
2. **O ML não bate o HAR-RV a prazo diário.** Empate técnico (+0,008) entre 40
   features com XGBoost e 4 coeficientes lineares.
3. **Nada aqui é negociável.** Nem por opções (o lucro está em eventos já
   cobrados), nem por execução (uma regra de relógio ganha), nem por
   dimensionamento (a volatilidade recente já dá tudo).

**Positivo, e real:**

4. **A volatilidade é previsível** — mas sobretudo por sazonalidade
   intradiária, e isso já se sabia há décadas.
5. **Existe sinal residual genuíno** depois de tirar calendário e inércia:
   Spearman 0,17 a 1h, IC95% [0,153, 0,185].
6. **O modelo detecta eventos de cauda** 2,98× melhor que o acaso.

O ponto 6 é competência real que não se converteu em valor em nenhuma das três
aplicações testadas. Isso não o torna falso — torna-o insuficiente.

## O que este projecto realmente demonstra

Não é um sistema de trading. É um **aparelho de medição honesto**: protocolo de
três partições com embargo, erros-padrão sobre dias e não sobre linhas,
correcção para testes múltiplos, verificações de calibração, e comparação
sistemática contra a alternativa gratuita mais simples.

Foi esse aparelho que apanhou, uma a uma, quatro conclusões erradas que eu
próprio tinha produzido. O valor está aí.
