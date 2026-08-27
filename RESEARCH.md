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
