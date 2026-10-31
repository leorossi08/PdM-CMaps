"""Chapter 11: Prognóstico de Vida Útil Restante (RUL): Formulação e Fenômenos de Paridade."""

def get_chapter():
    md = r"""# Capítulo 11: Prognóstico de Vida Útil Restante (RUL): Formulação e Fenômenos de Paridade

## 11.1 A Função Alvo Linear por Partes da NASA
Na modelagem de Vida Útil Restante (*Remaining Useful Life* - RUL), a abordagem ingênua consistiria em definir o alvo como a contagem regressiva linear estrita até o fim de vida:
$$\text{RUL}_{linear}(t) = \text{EOL} - t$$

Contudo, Saxena et al. [3] e a literatura canônica de propulsão estabeleceram que, durante as etapas iniciais de operação de uma turbina aeronáutica, o desgaste mecânico é infinitesimal e indistinguível do ruído térmico normal de combustão. Tentar forçar um estimador supervisionado a diferenciar se um motor no ciclo 20 tem vida restante de 300 ou 400 ciclos introduz gradientes espúrios que desestabilizam o treinamento.

Por essa razão, adota-se a formulação **linear por partes com saturação superior** (*piecewise linear target*):
$$\text{RUL}_{clip}(t) = \min(RUL_{max}, \; \text{EOL} - t), \quad \text{onde tipicamente } RUL_{max} = 125 \text{ ciclos}$$

```
+---------------------------------------------------------------------------------------------------+
|                     PERFIL DO ALVO RUL LINEAR POR PARTES (CONVENÇÃO NASA C-MAPSS)                 |
|                                                                                                   |
|   RUL                                                                                             |
|   125 |--------------------+ (Platô Saudável: Desgaste Mecânico Não-Detectável)                  |
|       |                     \                                                                     |
|       |                      \                                                                    |
|       |                       \  (Zona de Degradação Ativa: Rastreio Físico Monotônico)          |
|       |                        \                                                                  |
|       |                         \                                                                 |
|     0 +--------------------------+--------------------> Ciclos                                    |
|       0                         t_alerta             EOL                                          |
+---------------------------------------------------------------------------------------------------+
```

## 11.2 Análise do Gráfico de Paridade de 45° e o Fenômeno da Coluna em $x=125$
Ao analisar a projeção dos valores reais contra as previsões no gráfico de paridade da frota hold-out:

![Figura 11.1: Envelope de Paridade de 45 graus do modelo XGBoost sobre a frota hold-out de teste (62 motores) com bandas de +-20 ciclos.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/rul_best_model_parity_envelope.png)

A Figura 11.1 revela duas regiões de comportamento distintas:
1. **A Coluna Vertical em $x = 125$:** Observa-se uma concentração conspícua de pontos distribuídos verticalmente na abscissa $x = 125$. Esse fenômeno é o **resultado matemático esperado da saturação do alvo**: enquanto o motor se encontra em qualquer ciclo de sua juventude operacional ($t < \text{EOL} - 125$), o rótulo verdadeiro permanece travado em $125$. O modelo, sensível a variações pontuais de pressão e temperatura entre diferentes voos, oscila sua previsão ligeiramente em torno desse patamar nominal.
2. **A Faixa de Degradação Ativa ($x < 125$):** Assim que o ativo ingressa no regime de desgaste mensurável ($x < 125$), os pontos alinham-se com extrema precisão à bissetriz ideal de 45°, situando-se rigorosamente contidos no interior das bandas de tolerância de $\pm 20$ ciclos.

![Figura 11.2: Previsão de RUL ponto a ponto do melhor estimador (XGBoost) contra os valores reais da NASA.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/rul_best_model_actual_vs_predicted.png)

A Figura 11.2 ilustra a curva temporal ciclo a ciclo para a turbina `#179`. Nota-se a aderência da estimativa suavizada à linha de vida real ao longo de mais de 400 ciclos.

## 11.3 Suavização de Tendência por Mediana Móvel ($W=15$)
Como o motor C-MAPSS FD004 altera bruscamente de altitude e empuxo a cada ciclo, as previsões brutas de RUL pontuais apresentam pequenos choques transitórios. Para eliminar essas perturbações sem introduzir atrasos de fase prejudiciais, a Camada 6 aplica um filtro de **mediana móvel deslizante de comprimento $W = 15$ ciclos**:
$$\widetilde{\text{RUL}}(t) = \text{mediana}_{\tau \in [0, W-1]} \widehat{\text{RUL}}(t - \tau)$$

A mediana móvel rejeita categoricamente outliers isolados de transição de voo, proporcionando uma trajetória suave e estável para a tomada de decisão no cockpit industrial.

### Exemplo Teórico 11.1: Impacto do Platô na Função de Perda Quadrática
Considere a perda de treino com alvo linear estrito vs linear por partes:
$$L_{linear} = \frac{1}{2} (\hat{y} - (\text{EOL} - t))^2 \implies \frac{\partial L}{\partial \hat{y}} = \hat{y} - (\text{EOL} - t)$$
Em $t = 10$ para um motor com $\text{EOL} = 300$, o alvo é $290$. Para outro motor com $\text{EOL} = 200$, o alvo é $190$. Como ambos apresentam exatamente a mesma telemetria saudável de fábrica, o estimador recebe gradientes opostos ($\nabla L_1 > 0$ e $\nabla L_2 < 0$) para entradas idênticas, degradando a convergência.
Com o alvo truncado em $RUL_{max} = 125$, ambos recebem exatamente o mesmo rótulo $125$, anulando a contradição de gradientes na fase saudável.

### Código Python 11.1: Alvo RUL por Partes e Filtro de Tendência por Mediana
```python
def create_piecewise_rul_target(df: pd.DataFrame, max_rul: int = 125) -> pd.Series:
    '''Calcula o alvo de RUL linear por partes conforme padronização NASA.'''
    eol_per_unit = df.groupby("unit_id")["cycle"].transform("max")
    linear_rul = eol_per_unit - df["cycle"]
    clipped_rul = linear_rul.clip(upper=max_rul)
    return clipped_rul

def apply_median_trajectory_filter(predictions: pd.Series, window: int = 15) -> pd.Series:
    '''Aplica filtro causal de mediana móvel para eliminar transientes de voo.'''
    return predictions.rolling(window=window, min_periods=1).median()
```
"""
    return md

print("Loaded chapter 11 module.")
