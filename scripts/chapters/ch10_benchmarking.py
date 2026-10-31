"""Chapter 10: Modelagem e Benchmark Multi-Algoritmo (Reconstrução Térmica de T50)."""

from book_tables import load_tables

def get_chapter():
    tables = load_tables()
    md = r"""# Capítulo 10: Modelagem e Benchmark Multi-Algoritmo (Reconstrução Térmica de T50)

## 10.1 Taxonomia Abrangente das Famílias Algorítmicas Avaliadas
A literatura contemporânea de aprendizado de máquina frequentemente restringe-se a comparações pontuais entre poucos modelos selecionados [2]. Para estabelecer um padrão rigoroso de benchmark científico, a plataforma avalia **14 algoritmos concorrentes** distribuídos em seis grandes famílias paradigmáticas [2], [6], [7], [8], [9]:

1. **Linha de Base Empírica:** Média histórica de RUL.
2. **Modelos Lineares Regularizados:** Regressão Ridge com validação cruzada para o parâmetro $\ell_2$.
3. **Métodos de Kernel e Margem:** Máquinas de Vetores de Suporte (*Support Vector Regression* - SVR).
4. **Ensembles por Bagging:** Random Forest (Breiman) e Extra Trees (Geurts et al.).
5. **Gradient Boosted Decision Trees (GBDTs):** XGBoost (Chen & Guestrin [6]), LightGBM (Ke et al. [7]), CatBoost (Prokhorenkova et al. [8]) e HistGradientBoosting.
6. **Convoluções Determinísticas:** MiniRocket acoplado a classificador/regressor RidgeCV [5].
7. **Redes Neurais Profundas (*Deep Learning*):** Perceptron Multicamadas (MLP), Redes Convolucionais 1D (PyTorch 1D CNN), Redes Recorrentes (PyTorch LSTM) e Redes de Autoatenção (PyTorch Transformer [9]).

## 10.2 A Função de Perda Assimétrica da NASA ($S_{NASA}$)
Em ambientes industriais e aeronáuticos, os erros preditivos de RUL possuem assimetria de risco crítica:
* **Previsão Adiantada ($\hat{y} < y$):** O modelo prevê falha antes de ela ocorrer. Consequência: agendamento preventivo antecipado, com perda aceitável de ciclos úteis de operação.
* **Previsão Atrasada ($\hat{y} > y$):** O modelo prevê vida útil superior à realidade. Consequência: risco catastrófico de quebra em pleno voo.

Para modelar essa assimetria com rigor operacional, Saxena et al. [3] definiram a função exponencial de pontuação da NASA:
$$s_i = \begin{cases} 
\exp\left(-\frac{\hat{y}_i - y_i}{a_1}\right) - 1, & \text{para } \hat{y}_i - y_i < 0 \quad (\text{adiantada, com } a_1 = 13) \\ 
\exp\left(\frac{\hat{y}_i - y_i}{a_2}\right) - 1, & \text{para } \hat{y}_i - y_i \ge 0 \quad (\text{atrasada, com } a_2 = 10) 
\end{cases}$$
$$S_{NASA} = \sum_{i=1}^N s_i$$

A constante de penalidade para previsões atrasadas ($a_2 = 10$) penaliza o erro com taxa de crescimento exponencial significativamente mais agressiva do que para previsões conservadoras ($a_1 = 13$).

## 10.3 Resultados Comparativos Consolidados na Frota Hold-out
A tabela abaixo consolida os resultados preditivos apurados estritamente sobre os **62 motores mantidos no teste hold-out disjunto (15.890 ciclos totais)**:

""" + tables["bench_md"] + r"""

![Figura 10.1: Benchmark comparativo de R2 no teste hold-out de todos os modelos avaliados (GBDTs, Ensembles, Redes Neurais, SVM e Transformers).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/rul_model_r2_comparison.png)

A Figura 10.1 sintetiza o panorama de desempenho:
1. **Supremacia dos GBDTs:** Os estimadores baseados em árvores impulsionadas (XGBoost, LightGBM, HistGradientBoosting e Random Forest) lideram com folga estatística e estabilidade em validação cruzada agrupada por motor.
2. **Desempenho e Eficiência de Deep Learning vs GBDTs:** Enquanto arquiteturas neurais sequenciais exigem tempos de processamento substancialmente mais elevados, os modelos baseados em árvores com engenharia causal de atributos oferecem acurácia de ponta com latência de treino sub-segundo, alinhando-se às conclusões do artigo do IEEE CIM 2026 [2].

### Exemplo Teórico 10.1: Cálculo Comparativo de Penalidade da NASA
Considere dois estimadores com erro absoluto idêntico de 10 ciclos:
* **Estimador A (Adiantado):** $\Delta = \hat{y} - y = -10$ ciclos.
  $$s_A = \exp\left(-\frac{-10}{13}\right) - 1 = \exp(0,7692) - 1 = 2,158 - 1 = \mathbf{1,16}$$
* **Estimador B (Atrasado):** $\Delta = \hat{y} - y = +10$ ciclos.
  $$s_B = \exp\left(\frac{10}{10}\right) - 1 = \exp(1,0) - 1 = 2,718 - 1 = \mathbf{1,72}$$

A penalidade do modelo atrasado é $48\%$ superior. Caso o erro cresça para 30 ciclos:
$$s_{A,30} = \exp(30/13) - 1 = \exp(2,308) - 1 = 10,05 - 1 = \mathbf{9,05}$$
$$s_{B,30} = \exp(30/10) - 1 = \exp(3,0) - 1 = 20,08 - 1 = \mathbf{19,08}$$
A penalização pelo atraso atinge mais que o dobro ($+110\%$), punindo com severidade projeções excessivamente otimistas.

### Código Python 10.1: Função de Perda Assimétrica da NASA
```python
def compute_nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    '''Calcula a métrica oficial de pontuação assimétrica da NASA e porcentagem de previsões atrasadas.'''
    diff = y_pred - y_true
    scores = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,  # Previsão adiantada (conservadora)
        np.exp(diff / 10.0) - 1.0    # Previsão atrasada (crítica)
    )
    total_score = float(np.sum(scores))
    late_pct = float(np.mean(diff > 0.0) * 100.0)
    return total_score, late_pct
```
"""
    return md

print("Loaded chapter 10 module.")
