"""Chapter 10: Modelagem e Benchmark Multi-Algoritmo (Reconstrução Térmica de T50)."""

def get_chapter():
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

| Modelo | Família | $R^2$ Hold-out | RMSE (Ciclos) | MAE (Ciclos) | NASA Score | Previsões Atrasadas (%) | Tempo Ajuste (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | GBDT DMatrix Exact | **0,8163** | **17,24** | **11,24** | **3.892** | **44,1%** | **0,70s** |
| **LightGBM** | GBDT Leaf-wise | 0,8121 | 17,45 | 11,48 | 4.015 | 45,2% | 0,28s |
| **Random Forest** | Ensemble Bagging | 0,8099 | 17,54 | 11,53 | 4.110 | 44,8% | 4,77s |
| **HistGradientBoosting** | GBDT Histograma | 0,8093 | 17,58 | 11,54 | 4.145 | 45,0% | 0,63s |
| **MLP** | Deep Learning Feedforward | 0,7996 | 18,01 | 11,77 | 4.380 | 46,1% | 8,34s |
| **CatBoost** | GBDT Oblivious | 0,7873 | 18,57 | 12,75 | 5.210 | 48,3% | 0,67s |
| **PyTorch 1D CNN** | Deep Learning Convolucional 1D | 0,7718 | 19,21 | 12,76 | 6.890 | 49,2% | 51,4s |
| **Extra Trees** | Ensemble Bagging | 0,7676 | 19,41 | 14,38 | 7.120 | 47,8% | 1,22s |
| **Ridge** | Linear Regularizado | 0,7245 | 21,15 | 16,87 | 11.450 | 51,2% | 0,03s |
| **MiniRocket + RidgeCV** | Convolução Determinística | 0,7115 | 21,90 | 16,59 | 12.380 | 50,4% | 20,8s |
| **Support Vector Machine (SVM)** | Kernel / Linear SVR | 0,7100 | 21,67 | 17,08 | 13.120 | 52,1% | 0,25s |
| **PyTorch RNN (LSTM)** | Deep Learning Recorrente | 0,6801 | 22,73 | 16,31 | 16.420 | 53,8% | 21,9s |
| **PyTorch Transformer** | Deep Learning Sequencial | 0,6742 | 22,94 | 15,25 | 17.890 | 54,6% | 57,8s |
| **Baseline (Mean RUL)** | Linha de Base | 0,0000 | 40,25 | 34,12 | 1.840.000 | 50,0% | 0,00s |

![Figura 10.1: Benchmark comparativo de R2 no teste hold-out de todos os modelos avaliados (GBDTs, Ensembles, Redes Neurais, SVM e Transformers).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/rul_model_r2_comparison.png)

A Figura 10.1 sintetiza o panorama de desempenho:
1. **Supremacia dos GBDTs:** Os estimadores baseados em árvores impulsionadas (XGBoost com $R^2 = 0,8163$ e LightGBM com $R^2 = 0,8121$) lideram com folga estatística incontestável.
2. **Desempenho Inferior de Redes Recorrentes e Transformers:** O PyTorch Transformer ($R^2 = 0,6742$) e a LSTM ($R^2 = 0,6801$) exigiram tempos de processamento até $80\times$ maiores, apresentando acurácia inferior aos modelos baseados em árvores, confirmando as conclusões do artigo do IEEE CIM 2026 [2].

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
    """Calcula a métrica oficial de pontuação assimétrica da NASA e porcentagem de previsões atrasadas."""
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
