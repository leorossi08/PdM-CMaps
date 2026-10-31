"""Chapter 3: Engenharia de Dados, Limpeza e Particionamento Disjunto de Frota."""

def get_chapter():
    md = r"""# Capítulo 3: Engenharia de Dados, Limpeza e Particionamento Disjunto de Frota

## 3.1 Identificação de Sensores Invariantes e Físico-Determinísticos
Uma das fragilidades metodológicas mais frequentes em pipelines ingênuos de Machine Learning aplicados ao C-MAPSS consiste em submeter cegamente todos os 21 sensores aos estimadores estatísticos, sem prévia análise termodinâmica [2], [3].

No subconjunto FD004, verifica-se a existência de dois grupos específicos de canais invariantes:
1. **Canais Analiticamente Determinísticos ($s_1: T_2$ e $s_5: P_2$):**  
   Os sensores $s_1$ e $s_5$ registram a condição do ar estagnado imediatamente antes do bordo de ataque das palhetas do ventilador. Pela termodinâmica dos fluidos compressíveis ideais:
   $$T_2 = T_{amb} \left(1 + \frac{\gamma - 1}{2} M^2\right)$$
   $$P_2 = P_{amb} \left(1 + \frac{\gamma - 1}{2} M^2\right)^{\frac{\gamma}{\gamma - 1}}$$
   Como o ar de admissão não sofreu transformação mecânica no interior da turbomáquina, esses dois sensores constituem funções exatas da altitude e do número de Mach. Ao ajustar um modelo preditivo sobre $T_2$ e $P_2$, o erro residual é idêntico a zero ($\text{RMSE} \approx 10^{-13}$) e $R^2 \equiv 1,0000$. Esse resultado não reflete aprendizado útil de máquina, mas mera redundância da equação de estado atmosférica.
2. **Canais de Referência e Comandos Constantes FADEC ($s_{18}: Nf\_dmd$ e $s_{19}: PCNfR\_dmd$):**  
   Nos envelopes de regime estabilizado, o controle FADEC fixa patamares pré-programados de rotação demandada, resultando em variância nula ($\sigma^2 = 0$).

A inclusão de colunas com variância nula acarreta instabilidades numéricas severas em algoritmos baseados em inversão de matrizes e normalização $Z$-score:
$$Z_t = \frac{x_t - \mu_x}{\sigma_x + \epsilon}$$

## 3.2 O Perigo do Vazamento Temporal (*Data Leakage*) e a Falácia do K-Fold Aleatório
Em problemas convencionais de séries temporais financeiras ou demanda de energia, costuma-se aplicar o particionamento temporal cronológico [2]. Contudo, no contexto de frotas industriais de turbinas aeronáuticas, a aplicação de corte por linha aleatória (*random train-test split*) constitui **falha metodológica crítica**:
* Caso observações de um mesmo motor `#k` sejam distribuídas aleatoriamente entre os conjuntos de treino e teste (por exemplo, ciclos ímpares no treino e ciclos pares no teste), o algoritmo tem acesso indevido às tolerâncias dimensionais específicas de montagem, folgas basais de fabricação e ruído estocástico intrínseco daquela unidade.
* Isso produz métricas de validação cruzada inflacionadas ($R^2 > 0,95$), mas leva ao colapso preditivo quando o modelo é implantado para monitorar um motor novo recém-saído da linha de montagem.

```
+---------------------------------------------------------------------------------------------------+
|                        PARTICIONAMENTO ALEATÓRIO (COM VAZAMENTO) VS DISJUNTO                      |
|                                                                                                   |
|  [Vazamento Grave: K-Fold por Linha / Ciclo Aleatório]                                            |
|    Motor 1: [Treino: Ciclo 1, 2, 4]  -->  [Teste: Ciclo 3, 5]  ==> VAZAMENTO DE BIAS DA UNIDADE!   |
|    Motor 2: [Treino: Ciclo 2, 5, 6]  -->  [Teste: Ciclo 1, 4]  ==> INFLAÇÃO ARTIFICIAL DE MÉTRICA! |
|                                                                                                   |
|  [Protocolo Científico Disjunto: Engine Holdout Split]                                            |
|    Frota de Treino (75% = 187 Motores): [Motor 1, 2, ..., 187 do Nascimento ao Fim de Vida]      |
|    Frota Hold-out  (25% = 62 Motores):  [Motor 188, ..., 249 ISOLADOS - NUNCA VISTOS PELO MODELO] |
+---------------------------------------------------------------------------------------------------+
```

### Protocolo de Validação Disjunta por Frota (*Engine Hold-out Split*)
Para garantir fidelidade operacional estrita, o ecossistema estabelece o isolamento completo por turbina:
* **Conjunto de Treinamento e Validação:** $75\%$ dos motores da frota (187 motores completos em FD004), englobando todos os seus ciclos do primeiro voo ao esgotamento mecânico.
* **Conjunto de Teste Hold-out:** $25\%$ dos motores da frota (62 motores inteiros em FD004), preservados rigorosamente sem contato com nenhum estimador de parâmetros, normalizador ou regressor de RUL.
* **Unidades de Teste Longitudinais:** Turbinas de vida útil prolongada, como o motor `#179` (vida total de 435 ciclos), são reservadas integralmente no hold-out para validação contínua de telemetria crua e análise cega do primeiro alerta preditivo.

### Exemplo Teórico 3.1: Singularidade da Matriz de Covariância por Canais Constantes
Considere o problema de regressão linear regularizada (Ridge) para o vetor de parâmetros $\boldsymbol{\beta}$:
$$\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

Se a matriz de atributos $\mathbf{X} \in \mathbb{R}^{N \times p}$ contém uma coluna constante $c$ tal que $\mathbf{x}_j = [c, c, \dots, c]^T$, a coluna $\mathbf{x}_j$ é linearmente dependente da coluna de intercepto $\mathbf{1}$. 

Na ausência de regularização ($\lambda = 0$), a matriz de Gram $\mathbf{X}^T \mathbf{X}$ possui determinante nulo ($\det(\mathbf{X}^T \mathbf{X}) = 0$), tornando a matriz estritamente não-invertível. Com regularização fraca, o número de condicionamento da matriz $\kappa(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}) = \frac{\sigma_{max}}{\sigma_{min}}$ cresce para valores superiores a $10^{15}$, provocando catastrófica amplificação numérica de erros de arredondamento em ponto flutuante.

### Código Python 3.1: Particionamento Disjunto por Unidade e Filtragem de Canais
```python
def prepare_disjoint_fleet_data(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    '''Realiza limpeza de sensores invariantes e particionamento disjunto de frota.'''
    # 1. Identificar e descartar sensores com desvio padrão nulo ou determinísticos
    stds = df[[f"s{i}" for i in range(1, 22)]].std()
    invariant_cols = stds[stds < 1e-6].index.tolist()
    deterministic_cols = ["s1", "s5"]  # T2 e P2 atmosféricos
    drop_cols = list(set(invariant_cols + deterministic_cols))
    
    clean_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # 2. Particionamento disjunto por unidade de motor
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(clean_df, groups=clean_df["unit_id"]))
    
    train_fleet = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    test_fleet = clean_df.iloc[test_idx].copy().reset_index(drop=True)
    
    # Validação formal de disjunção
    train_units = set(train_fleet["unit_id"].unique())
    test_units = set(test_fleet["unit_id"].unique())
    assert train_units.isdisjoint(test_units), "ERRO CRÍTICO: Vazamento de unidades entre treino e teste!"
    return train_fleet, test_fleet
```
"""
    return md

print("Loaded chapter 3 module.")
