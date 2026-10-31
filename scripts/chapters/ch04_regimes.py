"""Chapter 4: Desacoplamento Físico-Atmosférico de Regimes de Voo & O Paradoxo do Pooled R²."""

def get_chapter():
    md = r"""# Capítulo 4: Desacoplamento Físico-Atmosférico de Regimes de Voo & O Paradoxo do Pooled R²

## 4.1 A Dinâmica do Envelope Multi-Regime
No conjunto C-MAPSS FD004, a turbina não opera em bancada estática de ensaios ao nível do mar, mas percorre dinamicamente o envelope de voo comercial completo [3]. A cada missão, os parâmetros exógenos posicionam o motor em um dos seis envelopes operacionais discretos de altitude e velocidade:

```
+----------------------------------------------------------------------------------------------------+
| OS SEIS ENVELOPES OPERACIONAIS DE VOO DO MOTOR TURBOFAN (C-MAPSS FD004)                           |
|                                                                                                    |
|  Regime 0: Baixa Altitude e Velocidade Moderada    (Alt: 0 - 10.000 ft, Mach: 0.00 - 0.25, TRA: 100)  |
|  Regime 1: Decolagem e Subida Inicial ao Nível Mar (Alt: 0 ft, Mach: 0.00, TRA: 100)              |
|  Regime 2: Cruzeiro Transônico de Média Altitude   (Alt: 20.000 - 25.000 ft, Mach: 0.62 - 0.70)   |
|  Regime 3: Cruzeiro Econômico em Teto Intermediário (Alt: 30.000 - 35.000 ft, Mach: 0.70 - 0.80)  |
|  Regime 4: Cruzeiro Estratosférico de Alta Altitude (Alt: 40.000 - 42.000 ft, Mach: 0.80 - 0.84)  |
|  Regime 5: Descida e Aproximação com Baixo Empuxo  (Alt: 10.000 - 25.000 ft, Mach: 0.25 - 0.50)  |
+----------------------------------------------------------------------------------------------------+
```

Caso o analista trace a temperatura crua de exaustão ($T_{50}$) contra os ciclos temporais sem prévio desacoplamento, a curva aparentará um sinal ruidoso de dente de serra oscilando entre $1040^\circ\text{C}$ e $1400^\circ\text{C}$. Um operador ingênuo confundiria essa oscilação com falhas repetidas da máquina a cada transição de altitude.

## 4.2 Clusterização Não-Supervisionada de Regimes via K-Means
Para que a plataforma reconheça de maneira determinística e autônoma o regime instantâneo de voo, a Camada 2 aplica o algoritmo de agrupamento K-Means sobre o vetor de condições operacionais padronizadas:
$$\mathbf{z}_t = \left[\frac{\text{setting\_1}_t - \mu_1}{\sigma_1}, \; \frac{\text{setting\_2}_t - \mu_2}{\sigma_2}, \; \frac{\text{setting\_3}_t - \mu_3}{\sigma_3}\right]^T$$

O algoritmo minimiza a inércia intra-cluster para $K=6$ centróides:
$$\arg\min_{\mathbf{C}} \sum_{j=1}^6 \sum_{\mathbf{z}_t \in C_j} \|\mathbf{z}_t - \mathbf{c}_j\|^2$$

Cada ciclo de telemetria recebe um rótulo discreto $r_t \in \{0, 1, 2, 3, 4, 5\}$, permitindo o desacoplamento termodinâmico completo.

## 4.3 A Decomposição da Variância por ANOVA (Análise de Variância)
Para comprovar cientificamente que a quase totalidade da variação observada nos sensores é de origem atmosférica e exógena, aplica-se a decomposição de variância unidirecional de Fisher (ANOVA) [4]:

$$\text{Variância Total: } \text{Var}_{total} = \frac{1}{N-1} \sum_{i=1}^N (y_i - \bar{y})^2$$
$$\text{Soma dos Quadrados Total: } SS_{total} = SS_{between} + SS_{within}$$
onde:
$$SS_{between} = \sum_{j=1}^K n_j (\bar{y}_j - \bar{y})^2 \quad \text{(Variação devida à Altitude, Mach e Demanda)}$$
$$SS_{within} = \sum_{j=1}^K \sum_{i \in C_j} (y_{ij} - \bar{y}_j)^2 \quad \text{(Variação decorrente de Desgaste Mecânico e Ruído)}$$

Os resultados da decomposição de variância calculados na frota hold-out (com médias por regime aprendidas estritamente no conjunto de treino) comprovam a dominância das condições operacionais:

| Sensor | Grandeza Física | Variância Total | Entre Regimes (%) | Intra-Regime (%) | Baseline OOS $R^2$ (RMSE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`T50`** | Temperatura Saída LPT (°C) | $14.278,97$ | $99,44\%$ | $0,56\%$ | $0,9943$ ($9,05^\circ\text{C}$) |
| **`P30`** | Pressão Saída HPC (psia) | $13.540,73$ | $99,98\%$ | $0,02\%$ | $0,9998$ ($1,48\text{ psia}$) |
| **`BPR`** | Razão de Bypass (adim.) | $0,5057$ | $99,19\%$ | $0,81\%$ | $0,9917$ ($0,065$) |
| **`Nc`** | Rotação do Núcleo (rpm) | $71.836,87$ | $99,69\%$ | $0,31\%$ | $0,9969$ ($14,88\text{ rpm}$) |

A quase totalidade da dispersão dos sensores (entre $99,19\%$ e $99,98\%$) decorre diretamente das manobras de voo e variações atmosféricas ISA, e não de desgaste de curto prazo.

![Figura 4.1: Validação cruzada do Sensor Virtual T50 estritamente intra-regime durante a fase saudável (H=50 ciclos).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/healthy_t50_by_regime.png)

A Figura 4.1 comprova o ajuste homogêneo do estimador em cada envelope isolado. Sem a interferência da amplitude de altitude, os resíduos mantêm-se simétricos em torno de zero em todos os seis regimes.

![Figura 4.2: Validação cruzada do canal de pressurização estática P30 intra-regime durante a fase saudável.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/healthy_p30_by_regime.png)

A Figura 4.2 demonstra comportamento análogo para o canal de pressão de descarga do compressor ($P_{30}$), confirmando que a calibração do modelo preserva a estabilidade das medições mecânicas em todas as faixas de voo.

## 4.4 O Paradoxo do Pooled R²: Efeito de Escala vs. Overfitting
O coeficiente de determinação clássico $R^2$ é definido como:
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{MSE}}{\text{Var}_{total}}$$

Ao avaliar o Sensor Virtual de $T_{50}$ sobre toda a frota de teste hold-out (15.890 ciclos), obtém-se:
$$R^2_{pooled} = 1 - \frac{7,026^2}{14.278,96} = 1 - \frac{49,36}{14.278,96} = \mathbf{0,99654}$$

Um auditor desavisado poderia interpretar esse coeficiente como evidência de *overfitting* ou memorização de dados. Todavia, a análise físico-matemática demonstra que **não se trata de sobreajuste**:
1. **Magnitude da Variância Global:** A oscilação térmica entre o nível do mar e o cruzeiro atinge mais de $350^\circ\text{C}$ ($\text{Var}_{total} = 14.278^\circ\text{C}^2$). Como o modelo conhece a altitude e o Mach, ele explica deterministicamente a dispersão global.
2. **Erro Físico Absoluto:** O erro médio quadrático ($\text{RMSE} = 7,02^\circ\text{C}$) é perfeitamente condizente com a incerteza instrumental de termopares e idêntico entre o treino ($\text{RMSE} = 6,8^\circ\text{C}$) e o teste hold-out ($\text{RMSE} = 7,02^\circ\text{C}$).
3. **Validação Estrita Intra-Regime:** Ao restringir o cálculo do $R^2$ a cada regime individual durante a fase saudável ($H=50$), a variância local cai para cerca de $30^\circ\text{C}^2$, e o $R^2$ intra-regime assume valores na faixa realista de **$0,41$ a $0,58$**.

### Exemplo Teórico 4.1: Decomposição Numérica de Variância e o Efeito Simpson
Suponha duas condições de voo: Regime A (Nível do Mar, $\bar{y}_A = 1380^\circ\text{C}$, $\sigma_A^2 = 25$) e Regime B (Cruzeiro a 40.000 ft, $\bar{y}_B = 1050^\circ\text{C}$, $\sigma_B^2 = 25$), com $n_A = n_B = 1000$ ciclos.
A média global é $\bar{y} = 1215^\circ\text{C}$.
$$SS_{between} = 1000 \cdot (1380 - 1215)^2 + 1000 \cdot (1050 - 1215)^2 = 2000 \cdot (165)^2 = 54.450.000$$
$$SS_{within} = 1000 \cdot 25 + 1000 \cdot 25 = 50.000$$
$$\text{Proporção Explicada} = \frac{54.450.000}{54.500.000} = 99,908\%$$

Mesmo que o regressor cometa um erro local de $\text{MSE} = 15$ dentro de cada regime ($R^2_{local} = 1 - 15/25 = 0,40$), o $R^2$ agregado global calculado é:
$$R^2_{global} = 1 - \frac{15}{27.250} = 0,99945$$
Essa discrepância numérica ilustra uma variante física do Paradoxo de Simpson, provando que métricas agregadas devem ser complementadas por avaliações locais intra-regime.

### Código Python 4.1: Clusterização de Regimes e Decomposição ANOVA
```python
def cluster_and_decompose_variance(df: pd.DataFrame, target_col: str = "s4", n_clusters: int = 6):
    '''Clusteriza configurações de voo e calcula decomposição de variância ANOVA.'''
    features = ["setting_1", "setting_2", "setting_3"]
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["regime"] = kmeans.fit_predict(scaled_x)
    
    y = df[target_col].to_numpy()
    grand_mean = y.mean()
    total_ss = np.sum((y - grand_mean) ** 2)
    
    between_ss = sum(
        len(group) * (group[target_col].mean() - grand_mean) ** 2
        for _, group in df.groupby("regime")
    )
    within_ss = total_ss - between_ss
    pct_between = (between_ss / total_ss) * 100.0
    pct_within = (within_ss / total_ss) * 100.0
    return df, {"pct_between": pct_between, "pct_within": pct_within, "total_variance": total_ss / (len(y) - 1)}
```
"""
    return md

print("Loaded chapter 4 module.")
