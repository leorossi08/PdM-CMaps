"""Chapter 6: Condicionamento e Filtragem de Sinais (EWMA Causal e Supressão de Ruído)."""

def get_chapter():
    md = r"""# Capítulo 6: Condicionamento e Filtragem de Sinais (EWMA Causal e Supressão de Ruído)

## 6.1 A Natureza do Ruído em Telemetria Aeronáutica
Sensores de turbofans operam em ambientes termomecânicos severos, caracterizados por vibração estrutural de alta frequência, transientes de combustão e turbulência de escoamento [3]. Como consequência, o sinal cru $y_t$ apresenta uma componente estocástica sobreposta conhecida como **jitter térmico**:
$$y_t = y_{real,t} + \eta_t, \quad \text{com } \eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$$

Se um sistema preditivo monitorar diretamente o sinal instantâneo $y_t$, perturbações de alta frequência cruzarão acidentalmente limites estáticos de tolerância, disparando alarmes falsos frequentes.

## 6.2 O Filtro Exponencial Móvel Ponderado (EWMA Causal)
Para atenuar o jitter instrumental sem violar a causalidade temporal, a plataforma emprega o filtro exponencial EWMA (*Exponentially Weighted Moving Average*), estruturado conforme as diretrizes do Slide 17 da monografia de Juliano Pádua [1] e de Montgomery [4]:
$$S_t = \alpha y_t + (1 - \alpha) S_{t-1}, \quad \text{para } t \ge 1, \; S_0 = y_0$$
onde $\alpha \in (0, 1]$ representa o fator de suavização exponencial.

Expandindo recursivamente a equação temporal:
$$S_t = \alpha \sum_{i=0}^{t-1} (1 - \alpha)^i y_{t-i} + (1 - \alpha)^t y_0$$

Os pesos atribuídos às observações passadas decaem geometricamente a uma taxa $(1 - \alpha)$, conferindo primazia à informação recente com consumo de memória estritamente $O(1)$.

## 6.3 Derivação da Supressão de Variância e do Atraso de Fase
A teoria de processos estocásticos permite deduzir analiticamente a atenuação de variância proporcionada pelo filtro linear [4]:
$$\text{Var}(S_t) = \text{Var}\left(\alpha \sum_{i=0}^\infty (1 - \alpha)^i y_{t-i}\right) = \alpha^2 \sigma_y^2 \sum_{i=0}^\infty (1 - \alpha)^{2i}$$

Aplicando a soma da série geométrica infinita $\sum_{i=0}^\infty q^i = \frac{1}{1 - q}$ para $q = (1 - \alpha)^2$:
$$\frac{\sigma_{EWMA}^2}{\sigma_y^2} = \frac{\alpha^2}{1 - (1 - \alpha)^2} = \frac{\alpha^2}{2\alpha - \alpha^2} = \frac{\alpha}{2 - \alpha}$$

Para o parâmetro de projeto adotado na plataforma (**$\alpha = 0,35$**):
$$\frac{\sigma_{EWMA}^2}{\sigma_y^2} = \frac{0,35}{2 - 0,35} = \frac{0,35}{1,65} \approx 0,2121 \quad \implies \quad \mathbf{-78,8\% \text{ de redução de variância}}$$

O atraso de fase temporal (*phase lag*) introduzido pelo filtro é dado por:
$$\tau \approx \frac{1}{\alpha} = \frac{1}{0,35} \approx 2,85 \text{ ciclos}$$

Ao contrário de filtros de média móvel simples de janela longa ($W=15$), que introduzem atraso de mais de 7 ciclos, o EWMA com $\alpha = 0,35$ reage em menos de três ciclos a transições térmicas verdadeiras.

![Figura 6.1: Comparativo de filtragem exponencial causal EWMA (alpha=0.70, 0.35, 0.10) na supressão de jitter térmico.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/signal_filtering_ewma_comparison.png)

A Figura 6.1 ilustra o comportamento comparativo das três configurações de suavização sobre a série temporal de telemetria. 
* O filtro leve ($\alpha = 0,70$) preserva velocidade, porém retém jitter residual excessivo (redução de apenas $46\%$).
* O filtro recomendado ($\alpha = 0,35$) elimina $78,8\%$ do ruído, estabilizando a linha de base nos limites nominais.
* O filtro pesado ($\alpha = 0,10$) gera atraso inaceitável de $10$ ciclos, atrasando o diagnóstico de transientes de falha.

![Figura 6.2: Filtragem EWMA no canal térmico T50 e resíduos centrados em zero.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/signal_filtering_ewma_t50.png)

A Figura 6.2 demonstra a resposta do filtro no canal de temperatura de exaustão ($T_{50}$). Observa-se que a série filtrada rastreia a evolução térmica suave sem distorção de amplitude.

![Figura 6.3: Filtragem EWMA no canal pneumático P30 e atenuação de 78.8% de variância.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/signal_filtering_ewma_p30.png)

A Figura 6.3 exibe a atenuação do ruído de pressurização no canal $P_{30}$, evidenciando resíduos com distribuição estável ao longo da vida operacional.

![Figura 6.4: Filtragem EWMA no canal cinemático Nc (velocidade física do núcleo N2).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/signal_filtering_ewma_nc.png)

A Figura 6.4 comprova a preservação dos transientes cinemáticos no canal de rotação física do eixo de alta ($N_c$).

![Figura 6.5: Filtragem EWMA no canal aerodinâmico BPR (razão de bypass).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/signal_filtering_ewma_bpr.png)

A Figura 6.5 documenta a estabilização da razão de derivação aerodinâmica ($BPR$), demonstrando que o filtro atenua oscilações turbulentas sem suprimir a tendência de longo prazo.

### Exemplo Teórico 6.1: Derivação Passo a Passo da Redução de Ruído
Considere uma série temporal onde o ruído de medição térmico possui desvio padrão amostral $\sigma_y = 5,00^\circ\text{C}$ ($\text{Var}_y = 25,00^\circ\text{C}^2$).
Aplicando o filtro EWMA com $\alpha = 0,35$:
$$\text{Var}_{EWMA} = \frac{0,35}{2 - 0,35} \cdot 25,00 = 0,21212 \cdot 25,00 = 5,303^\circ\text{C}^2$$
$$\sigma_{EWMA} = \sqrt{5,303} = 2,303^\circ\text{C}$$
A dispersão do sinal cai de $\pm 15^\circ\text{C}$ (intervalo de $\pm 3\sigma$) para apenas $\pm 6,91^\circ\text{C}$. Essa contração de $54\%$ na amplitude do desvio padrão é o fator determinante que possibilita a operação de envelopes estocásticos estreitos sem falso alarme.

### Código Python 6.1: Filtro Causal EWMA Vetorizado
```python
def apply_causal_ewma(series: pd.Series, alpha: float = 0.35) -> pd.Series:
    """Aplica filtro exponencial ponderado móvel estritamente causal."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("O fator de suavização alpha deve pertencer ao intervalo (0, 1]")
    return series.ewm(alpha=alpha, adjust=False).mean()

def filter_engine_telemetry(engine_df: pd.DataFrame, sensor_cols: list[str], alpha: float = 0.35) -> pd.DataFrame:
    """Filtra canais de telemetria por unidade de motor mantendo ordenação temporal."""
    filtered = engine_df.sort_values("cycle").copy()
    for col in sensor_cols:
        filtered[f"{col}_ewma"] = apply_causal_ewma(filtered[col], alpha=alpha)
    return filtered
```
"""
    return md

print("Loaded chapter 6 module.")
