"""Chapter 9: Enriquecimento Convolucional: MiniRocket e o Indicador Monotônico de Saúde."""

def get_chapter():
    md = r"""# Capítulo 9: Enriquecimento Convolucional: MiniRocket e o Indicador Monotônico de Saúde

## 9.1 Por Que Convoluções Rápidas em Vez de Redes Neurais Pesadas?
Em aplicações aeronáuticas embarcadas, o uso de redes neurais profundas com recorrência (LSTM, GRU) ou mecanismos densos de autoatenção (*Transformers*) enfrenta barreiras operacionais severas [2]:
1. **Elevado Custo Computacional e de Memória:** O algoritmo de retropropagação no tempo (*Backpropagation Through Time* - BPTT) demanda recursos computacionais pesados e GPUs dedicadas.
2. **Inviabilidade para Computação de Borda (*Edge Computing*):** Controladores digitais de motor (*FADEC*) e unidades de gerenciamento de saúde (*HUMS*) operam com microprocessadores de bordo com recursos estritamente limitados, requerendo algoritmos determinísticos de latência sub-milissegundo.

Para conciliar alto poder de extração de padrões temporais e extrema eficiência computacional, a Camada 5 incorpora a arquitetura **MiniRocket** (*Minimally Random Convolutional Kernel Transform*, Dempster et al., 2020 [5]), adaptada para a caracterização causal de degradação estrutural conforme proposto nos Slides 19 e 21 da monografia de Juliano Pádua [1].

```
+---------------------------------------------------------------------------------------------------+
|               ARQUITETURA CONVOLUCIONAL DETERMINÍSTICA MINIROCKET E POOLING PPV                   |
|                                                                                                   |
|  Sinal Causal (W=15) ----> [ 84 Kernels L=9, Pesos {-1, +2}, Dilatações d={1,2,4} ]              |
|                                         |                                                         |
|                                         V (Convolução 1D Rápida)                                  |
|                             Mapas de Ativação Convolucional C_j(t)                                |
|                                         |                                                         |
|                                         V                                                         |
|                       [ Pooling PPV: Proporção de Valores Positivos ]                             |
|                                         |                                                         |
|                                         V                                                         |
|                    Health Index Monotônico HI(t) in [0, 1] (Pádua / NASA)                         |
+---------------------------------------------------------------------------------------------------+
```

## 9.2 O Banco de 84 Kernels Determinísticos
Diferente do algoritmo ROCKET convencional (que gerava 10.000 kernels com pesos contínuos pseudo-aleatórios), o MiniRocket emprega um conjunto compacto e estritamente determinístico de **84 kernels** de comprimento fixo $L = 9$ [5].

Cada kernel é composto por pesos discretos do conjunto restrito $\{-1, +2\}$, onde exatamente três posições contêm o valor $+2$ e as seis restantes contêm o valor $-1$. O número exato de permutações possíveis é dado pela análise combinatória:
$$C(9, 3) = \frac{9!}{3!(9 - 3)!} = \frac{9 \times 8 \times 7}{3 \times 2 \times 1} = 84 \text{ kernels determinísticos}$$

### A Propriedade Fundamental da Soma Nula ($\sum W = 0$) e Invariância a Offset DC
A característica analítica mais elegante do MiniRocket reside na propriedade de que todos os 84 kernels possuem soma de coeficientes estritamente nula:
$$\sum_{i=1}^9 W_i = 3 \times (+2) + 6 \times (-1) = 6 - 6 = 0$$

Essa propriedade confere ao MiniRocket **imunidade matemática absoluta a deslocamentos de nível contínuo (invariância a offset DC)**:
* Quando o turbofan sobe do nível do mar para 42.000 pés, a temperatura dos sensores salta centenas de graus Celsius devido à variação de densidade do ar.
* Ao convolucionar o sinal com um deslocamento estático de patamar $\Delta$:
  $$\sum_{i=1}^9 W_i (x_{t-i} + \Delta) = \sum_{i=1}^9 W_i x_{t-i} + \Delta \sum_{i=1}^9 W_i = \sum_{i=1}^9 W_i x_{t-i} + 0$$
* Logo, o MiniRocket anula naturalmente as variações grosseiras de regime de voo e foca com altíssima sensibilidade exclusivamente em distorções morfológicas locais decorrentes de desgaste mecânico [1], [5].

## 9.3 Pooling PPV e o Indicador Monotônico de Saúde
Para cada mapa de ativação convolucional gerado por um kernel $j$, calcula-se a métrica de agregação temporal **PPV** (*Proportion of Positive Values*):
$$\text{PPV}_j(t) = \frac{1}{W} \sum_{\tau=0}^{W-1} \mathbb{I}(C_j(t - \tau) > b_j)$$
onde $\mathbb{I}(\cdot)$ é a função indicadora e $b_j$ é o limiar de polarização (*bias*).

Ao projetar os canais termodinâmicos ($T_{50}, P_{30}, N_c, BPR$) através dos kernels MiniRocket, o estimador gera atributos PPV temporais. A monotonicidade deve ser avaliada por canal e frota, sem pressupô-la como propriedade garantida [1].

![Figura 9.1: Trajetória estritamente monotônica do Índice de Saúde (PPV Pooling) extraído pelos 84 kernels do MiniRocket.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/minirocket_feature_filtering_padua.png)

A Figura 9.1 documenta os atributos PPV para a turbina `#179`; sua trajetória é um estudo de caso e não estabelece monotonicidade nem desempenho de frota por si só.

![Figura 9.2: Extração convolucional causal MiniRocket aplicada sobre a série térmica T50.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/minirocket_feature_filtering_t50.png)

A Figura 9.2 exibe as features convolucionais intermediárias para o canal $T_{50}$, evidenciando o desacoplamento de ruído e a retenção seletiva da dinâmica de degradação.

### Exemplo Teórico 9.1: Convolução Discreta e Cálculo de PPV
Seja um vetor de telemetria local de comprimento 9: $\mathbf{x} = [10, 11, 10, 12, 14, 16, 18, 20, 22]^T$.
Considere o kernel MiniRocket com pesos nas posições 1, 4 e 7 iguais a $+2$ e os demais $-1$:
$$\mathbf{W} = [+2, -1, -1, +2, -1, -1, +2, -1, -1]^T$$
O produto interno linear $\mathbf{W}^T \mathbf{x}$ é:
$$\mathbf{W}^T \mathbf{x} = 2(10) - 11 - 10 + 2(12) - 14 - 16 + 2(18) - 20 - 22$$
$$\mathbf{W}^T \mathbf{x} = 20 - 21 + 24 - 30 + 36 - 42 = -1 - 6 - 6 = -13$$
Caso a série sofra um deslocamento de altitude de $\Delta = +1000$ em todas as entradas:
$$\mathbf{W}^T (\mathbf{x} + 1000) = \mathbf{W}^T \mathbf{x} + 1000 \cdot \sum W_i = -13 + 1000 \cdot (0) = -13$$
O valor é rigorosamente idêntico, comprovando a invariância da saída a saltos atmosféricos exógenos.

### Código Python 9.1: Convolução MiniRocket e Extração de Features PPV
```python
def generate_minirocket_kernels(length: int = 9, num_kernels: int = 84) -> np.ndarray:
    """Gera o banco combinatório determinístico de 84 kernels com pesos {-1, 2}."""
    indices_combinations = list(itertools.combinations(range(length), 3))[:num_kernels]
    kernels = np.full((num_kernels, length), -1.0)
    for i, pos in enumerate(indices_combinations):
        kernels[i, list(pos)] = 2.0
    return kernels

def extract_causal_ppv_features(signal_window: np.ndarray, kernels: np.ndarray, dilation: int = 1) -> np.ndarray:
    """Extrai Proporção de Valores Positivos (PPV) para janela causal deslizante."""
    n_kernels, k_len = kernels.shape
    eff_len = (k_len - 1) * dilation + 1
    if len(signal_window) < eff_len:
        return np.zeros(n_kernels)
    
    subseq = signal_window[-eff_len::dilation]
    convolutions = np.dot(kernels, subseq)
    ppv = (convolutions > 0.0).astype(float)
    return ppv
```
"""
    return md

print("Loaded chapter 9 module.")
