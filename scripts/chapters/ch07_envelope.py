"""Chapter 7: Detecção de Anomalias por Envelope ('Embrulho') e Casco Geométrico."""

def get_chapter():
    md = r"""# Capítulo 7: Detecção de Anomalias por Envelope ("Embrulho") e Casco Geométrico

## 7.1 O Conceito do "Embrulho" de Juliano Pádua
Na monografia de Juliano Pádua [1], o conceito de "Embrulho" (ou Envelope) é concebido como um invólucro estocástico de tolerância dinâmica ajustado em torno do comportamento nominal da máquina. Qualquer sinal que rompa esse invólucro de maneira persistente deixa de ser classificado como variação operacional admissível e passa a constituir evidência de patologia mecânica [1], [4].

Na plataforma, esse conceito é implementado através de duas formulações matemáticas sinérgicas:
1. **O Envelope Shewhart de Controle Estatístico de Processos (SPC):** Baseado na inferência da variância dos resíduos [4].
2. **O Envelope Geométrico Digital (Rolling Convex Hull):** Baseado no rastreio da envoltória convexa de picos e vales locais [1].

## 7.2 O Envelope Shewhart SPC ($\mu_e \pm 3\sigma_e$)
Seja $e(t) = y(t) - \hat{y}(t)$ o resíduo gerado pelo Sensor Virtual. Durante a janela de calibração saudável ($t \in [1, H]$), estimam-se a média e o desvio padrão amostrais basais:
$$\mu_e = \frac{1}{H} \sum_{t=1}^H e(t), \quad \sigma_e = \sqrt{\frac{1}{H - 1} \sum_{t=1}^H (e(t) - \mu_e)^2}$$

Os limites nominais de controle estatístico de Shewhart [4] são definidos como:
$$\text{UCL} = \mu_e + 3\sigma_e \quad \text{(Upper Control Limit - Limite Superior)}$$
$$\text{LCL} = \mu_e - 3\sigma_e \quad \text{(Lower Control Limit - Limite Inferior)}$$

Sob a hipótese nula de estabilidade estrutural (distribuição normal dos resíduos):
$$P(\mu_e - 3\sigma_e \le e(t) \le \mu_e + 3\sigma_e) = 1 - 2\Phi(-3) \approx 0,9973 \quad (99,73\%)$$

A probabilidade teórica de uma leitura puramente saudável extrapolar a banda por flutuação amostral isolada é de apenas $\alpha = 0,27\%$ (1 evento a cada 370 ciclos).

## 7.3 O Envelope Geométrico Digital (Rolling Convex Hull)
Em conjunto com o critério paramétrico de Shewhart, a Camada 4 computa o casco envolvente móvel (*rolling hull*) sobre uma janela local deslizante de comprimento $W = 15$ ciclos [1]:
$$\text{Upper\_Hull}(t) = \max_{\tau \in [0, W-1]} y(t - \tau) + \delta$$
$$\text{Lower\_Hull}(t) = \min_{\tau \in [0, W-1]} y(t - \tau) - \delta$$
onde $\delta$ representa a tolerância geométrica admissível. Enquanto o envelope Shewhart monitora o afastamento da média, o envelope geométrico captura variações na modulação de amplitude de pico a pico.

## 7.4 A Regra de Persistência Industrial ($k \ge 2$) e a Fronteira de Pareto
Se um sistema de controle adotar a regra ingênua de violação unitária ($k=1$), qualquer perturbação espúria isolada que cruze a fronteira disparará alarme de parada mandatória de voo. 

Para avaliar esse impacto, todos os **62 motores mantidos estritamente no conjunto de teste hold-out** foram submetidos a diferentes critérios de persistência temporal:

| Critério de Alarme | Falsos Alarmes Saudáveis ($t \le 50$) | Taxa de Detecção da Falha | Lead Time Médio na Frota | Diagnóstico Industrial |
| :--- | :---: | :---: | :---: | :--- |
| **$k=1$ (Ponto Único)** | **$11,3\%$ (7 motores)** | $100,0\%$ | $90,6$ ciclos | **Inviável**: 1 a cada 9 motores saudáveis sofreria parada indevida. |
| **$k=2$ (Persistência)** | **$0,0\%$ (ZERO)** | **$96,8\%$** | **$33,0$ ciclos** | **Fronteira Ótima**: Elimina 100% dos alarmes falsos com alta sensibilidade. |
| **$k=3$ (3 Consecutivos)** | **$0,0\%$ (ZERO)** | $93,5\%$ | $24,6$ ciclos | **Conservador**: Reduz a margem temporal de manutenção. |
| **EWMA ($\alpha=0,35$)** | Recalcular com UCL variável no tempo | Recalcular | Recalcular | Valores anteriores usavam corte fixo inadequado de 3,0. |

```
+---------------------------------------------------------------------------------------------------+
|                     MÁQUINA DE ESTADOS FINITA: TRAVA DE PERSISTÊNCIA INDUSTRIAL (k=2)             |
|                                                                                                   |
|    +--------------------+       |e(t)| > 3σ        +-----------------------+                      |
|    |  ESTADO 0: NORMAL  | ---------------------->  |  ESTADO 1: SOB SUSPEITA|                      |
|    |  (Contador k = 0)  | <----------------------  |  (Contador k = 1)     |                      |
|    +--------------------+       |e(t)| <= 3σ       +-----------------------+                      |
|                                (Ruído Espúrio /               |                                   |
|                                 Ciclo 173)                    | |e(t+1)| > 3σ                     |
|                                                               V (Persistência Confirmada)         |
|                                                    +-----------------------+                      |
|                                                    |  ESTADO 2: ALARME REAL|                      |
|                                                    |  (Ciclo 222: Lead Time|                      |
|                                                    |   de +213 ciclos!)    |                      |
|                                                    +-----------------------+                      |
+---------------------------------------------------------------------------------------------------+
```

![Figura 7.1: Envelope Shewhart SPC do Sensor Virtual T50: Ruído no Ciclo 173 (+0.14C) vs Alarme de Degradação no Ciclo 222 (+213 ciclos de lead time).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/t50_time_series_envelope_detection.png)

A Figura 7.1 documenta a trajetória longitudinal da turbina `#179`. Observa-se que no Ciclo 173 ocorre um rompimento isolado efêmero ($+0,14^\circ\text{C}$), prontamente absorvido pela trava de persistência como ruído gaussiano. No Ciclo 222, estabelece-se o rompimento persistente contínuo, deflagrando o alarme preditivo com **$+213$ ciclos de antecedência** antes do colapso no ciclo 435.

![Figura 7.2: Envelope SPC do canal pneumático P30: Primeiro alarme antecipado no Ciclo 191 (+244 ciclos de folga).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/p30_time_series_envelope_detection.png)

A Figura 7.2 comprova a sensibilidade antecipada do canal pneumático $P_{30}$, cujo envelope é rompido já no Ciclo 191, conferindo $+244$ ciclos de folga operacional.

![Figura 7.3: Envelope SPC do canal cinemático Nc: Primeiro alarme mecânico no Ciclo 187 (+248 ciclos de folga).](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/nc_time_series_envelope_detection.png)

A Figura 7.3 exibe o primeiro sinal físico global de degradação registrado na frota: o rompimento cinemático no eixo N2 no Ciclo 187 (+248 ciclos de lead time).

![Figura 7.4: Envelope SPC do canal aerodinâmico BPR: Alarme no Ciclo 215 confirmando estrangulamento do núcleo.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/bpr_time_series_envelope_detection.png)

A Figura 7.4 retrata a divergência da razão de bypass ($BPR$) no Ciclo 215, corroborando a redistribuição anormal de fluxo aerodinâmico.

![Figura 7.5: Casco geométrico digital móvel (Rolling Convex Hull, W=15 ciclos) para rastreio de picos e vales.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/t50_geometric_envelope_detection.png)

A Figura 7.5 apresenta a envoltória geométrica móvel dos picos e vales locais para o canal $T_{50}$, evidenciando o alargamento da amplitude do casco à medida que a falha se propaga.

### Exemplo Teórico 7.1: Rastreamento Numérico do Motor #179
No motor `#179` (vida útil até o ciclo 435), os parâmetros de calibração saudável ($H=50$) para $T_{50}$ são:
$$\mu_e = -0,08^\circ\text{C}, \quad \sigma_e = 2,12^\circ\text{C} \quad \implies \quad \text{UCL} = +6,28^\circ\text{C}$$

* **Ciclo 173:** Resíduo instantâneo $e(173) = +6,42^\circ\text{C} > \text{UCL}$.  
  O contador passa para $k = 1$ (Estado Sob Suspeita). No Ciclo 174, $e(174) = +2,15^\circ\text{C} < \text{UCL}$. O contador zera imediatamente ($k = 0$). O evento é descartado como ruído aleatório de medição.
* **Ciclo 222:** Resíduo $e(222) = +7,26^\circ\text{C} > \text{UCL}$ ($k = 1$).  
  No Ciclo 223, $e(223) = +8,14^\circ\text{C} > \text{UCL}$ ($k = 2 \ge k_{crit}$).  
  O alarme industrial é formalmente consolidado. A turbina opera até o ciclo 435, propiciando um horizonte de intervenção programada de:
  $$\text{Lead Time} = 435 - 222 = \mathbf{+213\text{ ciclos}}$$

### Código Python 7.1: Detector de Envelope Shewhart com Trava de Persistência
```python
class EnvelopeAnomalyDetector:
    """Implementa corridor Shewhart SPC com persistência temporal k>=2."""

    def __init__(self, k_sigma: float = 3.0, min_consecutive: int = 2):
        self.k_sigma = k_sigma
        self.min_consecutive = min_consecutive
        self.ucl = None
        self.lcl = None

    def calibrate(self, healthy_residuals: np.ndarray):
        """Calcula limites de controle estatístico sobre resíduos saudáveis."""
        mu = np.mean(healthy_residuals)
        sigma = np.std(healthy_residuals)
        self.ucl = mu + self.k_sigma * sigma
        self.lcl = mu - self.k_sigma * sigma
        return self

    def score_trace(self, residuals: np.ndarray) -> tuple[np.ndarray, int | None]:
        """Avalia violações consecutivas e retorna índice do primeiro alarme."""
        violations = (residuals > self.ucl) | (residuals < self.lcl)
        alarm_idx = None
        consecutive = 0
        for i, viol in enumerate(violations):
            if viol:
                consecutive += 1
                if consecutive >= self.min_consecutive and alarm_idx is None:
                    alarm_idx = i
            else:
                consecutive = 0
        return violations, alarm_idx
```
"""
    return md

print("Loaded chapter 7 module.")
