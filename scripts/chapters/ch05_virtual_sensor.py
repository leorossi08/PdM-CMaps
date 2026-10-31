"""Chapter 5: Sensoriamento Virtual Calibrado Multi-Sensor (T50, P30, BPR, Nc)."""

def get_chapter():
    md = r"""# Capítulo 5: Sensoriamento Virtual Calibrado Multi-Sensor (T50, P30, BPR, Nc)

## 5.1 O Conceito de Sensor Virtual como Gêmeo Digital Estatístico
Em sistemas industriais avançados, um **Sensor Virtual** (*soft sensor* ou sensor de inferência) consiste em um estimador matemático treinado para reproduzir o comportamento de um sensor físico com base em grandezas de contorno e controle [1].

Na plataforma desenvolvida, o Sensor Virtual assume dupla função:
1. **Gêmeo Digital de Normalidade:** É calibrado exclusivamente na **janela operacional saudável** do ativo ($t \in [1, H]$ ciclos, sendo adotada a convenção padrão de $H = 50$ ciclos estabelecida por Saxena et al. [3] e Pádua [1]).
2. **Gerador de Resíduos Puros de Degradação:** Quando o equipamento entra em degradação mecânica, o motor real perde rendimento isentrópico. No entanto, o Sensor Virtual — por desconhecer a falha — continua projetando a temperatura ou pressão que um motor *novo* exibiria sob aquelas mesmas condições de altitude e velocidade.

A divergência entre o sinal físico real medido e a estimativa do gêmeo digital define o **Resíduo de Degradação**:
$$e(t) = y(t) - \hat{y}(t)$$

```
+---------------------------------------------------------------------------------------------------+
|               FLUXO DE GERAÇÃO DE RESÍDUOS E GÊMEO DIGITAL ESTATÍSTICO DE LINHA DE BASE           |
|                                                                                                   |
|  Medição Real (y_t)   -----\                                                                      |
|                             \--> e_t = y_t - y_hat_t   ---------------------------------> [SPC]   |
|  Condições de Voo     -----> [ Sensor Virtual ]                                        (Envelope) |
|  (setting_1, 2, 3)           (Calibrado em H=50)                                                  |
|                                                                                                   |
|  Fase 1: Motor Saudável (Ciclos 1 a 180)  --> y_t ≈ y_hat_t  ==> e_t ~ N(0, σ²)   (Ruído Branco)  |
|  Fase 2: Motor Degradado (Ciclos 220 a 435) --> y_t >> y_hat_t ==> e_t > +3σ       (Dano Térmico)  |
+---------------------------------------------------------------------------------------------------+
```

## 5.2 Modelagem dos Quatro Canais Termodinâmicos Estratégicos
Embora a plataforma avalie todos os 21 canais de instrumentação (conforme documentado na Figura 5.1), quatro sensores cobrem as principais manifestações físicas de dano:
1. **`T50` (Sensor Virtual Térmico - Temperatura de Exaustão da LPT):** O estrangulamento do fluxo induz o controlador FADEC a injetar combustível adicional para sustentar o empuxo, elevando a temperatura de descarga da turbina em até $+20^\circ\text{C}$ no final de vida.
2. **`P30` (Sensor Virtual Pneumático - Pressão de Descarga do HPC):** Mede a pressão estática na saída do compressor de alta, constituindo o indicador mais precoce de perda de estanqueidade das palhetas (*tip clearance*).
3. **`BPR` (Sensor Virtual Aerodinâmico - Razão de Bypass):** A restrição de vazão no núcleo força parcela crescente de ar para o duto secundário externo.
4. **`Nc` (Sensor Virtual Cinemático - Rotação do Eixo de Alta Pressão N2):** Reflete a sobrecarga mecânica e elevação de atrito viscoso nos mancais e vedações de labirinto.

![Figura 5.1: Perfil de acurácia R2 do Sensor Virtual calibrado nos 21 canais físicos de instrumentação.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/sensor_r2_by_channel.png)

A Figura 5.1 ilustra a acurácia do Sensor Virtual em todos os canais. Observa-se que canais atmosféricos atingem $R^2 \approx 1,0$, enquanto variáveis com ruído dinâmico intrínseco (como entalpia de sangria) mantêm-se em patamares estáveis de modelagem.

![Figura 5.2: Envelope de Paridade de 45 graus para o Sensor Virtual T50 na fase de calibração saudável.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/t50_parity_envelope.png)

A Figura 5.2 apresenta o gráfico de paridade real versus previsto para o sensor térmico $T_{50}$ durante a fase saudável. A aderência rigorosa à bissetriz confirma que o estimador reproduz a assinatura térmica nominal sem viés.

![Figura 5.3: Envelope de Paridade de 45 graus para o Sensor Virtual P30 demonstrando dispersão centrada.](file:///home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide_fd004/figures/p30_parity_envelope.png)

A Figura 5.3 comprova a mesma simetria no canal pneumático $P_{30}$, assegurando ausência de desvios sistemáticos no início da operação da turbina.

## 5.3 Sensibilidade da Janela de Calibração Saudável: $H=50$ vs $H=100$ Ciclos
A escolha do horizonte $H$ define a sensibilidade do gêmeo digital:

| Sensor | Domínio Físico | Primeiro Alarme ($H=50$) | Lead Time ($H=50$) | Primeiro Alarme ($H=100$) | Lead Time ($H=100$) | Diagnóstico Industrial |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Nc`** | Cinemático | **Ciclo 187** | **+248 ciclos** | **Ciclo 211** | **+224 ciclos** | Primeira assinatura mecânica de atrito no eixo N2. |
| **`P30`** | Pneumático | **Ciclo 191** | **+244 ciclos** | **Ciclo 200** | **+235 ciclos** | Perda de pressurização no compressor (HPC). |
| **`BPR`** | Aerodinâmico | **Ciclo 215** | **+220 ciclos** | **Ciclo 219** | **+216 ciclos** | Alteração da repartição de vazão entre duto e núcleo. |
| **`T50`** | Térmico | **Ciclo 222** | **+213 ciclos** | **Ciclo 222** | **+213 ciclos** | **Convergência invariante no ciclo 222**: sobreaquecimento térmico. |

### Exemplo Teórico 5.1: Geração de Resíduo Térmico em Voo
Seja um ponto operacional no Ciclo 350 com $\text{setting\_1} = 35.000\text{ ft}$, $\text{setting\_2} = 0,80$, $\text{setting\_3} = 100^\circ$.
O Sensor Virtual, calibrado nos primeiros 50 ciclos, prevê para essa condição um motor novo com temperatura nominal:
$$\hat{y}(350) = 1125,40^\circ\text{C}$$
Contudo, devido ao desgaste acumulado no compressor, o termopar $T_{50}$ registra:
$$y(350) = 1139,80^\circ\text{C}$$
O resíduo puro de degradação resultante é:
$$e(350) = 1139,80 - 1125,40 = +14,40^\circ\text{C}$$
Como o desvio padrão histórico saudável é $\sigma_e = 2,10^\circ\text{C}$, o resíduo padronizado atinge:
$$Z = \frac{14,40}{2,10} = +6,86\sigma$$
Ultrapassando o limiar de três sigmas ($+3\sigma$) com margem estatística avassaladora.

### Código Python 5.1: Calibração e Inferência do Sensor Virtual de Linha de Base
```python
class VirtualSensorBaseline:
    '''Gêmeo digital estatístico calibrado estritamente na fase saudável.'''

    def __init__(self, healthy_limit: int = 50):
        self.healthy_limit = healthy_limit
        self.models = {}

    def fit(self, train_df: pd.DataFrame, target_col: str, feature_cols: list[str]):
        '''Ajusta regressores Ridge regularizados por regime de voo.'''
        healthy_df = train_df[train_df["cycle"] <= self.healthy_limit]
        for regime_id, group in healthy_df.groupby("regime"):
            model = RidgeCV(alphas=np.logspace(-3, 3, 10))
            model.fit(group[feature_cols], group[target_col])
            self.models[regime_id] = model
        return self

    def predict_residuals(self, test_df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> np.ndarray:
        '''Gera resíduos de degradação: e(t) = y(t) - y_hat(t).'''
        residuals = np.zeros(len(test_df))
        for regime_id, group in test_df.groupby("regime"):
            idx = group.index
            model = self.models.get(regime_id)
            if model is not None:
                preds = model.predict(group[feature_cols])
                residuals[idx] = group[target_col].to_numpy() - preds
        return residuals
```
"""
    return md

print("Loaded chapter 5 module.")
