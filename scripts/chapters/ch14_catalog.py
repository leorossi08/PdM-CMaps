"""Chapter 14: Catálogo Técnico e Síntese da Evidência Empírica."""

def get_chapter():
    md = r"""# Capítulo 14: Catálogo Técnico e Síntese da Evidência Empírica

Este capítulo consolida a matriz de rastreabilidade integral entre os fenômenos físicos investigados, as camadas da esteira algorítmica, as evidências gráficas geradas e as decisões operacionais de engenharia:

## 14.1 Matriz de Rastreabilidade Físico-Algorítmica

| Capítulo | Camada da Esteira | Fenômeno Físico / Algorítmico | Evidência Gráfica | Teste Estatístico Associado | Decisão Operacional |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Cap. 4** | Camada 2: Desacoplamento | Decomposição OOS por regime após ajuste padronizado | **Figura 4.1 e 4.2** (`healthy_t50_by_regime.png`) | Baseline com médias aprendidas no treino | Clusterização de regimes prévia à modelagem preditiva |
| **Cap. 5** | Camada 3: Sensor Virtual | Gêmeo digital estatístico de linha de base calibrado em $H=50$ | **Figura 5.1 a 5.3** (`sensor_r2_by_channel.png`) | Resíduos centrados em zero ($\mu_e = -0,08^\circ\text{C}$) | Geração de resíduos puros de degradação estrutural |
| **Cap. 6** | Camada 1: Condicionamento | Supressão de $78,8\%$ de jitter térmico com $\alpha=0,35$ | **Figura 6.1 a 6.5** (`signal_filtering_ewma_comparison.png`) | Atenuação analítica de variância $\frac{\alpha}{2-\alpha}$ | Estabilização de leitura sem atraso de fase excessivo |
| **Cap. 7** | Camada 4: Envelopes | Eliminação de 100% dos falsos alarmes com persistência $k=2$ | **Figura 7.1 a 7.5** (`t50_time_series_envelope_detection.png`) | Controle Estatístico Shewhart $\pm 3\sigma$ | Emissão de alarme preditivo com $+213$ ciclos de antecedência |
| **Cap. 8** | Fundamentação Científica | Estudo de caso da trajetória de resíduos e análise de autocorrelação | **Figura 8.1 a 8.4** (`thesis_multisensor_physical_cascade.png`) | Estatísticas descritivas; inferência por motor | Evidência mecanística sem tratar ciclos como réplicas independentes |
| **Cap. 9** | Camada 5: MiniRocket | Trajetória monotônica de integridade via 84 kernels de soma zero | **Figura 9.1 e 9.2** (`minirocket_feature_filtering_padua.png`) | Invariância a offset DC de altitude | Extração de Health Index determinístico sem backpropagation |
| **Cap. 10** | Camada 6: Modelagem | Liderança de GBDTs (XGBoost com $R^2=0,8163$, tempo de $0,70$s) | **Figura 10.1** (`rul_model_r2_comparison.png`) | Pontuação Assimétrica da NASA ($S=3.892$) | Seleção de estimadores com menor taxa de previsões atrasadas |
| **Cap. 11** | Camada 6: RUL | Aderência estrita à bissetriz no regime de desgaste ativo | **Figura 11.1 e 11.2** (`rul_best_model_parity_envelope.png`) | Alvo linear por partes com saturação em $125$ | Projeção suave de vida restante com filtro de mediana |
| **Cap. 11B**| Validação de Hipóteses | Validação de 20 hipóteses formais com intervalos de confiança | **Tabelas 11B.1 a 11B.3** (Benchmark & Hipóteses) | Reamostragem Bootstrap ($B=500$ frotas) | Demonstração inequívoca da robustez na frota hold-out |
| **Cap. 12** | Diálogo IEEE CIM 2026 | Resposta formal às 3 RQs de Epifani, Falcetta e Roveri | **Diagrama 12.1** (Matriz Comparativa Deep vs GBDT) | Testes pareados de postos de Wilcoxon | Desmistificação de redes profundas em séries tabulares |
| **Cap. 13** | Camada 7: Cockpit SOP | Matriz de decisão em 4 níveis executável em chão de fábrica | **Dashboard Autocontido** (`dashboard.html`) | Motor de regras condicionais hierárquicas | Transição controlada: Alerta $\to$ Agendamento $\to$ Parada |

Esta síntese comprova que cada visualização gráfica, formulação matemática e teste de hipótese integra harmoniosamente o ecossistema "Do Sinal à Decisão", conferindo rigor científico e valor prático industrial à obra.
"""
    return md

print("Loaded chapter 14 module.")
