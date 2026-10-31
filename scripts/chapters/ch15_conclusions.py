"""Chapter 15: Conclusões, Implantação em Borda (Edge Computing) e Linhas Futuras."""

def get_chapter():
    md = r"""# Capítulo 15: Conclusões, Implantação em Borda (Edge Computing) e Linhas Futuras

## 15.1 Síntese das Contribuições Científicas e de Engenharia
O desenvolvimento do ecossistema **PdM-CMaps** consolida uma resposta metodológica e tecnológica completa aos principais gargalos da manutenção preditiva de turbofans aeronáuticos. Estruturado sobre o paradigma **"Do Sinal à Decisão"** de **Juliano Pádua** [1] e estabelecendo diálogo empírico com a literatura contemporânea em Long-Term Time Series Forecasting (**Epifani, Falcetta e Roveri, IEEE CIM 2026** [2]), o tratado apresenta seis contribuições nucleares:

1. **Desacoplamento de Regimes:** O K-Means padronizado separa entre $99,19\%$ e $99,98\%$ da variância global nos 6 regimes ISA, com baseline OOS $R^2 = 0,9943$ para $T_{50}$, comprovando que o $R^2$ bruto elevado decorre de altitude/Mach e não de capacidade prognóstica precoce.
2. **Sensoriamento Virtual Exógeno Canônico:** A calibração de modelos exógenos (`setting_1..3`, $T_2, P_2, P_{15}, \text{epr}$) sobre a janela saudável ($H=50$) elimina a contaminação por co-degradação (*fault spillover*), gerando resíduos puros de degradação nos canais $T_{50}, P_{30}, N_c$ e $BPR$.
3. **Condicionamento de Sinal Causal e Ótimo:** A parametrização do filtro EWMA ($\alpha=0,35$) assegurou atenuação do jitter instrumental com latência de apenas $2,85$ ciclos e custo de memória $O(1)$.
4. **Envelopes Estocásticos e Limites Empíricos:** A calibração empírica de limites de controle EWMA e restrição à fase online ($cycle > H$) suprime alarmes precoces espúrios, garantindo lead time médio operacional efetivo na frota.
5. **Enriquecimento Convolucional Determinístico:** O emprego dos 84 kernels invariantes a offset DC do MiniRocket proporcionou atributos de Proporção de Valores Positivos (PPV) informativos para projeção linearizada de RUL ($R^2 = 0,7198$) sem necessidade de backpropagation.
6. **Vindicação Empírica dos Modelos GBDTs:** Em consonância direta com Epifani et al. [2], comprovou-se que estimadores baseados em árvores (XGBoost com $R^2=0,8156$, RMSE $=17,32$ ciclos) superam estatisticamente ($p < 0,05$ verificado via Wilcoxon pareado por motor e FDR Benjamini-Hochberg) arquiteturas profundas de Deep Learning na tarefa de prognóstico tabular de telemetria.

## 15.2 Viabilidade para Computação de Borda (*Edge Computing*)
Os requisitos de certificação aeronáutica (DO-178C) impõem restrições determinísticas severas sobre o software embarcado nos computadores de controle do motor (*Full Authority Digital Engine Control* - FADEC) e nos sistemas de gerenciamento de saúde (*Health and Usage Monitoring Systems* - HUMS) [3]:
* **Latência de Inferência:** O pipeline composto por filtro EWMA, inferência de K-Means, resíduo Ridge e previsão XGBoost executa em **menos de $0,15$ milissegundos por ciclo de telemetria** em processadores de uso geral.
* **Pegada de Memória:** Como a árvore de decisão do XGBoost e os regressores Ridge operam com tabelas estáticas de limiares compiláveis diretamente para código ANSI C puro (sem dependência de interpretadores Python ou frameworks pesados de tensores como PyTorch/TensorFlow), o consumo de memória RAM embarcada é inferior a $500\text{ KB}$.
* **Determinismo Temporal:** Ao contrário de redes neurais com loops dinâmicos de autoatenção ou camadas recorrentes que variam com o tamanho da janela histórica, as árvores de decisão executam em tempo constante $O(profundidade)$, garantindo o cumprimento estrito de janelas de tempo real rígido (*hard real-time*).

## 15.3 Linhas de Pesquisa Futuras
Como desdobramentos promissores desta investigação, destacam-se:
1. **Redes Neurais Informadas pela Física (*Physics-Informed Neural Networks* - PINNs):** Incorporação das equações diferenciais de transferência de calor e balanço de massa do Ciclo Brayton diretamente na função de perda de modelos híbridos.
2. **Aprendizado Federado de Frotas (*Federated Fleet Learning*):** Treinamento colaborativo e contínuo de estimadores preditivos entre diferentes companhias aéreas sem o compartilhamento centralizado de dados sensíveis ou proprietários de telemetria de voo.
3. **Adaptação Online a Manutenções Parciais (*Continual Learning*):** Ajuste dinâmico dos limites de envelope após intervenções pontuais de manutenção (como troca isolada de lâminas de turbina ou lavagem de compressor em linha), atualizando a linha de base saudável sem exigir recalibração completa do modelo.
"""
    return md

print("Loaded chapter 15 module.")
