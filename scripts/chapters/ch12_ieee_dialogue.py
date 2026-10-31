"""Chapter 12: Diálogo Científico com o Artigo do IEEE CIM 2026 (The Good, the Bad, and the Ugly)."""

def get_chapter():
    md = r"""# Capítulo 12: Diálogo Científico com o Artigo do IEEE CIM 2026 (The Good, the Bad, and the Ugly)

Em publicação de destaque na *IEEE Computational Intelligence Magazine* (Vol. 21, No. 1, Fevereiro de 2026), Epifani, Falcetta e Roveri apresentaram um exame crítico seminal intitulado *"Long-Term Time Series Forecasting: The Good, the Bad, and the Ugly"* [2]. O artigo questiona três dogmas amplamente difundidos na literatura de inteligência artificial aplicada a séries temporais. 

Abaixo, o ecossistema PdM-CMaps estabelece um diálogo científico rigoroso com cada uma das três Questões de Pesquisa (*Research Questions* - RQs) formuladas pelos autores:

```
+---------------------------------------------------------------------------------------------------+
|               MATRIZ COMPARATIVA: DEEP LEARNING VS GBDTS EM SÉRIES TEMPORAIS INDUSTRIAIS          |
|                                                                                                   |
|  Critério de Avaliação       | Deep Learning / Transformers (PyTorch) | GBDTs (XGBoost / LightGBM)|
|  --------------------------- | -------------------------------------- | ------------------------- |
|  Acurácia R2 na Frota        | 0.6742 a 0.6801                        | 0.8121 a 0.8163 (SUPERIOR)|
|  Erro RMSE Hold-out          | 22.73 a 22.94 ciclos                   | 17.24 a 17.45 ciclos      |
|  Sensibilidade a Assimetria  | NASA Score: 16.420 a 17.890            | NASA Score: 3.892 a 4.015 |
|  Previsões Atrasadas (Risco) | 53.8% a 54.6% (PERIGOSO)               | 44.1% a 45.2% (SEGURO)    |
|  Tempo de Treinamento        | 21.9s a 57.8s                          | 0.28s a 0.70s (80x MAIS)  |
|  Exigência Computacional     | GPUs pesadas / Memória VRAM            | CPU Standard / Microcontrol|
|  Interpretabilidade Física   | Caixa-preta densa                      | Importância e SHAP exatos |
+---------------------------------------------------------------------------------------------------+
```

## 12.1 Resposta à RQ1: Realismo dos Datasets e a Armadilha da Deriva de Regime
* **A Crítica de Epifani et al. [2]:** Os autores alertam que a maioria dos benchmarks acadêmicos avalia modelos em séries univariadas artificiais ou conjuntos estáticos e homogêneos, omitindo a existência de variações abruptas de regime de operação que caracterizam a vida real.
* **A Evidência Experimental do PdM-CMaps:** Ao selecionar deliberadamente o subconjunto **FD004** (6 regimes de voo simultâneos variando de 0 a 42.000 pés com 2 modos de falha ativos), a plataforma opera sob o padrão máximo de severidade industrial da NASA. Como comprovado pela decomposição ANOVA do Capítulo 4, $99,45\%$ da oscilação dos sensores é exógena (atmosférica). Redes neurais que tentam ajustar séries brutas sem o desacoplamento prévio por K-Means colapsam frente à deriva não-estacionária, enquanto a abordagem física de 7 camadas preserva a integridade do prognóstico.

## 12.2 Resposta à RQ2: GBDTs vs Deep Learning e Transformers em Séries Tabulares
* **A Crítica de Epifani et al. [2]:** O artigo questiona a hegemonia presumida de arquiteturas neurais complexas (Transformers, LSTMs), demonstrando que modelos lineares simples ou árvores de decisão frequentemente igualam ou superam redes profundas em séries temporais estruturadas.
* **A Evidência Experimental do PdM-CMaps:** O benchmark com 14 modelos na frota hold-out confirma cabalmente a tese dos autores da IEEE:
  * O modelo **XGBoost** atingiu **$R^2 \approx 0,839$** com tempo de treinamento sub-segundo ($< 1$ segundo).
  * Em contraste, o **PyTorch Transformer** ($R^2 \approx 0,773$) e a **PyTorch LSTM** ($R^2 \approx 0,794$) exigiram tempos de processamento até $50\times$ maiores em CPU.
  * O teste não-paramétrico pareado de Wilcoxon em nível de motor com controle Benjamini-Hochberg ($H_5$) atesta a superioridade estatística do XGBoost sobre o Transformer com $p = 5,46 \times 10^{-5}$ e sobre a LSTM com $p = 1,54 \times 10^{-6}$ (Hodges-Lehmann $\Delta \approx 2,17$ ciclos de vantagem para o XGBoost).

Em séries temporais de telemetria industrial com forte estrutura tabular e descontinuidades físicas de regime, os algoritmos baseados em árvores particionam o espaço com hiperplanos ortogonais sem sofrer com desvanecimento de gradientes ou perturbações de escala.

## 12.3 Resposta à RQ3: A Primazia da Decisão Operacional sobre Métricas Estatísticas
* **A Crítica de Epifani et al. [2]:** Os autores denunciam que a ênfase exclusiva em métricas agregadas simétricas de erro (como MSE ou MAE) mascara a utilidade operacional prática das previsões para os tomadores de decisão humanos.
* **A Evidência Experimental do PdM-CMaps:** Como demonstrado nos Capítulos 7 e 10, a métrica isolada de $R^2$ pouco informaria ao operador de manutenção sobre a segurança do voo. 
  * O ecossistema avalia a função assimétrica oficial da NASA ($S_{NASA}$), penalizando severamente projeções otimistas atrasadas.
  * O verdadeiro valor de engenharia do sistema traduz-se no **Lead Time Operacional Útil**: a capacidade de acusar a falha com **$+213$ a $+248$ ciclos de antecedência** mantendo **zero falsos alarmes ($FAR = 0,0\%$)**, viabilizando o agendamento preventivo seguro de peças e equipes de hangar.

### Exemplo Teórico 12.1: Falha Catastrófica de Transformers sem Invariância Atmosférica
Considere uma camada de autoatenção escalada (*Scaled Dot-Product Attention* [9]):
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
Se as entradas contêm o salto bruto de altitude de centenas de graus Celsius, as magnitudes das consultas $\mathbf{Q}$ e chaves $\mathbf{K}$ tornam-se desproporcionalmente grandes. A função *softmax* satura nas regiões assintóticas com gradientes praticamente nulos ($\nabla \text{softmax} \approx 0$), bloqueando a propagação do erro e impedindo a rede de aprender o desgaste sutil de menos de $1\%$ da variância mecânica.
"""
    return md

print("Loaded chapter 12 module.")
