"""Full textual and code content for all chapters of the treatise."""

def get_front_matter():
    md = """# Tratado Científico de Engenharia Preditiva: Da Telemetria Bruta ao Cockpit de Decisão

## Fundamentação Metodológica, Modelagem Físico-Estatística e Diálogo com o Estado da Arte em Long-Term Time Series Forecasting (IEEE 2026)

> **Plataforma de Referência:** [PdM-CMaps](file:///home/leorossi/PdM-CMaps/PdM-CMaps)  
> **Fundamentação Conceitual:** Monografia de Juliano Pádua (*"Inteligência geoespacial - do ambiente à decisão"*) [1]  
> **Referência do Estado da Arte:** Epifani, Falcetta & Roveri, *"Long-Term Time Series Forecasting: The Good, the Bad, and the Ugly"*, *IEEE Computational Intelligence Magazine*, Vol. 21, No. 1, Fevereiro de 2026 [2]  
> **Base Experimental:** NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS FD001 e FD004 com 6 regimes operacionais severos de voo) [3]  
> **Ambiente Operacional:** Cockpit Industrial Autocontido Plotly ([dashboard.html](file:///home/leorossi/PdM-CMaps/PdM-CMaps/dashboard.html))

---

### Resumo Executivo e Epígrafe
A manutenção preditiva (*Predictive Maintenance* - PdM) em sistemas aeronáuticos e de turbomáquinas de alta complexidade representa um dos desafios mais expressivos da engenharia contemporânea. A necessidade de antecipar falhas estruturais catastróficas, evitar paradas não-programadas de frotas e maximizar a vida útil de componentes mecânicos requer uma esteira analítica contínua e integrada.

Este tratado científico apresenta a fundamentação teórica, o desenvolvimento algorítmico, a validação experimental e o diálogo com o estado da arte do ecossistema **PdM-CMaps**. Fundamentada na arquitetura conceitual de **Juliano Pádua** (*"Do Sinal à Decisão"*) [1] e respondendo rigorosamente às três questões de pesquisa de **Epifani, Falcetta e Roveri (IEEE CIM 2026)** [2], a obra demonstra que a integração de princípios termodinâmicos (Atmosfera ISA e Ciclo Brayton), desacoplamento de regimes operacionais, filtragem causal EWMA, corredores estocásticos de Shewhart com travas de persistência e convoluções determinísticas MiniRocket supera categoricamente arquiteturas cegas de Deep Learning e Transformers em dados tabulares complexos de séries temporais industriais.

---

## Sumário Geral

- [Capítulo 1: Introdução e Arquitetura "Do Sinal à Decisão"](#capítulo-1-introdução-e-arquitetura-do-sinal-à-decisão)
- [Capítulo 2: Os Dados Brutos (Raw Telemetry) do NASA C-MAPSS](#capítulo-2-os-dados-brutos-raw-telemetry-do-nasa-c-mapss)
- [Capítulo 3: Engenharia de Dados, Limpeza e Particionamento Disjunto de Frota](#capítulo-3-engenharia-de-dados-limpeza-e-particionamento-disjunto-de-frota)
- [Capítulo 4: Desacoplamento Físico-Atmosférico de Regimes de Voo & O Paradoxo do Pooled R²](#capítulo-4-desacoplamento-físico-atmosférico-de-regimes-de-voo--o-paradoxo-do-pooled-r)
- [Capítulo 5: Sensoriamento Virtual Calibrado Multi-Sensor (T50, P30, BPR, Nc)](#capítulo-5-sensoriamento-virtual-calibrado-multi-sensor-t50-p30-bpr-nc)
- [Capítulo 6: Condicionamento e Filtragem de Sinais (EWMA Causal e Supressão de Ruído)](#capítulo-6-condicionamento-e-filtragem-de-sinais-ewma-causal-e-supressão-de-ruído)
- [Capítulo 7: Detecção de Anomalias por Envelope ("Embrulho") e Casco Geométrico](#capítulo-7-detecção-de-anomalias-por-envelope-embrulho-e-casco-geométrico)
- [Capítulo 8: Fundamentação Científica: Prova de Degradação Real vs Ruído de Medição](#capítulo-8-fundamentação-científica-prova-de-degradação-real-vs-ruído-de-medição)
- [Capítulo 9: Enriquecimento Convolucional: MiniRocket e o Indicador Monotônico de Saúde](#capítulo-9-enriquecimento-convolucional-minirocket-e-o-indicador-monotônico-de-saúde)
- [Capítulo 10: Modelagem e Benchmark Multi-Algoritmo (Reconstrução Térmica de T50)](#capítulo-10-modelagem-e-benchmark-multi-algoritmo-reconstrução-térmica-de-t50)
- [Capítulo 11: Prognóstico de Vida Útil Restante (RUL): Formulação e Fenômenos de Paridade](#capítulo-11-prognóstico-de-vida-útil-restante-rul-formulação-e-fenômenos-de-paridade)
- [Capítulo 11B: Validação Estatística de Hipóteses (P-Values e Bootstrap CI 95% na Frota Hold-out)](#capítulo-11b-validação-estatística-de-hipóteses-p-values-e-bootstrap-ci-95-na-frota-hold-out)
- [Capítulo 12: Diálogo Científico com o Artigo do IEEE CIM 2026 (The Good, the Bad, and the Ugly)](#capítulo-12-diálogo-científico-com-o-artigo-do-ieee-cim-2026-the-good-the-bad-and-the-ugly)
- [Capítulo 13: O Cockpit Industrial Plotly e a Matriz Operacional de Tomada de Decisão (SOP)](#capítulo-13-o-cockpit-industrial-plotly-e-a-matriz-operacional-de-tomada-de-decisão-sop)
- [Capítulo 14: Catálogo Técnico e Síntese da Evidência Empírica](#capítulo-14-catálogo-técnico-e-síntese-da-evidência-empírica)
- [Capítulo 15: Conclusões, Implantação em Borda (Edge Computing) e Linhas Futuras](#capítulo-15-conclusões-implantação-em-borda-edge-computing-e-linhas-futuras)
- [Referências Bibliográficas Completas](#referências-bibliográficas-completas)

---
"""
    return md

print("Loaded front matter module.")
