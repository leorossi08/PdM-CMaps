# PdM-CMaps: Plataforma de Manutenção Preditiva & Inteligência Analítica

Implementação em Python modular e orientada a scripts para análise de dados de degradação de turbofans (**NASA C-MAPSS**), integrando os conceitos e a taxonomia da monografia e apresentação de **Juliano Pádua** (*"Inteligência geoespacial - do ambiente à decisão"*).

A arquitetura do projeto cobre a esteira completa **"Do Sinal à Decisão"**: ingestão de telemetria bruta, desacoplamento de regimes de voo, comparação de filtragem de sinal (EWMA e MiniRocket), modelagem com os principais algoritmos de Machine Learning (incluindo CatBoost, LightGBM, XGBoost, Random Forest, MLP e MiniRocket), sensoriamento virtual calibrado e **detecção de anomalias por Envelope ("Embrulho")**, marcando o **Primeiro Alarme**, a **Quebra do Produto** e o **Lead Time** operacional.

---

## 1. Ambiente e Instalação

O projeto requer **Python 3.10+** (validado com Python 3.14). Recomenda-se utilizar um ambiente virtual:

```bash
# Acessar a pasta do repositório
cd PdM-CMaps

# Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências completas
pip install -r requirements.txt
```

### Principais Dependências Instaladas
- **Machine Learning / GBDTs**: `catboost`, `xgboost`, `lightgbm`, `scikit-learn`
- **Deep Learning & Convoluções**: `torch` (usado nativamente no `MiniRocket` e `LSTM`)
- **Manipulação de Dados e Visualização**: `pandas`, `numpy`, `scipy`, `matplotlib`
- **Testes**: `pytest`

---

## 2. Visão dos Scripts de Execução

O repositório disponibiliza o orquestrador geral `run.py` na raiz e os módulos especializados em `scripts/`:

| Script | Finalidade | Principais Saídas |
| :--- | :--- | :--- |
| [`run.py`](run.py) | **Ponto de entrada geral**: Limpa resultados legados, executa o pipeline preditivo completo, compila o dashboard Plotly offline e inicializa o servidor local. | `results/`<br>`dashboard.html`<br>Servidor HTTP local |
| [`scripts/run_maintenance_guide.py`](scripts/run_maintenance_guide.py) | Esteira analítica: desacoplamento de regimes, sensor virtual, filtros de sinal, MiniRocket, envelopes de anomalia e benchmark de RUL. | `figures/*.png`<br>`tables/*.csv` |
| [`scripts/build_factory_dashboard.py`](scripts/build_factory_dashboard.py) | Compilador do dashboard interativo em Plotly (100% offline, autocontido, sinais crus, envelopes SPC e métricas). | `results/dashboard/index.html`<br>`dashboard.html` |
| [`scripts/run_r2_investigation.py`](scripts/run_r2_investigation.py) | Investigação estrutural de $R^2$, decomposição de variância, curvas de aprendizado, permutation importance e grouped cross-validation entre subsets. | `r2_investigation/figures/`<br>`r2_investigation/tables/` |

---

## 3. Como Executar (Comando Unico e Saida Verbose)

Para executar a esteira completa com limpeza automática, treinamento dos modelos (incluindo MiniRocket, CatBoost, XGBoost, LightGBM, Random Forest), compilação do dashboard e inicialização do servidor HTTP:

```bash
# Execucao geral completa no subset FD004 (padrao industrial multi-regime)
.venv/bin/python run.py
```

O comando realiza automaticamente:
1. Limpeza de diretórios anteriores em `results/` e do arquivo `dashboard.html`.
2. Execução das 7 etapas analíticas com logs detalhados no console.
3. Compilação do dashboard interativo offline de 4.94 MB.
4. Inicialização do servidor HTTP na porta 8050, acessível em:
   - `http://localhost:8050/dashboard.html`
   - `http://127.0.0.1:8050/dashboard.html`

### Execução em outros subconjuntos ou sem servidor HTTP:

```bash
# Executar no FD001 sem subir servidor HTTP local
.venv/bin/python run.py --subset FD001 --no-serve

# Executar especificando porta customizada para o servidor
.venv/bin/python run.py --port 8080
```

#### Exemplo do que é exibido no console (Status Verbose):

```text
=======================================================================
[maintenance] STARTING PdM PIPELINE: Subset=FD001 | Seed=42 | Healthy limit=50 cycles
=======================================================================

[maintenance] [STEP 1/7] Loading FD001 data and partitioning disjoint engine holdout...
  -> Total rows: 20631 | Engines: 100 (Train: 75, Test: 25)

[maintenance] [STEP 2/7] Decoupling operating flight regimes (1 clusters)...
  -> T50 variance decomposition: Between regimes = 0.0%, Within regime = 100.0%

[maintenance] [STEP 3/7] Fitting T50 Virtual Sensor & evaluating fleet SPC failure detection...
  -> Virtual Sensor T50: Fleet detection rate = 100.0% | Mean lead time = 35.8 cycles

[maintenance] [STEP 4/7] Generating signal filtering & MiniRocket features for Engine 81...
  -> Generated EWMA comparison (Slide 17) and MiniRocket PPV responses (Slides 19 & 21)

[maintenance] [STEP 5/7] Running Envelope ('Embrulho') Anomaly Detection on Engine 81...
  -> Time-series envelope: Alarm at cycle 143 | Breakage at cycle 240 | Lead time: 97 cycles
  -> Geometric hull envelope: Alarm at cycle 197 | Lead time: 43 cycles

[maintenance] [STEP 6/7] Profiling full sensor fleet across channels and operating regimes...
  [channels] [01/21] T2         -> Full Lifecycle R² = n/a (constant), RMSE = 0.0000
  [channels] [02/21] T24        -> Full Lifecycle R² = 0.3969, RMSE = 0.3886
  [channels] [03/21] T30        -> Full Lifecycle R² = 0.3251, RMSE = 5.0295
  [channels] [04/21] T50        -> Full Lifecycle R² = 0.4621, RMSE = 6.6225
  ...
  [channels] [13/21] NRf        -> Full Lifecycle R² = 0.5670, RMSE = 0.0485
  [channels] Complete. Saved sensor metrics to sensor_channel_metrics.csv

[maintenance] [STEP 7/7] Benchmarking RUL models on held-out engines (CatBoost, LightGBM, XGBoost, etc.)...
  [benchmark] [1/8] Fitting 'Ridge' on 15749 samples...
  [benchmark] -> 'Ridge' done in 0.01s | R²: 0.7991 | RMSE: 18.74 | MAE: 15.19
  [benchmark] [2/8] Fitting 'Random Forest' on 15749 samples...
  [benchmark] -> 'Random Forest' done in 1.10s | R²: 0.8733 | RMSE: 14.88 | MAE: 9.96
  [benchmark] [3/8] Fitting 'Extra Trees' on 15749 samples...
  [benchmark] -> 'Extra Trees' done in 0.27s | R²: 0.8767 | RMSE: 14.68 | MAE: 9.96
  [benchmark] [4/8] Fitting 'HistGradientBoosting' on 15749 samples...
  [benchmark] -> 'HistGradientBoosting' done in 0.23s | R²: 0.8760 | RMSE: 14.72 | MAE: 9.89
  [benchmark] [5/8] Fitting 'XGBoost' on 15749 samples...
  [benchmark] -> 'XGBoost' done in 0.20s | R²: 0.8756 | RMSE: 14.75 | MAE: 9.89
  [benchmark] [6/8] Fitting 'CatBoost' on 15749 samples...
  [benchmark] -> 'CatBoost' done in 0.55s | R²: 0.8755 | RMSE: 14.75 | MAE: 9.93
  [benchmark] [7/8] Fitting 'LightGBM' on 15749 samples...
  [benchmark] -> 'LightGBM' done in 0.53s | R²: 0.8758 | RMSE: 14.73 | MAE: 9.89
  [benchmark] [8/8] Fitting 'MLP' on 15749 samples...
  [benchmark] -> 'MLP' done in 2.66s | R²: 0.8827 | RMSE: 14.32 | MAE: 9.88

  [maintenance] Fitting MiniRocket + RidgeCV (1000 causal kernels, seq_len=15)...
  -> MiniRocket RUL complete: R² = 0.8409, fit time = 7.67s

=======================================================================
[maintenance] PIPELINE EXECUTION COMPLETE! Summary of generated tables:
=======================================================================
  - Table: regime_variance              (1 rows)
  - Table: virtual_sensor_regimes       (1 rows)
  - Table: anomaly_summary              (1 rows)
  - Table: rul_holdout                  (10 rows)
  - Table: rul_grouped_cv               (8 rows)
  - Table: sensor_channel_metrics       (21 rows)
  - Table: rul_minirocket               (1 rows)

[maintenance] Tables written to: /home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide/tables
[maintenance] Figures written to: /home/leorossi/PdM-CMaps/PdM-CMaps/results/maintenance_guide/figures
```

---

### B. Execução em Múltiplos Regimes de Voo: Dataset FD004

O dataset FD004 possui 6 regimes de voo e dois modos de falha (HPC e fan). Para executá-lo com desacoplamento automático via K-Means e sensoriamento virtual:

```bash
.venv/bin/python scripts/run_maintenance_guide.py \
    --subset FD004 \
    --output-dir results/maintenance_guide_fd004 \
    --healthy-cycle-limit 50 \
    --include-minirocket
```

> [!NOTE]
> **Interpretação dos $R^2$ em FD004 (Efeito de Escala vs Overfitting)**:
> Em datasets multi-regime, a variação de altitude (0 a 42.000 ft) e Mach responde por **99,4% da variância total** dos sensores térmicos ($\text{Var}_{total} \approx 14.278^\circ\text{C}^2$).
> O sensor virtual capta esses regimes através dos parâmetros de voo (`setting_1, setting_2, setting_3`).
> Por isso, o **$R^2$ Agregado (*Pooled*)** atinge $\approx 0.9965$ no ciclo de vida completo.
> **Isto NÃO é overfitting**:
> - A avaliação é feita em **62 motores mantidos estritamente em teste** (hold-out disjunto nunca visto no treino).
> - O erro de teste ($\text{RMSE} = 7.02^\circ\text{C}$) é idêntico ao erro de treino ($\text{RMSE} \approx 6.8^\circ\text{C}$).
> - **Dentro de cada regime individual (*Within-Regime*)**, o $R^2$ do sensor virtual na janela saudável fica na faixa realista de **$0.41$ a $0.56$** (conforme registrado em `healthy_t50_by_regime.csv`).
> - Modelos de predição de vida útil restante (**RUL**) alcançam $R^2$ consistente entre **0.77 e 0.82** (XGBoost $0.8156$, LightGBM $0.8121$, Random Forest $0.8104$, CatBoost $0.7882$, MiniRocket $0.7788$).

---

### C. Execução Opcional com Redes Neurais Recorrentes (PyTorch LSTM)

Para fins de benchmark contra redes sequenciais pesadas:

```bash
.venv/bin/python scripts/run_maintenance_guide.py \
    --subset FD001 \
    --output-dir results/maintenance_guide_lstm \
    --include-lstm
```

---

### D. Execução da Investigação Estrutural de $R^2$

Para comparar os 4 subconjuntos da NASA (FD001 a FD004) e avaliar o impacto de regimes operacionais e limites de corte de RUL:

```bash
# Diagnóstico baseline rápido
.venv/bin/python scripts/run_r2_investigation.py \
    --output-dir results/r2_investigation

# Diagnóstico estendido (curvas de aprendizado, permutation importance e capacity sweep)
.venv/bin/python scripts/run_r2_investigation.py \
    --output-dir results/r2_investigation_extended \
    --extended
```

---

### E. Painel Industrial Interativo (Plotly Dashboard: Do Sinal à Decisão)

Para compilar e visualizar o cockpit interativo em Plotly (100% offline, contendo gráficos de valores reais vs previstos sem média, envelope dinâmico SPC, casco geométrico, MiniRocket, paridade de 45° e matriz de decisão industrial):

```bash
# Gerar o dashboard standalone (results/dashboard/index.html e dashboard.html na raiz)
.venv/bin/python scripts/build_factory_dashboard.py \
    --data-dir results/maintenance_guide_fd004 \
    --unit-id 179

# Ou gerar e servir localmente na porta 8050
.venv/bin/python scripts/build_factory_dashboard.py \
    --data-dir results/maintenance_guide_fd004 \
    --unit-id 179 \
    --serve --port 8050
```

O dashboard fica acessível no navegador em:
- **Servidor local**: `http://localhost:8050/dashboard.html`
- **Arquivo local direto**: abra `dashboard.html` no navegador

---

## 4. Parâmetros e Argumentos dos Scripts

### Flags de `scripts/run_maintenance_guide.py`:
- `--subset`: Subconjunto C-MAPSS a executar: `FD001`, `FD002`, `FD003` ou `FD004` (padrão: `FD004`).
- `--output-dir`: Diretório base de saída onde serão criadas as pastas `figures/` e `tables/` (padrão: `results/maintenance_guide`).
- `--healthy-cycle-limit`: Limite de ciclos saudáveis para calibração do sensor virtual e linha divisória no gráfico (padrão: `50`).
- `--include-minirocket`: Ativa a extração convolucional causal MiniRocket + RidgeCV para predição de RUL.
- `--include-lstm`: Ativa o treinamento sequencial PyTorch LSTM.
- `--seed`: Semente de reprodutibilidade para particionamento e modelos (padrão: `42`).

---

## 5. Arquivos e Visualizações Geradas

Cada execução gera um pacote estruturado em `figures/` e `tables/`:

### Figuras em `figures/`:
1. **`t50_time_series_envelope_detection.png`**: Detecção de anomalia por Envelope de Séries Temporais (estilo Power BI $\pm 3\sigma$), destacando o **Primeiro Alarme** (ciclo 143), a **Quebra do Produto** (ciclo 240) e o **Lead Time** (97 ciclos de antecedência).
2. **`t50_geometric_envelope_detection.png`**: Envelope geométrico contínuo de picos e vales (*Upper/Lower Hull*) acusando quebra de teto térmico e modulação de amplitude mecânica (alarme no ciclo 197).
3. **`t50_failure_detection.png`**: Trajetória real vs sensor virtual de $T_{50}$ e pontuação estatística EWMA $|Z| \ge 3.0\sigma$ com região de calibração saudável e online monitoring.
4. **`signal_filtering_ewma_comparison.png`**: Sinal cru versus suavizado por EWMA ($\alpha \in \{0.10, 0.35, 0.70\}$) espelhando o Slide 17 da apresentação de Pádua.
5. **`minirocket_feature_filtering_padua.png`**: Respostas dos kernels causais $L=9$, limiar de bias $b$ e ativação monotônica da métrica PPV ao longo dos ciclos operacionais (Slides 19 e 21).
6. **`rul_model_r2_comparison.png`**: Gráfico de barras comparando os modelos aprovados no critério de alta fidelidade ($R^2 \ge 0.75$), com destaque para o melhor modelo.
7. **`rul_best_model_parity_envelope.png`**: Gráfico de paridade Real vs Predito do melhor modelo com corredor sombreado de tolerância ("Embrulho de Paridade").
8. **`rul_minirocket_parity_envelope.png`**: Gráfico de paridade Real vs Predito do MiniRocket + RidgeCV.
9. **`t50_parity_envelope.png`**: Paridade Real vs Predito do sensor virtual $T_{50}$.
10. **`healthy_t50_by_regime.png`**: Acurácia do sensor virtual segmentada por regime de voo.

### Tabelas em `tables/`:
- `rul_holdout_benchmark.csv`: Tabela com métricas completas ($R^2$, RMSE, MAE, NASA Score, tempo de treino) para todos os modelos tabulares e convolucionais.
- `envelope_summary.csv`: Resumo do alarme do envelope temporal, ciclo de falha e lead time.
- `geometric_envelope_summary.csv`: Resumo da modulação geométrica e ultrapassagem de teto físico.
- `fleet_anomalies.csv`: Métricas de detecção por motor em toda a frota de teste.
- `sensor_channel_metrics.csv`: Performance de reconstrução para todos os 21 sensores.

---

## 6. Limpeza e Reexecução

Para limpar todas as pastas de resultados e rodar uma nova bateria de testes limpa do zero:

```bash
# Apagar todos os diretórios gerados em results
rm -rf results/*

# Executar novamente com o pipeline completo
.venv/bin/python scripts/run_maintenance_guide.py \
    --subset FD001 \
    --output-dir results/maintenance_guide \
    --healthy-cycle-limit 50 \
    --include-minirocket
```

---

## 7. Suíte de Testes Automatizados

O repositório conta com **15 testes unitários e de integração** automatizados, cobrindo validação de esquema, prevenção de vazamento entre motores, propriedades dos kernels do MiniRocket, causalidade e detectores de envelope:

```bash
.venv/bin/python -m pytest tests/ -v
```

Resultado esperado:
```text
============================== 15 passed in 3.28s ==============================
```

Para a apresentação executiva e aprofundamento teórico dos resultados, consulte o documento `resumo.md` (notas locais, não versionado).
