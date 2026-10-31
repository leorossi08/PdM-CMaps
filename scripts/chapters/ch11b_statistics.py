"""Chapter 11B: Validação Estatística de Hipóteses (P-Values e Bootstrap CI 95% na Frota Hold-out)."""

from book_tables import load_tables

def get_chapter():
    tables = load_tables()
    md = r"""# Capítulo 11B: Validação Estatística de Hipóteses (P-Values e Bootstrap CI 95% na Frota Hold-out)

Esta análise usa testes pareados a nível de motor, ajuste Benjamini-Hochberg e intervalos BCa por bootstrap de clusters ($B=2.000$), evitando tratar ciclos correlacionados como réplicas independentes.

## 11B.1 Matriz de Testes de Hipótese Científica e p-valores Extraídos
A tabela abaixo consolida os testes estatísticos formais aplicados aos dados de telemetria da frota:

""" + tables["hyp_md"] + r"""

### Análise e Interpretação Científica dos Resultados:
1. **$H_1$ (Dominância de Altitude e Mach na Variabilidade Térmica):** A decomposição de variância por regimes é uma descrição da partição analisada; os valores históricos ($99,45\%$, $F = 14.224,1$) foram calculados no conjunto global usado pelo relatório e não devem ser atribuídos ao motor hold-out #179 nem tratados como inferência independente por motor. A clusterização agora padroniza as configurações operacionais.
2. **$H_2$ (Divergência Térmica Real vs Ruído Amostral):** Os resultados históricos do teste de Welch ($p = 3,61 \times 10^{-14}$; $t=8,54$) foram removidos como evidência inferencial porque tratavam ciclos de um único motor como observações independentes. A análise atual agrega médias saudáveis e degradadas por motor e aplica teste pareado; consulte a tabela recalculada antes de afirmar significância.
3. **$H_{3a}$ e $H_{3b}$ (Estrutura de Autocorrelação de Wold):** Não rejeitar H0 no Ljung-Box não comprova ausência de autocorrelação. Tratar os resultados como diagnóstico limitado, considerando a dependência temporal e o poder do teste.
4. **$H_4$ (Superioridade sobre Linhas de Base):** A comparação agora usa uma diferença de erro médio por motor (uma observação por unidade) e Wilcoxon pareado. As conclusões de significância serão reportadas com valores recalculados e ajuste Benjamini–Hochberg; os p-valores históricos por ciclo não são evidência válida.
5. **$H_5$ (Comparação entre arquiteturas):** As diferenças de MAE são agregadas por motor antes do teste pareado, com ajuste Benjamini–Hochberg. Os p-valores históricos por ciclo foram retirados por pseudo-replicação; não sustentam superioridade inferencial.

## 11B.2 Intervalos de Confiança BCa por Cluster (Bootstrap, B=2.000, 95% CI)
Para quantificar a incerteza em nível de frota, aplica-se reamostragem clusterizada por motor e intervalo BCa ($B=2.000$).

""" + tables["boot_md"] + r"""

Nota-se que os intervalos de confiança de $95\%$ para o XGBoost ($R^2 \in [0,7619; 0,8630]$) e LightGBM ($R^2 \in [0,7568; 0,8532]$) não interceptam as bandas das arquiteturas neurais sequenciais (Transformer com $R^2 \in [0,5791; 0,7678]$ e LSTM com $R^2 \in [0,5826; 0,7722]$), confirmando a separação estatística de desempenho.

## 11B.3 Benchmark Completo na Frota Hold-out (14 Estimadores)
Abaixo resume-se o desempenho integral de todos os 14 modelos avaliados:

""" + tables["bench_md"] + r"""

### Exemplo Teórico 11B.1: Procedimento de Bootstrap Clusterizado por Unidade
Em dados de frotas industriais, a reamostragem bootstrap simples de linhas viola a independência das observações. Deve-se aplicar o **Cluster Bootstrap** em nível de motor:
Seja $\mathcal{U} = \{u_1, u_2, \dots, u_{62}\}$ o conjunto de identidades dos motores da frota hold-out.
A cada iteração $b \in \{1, \dots, 500\}$:
1. Sorteiam-se $62$ identidades de motores com reposição: $\mathcal{U}^{*(b)} = \{u_1^*, \dots, u_{62}^*\}$.
2. Concatena-se a totalidade dos ciclos associados aos motores sorteados, compondo o conjunto $\mathcal{D}^{*(b)}$.
3. Avalia-se a métrica de interesse $\theta^{*(b)} = R^2(\mathcal{D}^{*(b)})$.

O intervalo de confiança não-paramétrico de $95\%$ é obtido pelos percentis empíricos $2,5\%$ e $97,5\%$:
$$\text{IC}_{95\%} = \left[ \theta^*_{(0,025)}, \; \theta^*_{(0,975)} \right]$$

### Código Python 11B.1: Algoritmo de Bootstrap Clusterizado de Frota
```python
def compute_clustered_fleet_bootstrap(df_predictions: pd.DataFrame, n_bootstraps: int = 500, seed: int = 42):
    """Calcula intervalos de confiança de 95% via bootstrap clusterizado por turbina."""
    np.random.seed(seed)
    unique_engines = df_predictions["unit_id"].unique()
    n_engines = len(unique_engines)
    
    r2_samples, rmse_samples, mae_samples = [], [], []
    for _ in range(n_bootstraps):
        sampled_engines = np.random.choice(unique_engines, size=n_engines, replace=True)
        sample_df = pd.concat([df_predictions[df_predictions["unit_id"] == eng] for eng in sampled_engines], ignore_index=True)
        
        y_true = sample_df["y_true"].to_numpy()
        y_pred = sample_df["y_pred"].to_numpy()
        
        r2_samples.append(r2_score(y_true, y_pred))
        rmse_samples.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_samples.append(mean_absolute_error(y_true, y_pred))
        
    return {
        "r2_ci": (np.percentile(r2_samples, 2.5), np.percentile(r2_samples, 97.5)),
        "rmse_ci": (np.percentile(rmse_samples, 2.5), np.percentile(rmse_samples, 97.5)),
        "mae_ci": (np.percentile(mae_samples, 2.5), np.percentile(mae_samples, 97.5)),
    }
```
"""
    return md

print("Loaded chapter 11B module.")
