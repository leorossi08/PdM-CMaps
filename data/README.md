# NASA C-MAPSS Turbofan Degradation Dataset

Este diretório contém os dados brutos e metadados do conjunto aerotermodinâmico **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)**.

## 1. Proveniência e Citação
- **Autores:** Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
- **Publicação:** *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. IEEE International Conference on Prognostics and Health Management.
- **Origem:** NASA Ames Prognostics Data Repository.

## 2. Estrutura dos Arquivos
Os dados estão organizados nos arquivos de texto em formato tabular delimitado por espaços:

| Arquivo | Descrição | Motores Treino | Motores Teste | Condições de Voo | Modos de Falha |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `train_FD001.txt` / `test_FD001.txt` | Subconjunto Monorregime | 100 | 100 | 1 (Nível do Mar) | 1 (HPC Degradação) |
| `train_FD002.txt` / `test_FD002.txt` | Subconjunto Multirregime | 260 | 259 | 6 Envelopes | 1 (HPC Degradação) |
| `train_FD003.txt` / `test_FD003.txt` | Subconjunto Monorregime | 100 | 100 | 1 (Nível do Mar) | 2 (HPC + Fan) |
| `train_FD004.txt` / `test_FD004.txt` | **Subconjunto Complexo Principal** | 249 | 248 | 6 Envelopes | 2 (HPC + Fan) |
| `RUL_FD001.txt` ... `RUL_FD004.txt` | Ground-truth de RUL no final da trajetória de teste | - | - | - | - |

## 3. Descrição das Colunas
1. `unit_id`: Identificador único do motor (turbofan).
2. `cycle`: Ciclo de voo operacional (tempo discreto contínuo por voo).
3. `setting_1`: Altitude (0 a 42.000 pés).
4. `setting_2`: Número de Mach de voo (0.00 a 0.84).
5. `setting_3`: Ângulo da Manete de Potência / Throttle Resolver Angle (TRA: 60° a 100°).
6. `sensor_1` a `sensor_21`: 21 canais de sensores de telemetria (temperaturas totais, pressões totais e estáticas, rotações de eixos de baixa e alta pressão, fluxo de combustível, razão de desvio de ar / bypass ratio).

> [!NOTE]
> No conjunto **FD004**, o foco deste trabalho, os 6 regimes operacionais mascaram a degradação mecânica. Nosso framework realiza o desacoplamento físico antes de qualquer estimativa de saúde ou RUL.
