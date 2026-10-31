"""Chapter 2: Os Dados Brutos (Raw Telemetry) do NASA C-MAPSS."""

def get_chapter():
    md = r"""# Capítulo 2: Os Dados Brutos (Raw Telemetry) do NASA C-MAPSS

## 2.1 A Plataforma de Simulação C-MAPSS
O simulador **C-MAPSS** (*Commercial Modular Aero-Propulsion System Simulation*), desenvolvido pela divisão de propulsão e diagnósticos do NASA Glenn Research Center, estabeleceu-se como a plataforma de referência internacional para a modelagem fenomenológica e degradação progressiva em turbofans de alta razão de derivação da classe de 90.000 lbf de empuxo [3]. 

O motor simulado implementa uma formulação termodinâmica não-linear completa baseada em componentes integrados: ventilador (*fan*), compressor de baixa pressão (*LPC*), compressor de alta pressão (*HPC*), combustor anular, turbina de alta pressão (*HPT*), turbina de baixa pressão (*LPT*) e bocal de exaustão convergente-divergente.

O conjunto público de dados é particionado em quatro subconjuntos de avaliação (FD001 a FD004), cada qual impondo diferentes regimes de severidade operacional:

| Subconjunto | Motores Treino | Motores Teste | Regimes Operacionais | Modos de Falha Ativos | Severidade Industrial |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **FD001** | 100 | 100 | 1 (Nível do Mar constante) | 1 (Degradação no HPC) | Baixa (Benchmark Didático) |
| **FD002** | 260 | 259 | 6 (Envelope Completo de Voo) | 1 (Degradação no HPC) | Alta (Multi-Regime) |
| **FD003** | 100 | 100 | 1 (Nível do Mar constante) | 2 (Degradação no HPC e Fan) | Média (Multi-Falha) |
| **FD004** | 249 | 248 | 6 (Envelope Completo de Voo) | 2 (Degradação no HPC e Fan) | **Máxima (Desafio Industrial Real)** |

Enquanto a literatura acadêmica frequentemente restringe seus experimentos ao subconjunto simplificado FD001, este tratado foca fundamentalmente no **FD004**, no qual a sobreposição de múltiplos regimes de voo e modos de falha simultâneos exige desacoplamento rigoroso entre dinâmicas exógenas de voo e a mecânica de dano interno [2], [3].

## 2.2 As Três Variáveis de Configuração de Voo (Operational Settings)
A cada ciclo de inspeção (voo completo englobando decolagem, cruzeiro e pouso), o C-MAPSS registra três parâmetros operacionais fundamentais:
1. **`setting_1` (Altitude de Voo - $h$):** Varia continuamente entre $0$ pés (nível do mar) e $42.000$ pés (teto estratosférico de cruzeiro).
2. **`setting_2` (Número de Mach - $M$):** Varia entre $0,00$ (estático em solo) e $0,84$ (velocidade máxima de cruzeiro transônico).
3. **`setting_3` (Throttle Resolver Angle - $TRA$):** Ângulo de comando da manete de combustível em graus ($20^\circ$ a $100^\circ$), expressando o empuxo demandado pelo piloto ou piloto automático.

Pelas equações da atmosfera padrão ISA (*International Standard Atmosphere*):
$$T_{amb}(h) = T_0 - L \cdot h, \quad \text{para } h \le 11.000\text{ m}$$
$$P_{amb}(h) = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g \cdot M_{ar}}{R \cdot L}}$$
onde $T_0 = 288,15\text{ K}$, $P_0 = 101,325\text{ kPa}$, $L = 0,0065\text{ K/m}$, $g = 9,80665\text{ m/s}^2$, $M_{ar} = 0,0289644\text{ kg/mol}$ e $R = 8,31447\text{ J/(mol}\cdot\text{K)}$.

```
+---------------------------------------------------------------------------------------------------+
|               ESTAÇÕES TERMODINÂMICAS E POSICIONAMENTO DE SENSORES NO MOTOR TURBOFAN              |
|                                                                                                   |
|     Estação 2: Entrada do Fan [s1: T2, s5: P2]                                                    |
|        |                                                                                          |
|     [ FAN ] ===> Fluxo Secundário (Bypass) [s6: P15, s15: BPR] ==================> Bocal Frio     |
|        |                                                                                          |
|     Estação 2.4: Entrada do Compressor de Alta Pressão (HPC) [s2: T24]                            |
|        |                                                                                          |
|     [ LPC ] ---> [ HPC ] [s8: Nf, s9: Nc]                                                         |
|                    |                                                                              |
|                 Estação 3: Descarga do HPC / Difusor [s3: T30, s7: P30, s11: Ps30]                |
|                    |                                                                              |
|                 [ COMBUSTOR ] [s12: phi, s16: farB]                                               |
|                    |                                                                              |
|                 [ HPT ] ---> [ LPT ] [s17: htBleed, s20: W31, s21: W32]                           |
|                                |                                                                  |
|                             Estação 5: Descarga da Turbina de Baixa Pressão [s4: T50]             |
|                                |                                                                  |
|                             [ BOCAL DE EXAUSTÃO ] [s10: epr]                                      |
+---------------------------------------------------------------------------------------------------+
```

## 2.3 O Dicionário Físico dos 21 Sensores de Telemetria
Os 21 sensores monitorados cobrem o balanço de massa, momento e energia ao longo das estações termodinâmicas do motor:

| Canal | Símbolo | Grandeza Física | Unidade | Estação Física / Componente |
| :---: | :---: | :--- | :---: | :--- |
| **s1** | $T_2$ | Temperatura Total na Entrada do Fan | $^\circ\text{R}$ | Admissão de Ar Frontal (Estação 2) |
| **s2** | $T_{24}$ | Temperatura Total na Saída do LPC | $^\circ\text{R}$ | Saída LPC / Entrada HPC (Estação 2.4) |
| **s3** | $T_{30}$ | Temperatura Total na Saída do HPC | $^\circ\text{R}$ | Descarga do Compressor de Alta (Estação 3) |
| **s4** | $T_{50}$ | Temperatura Total na Saída da LPT | $^\circ\text{R}$ | Descarga da Turbina de Baixa (Estação 5) |
| **s5** | $P_2$ | Pressão Total na Entrada do Fan | psia | Entrada do Ventilador (Estação 2) |
| **s6** | $P_{15}$ | Pressão Total no Duto de Bypass | psia | Canal de Derivação Secundária (Estação 15) |
| **s7** | $P_{30}$ | Pressão Total na Saída do HPC | psia | Descarga do Compressor de Alta (Estação 3) |
| **s8** | $N_f$ | Velocidade Física do Eixo de Baixa (N1) | rpm | Eixo do Fan e Turbina de Baixa Pressão |
| **s9** | $N_c$ | Velocidade Física do Eixo de Alta (N2) | rpm | Eixo do HPC e Turbina de Alta Pressão |
| **s10** | $epr$ | Razão de Pressão do Motor ($P_{50}/P_2$) | adim. | Balanço Geral de Expansão |
| **s11** | $Ps_{30}$ | Pressão Estática na Saída do HPC | psia | Difusor da Câmara de Combustão |
| **s12** | $\phi$ | Razão Combustível/Ar Corrigida | pps/psia | Sistema de Controle de Combustível FADEC |
| **s13** | $NR_f$ | Velocidade Corrigida do Fan | rpm | Rotação Normalizada por $\sqrt{\theta}$ |
| **s14** | $NR_c$ | Velocidade Corrigida do Compressor de Alta | rpm | Rotação Normalizada por $\sqrt{\theta}$ |
| **s15** | $BPR$ | Razão de Derivação (*Bypass Ratio*) | adim. | Razão entre Vazão Bypass e Vazão Núcleo |
| **s16** | $farB$ | Razão Combustível/Ar no Queimador | adim. | Câmara de Combustão Primária |
| **s17** | $htBleed$ | Entalpia de Sangria de Ar | Btu/lbm | Duto de Sangria do Compressor |
| **s18** | $Nf\_dmd$ | Rotação Demandada do Fan | rpm | Comando de Referência FADEC |
| **s19** | $PCNfR\_dmd$ | Demanda Corrigida de Rotação do Fan | rpm | Comando Normalizado FADEC |
| **s20** | $W_{31}$ | Sangria de Ar para Refrigeração da HPT | lbm/s | Sistema Térmico de Palhetas de Alta |
| **s21** | $W_{32}$ | Sangria de Ar para Refrigeração da LPT | lbm/s | Sistema Térmico de Palhetas de Baixa |

### Exemplo Teórico 2.1: Cálculo das Condições de Entrada ISA em Voo de Cruzeiro
Seja um voo de cruzeiro em altitude de $h = 35.000\text{ ft}$ ($10.668\text{ m}$) e Mach $M = 0,80$.
Aplicando a equação ISA:
$$T_{amb} = 288,15 - 0,0065 \cdot 10668 = 218,81\text{ K} \quad (-54,34^\circ\text{C})$$
$$P_{amb} = 101,325 \cdot \left(1 - \frac{0,0065 \cdot 10668}{288,15}\right)^{5,25588} = 23,84\text{ kPa} \quad (3,458\text{ psia})$$

Pela conservação da entalpia estagnada na admissão (Estação 2):
$$T_2 = T_{amb} \left(1 + \frac{\gamma - 1}{2} M^2\right) = 218,81 \cdot (1 + 0,2 \cdot 0,8^2) = 246,82\text{ K} \quad (444,28^\circ\text{R})$$
$$P_2 = P_{amb} \left(1 + \frac{\gamma - 1}{2} M^2\right)^{\frac{\gamma}{\gamma - 1}} = 3,458 \cdot (1,128)^{3,5} = 5,27\text{ psia}$$

Em contrapartida, ao nível do mar ($h=0$, $M=0$), $T_2 = 518,67^\circ\text{R}$ e $P_2 = 14,696\text{ psia}$. 
Portanto, a temperatura de admissão varia mais de $74^\circ\text{R}$ ($41^\circ\text{C}$) e a pressão cai quase $65\%$ puramente devido à manobra de altitude, sem qualquer correlação com dano mecânico.

### Código Python 2.1: Ingestão de Telemetria e Validação de Schema
```python
def load_and_validate_cmapss_telemetry(file_path: str) -> pd.DataFrame:
    """Carrega arquivo bruto de telemetria C-MAPSS e atribui schema padronizado."""
    columns = [
        "unit_id", "cycle", "setting_1", "setting_2", "setting_3"
    ] + [f"s{i}" for i in range(1, 22)]
    
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=columns)
    assert not df.empty, "Arquivo de telemetria vazio ou corrompido"
    assert df["unit_id"].dtype in (int, np.int64), "Identificador de unidade deve ser inteiro"
    assert len(df.columns) == 26, f"Esperadas 26 colunas, obtidas {len(df.columns)}"
    return df
```
"""
    return md

print("Loaded chapter 2 module.")
