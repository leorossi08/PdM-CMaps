"""Chapter 13: O Cockpit Industrial Plotly e a Matriz Operacional de Tomada de Decisão (SOP)."""

def get_chapter():
    md = r"""# Capítulo 13: O Cockpit Industrial Plotly e a Matriz Operacional de Tomada de Decisão (SOP)

## 13.1 A Conclusão da Esteira: Da Predição à Ação Prática de Fábrica
De acordo com o princípio fundamental enunciado na monografia de Juliano Pádua (*"Do Sinal à Decisão"*) [1], nenhum algoritmo preditivo atinge seu objetivo final se permanecer confinado a um repositório de código ou a métricas abstratas de laboratório. O ciclo de inteligência de engenharia apenas se completa quando o prognóstico é traduzido em **Procedimentos Operacionais Padronizados** (*Standard Operating Procedures* - SOP) executáveis pelas equipes de chão de fábrica e manutenção de pista.

Para essa finalidade, a Camada 7 implementa o **Cockpit Industrial Plotly** ([dashboard.html](file:///home/leorossi/PdM-CMaps/PdM-CMaps/dashboard.html)), uma aplicação 100% autocontida, offline e interativa construída sem dependência de servidores externos ou conexão com a internet.

```
+---------------------------------------------------------------------------------------------------+
|               FLUXOGRAMA OPERACIONAL DO PROCEDIMENTO PADRÃO (SOP EM 4 ESTÁGIOS)                   |
|                                                                                                   |
|  [Telemetria Contínua]                                                                            |
|          |                                                                                        |
|          V                                                                                        |
|  Envelope Rompido?  ----( NÃO )----> [ NÍVEL 0: NORMALIDADE OPERACIONAL ]                         |
|          |                            Status: Verde | Liberação Irrestrita de Voo                 |
|       ( SIM )                                                                                     |
|          V                                                                                        |
|  Persistência k >= 2? --( NÃO )----> [ Spike Espúrio Rejeitado ] (Retorna a Nível 0)              |
|          |                                                                                        |
|       ( SIM )                                                                                     |
|          V                                                                                        |
|  RUL Estimado?                                                                                    |
|    |                                                                                              |
|    +---> RUL > 50 ciclos   ========> [ NÍVEL 1: ALERTA PREDITIVO INICIAL ]                        |
|    |                                  Status: Amarelo | Monitoramento Diário e Análise de Óleo    |
|    |                                                                                              |
|    +---> 20 <= RUL <= 50   ========> [ NÍVEL 2: AGENDAMENTO PREVENTIVO PRIORITÁRIO ]              |
|    |                                  Status: Laranja | Reserva de Peças e Janela de Hangar       |
|    |                                                                                              |
|    +---> RUL < 20 ciclos   ========> [ NÍVEL 3: PARADA MANDATÓRIA IMEDIATA ]                      |
|                                       Status: Vermelho | Remoção Compulsória da Turbina da Asa    |
+---------------------------------------------------------------------------------------------------+
```

## 13.2 A Matriz Operacional de Decisão em Quatro Níveis
O motor de regras decisórias categoriza cada turbina da frota em quatro estágios hierárquicos:

1. **Nível 0: Normalidade Operacional (Banda Verde):**  
   * **Critério:** Nenhum sensor em violação de persistência e $\text{RUL} > 80$ ciclos.
   * **Ação:** Voo irrestrito; rotina padrão de inspeção de pré-voo.
2. **Nível 1: Alerta Preditivo Inicial (Banda Amarela):**  
   * **Critério:** Primeiro alarme confirmado por persistência ($k \ge 2$) em ao menos um sensor (tipicamente $N_c$ ou $P_{30}$), com $\text{RUL} > 50$ ciclos.
   * **Ação:** Alocação da aeronave em rotas de baixa severidade climática; intensificação da análise espectrométrica de óleo lubrificante.
3. **Nível 2: Agendamento Preventivo Prioritário (Banda Laranja):**  
   * **Critério:** Múltiplos sensores rompem o envelope ou $\text{RUL}$ estimado situa-se entre $20$ e $50$ ciclos.
   * **Ação:** Reserva compulsória de módulos sobressalentes de compressor e turbina no almoxarifado central; bloqueio de janela programada de hangar para o próximo pernoite.
4. **Nível 3: Parada Mandatória Imediata (Banda Vermelha):**  
   * **Critério:** $\text{RUL} < 20$ ciclos ou rompimento térmico crítico no canal $T_{50}$ com desvio superior a $+15^\circ\text{C}$.
   * **Ação:** Interdição de voo comercial imediata; desmontagem do motor da asa para inspeção boroscópica e revisão completa.

### Exemplo Teórico 13.1: Auditoria Decisória do Motor #179
Avaliando a trajetória histórica do motor hold-out `#179` (vida total de 435 ciclos):
* **Ciclo 50:** Normalidade plena ($\text{RUL} = 125$ sat., resíduos $e \in \pm 0,5\sigma$). **Decisão: Nível 0 (Verde)**.
* **Ciclo 187:** O eixo $N_c$ rompe o envelope por $k=2$ ciclos consecutivos. **Decisão: Nível 1 (Amarelo)**. O sistema antecipa a falha com $+248$ ciclos de folga.
* **Ciclo 222:** O canal térmico $T_{50}$ cruza o limite estocástico $+3\sigma$. **Decisão: Nível 1 mantido com prioridade reforçada**.
* **Ciclo 385:** O estimador XGBoost indica $\text{RUL} = 48$ ciclos. **Decisão: Transição para Nível 2 (Laranja)**. Emissão de ordem de serviço para aquisição de palhetas de reposição e agendamento de hangar.
* **Ciclo 418:** O RUL estimado atinge $17$ ciclos. **Decisão: Transição mandatória para Nível 3 (Vermelho)**. A turbina é retirada da asa com 17 ciclos de segurança antes do colapso mecânico final.

### Código Python 13.1: Motor de Decisão Operacional do SOP de Fábrica
```python
class SOPDecisionEngine:
    """Motor de regras do Procedimento Operacional Padrão (SOP industrial)."""

    def evaluate_engine_status(self, rul_estimate: float, active_alarms: list[str], max_residual_t50: float) -> dict:
        """Determina o nível de ação, cor do indicador e diretriz técnica."""
        if rul_estimate < 20.0 or max_residual_t50 > 15.0:
            return {
                "level": 3,
                "status": "PARADA MANDATÓRIA IMEDIATA",
                "color": "#EF4444",  # Vermelho
                "action": "Interdição de voo. Remoção imediata da asa para revisão geral."
            }
        elif rul_estimate <= 50.0 or len(active_alarms) >= 2:
            return {
                "level": 2,
                "status": "AGENDAMENTO PREVENTIVO PRIORITÁRIO",
                "color": "#F97316",  # Laranja
                "action": "Reservar peças no almoxarifado e agendar hangar em até 10 dias."
            }
        elif len(active_alarms) >= 1 or rul_estimate <= 80.0:
            return {
                "level": 1,
                "status": "ALERTA PREDITIVO INICIAL",
                "color": "#EAB308",  # Amarelo
                "action": "Monitoramento diário intensificado e amostragem de óleo lubrificante."
            }
        else:
            return {
                "level": 0,
                "status": "NORMALIDADE OPERACIONAL",
                "color": "#10B981",  # Verde
                "action": "Operação irrestrita de voo. Manutenção preventiva de rotina."
            }
```
"""
    return md

print("Loaded chapter 13 module.")
