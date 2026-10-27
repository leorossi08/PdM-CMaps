#!/usr/bin/env python3
"""Ponto de Entrada Geral para o Pipeline de Manutencao Preditiva e Dashboard C-MAPSS.

Este script orquestra a execucao completa do projeto de TCC:
1. Limpeza de diretorios de resultados legados (garantindo reprodutibilidade experimental).
2. Execucao do pipeline analitico completo (desacoplamento de regimes, sensores virtuais,
   deteccao por envelope SPC, extracao convolucional MiniRocket e predicao de RUL).
3. Geracao do dashboard interativo autossuficiente em Plotly (offline, sem dependencia de CDN).
4. Inicializacao do servidor HTTP local para inspecao industrial 'do sinal a decisao'.

Padrao de formatacao: Rigor academico, sem emojis, compatibilidade com execucao desconectada.
"""

from __future__ import annotations

import argparse
import http.server
import os
from pathlib import Path
import shutil
import socketserver
import subprocess
import sys
import webbrowser


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"


def print_banner(title: str) -> None:
    """Imprime cabecalho estruturado sem emojis."""
    separator = "=" * 78
    print(f"\n{separator}", flush=True)
    print(f"[EXECUCAO GERAL] {title.upper()}", flush=True)
    print(f"{separator}", flush=True)


def clean_results_directories(output_dir: Path, dashboard_dir: Path) -> None:
    """Remove diretorios de resultados e dashboard anteriores para garantir reproducibilidade."""
    print("[FASE 0/3] Limpeza de resultados anteriores...", flush=True)
    targets = [
        output_dir,
        dashboard_dir,
        ROOT_DIR / "results-maitenance_guide",
        ROOT_DIR / "dashboard.html",
    ]
    for target in targets:
        if target.is_dir():
            print(f"  -> Removendo diretorio legado: {target.name}", flush=True)
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            print(f"  -> Removendo arquivo legado: {target.name}", flush=True)
            target.unlink(missing_ok=True)
            
    # Assegura a criacao limpa dos diretorios de destino
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    print("  -> Limpeza e preparacao de diretorios concluidas com sucesso.", flush=True)


def run_maintenance_pipeline(
    subset: str,
    output_dir: Path,
    seed: int,
    include_minirocket: bool,
    include_lstm: bool,
    healthy_cycle_limit: int,
    unit_id: int = 179,
) -> None:
    """Executa o pipeline analitico completo de Manutencao Preditiva."""
    print_banner(f"Fase 1/3: Execucao do Pipeline PdM C-MAPSS ({subset})")
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_maintenance_guide.py"),
        "--subset", subset,
        "--output-dir", str(output_dir),
        "--seed", str(seed),
        "--healthy-cycle-limit", str(healthy_cycle_limit),
        "--unit-id", str(unit_id),
    ]
    if include_minirocket:
        cmd.append("--include-minirocket")
    if include_lstm:
        cmd.append("--include-lstm")

    print(f"[FASE 1/3] Comando: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Falha na execucao do pipeline PdM (codigo de retorno: {proc.returncode})")
    print("[FASE 1/3] Pipeline de manutencao preditiva finalizado com exito.", flush=True)


def build_dashboard(data_dir: Path, output_dir: Path, unit_id: int) -> Path:
    """Gera o dashboard interativo Plotly em formato HTML standalone offline."""
    print_banner("Fase 2/3: Compilacao do Dashboard Interativo Offline")
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_factory_dashboard.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--unit-id", str(unit_id),
    ]
    print(f"[FASE 2/3] Comando: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Falha na geracao do dashboard (codigo de retorno: {proc.returncode})")
    dashboard_html = ROOT_DIR / "dashboard.html"
    print(f"[FASE 2/3] Dashboard compilado com sucesso em: {dashboard_html}", flush=True)
    return dashboard_html


def serve_dashboard(port: int, open_browser: bool) -> None:
    """Inicializa o servidor HTTP local para disponibilizar o dashboard."""
    print_banner(f"Fase 3/3: Servidor Local do Dashboard Industrial (Porta {port})")
    url = f"http://localhost:{port}/dashboard.html"
    alt_url = f"http://127.0.0.1:{port}/dashboard.html"
    
    print(f"[FASE 3/3] Servidor HTTP local inicializado em: {url}", flush=True)
    print(f"[FASE 3/3] Endereco alternativo: {alt_url}", flush=True)
    print(f"[FASE 3/3] O arquivo dashboard.html e 100% autocontido e offline.", flush=True)
    print("[FASE 3/3] Pressione Ctrl+C para encerrar o servidor HTTP a qualquer momento.\n", flush=True)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"[AVISO] Nao foi possivel abrir o navegador automaticamente: {exc}", flush=True)

    # Executa SimpleHTTPRequestHandler no diretorio raiz
    os.chdir(str(ROOT_DIR))
    
    # Permite reuso de porta imediatamente caso tenha sido fechada recentemente
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] Servidor HTTP encerrado pelo usuario (Ctrl+C).", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ponto de entrada geral para reproducao do TCC em Manutencao Preditiva (NASA C-MAPSS)."
    )
    parser.add_argument(
        "--subset",
        choices=("FD001", "FD002", "FD003", "FD004"),
        default="FD004",
        help="Subconjunto do C-MAPSS a ser processado (padrao: FD004).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente pseudoaleatoria para particionamento e reprodutibilidade (padrao: 42).",
    )
    parser.add_argument(
        "--healthy-cycle-limit",
        type=int,
        default=50,
        help="Limite de ciclos saudaveis para calibracao do sensor virtual (padrao: 50).",
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        default=179,
        help="Identificador do motor de teste para analise pontual no dashboard (padrao: 179).",
    )
    parser.add_argument(
        "--no-minirocket",
        dest="include_minirocket",
        action="store_false",
        help="Desativa o extrator de caracteristicas convolucionais MiniRocket.",
    )
    parser.set_defaults(include_minirocket=True)
    parser.add_argument(
        "--include-lstm",
        action="store_true",
        help="Inclui o treinamento do modelo de Deep Learning LSTM sequencial (mais lento).",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Nao remove resultados anteriores antes de rodar.",
    )
    parser.set_defaults(clean=True)
    parser.add_argument(
        "--no-serve",
        dest="serve",
        action="store_false",
        help="Gera todos os arquivos mas nao inicia o servidor HTTP local.",
    )
    parser.set_defaults(serve=True)
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Porta TCP do servidor HTTP local (padrao: 8050).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Abre automaticamente o navegador padrao no endereco do dashboard.",
    )

    args = parser.parse_args()

    output_dir = ROOT_DIR / "results" / f"maintenance_guide_{args.subset.lower()}"
    dashboard_dir = ROOT_DIR / "results" / "dashboard"

    print_banner(f"Inicializando Fluxo Geral: C-MAPSS {args.subset}")
    print(f"Configuracao experimental:", flush=True)
    print(f"  - Subconjunto: {args.subset}", flush=True)
    print(f"  - Semente: {args.seed}", flush=True)
    print(f"  - MiniRocket: {'Ativado' if args.include_minirocket else 'Desativado'}", flush=True)
    print(f"  - Deep Learning LSTM: {'Ativado' if args.include_lstm else 'Desativado'}", flush=True)
    print(f"  - Limpeza previa: {'Ativada' if args.clean else 'Desativada'}", flush=True)
    print(f"  - Servidor HTTP: {'Ativado na porta ' + str(args.port) if args.serve else 'Desativado'}", flush=True)

    # 1. Limpeza previa se solicitada
    if args.clean:
        clean_results_directories(output_dir, dashboard_dir)

    # 2. Execucao do pipeline analitico
    run_maintenance_pipeline(
        subset=args.subset,
        output_dir=output_dir,
        seed=args.seed,
        include_minirocket=args.include_minirocket,
        include_lstm=args.include_lstm,
        healthy_cycle_limit=args.healthy_cycle_limit,
        unit_id=args.unit_id,
    )

    # 3. Compilacao do dashboard
    build_dashboard(
        data_dir=output_dir,
        output_dir=dashboard_dir,
        unit_id=args.unit_id,
    )

    # 4. Servidor HTTP
    if args.serve:
        serve_dashboard(port=args.port, open_browser=args.open_browser)
    else:
        print_banner("Processamento Concluido com Exito")
        print(f"[INFO] Dashboard disponivel em: {ROOT_DIR / 'dashboard.html'}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
