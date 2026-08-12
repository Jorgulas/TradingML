"""Sincronizacao diaria local, pensada para ser chamada pelo Agendador de
Tarefas do Windows todas as manhas: puxa do GitHub o JSON de noticias que a
rotina cloud publicou durante a noite, ingere-o se existir, e corre SEMPRE o
pipeline diario a seguir -- mesmo que a parte de noticias falhe ou ainda nao
tenha chegado, run_daily.py continua (os booleanos ficam neutros nesse dia,
o que e' um degradar aceitavel, nao um bloqueio).

Uso (chamado pelo Agendador de Tarefas): py scripts/local_daily_sync.py
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run(cmd, **kwargs):
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=BASE_DIR, **kwargs)
    return result


def main():
    print(f"=== local_daily_sync {datetime.now().isoformat()} ===")

    pull = run(["git", "pull", "--ff-only", "origin", "master"])
    if pull.returncode != 0:
        print(f"AVISO: git pull falhou (exit {pull.returncode}) -- a continuar com o que ja esta local")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    news_file = BASE_DIR / "data" / "incoming" / f"{today}.json"
    if news_file.exists():
        result = run([PYTHON, "src/record_daily_features.py", "--file", str(news_file)])
        if result.returncode != 0:
            print(f"AVISO: ingestao de noticias falhou (exit {result.returncode}) -- a continuar sem elas hoje")
    else:
        print(f"AVISO: sem ficheiro de noticias para {today} ainda ({news_file}) -- a continuar sem elas")

    daily = run([PYTHON, "src/run_daily.py"])
    if daily.returncode != 0:
        print(f"ERRO: run_daily.py falhou (exit {daily.returncode})")
        sys.exit(1)

    print("=== local_daily_sync concluido ===")


if __name__ == "__main__":
    main()
