"""CLI validado para a tarefa diaria (agente Claude) escrever os booleanos de
noticias do dia. O LLM nunca toca em SQL: escreve um JSON, corre este script,
que valida estritamente e faz upsert atomico (tudo ou nada) numa unica
transacao SQLite. Em erro, mensagem especifica + exit code != 0, para o
agente poder corrigir e tentar de novo no mesmo run.

Uso:
    py src/record_daily_features.py --schema
    py src/record_daily_features.py --file data/incoming/2026-08-12.json
"""

import argparse
import difflib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from db import database


class ValidationError(Exception):
    pass


def _active_tickers(conn) -> set:
    rows = conn.execute("SELECT ticker FROM watchlist WHERE active = 1").fetchall()
    tickers = {r["ticker"] for r in rows}
    return tickers or set(config.TICKERS)


def print_schema(conn) -> None:
    tickers = sorted(_active_tickers(conn))
    example = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "assessments": {
            tickers[0]: {
                **{f: False for f in config.BOOLEAN_FEATURES},
                "notes": "breve justificacao opcional (o que leste, fonte, etc.)",
            }
        },
    }
    print("Watchlist ativa -- tem de aparecer EXATAMENTE uma vez cada em 'assessments', nem mais nem menos:")
    for t in tickers:
        print(f"  {t}")
    print()
    print("Campos obrigatorios por ticker (exatamente estes 5 booleanos + 'notes' opcional):")
    for f in config.BOOLEAN_FEATURES:
        print(f"  {f}  (true/false)")
    print()
    print("Formato do ficheiro JSON esperado (um so ficheiro para todos os tickers do dia):")
    print(json.dumps(example, indent=2, ensure_ascii=False))


def validate_payload(payload, active_tickers: set) -> dict:
    if not isinstance(payload, dict):
        raise ValidationError("payload de topo tem de ser um objeto JSON")

    date_str = payload.get("date")
    if not isinstance(date_str, str):
        raise ValidationError("campo 'date' em falta ou nao e string (esperado 'YYYY-MM-DD')")
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValidationError(f"'date'='{date_str}' nao e uma data valida no formato YYYY-MM-DD")

    today = datetime.now(timezone.utc).date()
    if parsed_date > today:
        raise ValidationError(f"'date'={date_str} esta no futuro (hoje e {today})")
    if (today - parsed_date) > timedelta(days=7):
        raise ValidationError(
            f"'date'={date_str} e mais de 7 dias no passado (hoje e {today}) -- confirma que nao e engano"
        )

    assessments = payload.get("assessments")
    if not isinstance(assessments, dict):
        raise ValidationError("campo 'assessments' em falta ou nao e um objeto")

    given_tickers = set(assessments.keys())
    unknown = given_tickers - active_tickers
    missing = active_tickers - given_tickers
    if unknown:
        hints = []
        for u in sorted(unknown):
            close = difflib.get_close_matches(u, list(active_tickers), n=1)
            hint = f" (querias dizer '{close[0]}'?)" if close else ""
            hints.append(f"'{u}'{hint}")
        raise ValidationError(f"tickers desconhecidos em 'assessments': {', '.join(hints)}")
    if missing:
        raise ValidationError(f"faltam tickers da watchlist ativa em 'assessments': {', '.join(sorted(missing))}")

    allowed_keys = set(config.BOOLEAN_FEATURES) | {"notes"}
    for ticker, values in assessments.items():
        if not isinstance(values, dict):
            raise ValidationError(f"assessments['{ticker}'] tem de ser um objeto")
        extra_keys = set(values.keys()) - allowed_keys
        if extra_keys:
            raise ValidationError(f"assessments['{ticker}'] tem campos desconhecidos: {', '.join(sorted(extra_keys))}")
        missing_keys = set(config.BOOLEAN_FEATURES) - set(values.keys())
        if missing_keys:
            raise ValidationError(f"assessments['{ticker}'] falta campos obrigatorios: {', '.join(sorted(missing_keys))}")
        for feat in config.BOOLEAN_FEATURES:
            if not isinstance(values[feat], bool):
                raise ValidationError(
                    f"assessments['{ticker}']['{feat}']={values[feat]!r} tem de ser true/false "
                    f"(booleano JSON), nao {type(values[feat]).__name__}"
                )
        notes = values.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise ValidationError(f"assessments['{ticker}']['notes'] tem de ser string ou omitido")

    return {"date": date_str, "assessments": assessments}


def ingest_payload(conn, clean: dict) -> None:
    """Upsert de todos os tickers do payload numa unica transacao (tudo ou
    nada): sqlite3 abre transacao implicita no primeiro INSERT/UPDATE e so
    fecha em commit()/rollback(), por isso o loop abaixo e' ja atomico."""
    now = datetime.now(timezone.utc).isoformat()
    date_str = clean["date"]
    cols = config.BOOLEAN_FEATURES
    col_list = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    update_clause = ", ".join(f"{c} = excluded.{c}" for c in cols)
    sql = (
        f"INSERT INTO news_features (ticker, date, {col_list}, notes, filled_by, filled_at) "
        f"VALUES (?, ?, {placeholders}, ?, 'llm_agent', ?) "
        f"ON CONFLICT(ticker, date) DO UPDATE SET {update_clause}, notes = excluded.notes, filled_at = excluded.filled_at"
    )
    try:
        for ticker, values in clean["assessments"].items():
            bool_values = [1 if values[f] else 0 for f in cols]
            conn.execute(sql, [ticker, date_str, *bool_values, values.get("notes"), now])
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def ingest_file(conn, path: Path) -> dict:
    if not path.exists():
        raise ValidationError(f"ficheiro nao encontrado: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"JSON invalido em {path}: {exc}")

    clean = validate_payload(payload, _active_tickers(conn))
    ingest_payload(conn, clean)
    return clean


def archive_file(path: Path) -> Path:
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.ARCHIVE_DIR / path.name
    path.replace(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", action="store_true", help="mostra o formato esperado e a watchlist ativa")
    parser.add_argument("--file", type=str, help="caminho para o JSON do dia a ingerir")
    args = parser.parse_args()

    conn = database.get_connection()

    if args.schema:
        print_schema(conn)
        conn.close()
        return

    if not args.file:
        parser.error("indica --schema ou --file <caminho>")

    path = Path(args.file)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        clean = ingest_file(conn, path)
    except ValidationError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        database.log_run(conn, today, "record_daily_features", "ERROR", str(exc))
        conn.close()
        sys.exit(1)

    for ticker in sorted(clean["assessments"].keys()):
        print(f"  {ticker}: OK")
    dest = archive_file(path)
    n = len(clean["assessments"])
    print(f"news_features atualizado para {n} tickers em {clean['date']}. Ficheiro arquivado em {dest}")
    database.log_run(conn, clean["date"], "record_daily_features", "OK", f"{n} tickers")
    conn.close()


if __name__ == "__main__":
    main()
