import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import record_daily_features as rdf


def _valid_payload(date="2026-08-12"):
    return {
        "date": date,
        "assessments": {
            t: {f: False for f in config.BOOLEAN_FEATURES} for t in config.TICKERS
        },
    }


def test_valid_payload_passes(conn):
    clean = rdf.validate_payload(_valid_payload(), rdf._active_tickers(conn))
    assert clean["date"] == "2026-08-12"
    assert set(clean["assessments"].keys()) == set(config.TICKERS)


def test_unknown_ticker_rejected_with_suggestion(conn):
    payload = _valid_payload()
    payload["assessments"]["NVDIA"] = payload["assessments"].pop("NVDA")
    with pytest.raises(rdf.ValidationError, match="NVDIA.*NVDA"):
        rdf.validate_payload(payload, rdf._active_tickers(conn))


def test_missing_ticker_rejected(conn):
    payload = _valid_payload()
    del payload["assessments"]["JNJ"]
    with pytest.raises(rdf.ValidationError, match="JNJ"):
        rdf.validate_payload(payload, rdf._active_tickers(conn))


def test_wrong_type_rejected(conn):
    payload = _valid_payload()
    payload["assessments"]["AAPL"]["good_company_news"] = "true"
    with pytest.raises(rdf.ValidationError, match="good_company_news"):
        rdf.validate_payload(payload, rdf._active_tickers(conn))


def test_unknown_field_rejected(conn):
    payload = _valid_payload()
    payload["assessments"]["AAPL"]["confidence"] = 0.9
    with pytest.raises(rdf.ValidationError, match="confidence"):
        rdf.validate_payload(payload, rdf._active_tickers(conn))


def test_missing_required_field_rejected(conn):
    payload = _valid_payload()
    del payload["assessments"]["AAPL"]["macro_event_today"]
    with pytest.raises(rdf.ValidationError, match="macro_event_today"):
        rdf.validate_payload(payload, rdf._active_tickers(conn))


def test_future_date_rejected(conn):
    payload = _valid_payload(date="2099-01-01")
    with pytest.raises(rdf.ValidationError, match="futuro"):
        rdf.validate_payload(payload, rdf._active_tickers(conn))


def test_bad_date_format_rejected(conn):
    payload = _valid_payload(date="12-08-2026")
    with pytest.raises(rdf.ValidationError):
        rdf.validate_payload(payload, rdf._active_tickers(conn))


def test_ingest_payload_upserts_all_tickers_atomically(conn):
    clean = rdf.validate_payload(_valid_payload(), rdf._active_tickers(conn))
    rdf.ingest_payload(conn, clean)

    count = conn.execute("SELECT COUNT(*) FROM news_features WHERE date='2026-08-12'").fetchone()[0]
    assert count == len(config.TICKERS)


def test_resubmitting_same_date_updates_in_place_not_duplicates(conn):
    payload = _valid_payload()
    clean = rdf.validate_payload(payload, rdf._active_tickers(conn))
    rdf.ingest_payload(conn, clean)

    payload["assessments"]["AAPL"]["good_company_news"] = True
    clean2 = rdf.validate_payload(payload, rdf._active_tickers(conn))
    rdf.ingest_payload(conn, clean2)

    total = conn.execute("SELECT COUNT(*) FROM news_features").fetchone()[0]
    assert total == len(config.TICKERS)
    value = conn.execute(
        "SELECT good_company_news FROM news_features WHERE ticker='AAPL' AND date='2026-08-12'"
    ).fetchone()[0]
    assert value == 1
