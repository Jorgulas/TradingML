import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import features


def _synthetic_prices(n=30, start_close=100.0, step=1.0, start_volume=1_000_000, vol_step=1000):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [start_close + i * step for i in range(n)]
    volumes = [start_volume + i * vol_step for i in range(n)]
    return pd.DataFrame({"Close": closes, "Volume": volumes}, index=dates)


def test_ret_1d_exact_value_on_linear_series():
    df = _synthetic_prices(n=6, start_close=100.0, step=1.0)
    feats = features.compute_technical_features(df)
    # shifted.iloc[2] = raw.iloc[1] = (close[1]-close[0])/close[0] = (101-100)/100
    assert feats.iloc[2]["ret_1d"] == pytest.approx(0.01)


def test_leading_rows_are_nan_and_get_dropped():
    df = _synthetic_prices(n=10)
    feats = features.compute_technical_features(df).dropna(how="all")
    # a primeira linha (sem nenhum dia anterior) tem de desaparecer ao dropna(how="all")
    assert df.index[0] not in feats.index


def test_long_window_columns_nan_until_enough_history():
    df = _synthetic_prices(n=60)
    feats = features.compute_technical_features(df)
    # sma200_ratio precisa de 200 fechos anteriores -- com so' 60 dias, nunca disponivel
    assert feats["sma200_ratio"].notna().sum() == 0
    # sma5_ratio precisa so' de 5 -- deve estar disponivel para a maioria das linhas
    assert feats["sma5_ratio"].notna().sum() > 40


def test_rsi_bounded_between_0_and_100():
    df = _synthetic_prices(n=80, step=1.0)
    feats = features.compute_technical_features(df)
    valid = feats["rsi14"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 100).all()


def test_technical_features_do_not_leak_same_day_price():
    """Regressao anti-leakage: mutar o Close do PROPRIO dia D nao pode mudar
    a linha de features calculada PARA D (que so' pode usar dados ate' D-1)."""
    df = _synthetic_prices(n=40)
    mutate_idx = 25

    feats_original = features.compute_technical_features(df)

    df_mutated = df.copy()
    df_mutated.iloc[mutate_idx, df_mutated.columns.get_loc("Close")] = 99999.0
    feats_mutated = features.compute_technical_features(df_mutated)

    mutated_date = df.index[mutate_idx]
    pd.testing.assert_series_equal(
        feats_original.loc[mutated_date], feats_mutated.loc[mutated_date], check_names=False
    )

    # sanity check complementar: o dia SEGUINTE usa o close de D, por isso
    # tem mesmo de mudar -- confirma que o shift(1) esta' de facto a acontecer
    # e nao estamos so' a passar o teste por os dois lados estarem parados.
    next_date = df.index[mutate_idx + 1]
    assert not feats_original.loc[next_date].equals(feats_mutated.loc[next_date])


def test_get_news_aggregate_defaults_to_zero_when_no_assessment(conn):
    result = features.get_news_aggregate(conn, "AAPL", "2024-01-15", window_days=10)
    assert all(v == 0.0 for v in result.values())


def test_get_news_aggregate_short_window_is_just_todays_row(conn):
    import config
    now = "2024-01-15T00:00:00+00:00"
    cols = ", ".join(config.BOOLEAN_FEATURES)
    placeholders = ", ".join("?" for _ in config.BOOLEAN_FEATURES)
    conn.execute(
        f"INSERT INTO news_features (ticker, date, {cols}, filled_at) VALUES (?, ?, {placeholders}, ?)",
        ["AAPL", "2024-01-15", 1, 0, 0, 1, 0, now],
    )
    conn.commit()
    result = features.get_news_aggregate(conn, "AAPL", "2024-01-15", window_days=1)
    assert result["good_company_news"] == 1.0
    assert result["bad_company_news"] == 0.0
    assert result["sector_momentum_positive"] == 1.0


def test_build_feature_vector_none_when_technical_history_missing(conn):
    vector = features.build_feature_vector(conn, "AAPL", "2024-01-15", "SHORT")
    assert vector is None


def test_get_news_aggregate_long_window_averages_last_10_days(conn):
    import config
    cols = ", ".join(config.BOOLEAN_FEATURES)
    placeholders = ", ".join("?" for _ in config.BOOLEAN_FEATURES)
    dates = pd.date_range("2024-02-01", periods=12, freq="D").strftime("%Y-%m-%d")
    # 3 dos ultimos 10 dias com good_company_news=True, o resto False
    good_news_dates = {dates[9], dates[10], dates[11]}
    for d in dates:
        good = 1 if d in good_news_dates else 0
        conn.execute(
            f"INSERT INTO news_features (ticker, date, {cols}, filled_at) VALUES (?, ?, {placeholders}, ?)",
            ["AAPL", d, good, 0, 0, 0, 0, "2024-02-01T00:00:00+00:00"],
        )
    conn.commit()

    result = features.get_news_aggregate(conn, "AAPL", dates[11], window_days=10)
    # janela = ultimos 10 dias com date <= dates[11] -> dates[2..11], dos quais
    # dates[9],[10],[11] tem good_company_news=True -> 3/10
    assert result["good_company_news"] == pytest.approx(0.3)
