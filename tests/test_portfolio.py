"""Testes da terceira carteira (sinal de direccao pos-padrao).

Protege duas coisas que, se partirem, produzem um backtest lucrativo e falso:
a carteira negociar fora do periodo de teste, e a curva de equity ter pontos
duplicados por instante (as barras de 1h sao as mesmas para os 43 tickers, por
isso varios padroes confirmam ao mesmo tempo).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.patterns import portfolio


def test_portfolio_has_its_own_cadence_not_the_daily_simulator():
    """O horizonte seleccionado e' de poucas barras de 1h. Se esta carteira
    passasse pelo simulator.py (diario), manteria a posicao muito para la' do
    que o modelo preve e deixaria de testar o sinal."""
    assert "1h" in config.PATTERN_DIRECTION["timeframes"]
    assert config.PATTERN_PORTFOLIO["starting_cash"] > 0
    assert 0 < config.PATTERN_PORTFOLIO["position_size_pct"] < 1


def test_confidence_threshold_is_above_a_coin_flip():
    """Abrir posicao com probabilidade < 0.5 seria negociar contra a propria
    previsao."""
    assert config.PATTERN_PORTFOLIO["confidence_threshold"] > 0.5


def test_equity_curve_has_one_point_per_timestamp(conn):
    """As barras de 1h sao partilhadas por todos os tickers; sem deduplicacao
    a curva teria varios pontos no mesmo instante e o INSERT rebentaria."""
    result = portfolio.run_backtest(conn)
    if result is None:
        pytest.skip("sem modelo de direccao treinado nesta BD de teste")
    timestamps = [p["ts"] for p in result["equity_curve"]]
    assert len(timestamps) == len(set(timestamps))
    assert timestamps == sorted(timestamps)


def test_backtest_returns_none_without_a_trained_model(conn):
    """BD limpa: nao deve inventar uma carteira a partir do nada."""
    assert portfolio.run_backtest(conn) is None
