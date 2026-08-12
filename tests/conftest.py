import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import database


@pytest.fixture
def conn():
    connection = database.get_connection(":memory:")
    database.seed_watchlist(connection)
    database.seed_portfolio_state(connection)
    yield connection
    connection.close()
