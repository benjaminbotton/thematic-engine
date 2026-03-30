"""
Data Provider — fetches price/volume data from Polygon.io and caches in SQLite.

Usage:
    provider = PolygonDataProvider(api_key="...", db_path="data/thematic_engine.db")
    prices = provider.get_prices("AAPL", days=60)  # np.ndarray of closing prices
    ohlcv = provider.get_ohlcv("AAPL", days=60)    # DataFrame with full OHLCV

Respects Polygon rate limits (5 req/min on free, 100/min on starter).
Caches all data in SQLite so repeated queries don't hit the API.
"""

import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None


class PolygonDataProvider:
    """
    Fetches daily price data from Polygon.io REST API.
    Implements the DataProvider protocol expected by PairsEngine.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: str = "data/thematic_engine.db",
        rate_limit_per_min: int = 100,  # starter tier
    ):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY env var "
                "or pass api_key= parameter."
            )

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit_per_min
        self._last_request_time = 0.0
        self._request_count = 0
        self._minute_start = 0.0

        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                vwap REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
            ON daily_prices(ticker, date)
        """)
        conn.commit()
        conn.close()

    def _rate_limit_wait(self):
        """Simple rate limiter — space requests ~12s apart on free tier."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = 60.0 / self.rate_limit  # 12s for 5/min
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_from_polygon(
        self, ticker: str, from_date: str, to_date: str
    ) -> list[dict]:
        """Fetch daily bars from Polygon.io API."""
        if requests is None:
            raise ImportError("requests library required: pip install requests")

        self._rate_limit_wait()

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day"
            f"/{from_date}/{to_date}"
        )
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000,
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            return []

        results = []
        for bar in data.get("results", []):
            # Polygon timestamps are in milliseconds
            dt = datetime.fromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d")
            results.append({
                "ticker": ticker,
                "date": dt,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "vwap": bar.get("vw"),
            })

        return results

    def _cache_bars(self, bars: list[dict]):
        """Insert bars into SQLite, ignoring duplicates."""
        if not bars:
            return
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            """INSERT OR IGNORE INTO daily_prices
               (ticker, date, open, high, low, close, volume, vwap)
               VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :vwap)""",
            bars,
        )
        conn.commit()
        conn.close()

    def _get_cached(self, ticker: str, from_date: str, to_date: str) -> list[tuple]:
        """Query cached data from SQLite."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume, vwap
               FROM daily_prices
               WHERE ticker = ? AND date >= ? AND date <= ?
               ORDER BY date ASC""",
            (ticker, from_date, to_date),
        ).fetchall()
        conn.close()
        return rows

    def _get_latest_cached_date(self, ticker: str) -> Optional[str]:
        """Get the most recent cached date for a ticker."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT MAX(date) FROM daily_prices WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] else None

    def _ensure_data(self, ticker: str, days: int):
        """Fetch data from Polygon if cache is stale or missing."""
        to_date = date.today().isoformat()
        from_date = (date.today() - timedelta(days=days + 10)).isoformat()  # buffer

        latest = self._get_latest_cached_date(ticker)

        if latest:
            # Only fetch new data since last cached date
            fetch_from = (
                datetime.strptime(latest, "%Y-%m-%d").date() + timedelta(days=1)
            ).isoformat()
            if fetch_from > to_date:
                return  # fully up to date
        else:
            fetch_from = from_date

        bars = self._fetch_from_polygon(ticker, fetch_from, to_date)
        self._cache_bars(bars)

    # ------------------------------------------------------------------
    # Public interface (implements DataProvider protocol)
    # ------------------------------------------------------------------

    def get_prices(self, ticker: str, days: int, cache_only: bool = False) -> np.ndarray:
        """
        Return array of closing prices, most recent last.
        If cache_only=True, only reads from SQLite (no API calls).
        """
        if not cache_only:
            self._ensure_data(ticker, days)

        from_date = (date.today() - timedelta(days=days + 10)).isoformat()
        to_date = date.today().isoformat()
        rows = self._get_cached(ticker, from_date, to_date)

        if not rows:
            raise ValueError(f"No price data available for {ticker}")

        closes = np.array([r[4] for r in rows], dtype=float)  # index 4 = close
        return closes[-days:] if len(closes) > days else closes

    def get_ohlcv(self, ticker: str, days: int):
        """Return a pandas DataFrame with full OHLCV data."""
        if pd is None:
            raise ImportError("pandas required: pip install pandas")

        self._ensure_data(ticker, days)

        from_date = (date.today() - timedelta(days=days + 10)).isoformat()
        to_date = date.today().isoformat()
        rows = self._get_cached(ticker, from_date, to_date)

        df = pd.DataFrame(
            rows,
            columns=["date", "open", "high", "low", "close", "volume", "vwap"],
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df.tail(days)

    def get_latest_price(self, ticker: str) -> float:
        """Get the most recent closing price."""
        prices = self.get_prices(ticker, days=5)
        return float(prices[-1])

    def bulk_fetch(self, tickers: list[str], days: int):
        """Pre-fetch data for multiple tickers. Respects rate limits."""
        for i, ticker in enumerate(tickers):
            try:
                self._ensure_data(ticker, days)
                if (i + 1) % 10 == 0:
                    print(f"  Fetched {i + 1}/{len(tickers)} tickers...")
            except Exception as e:
                print(f"  Warning: failed to fetch {ticker}: {e}")

    def cache_stats(self) -> dict:
        """Return stats about the local cache."""
        conn = sqlite3.connect(self.db_path)
        ticker_count = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM daily_prices"
        ).fetchone()[0]
        row_count = conn.execute(
            "SELECT COUNT(*) FROM daily_prices"
        ).fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(date) FROM daily_prices"
        ).fetchone()[0]
        oldest = conn.execute(
            "SELECT MIN(date) FROM daily_prices"
        ).fetchone()[0]
        conn.close()
        return {
            "tickers": ticker_count,
            "rows": row_count,
            "latest_date": latest,
            "oldest_date": oldest,
            "db_path": str(self.db_path),
        }


# ---------------------------------------------------------------------------
# CLI: fetch data for all tickers in pods
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pod_manager import PodManager
    import yaml

    config_path = Path(__file__).parent.parent / "config"
    pm = PodManager(config_path)
    with open(config_path / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Set POLYGON_API_KEY environment variable first.")
        print("  export POLYGON_API_KEY=your_key_here")
        sys.exit(1)

    db_path = config_path.parent / settings["data"]["db_path"]
    provider = PolygonDataProvider(api_key=api_key, db_path=str(db_path))

    tickers = sorted(pm.all_tickers())
    print(f"Fetching {len(tickers)} tickers (60-day lookback)...")
    provider.bulk_fetch(tickers, days=90)

    stats = provider.cache_stats()
    print(f"\nCache: {stats['tickers']} tickers, {stats['rows']} rows")
    print(f"Date range: {stats['oldest_date']} to {stats['latest_date']}")
    print(f"DB: {stats['db_path']}")
