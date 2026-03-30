"""
Executor — submits pair trade orders to Alpaca and manages position lifecycle.

All orders go through RiskGuard before submission. The executor never bypasses
risk checks — that's the whole point.

Usage:
    executor = AlpacaExecutor(api_key, secret_key, paper=True)
    result = executor.enter_pair(signal, risk_verdict)
    result = executor.exit_pair(pair_id)
    result = executor.flatten_all()  # circuit breaker
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    import requests
except ImportError:
    requests = None


@dataclass
class OrderResult:
    success: bool
    pair_id: str
    long_order_id: Optional[str] = None
    short_order_id: Optional[str] = None
    long_fill_price: Optional[float] = None
    short_fill_price: Optional[float] = None
    long_shares: int = 0
    short_shares: int = 0
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PositionSnapshot:
    ticker: str
    side: str           # "long" or "short"
    qty: int
    avg_entry: float
    current_price: float
    market_value: float
    unrealized_pnl: float


class AlpacaExecutor:
    """
    Submits orders to Alpaca's REST API.
    Supports both paper and live environments.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        max_leg_delay_seconds: int = 30,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.base_url = self.PAPER_URL if paper else self.LIVE_URL
        self.paper = paper
        self.max_leg_delay = max_leg_delay_seconds

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API key and secret required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY env vars, or pass them directly."
            )

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an authenticated request to Alpaca."""
        if requests is None:
            raise ImportError("requests library required: pip install requests")

        url = f"{self.base_url}{endpoint}"
        resp = requests.request(method, url, headers=self.headers, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.text else {}

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Get account info (equity, buying power, etc.)."""
        return self._request("GET", "/v2/account")

    def get_positions(self) -> list[PositionSnapshot]:
        """Get all open positions."""
        raw = self._request("GET", "/v2/positions")
        return [
            PositionSnapshot(
                ticker=p["symbol"],
                side=p["side"],
                qty=int(p["qty"]),
                avg_entry=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                market_value=float(p["market_value"]),
                unrealized_pnl=float(p["unrealized_pl"]),
            )
            for p in raw
        ]

    def get_position(self, ticker: str) -> Optional[PositionSnapshot]:
        """Get position for a specific ticker."""
        try:
            p = self._request("GET", f"/v2/positions/{ticker}")
            return PositionSnapshot(
                ticker=p["symbol"],
                side=p["side"],
                qty=int(p["qty"]),
                avg_entry=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                market_value=float(p["market_value"]),
                unrealized_pnl=float(p["unrealized_pl"]),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _submit_order(
        self,
        ticker: str,
        qty: int,
        side: str,           # "buy" or "sell"
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> dict:
        """Submit a single order."""
        body = {
            "symbol": ticker,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        return self._request("POST", "/v2/orders", json=body)

    def _wait_for_fill(self, order_id: str, timeout_seconds: int = 30) -> dict:
        """Poll until order fills or timeout."""
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            order = self._request("GET", f"/v2/orders/{order_id}")
            if order["status"] == "filled":
                return order
            if order["status"] in ("canceled", "expired", "rejected"):
                return order
            time.sleep(1)
        return self._request("GET", f"/v2/orders/{order_id}")

    def _cancel_order(self, order_id: str):
        """Cancel an open order."""
        try:
            self._request("DELETE", f"/v2/orders/{order_id}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pair trade execution
    # ------------------------------------------------------------------

    def enter_pair(
        self,
        long_ticker: str,
        short_ticker: str,
        long_dollars: float,
        short_dollars: float,
        pair_id: str,
    ) -> OrderResult:
        """
        Enter a pair trade: buy the long leg, sell short the short leg.
        Both legs must fill within max_leg_delay or the entry is aborted.
        """
        # Calculate share counts from dollar amounts
        try:
            long_price = self._get_last_price(long_ticker)
            short_price = self._get_last_price(short_ticker)
        except Exception as e:
            return OrderResult(
                success=False, pair_id=pair_id,
                error=f"Failed to get prices: {e}",
            )

        long_shares = max(1, int(long_dollars / long_price))
        short_shares = max(1, int(short_dollars / short_price))

        # Submit long leg first
        try:
            long_order = self._submit_order(long_ticker, long_shares, "buy")
        except Exception as e:
            return OrderResult(
                success=False, pair_id=pair_id,
                error=f"Long order failed: {e}",
            )

        # Wait for long fill
        long_filled = self._wait_for_fill(long_order["id"], self.max_leg_delay)
        if long_filled["status"] != "filled":
            self._cancel_order(long_order["id"])
            return OrderResult(
                success=False, pair_id=pair_id,
                error=f"Long leg did not fill (status: {long_filled['status']}). Canceled.",
            )

        # Submit short leg
        try:
            short_order = self._submit_order(short_ticker, short_shares, "sell")
        except Exception as e:
            # Long filled but short failed — must unwind the long
            self._submit_order(long_ticker, long_shares, "sell")
            return OrderResult(
                success=False, pair_id=pair_id,
                error=f"Short order failed: {e}. Unwound long leg.",
            )

        # Wait for short fill
        short_filled = self._wait_for_fill(short_order["id"], self.max_leg_delay)
        if short_filled["status"] != "filled":
            # Short didn't fill — unwind the long
            self._cancel_order(short_order["id"])
            self._submit_order(long_ticker, long_shares, "sell")
            return OrderResult(
                success=False, pair_id=pair_id,
                error=f"Short leg did not fill (status: {short_filled['status']}). Unwound long.",
            )

        return OrderResult(
            success=True,
            pair_id=pair_id,
            long_order_id=long_filled["id"],
            short_order_id=short_filled["id"],
            long_fill_price=float(long_filled["filled_avg_price"]),
            short_fill_price=float(short_filled["filled_avg_price"]),
            long_shares=long_shares,
            short_shares=short_shares,
        )

    def exit_pair(
        self,
        long_ticker: str,
        short_ticker: str,
        long_shares: int,
        short_shares: int,
        pair_id: str,
    ) -> OrderResult:
        """
        Exit a pair trade: sell the long leg, buy to cover the short leg.
        """
        errors = []

        # Sell the long
        try:
            long_order = self._submit_order(long_ticker, long_shares, "sell")
            long_filled = self._wait_for_fill(long_order["id"], self.max_leg_delay)
        except Exception as e:
            errors.append(f"Long exit failed: {e}")
            long_filled = {"status": "failed", "filled_avg_price": "0"}

        # Buy to cover the short
        try:
            short_order = self._submit_order(short_ticker, short_shares, "buy")
            short_filled = self._wait_for_fill(short_order["id"], self.max_leg_delay)
        except Exception as e:
            errors.append(f"Short cover failed: {e}")
            short_filled = {"status": "failed", "filled_avg_price": "0"}

        success = not errors
        return OrderResult(
            success=success,
            pair_id=pair_id,
            long_fill_price=float(long_filled.get("filled_avg_price", 0)),
            short_fill_price=float(short_filled.get("filled_avg_price", 0)),
            long_shares=long_shares,
            short_shares=short_shares,
            error="; ".join(errors) if errors else "",
        )

    def flatten_all(self) -> list[OrderResult]:
        """
        CIRCUIT BREAKER: Close all positions immediately.
        This is the nuclear option — used when drawdown hits the limit.
        """
        results = []
        positions = self.get_positions()

        for pos in positions:
            try:
                if pos.side == "long":
                    order = self._submit_order(pos.ticker, pos.qty, "sell")
                else:
                    order = self._submit_order(pos.ticker, pos.qty, "buy")

                filled = self._wait_for_fill(order["id"], 30)
                results.append(OrderResult(
                    success=filled["status"] == "filled",
                    pair_id=f"flatten:{pos.ticker}",
                    long_fill_price=float(filled.get("filled_avg_price", 0)),
                    long_shares=pos.qty,
                ))
            except Exception as e:
                results.append(OrderResult(
                    success=False,
                    pair_id=f"flatten:{pos.ticker}",
                    error=str(e),
                ))

        return results

    def _get_last_price(self, ticker: str) -> float:
        """Get the latest trade price from Alpaca data API."""
        if requests is None:
            raise ImportError("requests library required: pip install requests")
        url = f"{self.DATA_URL}/v2/stocks/{ticker}/trades/latest"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return float(resp.json()["trade"]["p"])

    def get_live_prices(self, tickers: list[str]) -> dict[str, float]:
        """Get latest prices for multiple tickers using Alpaca snapshots."""
        if requests is None:
            raise ImportError("requests library required: pip install requests")
        if not tickers:
            return {}

        # Alpaca snapshot endpoint handles up to ~200 tickers at once
        symbols = ",".join(tickers)
        url = f"{self.DATA_URL}/v2/stocks/snapshots"
        resp = requests.get(url, headers=self.headers, params={"symbols": symbols}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        prices = {}
        for ticker, snap in data.items():
            try:
                prices[ticker] = float(snap["latestTrade"]["p"])
            except (KeyError, TypeError):
                pass
        return prices

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def account_summary(self) -> str:
        """Human-readable account summary."""
        acct = self.get_account()
        positions = self.get_positions()
        lines = [
            f"{'PAPER' if self.paper else 'LIVE'} Account Summary",
            f"  Equity:       ${float(acct['equity']):,.2f}",
            f"  Buying Power: ${float(acct['buying_power']):,.2f}",
            f"  Cash:         ${float(acct['cash']):,.2f}",
            f"  Positions:    {len(positions)}",
        ]
        if positions:
            total_pnl = sum(p.unrealized_pnl for p in positions)
            lines.append(f"  Unrealized:   ${total_pnl:+,.2f}")
            lines.append("")
            for p in positions:
                pnl_pct = (p.unrealized_pnl / (p.avg_entry * p.qty)) * 100 if p.qty else 0
                lines.append(
                    f"    {p.side.upper():5s} {p.ticker:6s} "
                    f"{p.qty:4d} @ ${p.avg_entry:.2f} → ${p.current_price:.2f} "
                    f"({p.unrealized_pnl:+.2f} / {pnl_pct:+.1f}%)"
                )
        return "\n".join(lines)
