"""
Backtest — walk-forward simulation of the thematic pairs strategy over historical data.

Replays every trading day:
  1. Weekly: re-run pair discovery on trailing 60 days
  2. Daily: update z-scores, check for entry/exit/stop signals
  3. Apply all risk rules (position sizing, gross cap, circuit breaker)
  4. Track P&L per pair, per pod, total portfolio

Usage:
    python3 src/backtest.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from pod_manager import PodManager
from pairs_engine import (
    PairsEngine, engle_granger_test, calculate_half_life,
    calculate_hurst, rolling_zscore
)


# ---------------------------------------------------------------------------
# Data fetcher — gets full history from Polygon
# ---------------------------------------------------------------------------

class BacktestDataProvider:
    """Fetches and caches full price history for backtesting."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._full_history: dict[str, dict[str, float]] = {}  # ticker -> {date_str: close}
        self._last_request = 0.0

    def fetch_full_history(self, ticker: str, from_date: str, to_date: str):
        """Fetch full daily history for a ticker."""
        import requests

        # Rate limit
        elapsed = time.time() - self._last_request
        if elapsed < 0.6:  # ~100 req/min
            time.sleep(0.6 - elapsed)
        self._last_request = time.time()

        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 5000}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            return {}

        history = {}
        for bar in data.get("results", []):
            dt = datetime.fromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d")
            history[dt] = bar["c"]

        self._full_history[ticker] = history
        return history

    def get_prices_as_of(self, ticker: str, as_of: str, lookback_days: int) -> np.ndarray:
        """Get closing prices up to (and including) as_of date."""
        history = self._full_history.get(ticker, {})
        if not history:
            raise ValueError(f"No data for {ticker}")

        # Get all dates <= as_of, sorted
        dates = sorted(d for d in history.keys() if d <= as_of)
        if len(dates) < 10:
            raise ValueError(f"Insufficient data for {ticker} as of {as_of}")

        # Take last lookback_days
        dates = dates[-lookback_days:]
        return np.array([history[d] for d in dates], dtype=float)

    def get_price_on(self, ticker: str, date_str: str) -> float:
        """Get closing price on a specific date."""
        history = self._full_history.get(ticker, {})
        if date_str in history:
            return history[date_str]
        # Find nearest prior date
        dates = sorted(d for d in history.keys() if d <= date_str)
        if dates:
            return history[dates[-1]]
        raise ValueError(f"No price for {ticker} on {date_str}")

    def get_trading_days(self, from_date: str, to_date: str) -> list[str]:
        """Get list of trading days from any ticker's history."""
        for ticker, history in self._full_history.items():
            if history:
                return sorted(d for d in history.keys() if from_date <= d <= to_date)
        return []


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass
class BacktestPosition:
    pair_id: str
    pod_id: str
    long_ticker: str
    short_ticker: str
    long_shares: int
    short_shares: int
    long_entry_price: float
    short_entry_price: float
    entry_date: str
    entry_zscore: float
    long_notional: float = 0.0
    short_notional: float = 0.0

    def __post_init__(self):
        self.long_notional = self.long_shares * self.long_entry_price
        self.short_notional = self.short_shares * self.short_entry_price

    def pnl(self, long_price: float, short_price: float) -> float:
        long_pnl = self.long_shares * (long_price - self.long_entry_price)
        short_pnl = self.short_shares * (self.short_entry_price - short_price)
        return long_pnl + short_pnl

    def gross(self, long_price: float, short_price: float) -> float:
        return abs(self.long_shares * long_price) + abs(self.short_shares * short_price)


@dataclass
class TradeRecord:
    pair_id: str
    pod_id: str
    long_ticker: str
    short_ticker: str
    action: str          # ENTER / EXIT / STOP
    entry_date: str
    exit_date: str = ""
    entry_zscore: float = 0.0
    exit_zscore: float = 0.0
    long_notional: float = 0.0
    short_notional: float = 0.0
    pnl: float = 0.0
    holding_days: int = 0


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class Backtester:

    def __init__(self, settings: dict, pm: PodManager, data: BacktestDataProvider):
        self.settings = settings
        self.pm = pm
        self.data = data

        self.capital = settings.get("capital_base", 100000)
        self.equity_curve: list[tuple[str, float]] = []
        self.positions: list[BacktestPosition] = []
        self.trades: list[TradeRecord] = []
        self.daily_pnl: list[tuple[str, float]] = []

        # Config
        spread_cfg = settings.get("pairs", {}).get("spread", {})
        self.entry_z = spread_cfg.get("entry_zscore", 2.0)
        self.exit_z = spread_cfg.get("exit_zscore", 0.5)
        self.stop_z = spread_cfg.get("stop_zscore", 3.5)
        self.z_lookback = spread_cfg.get("zscore_lookback", 20)
        self.discovery_lookback = settings.get("pairs", {}).get("discovery", {}).get("lookback_days", 60)

        sizing_cfg = settings.get("pairs", {}).get("sizing", {})
        self.default_leg_size = sizing_cfg.get("default_leg_size", 8000)
        self.max_leg_size = sizing_cfg.get("max_leg_size", 10000)

        hard_rules = settings.get("hard_rules", {})
        self.max_risk_pct = hard_rules.get("max_risk_per_trade_pct", 0.015)
        self.gap_multiplier = hard_rules.get("gap_multiplier", 2.0)
        self.max_gross_ratio = hard_rules.get("max_gross_to_capital_ratio", 1.20)
        self.circuit_breaker_dd = hard_rules.get("circuit_breaker_drawdown", 0.10)

        disc_cfg = settings.get("pairs", {}).get("discovery", {})
        self.max_pairs_per_pod = disc_cfg.get("max_pairs_per_pod", 4)
        self.max_total_pairs = disc_cfg.get("max_total_pairs", 15)
        self.coint_pvalue = disc_cfg.get("cointegration_pvalue", 0.05)
        self.min_half_life = spread_cfg.get("min_half_life", 2)
        self.max_half_life = spread_cfg.get("max_half_life", 15)
        self.hurst_threshold = settings.get("pairs", {}).get("regime", {}).get("hurst_threshold", 0.45)

    def run(self, start_date: str, end_date: str):
        """Run the full backtest."""
        trading_days = self.data.get_trading_days(start_date, end_date)
        if not trading_days:
            print("No trading days found!")
            return

        print(f"Backtesting {start_date} to {end_date} ({len(trading_days)} trading days)")
        print(f"Capital: ${self.capital:,.0f}")
        print()

        current_equity = self.capital
        peak_equity = self.capital
        valid_pairs = {}  # pair_id -> {long, short, hedge_ratio, ...}
        discovery_countdown = 0
        circuit_breaker_until = None  # day_idx when cooloff ends
        pod_losses = {}       # pod_id -> consecutive loss count
        pod_disabled_until = {}  # pod_id -> day_idx when re-enabled

        for day_idx, today in enumerate(trading_days):
            # --- Weekly pair discovery ---
            if discovery_countdown <= 0:
                valid_pairs = self._discover_pairs(today)
                discovery_countdown = 5  # every 5 trading days
                if day_idx == 0:
                    print(f"  [{today}] Initial discovery: {len(valid_pairs)} pairs")
            discovery_countdown -= 1

            # --- Update z-scores for all pairs ---
            pair_zscores = {}
            for pid, pair_info in valid_pairs.items():
                try:
                    lp = self.data.get_prices_as_of(pair_info["long"], today, self.z_lookback + 5)
                    sp = self.data.get_prices_as_of(pair_info["short"], today, self.z_lookback + 5)
                    n = min(len(lp), len(sp))
                    spread = lp[-n:] - pair_info["hedge_ratio"] * sp[-n:]
                    z = rolling_zscore(spread, self.z_lookback)
                    pair_zscores[pid] = z
                except Exception:
                    continue

            # --- Check exits/stops on existing positions ---
            to_close = []
            for pos in self.positions:
                z = pair_zscores.get(pos.pair_id)
                if z is None:
                    continue

                try:
                    lp = self.data.get_price_on(pos.long_ticker, today)
                    sp = self.data.get_price_on(pos.short_ticker, today)
                except Exception:
                    continue

                # Holding days
                entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
                today_dt = datetime.strptime(today, "%Y-%m-%d")
                holding_days = (today_dt - entry_dt).days

                # Current P&L for trailing stop
                current_pnl = pos.pnl(lp, sp)
                total_notional = pos.long_notional + pos.short_notional
                pnl_pct = current_pnl / total_notional if total_notional > 0 else 0

                should_close = False
                action = ""

                # Stop loss — spread blew out
                if abs(z) >= self.stop_z:
                    should_close = True
                    action = "STOP"
                # Hard loss stop — pair is losing more than 4% of notional
                elif pnl_pct < -0.04:
                    should_close = True
                    action = "LOSS_CUT"
                # Exit — mean reversion complete
                elif abs(z) <= self.exit_z:
                    should_close = True
                    action = "EXIT"
                # Trailing stop — was profitable, now giving back
                elif pnl_pct > 0.02 and abs(z) <= 1.0:
                    should_close = True
                    action = "TRAILING_STOP"
                # Max holding — 3x max half-life. If it hasn't reverted by now, it won't.
                elif holding_days >= 8:
                    should_close = True
                    action = "EXIT_TIME"

                if should_close:
                    pnl = pos.pnl(lp, sp)
                    self.trades.append(TradeRecord(
                        pair_id=pos.pair_id, pod_id=pos.pod_id,
                        long_ticker=pos.long_ticker, short_ticker=pos.short_ticker,
                        action=action, entry_date=pos.entry_date, exit_date=today,
                        entry_zscore=pos.entry_zscore, exit_zscore=z,
                        long_notional=pos.long_notional, short_notional=pos.short_notional,
                        pnl=pnl, holding_days=holding_days,
                    ))
                    to_close.append(pos)

            for pos in to_close:
                self.positions.remove(pos)
                # Track pod win/loss streaks
                trade = self.trades[-1] if self.trades else None
                if trade and trade.pair_id == pos.pair_id:
                    if trade.pnl > 0:
                        pod_losses[pos.pod_id] = 0
                    else:
                        pod_losses[pos.pod_id] = pod_losses.get(pos.pod_id, 0) + 1
                        if pod_losses[pos.pod_id] >= 3:
                            pod_disabled_until[pos.pod_id] = day_idx + 20
                            print(f"  [{today}] Pod {pos.pod_id} disabled (3 consecutive losses)")

            # Re-enable pods after cooloff
            for pod_id in list(pod_disabled_until.keys()):
                if day_idx >= pod_disabled_until[pod_id]:
                    del pod_disabled_until[pod_id]
                    pod_losses[pod_id] = 0
                    print(f"  [{today}] Pod {pod_id} re-enabled")

            # --- Check entries (skip during circuit breaker cooloff) ---
            for pid, z in pair_zscores.items():
                if circuit_breaker_until and day_idx < circuit_breaker_until:
                    break  # no new entries during cooloff

                # Pod disabled?
                pair_pod = valid_pairs[pid]["pod_id"]
                if pair_pod in pod_disabled_until:
                    continue

                # Already in this pair?
                if any(p.pair_id == pid for p in self.positions):
                    continue

                if abs(z) < self.entry_z:
                    continue

                pair_info = valid_pairs[pid]

                # Risk sizing
                max_loss = current_equity * self.max_risk_pct
                stop_pct = 0.08
                max_pair_notional = max_loss / (stop_pct * self.gap_multiplier)
                leg_size = min(self.default_leg_size, max_pair_notional / 2)
                leg_size = min(leg_size, self.max_leg_size)

                # Gross exposure check
                current_gross = self._current_gross(today)
                max_gross = current_equity * self.max_gross_ratio
                if current_gross + (leg_size * 2) > max_gross:
                    continue

                # Max positions
                if len(self.positions) >= self.max_total_pairs:
                    continue

                try:
                    lp = self.data.get_price_on(pair_info["long"], today)
                    sp = self.data.get_price_on(pair_info["short"], today)
                except Exception:
                    continue

                long_shares = max(1, int(leg_size / lp))
                short_shares = max(1, int(leg_size / sp))

                self.positions.append(BacktestPosition(
                    pair_id=pid, pod_id=pair_info["pod_id"],
                    long_ticker=pair_info["long"], short_ticker=pair_info["short"],
                    long_shares=long_shares, short_shares=short_shares,
                    long_entry_price=lp, short_entry_price=sp,
                    entry_date=today, entry_zscore=z,
                ))

            # --- Circuit breaker ---
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            if drawdown >= self.circuit_breaker_dd and not circuit_breaker_until:
                # Flatten everything
                for pos in list(self.positions):
                    try:
                        lp = self.data.get_price_on(pos.long_ticker, today)
                        sp = self.data.get_price_on(pos.short_ticker, today)
                        pnl = pos.pnl(lp, sp)
                        entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
                        today_dt = datetime.strptime(today, "%Y-%m-%d")
                        self.trades.append(TradeRecord(
                            pair_id=pos.pair_id, pod_id=pos.pod_id,
                            long_ticker=pos.long_ticker, short_ticker=pos.short_ticker,
                            action="CIRCUIT_BREAKER", entry_date=pos.entry_date, exit_date=today,
                            entry_zscore=pos.entry_zscore, exit_zscore=0,
                            long_notional=pos.long_notional, short_notional=pos.short_notional,
                            pnl=pnl, holding_days=(today_dt - entry_dt).days,
                        ))
                    except Exception:
                        pass
                self.positions.clear()
                # Cooloff: pause for 10 trading days, then reset peak to current equity
                circuit_breaker_until = day_idx + 10
                print(f"  [{today}] CIRCUIT BREAKER at {drawdown:.1%} — pausing 10 trading days")

            if circuit_breaker_until and day_idx >= circuit_breaker_until:
                # Resume trading, reset peak so we don't immediately re-trigger
                peak_equity = current_equity
                circuit_breaker_until = None
                print(f"  [{today}] Circuit breaker cooloff complete. Resuming. Peak reset to ${current_equity:,.0f}")

            # --- Mark-to-market ---
            unrealized = 0
            for pos in self.positions:
                try:
                    lp = self.data.get_price_on(pos.long_ticker, today)
                    sp = self.data.get_price_on(pos.short_ticker, today)
                    unrealized += pos.pnl(lp, sp)
                except Exception:
                    pass

            realized_today = sum(t.pnl for t in self.trades if t.exit_date == today)
            day_pnl = realized_today  # simplified — unrealized captured in equity
            current_equity = self.capital + sum(t.pnl for t in self.trades) + unrealized
            peak_equity = max(peak_equity, current_equity)

            self.equity_curve.append((today, current_equity))
            self.daily_pnl.append((today, day_pnl))

        # Close any remaining positions at end
        if self.positions:
            last_day = trading_days[-1]
            for pos in list(self.positions):
                try:
                    lp = self.data.get_price_on(pos.long_ticker, last_day)
                    sp = self.data.get_price_on(pos.short_ticker, last_day)
                    pnl = pos.pnl(lp, sp)
                    entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
                    last_dt = datetime.strptime(last_day, "%Y-%m-%d")
                    self.trades.append(TradeRecord(
                        pair_id=pos.pair_id, pod_id=pos.pod_id,
                        long_ticker=pos.long_ticker, short_ticker=pos.short_ticker,
                        action="EOY_CLOSE", entry_date=pos.entry_date, exit_date=last_day,
                        entry_zscore=pos.entry_zscore, exit_zscore=0,
                        long_notional=pos.long_notional, short_notional=pos.short_notional,
                        pnl=pnl, holding_days=(last_dt - entry_dt).days,
                    ))
                except Exception:
                    pass
            self.positions.clear()

    def _discover_pairs(self, as_of: str) -> dict:
        """Run pair discovery using data up to as_of date."""
        valid_pairs = {}

        for pod in self.pm.active_pods():
            pod_pairs = []
            candidates = list(pod.pair_candidates)

            for long_entry, short_entry in candidates:
                long_t = long_entry.ticker if hasattr(long_entry, 'ticker') else long_entry
                short_t = short_entry.ticker if hasattr(short_entry, 'ticker') else short_entry
                try:
                    lp = self.data.get_prices_as_of(long_t, as_of, self.discovery_lookback)
                    sp = self.data.get_prices_as_of(short_t, as_of, self.discovery_lookback)
                except Exception:
                    continue

                n = min(len(lp), len(sp))
                if n < 30:
                    continue
                lp = lp[-n:]
                sp = sp[-n:]

                # Cointegration test
                pvalue, hedge_ratio, residuals = engle_granger_test(lp, sp)
                if pvalue > self.coint_pvalue:
                    continue

                # Half-life
                hl = calculate_half_life(residuals)
                if hl < self.min_half_life or hl > self.max_half_life:
                    continue

                # Hurst
                hurst = calculate_hurst(residuals, max_lag=min(20, len(residuals) // 2))
                if hurst > self.hurst_threshold:
                    continue

                # Score (simplified)
                score = (1 - pvalue) * 0.3 + (1 - hl / self.max_half_life) * 0.3 + (1 - hurst) * 0.4
                pod_pairs.append({
                    "long": long_t, "short": short_t,
                    "pod_id": pod.pod_id, "hedge_ratio": hedge_ratio,
                    "score": score, "pvalue": pvalue, "half_life": hl, "hurst": hurst,
                })

            # Keep top N per pod
            pod_pairs.sort(key=lambda x: x["score"], reverse=True)
            for p in pod_pairs[:self.max_pairs_per_pod]:
                pid = f"{pod.pod_id}:{p['long']}-{p['short']}"
                valid_pairs[pid] = p

        # Trim globally
        if len(valid_pairs) > self.max_total_pairs:
            all_sorted = sorted(valid_pairs.items(), key=lambda x: x[1]["score"], reverse=True)
            # Guarantee 1 per pod
            kept = {}
            pods_seen = set()
            remaining = []
            for pid, info in all_sorted:
                if info["pod_id"] not in pods_seen:
                    kept[pid] = info
                    pods_seen.add(info["pod_id"])
                else:
                    remaining.append((pid, info))
            for pid, info in remaining:
                if len(kept) >= self.max_total_pairs:
                    break
                kept[pid] = info
            valid_pairs = kept

        return valid_pairs

    def _current_gross(self, today: str) -> float:
        """Calculate current gross exposure."""
        gross = 0
        for pos in self.positions:
            try:
                lp = self.data.get_price_on(pos.long_ticker, today)
                sp = self.data.get_price_on(pos.short_ticker, today)
                gross += pos.gross(lp, sp)
            except Exception:
                pass
        return gross

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self):
        """Print backtest results."""
        if not self.trades:
            print("No trades executed.")
            return

        total_pnl = sum(t.pnl for t in self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float('inf')

        avg_holding = np.mean([t.holding_days for t in self.trades])

        # Max drawdown
        peak = self.capital
        max_dd = 0
        max_dd_date = ""
        for dt, eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_date = dt

        # Sharpe (annualized, simplified)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_eq = self.equity_curve[i-1][1]
                curr_eq = self.equity_curve[i][1]
                returns.append((curr_eq - prev_eq) / prev_eq)
            if returns and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.capital
        total_return = (final_equity - self.capital) / self.capital * 100

        print("=" * 65)
        print("BACKTEST RESULTS")
        print("=" * 65)
        print()
        print(f"  Period:         {self.equity_curve[0][0]} to {self.equity_curve[-1][0]}")
        print(f"  Starting:       ${self.capital:,.0f}")
        print(f"  Ending:         ${final_equity:,.0f}")
        print(f"  Total Return:   {total_return:+.1f}%  (${total_pnl:+,.0f})")
        print(f"  Sharpe Ratio:   {sharpe:.2f}")
        print(f"  Max Drawdown:   {max_dd:.1%} (on {max_dd_date})")
        print()
        print(f"  Total Trades:   {len(self.trades)}")
        print(f"  Win Rate:       {win_rate:.0f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg Win:        ${avg_win:+,.0f}")
        print(f"  Avg Loss:       ${avg_loss:+,.0f}")
        print(f"  Profit Factor:  {profit_factor:.2f}")
        print(f"  Avg Holding:    {avg_holding:.1f} days")
        print()

        # By exit type
        exit_types = {}
        for t in self.trades:
            exit_types[t.action] = exit_types.get(t.action, 0) + 1
        print("  Exit Types:")
        for action, count in sorted(exit_types.items()):
            action_pnl = sum(t.pnl for t in self.trades if t.action == action)
            print(f"    {action:20s}  {count:3d} trades  ${action_pnl:+,.0f}")

        # By pod
        print()
        print("  By Pod:")
        pod_stats = {}
        for t in self.trades:
            if t.pod_id not in pod_stats:
                pod_stats[t.pod_id] = {"trades": 0, "pnl": 0, "wins": 0}
            pod_stats[t.pod_id]["trades"] += 1
            pod_stats[t.pod_id]["pnl"] += t.pnl
            if t.pnl > 0:
                pod_stats[t.pod_id]["wins"] += 1

        for pod_id, stats in sorted(pod_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            print(f"    {pod_id:30s}  {stats['trades']:3d} trades  {wr:.0f}% WR  ${stats['pnl']:+,.0f}")

        # Top 5 trades
        print()
        sorted_trades = sorted(self.trades, key=lambda x: x.pnl, reverse=True)
        print("  Top 5 Winners:")
        for t in sorted_trades[:5]:
            print(f"    {t.long_ticker}/{t.short_ticker}  {t.entry_date}→{t.exit_date}  "
                  f"z={t.entry_zscore:+.1f}→{t.exit_zscore:+.1f}  ${t.pnl:+,.0f}  ({t.action})")

        print()
        print("  Top 5 Losers:")
        for t in sorted_trades[-5:]:
            print(f"    {t.long_ticker}/{t.short_ticker}  {t.entry_date}→{t.exit_date}  "
                  f"z={t.entry_zscore:+.1f}→{t.exit_zscore:+.1f}  ${t.pnl:+,.0f}  ({t.action})")

        # Monthly returns
        print()
        print("  Monthly Returns:")
        monthly = {}
        for dt, eq in self.equity_curve:
            month = dt[:7]
            monthly[month] = eq
        prev_eq = self.capital
        for month in sorted(monthly.keys()):
            eq = monthly[month]
            ret = (eq - prev_eq) / prev_eq * 100
            bar = "█" * int(abs(ret) * 2)
            sign = "+" if ret >= 0 else "-"
            color_indicator = "▲" if ret >= 0 else "▼"
            print(f"    {month}  {color_indicator} {ret:+6.2f}%  ${eq - prev_eq:+,.0f}  {bar}")
            prev_eq = eq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load env
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k] = v

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Set POLYGON_API_KEY to run backtest")
        sys.exit(1)

    # Load config
    config_path = Path(__file__).parent.parent / "config"
    with open(config_path / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    pm = PodManager(config_path)
    data = BacktestDataProvider(api_key)

    # Fetch all ticker history
    all_tickers = sorted(pm.all_tickers())
    from_date = "2024-11-01"  # need ~60 days before Jan 2025 for lookback
    to_date = "2025-12-31"

    print(f"Fetching {len(all_tickers)} tickers ({from_date} to {to_date})...")
    fetched = 0
    for ticker in all_tickers:
        try:
            history = data.fetch_full_history(ticker, from_date, to_date)
            if history:
                fetched += 1
                if fetched % 20 == 0:
                    print(f"  {fetched}/{len(all_tickers)} fetched...")
        except Exception as e:
            print(f"  {ticker}: {e}")

    print(f"  {fetched}/{len(all_tickers)} tickers with data")
    print()

    # Run backtest
    bt = Backtester(settings, pm, data)
    bt.run("2025-01-02", "2025-12-31")
    print()
    bt.report()
