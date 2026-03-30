"""
Backtest V2 — with dynamic allocation + simulated LLM pod generation.

Key differences from V1:
  - Pods are generated dynamically per quarter (simulating LLM output)
  - Capital allocated by momentum — winning pods get more, losers get less
  - Tighter exits: max 8 days hold (2-3x half-life), 4% loss cut
  - Trailing stop locks in profits
  - Pod-level kill switch after 3 consecutive losses
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

from pairs_engine import engle_granger_test, calculate_half_life, calculate_hurst, rolling_zscore
from allocator import DynamicAllocator
from pod_generator import get_backtest_pods, GeneratedPod
from backtest import BacktestDataProvider, BacktestPosition, TradeRecord


class BacktesterV2:

    def __init__(self, settings: dict, data: BacktestDataProvider):
        self.settings = settings
        self.data = data
        self.capital = settings.get("capital_base", 100000)
        self.equity_curve: list[tuple[str, float]] = []
        self.positions: list[BacktestPosition] = []
        self.trades: list[TradeRecord] = []

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
        trading_days = self.data.get_trading_days(start_date, end_date)
        if not trading_days:
            print("No trading days found!")
            return

        print(f"Backtesting {start_date} to {end_date} ({len(trading_days)} trading days)")
        print(f"Capital: ${self.capital:,.0f}")
        print(f"Mode: Dynamic allocation + LLM-generated pods")
        print()

        current_equity = self.capital
        peak_equity = self.capital
        valid_pairs = {}
        discovery_countdown = 0
        circuit_breaker_until = None
        active_pods: list[GeneratedPod] = []
        allocator = None
        last_pod_check = ""

        for day_idx, today in enumerate(trading_days):

            # --- Check for new pods (quarterly catalyst refresh) ---
            new_pods = get_backtest_pods(today)
            new_pod_ids = {p.pod_id for p in new_pods}
            old_pod_ids = {p.pod_id for p in active_pods}
            if new_pod_ids != old_pod_ids:
                active_pods = new_pods
                pod_ids = [p.pod_id for p in active_pods]
                if allocator is None:
                    allocator = DynamicAllocator(pod_ids, self.capital)
                else:
                    # Add new pods to allocator
                    for pid in pod_ids:
                        if pid not in allocator.pods:
                            from allocator import PodPerformance
                            allocator.pods[pid] = PodPerformance(pod_id=pid)
                discovery_countdown = 0  # force rediscovery
                added = new_pod_ids - old_pod_ids
                removed = old_pod_ids - new_pod_ids
                if added:
                    print(f"  [{today}] New pods: {', '.join(added)}")
                if removed:
                    print(f"  [{today}] Expired pods: {', '.join(removed)}")

            if not active_pods or not allocator:
                self.equity_curve.append((today, current_equity))
                continue

            # --- Weekly pair discovery ---
            if discovery_countdown <= 0:
                valid_pairs = self._discover_pairs(today, active_pods)
                discovery_countdown = 5
                if day_idx == 0 or discovery_countdown == 5:
                    pass  # silent
            discovery_countdown -= 1

            # --- Update z-scores ---
            pair_zscores = {}
            pair_half_lives = {}
            for pid, pair_info in valid_pairs.items():
                try:
                    lp = self.data.get_prices_as_of(pair_info["long"], today, self.z_lookback + 5)
                    sp = self.data.get_prices_as_of(pair_info["short"], today, self.z_lookback + 5)
                    n = min(len(lp), len(sp))
                    spread = lp[-n:] - pair_info["hedge_ratio"] * sp[-n:]
                    z = rolling_zscore(spread, self.z_lookback)
                    pair_zscores[pid] = z
                    pair_half_lives[pid] = pair_info.get("half_life", 5)
                except Exception:
                    continue

            # --- Exits ---
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

                entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
                today_dt = datetime.strptime(today, "%Y-%m-%d")
                holding_days = (today_dt - entry_dt).days
                current_pnl = pos.pnl(lp, sp)
                total_notional = pos.long_notional + pos.short_notional
                pnl_pct = current_pnl / total_notional if total_notional > 0 else 0

                # Get this pair's half-life for adaptive max hold
                hl = pair_half_lives.get(pos.pair_id, 4)
                max_hold = max(5, int(hl * 2.5))  # 2.5x half-life, min 5 days

                should_close = False
                action = ""

                if abs(z) >= self.stop_z:
                    should_close, action = True, "STOP"
                elif pnl_pct < -0.04:
                    should_close, action = True, "LOSS_CUT"
                elif abs(z) <= self.exit_z:
                    should_close, action = True, "EXIT"
                elif pnl_pct > 0.02 and abs(z) <= 1.0:
                    should_close, action = True, "TRAILING_STOP"
                elif holding_days >= max_hold:
                    should_close, action = True, "EXIT_TIME"

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
                trade = self.trades[-1]
                allocator.record_trade(pos.pod_id, trade.pnl, today, day_idx)
                allocator._allocations_cache = None

            # --- Entries ---
            for pid, z in pair_zscores.items():
                if circuit_breaker_until and day_idx < circuit_breaker_until:
                    break

                pair_info = valid_pairs[pid]
                pod_id = pair_info["pod_id"]

                if not allocator.is_pod_active(pod_id, day_idx):
                    continue
                if any(p.pair_id == pid for p in self.positions):
                    continue
                if abs(z) < self.entry_z:
                    continue

                # Dynamic sizing from allocator
                leg_size = allocator.get_leg_size(pod_id, self.default_leg_size, day_idx)

                # Risk cap
                max_loss = current_equity * self.max_risk_pct
                max_pair_notional = max_loss / (0.08 * self.gap_multiplier)
                leg_size = min(leg_size, max_pair_notional / 2, self.max_leg_size)

                # Gross check
                current_gross = sum(
                    pos.gross(
                        self.data.get_price_on(pos.long_ticker, today),
                        self.data.get_price_on(pos.short_ticker, today)
                    ) for pos in self.positions
                ) if self.positions else 0

                if current_gross + (leg_size * 2) > current_equity * self.max_gross_ratio:
                    continue
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
                    pair_id=pid, pod_id=pod_id,
                    long_ticker=pair_info["long"], short_ticker=pair_info["short"],
                    long_shares=long_shares, short_shares=short_shares,
                    long_entry_price=lp, short_entry_price=sp,
                    entry_date=today, entry_zscore=z,
                ))

            # --- Circuit breaker ---
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0
            if drawdown >= self.circuit_breaker_dd and not circuit_breaker_until:
                for pos in list(self.positions):
                    try:
                        lp = self.data.get_price_on(pos.long_ticker, today)
                        sp = self.data.get_price_on(pos.short_ticker, today)
                        pnl = pos.pnl(lp, sp)
                        hd = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(pos.entry_date, "%Y-%m-%d")).days
                        self.trades.append(TradeRecord(
                            pair_id=pos.pair_id, pod_id=pos.pod_id,
                            long_ticker=pos.long_ticker, short_ticker=pos.short_ticker,
                            action="CIRCUIT_BREAKER", entry_date=pos.entry_date, exit_date=today,
                            entry_zscore=pos.entry_zscore, exit_zscore=0,
                            long_notional=pos.long_notional, short_notional=pos.short_notional,
                            pnl=pnl, holding_days=hd,
                        ))
                        allocator.record_trade(pos.pod_id, pnl, today, day_idx)
                    except Exception:
                        pass
                self.positions.clear()
                circuit_breaker_until = day_idx + 10
                print(f"  [{today}] CIRCUIT BREAKER at {drawdown:.1%}")

            if circuit_breaker_until and day_idx >= circuit_breaker_until:
                peak_equity = current_equity
                circuit_breaker_until = None
                print(f"  [{today}] Resuming after circuit breaker. Peak reset to ${current_equity:,.0f}")

            # --- Mark to market ---
            unrealized = 0
            for pos in self.positions:
                try:
                    lp = self.data.get_price_on(pos.long_ticker, today)
                    sp = self.data.get_price_on(pos.short_ticker, today)
                    unrealized += pos.pnl(lp, sp)
                except Exception:
                    pass

            current_equity = self.capital + sum(t.pnl for t in self.trades) + unrealized
            peak_equity = max(peak_equity, current_equity)
            self.equity_curve.append((today, current_equity))

        # Close remaining
        if self.positions:
            last_day = trading_days[-1]
            for pos in list(self.positions):
                try:
                    lp = self.data.get_price_on(pos.long_ticker, last_day)
                    sp = self.data.get_price_on(pos.short_ticker, last_day)
                    pnl = pos.pnl(lp, sp)
                    hd = (datetime.strptime(last_day, "%Y-%m-%d") - datetime.strptime(pos.entry_date, "%Y-%m-%d")).days
                    self.trades.append(TradeRecord(
                        pair_id=pos.pair_id, pod_id=pos.pod_id,
                        long_ticker=pos.long_ticker, short_ticker=pos.short_ticker,
                        action="EOY_CLOSE", entry_date=pos.entry_date, exit_date=last_day,
                        entry_zscore=pos.entry_zscore, exit_zscore=0,
                        long_notional=pos.long_notional, short_notional=pos.short_notional,
                        pnl=pnl, holding_days=hd,
                    ))
                except Exception:
                    pass
            self.positions.clear()

        # Print allocator final state
        if allocator:
            print()
            print(allocator.summary(len(trading_days)))

    def _discover_pairs(self, as_of: str, pods: list[GeneratedPod]) -> dict:
        valid_pairs = {}
        for pod in pods:
            pod_pairs = []
            for lt in pod.long_tickers:
                for st in pod.short_tickers:
                    long_t = lt["ticker"]
                    short_t = st["ticker"]
                    try:
                        lp = self.data.get_prices_as_of(long_t, as_of, self.discovery_lookback)
                        sp = self.data.get_prices_as_of(short_t, as_of, self.discovery_lookback)
                    except Exception:
                        continue
                    n = min(len(lp), len(sp))
                    if n < 30:
                        continue
                    lp, sp = lp[-n:], sp[-n:]
                    pvalue, hedge_ratio, residuals = engle_granger_test(lp, sp)
                    if pvalue > self.coint_pvalue:
                        continue
                    hl = calculate_half_life(residuals)
                    if hl < self.min_half_life or hl > self.max_half_life:
                        continue
                    hurst = calculate_hurst(residuals, max_lag=min(20, len(residuals) // 2))
                    if hurst > self.hurst_threshold:
                        continue
                    score = (1 - pvalue) * 0.3 + (1 - hl / self.max_half_life) * 0.3 + (1 - hurst) * 0.4
                    pod_pairs.append({
                        "long": long_t, "short": short_t, "pod_id": pod.pod_id,
                        "hedge_ratio": hedge_ratio, "score": score,
                        "half_life": hl, "hurst": hurst,
                    })
            pod_pairs.sort(key=lambda x: x["score"], reverse=True)
            for p in pod_pairs[:self.max_pairs_per_pod]:
                pid = f"{pod.pod_id}:{p['long']}-{p['short']}"
                valid_pairs[pid] = p
        # Global trim with pod guarantee
        if len(valid_pairs) > self.max_total_pairs:
            all_sorted = sorted(valid_pairs.items(), key=lambda x: x[1]["score"], reverse=True)
            kept, pods_seen, remaining = {}, set(), []
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

    def report(self):
        """Reuse V1 report logic."""
        from backtest import Backtester
        # Monkey-patch the report
        fake = Backtester.__new__(Backtester)
        fake.capital = self.capital
        fake.equity_curve = self.equity_curve
        fake.trades = self.trades
        fake.positions = self.positions
        fake.daily_pnl = []
        fake.report()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
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
        print("Set POLYGON_API_KEY")
        sys.exit(1)

    config_path = Path(__file__).parent.parent / "config"
    with open(config_path / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    data = BacktestDataProvider(api_key)

    # Collect all tickers from all quarterly pods
    all_tickers = set()
    from pod_generator import CATALYSTS_2025
    for pods in CATALYSTS_2025.values():
        for pod in pods:
            for t in pod.long_tickers:
                all_tickers.add(t["ticker"])
            for t in pod.short_tickers:
                all_tickers.add(t["ticker"])

    all_tickers = sorted(all_tickers)
    print(f"Fetching {len(all_tickers)} tickers (2024-11-01 to 2025-12-31)...")
    fetched = 0
    for ticker in all_tickers:
        try:
            h = data.fetch_full_history(ticker, "2024-11-01", "2025-12-31")
            if h:
                fetched += 1
                if fetched % 20 == 0:
                    print(f"  {fetched}/{len(all_tickers)} fetched...")
        except Exception as e:
            print(f"  {ticker}: {e}")
    print(f"  {fetched}/{len(all_tickers)} tickers with data")
    print()

    bt = BacktesterV2(settings, data)
    bt.run("2025-01-02", "2025-12-31")
    print()
    bt.report()
