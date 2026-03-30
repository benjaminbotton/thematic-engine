"""
Engine — the main orchestrator that ties everything together.

This is the entry point. It runs the signal loop:
  1. Load pods and settings
  2. Fetch/update price data
  3. Discover or refresh pairs
  4. Update spreads and z-scores
  5. Generate signals
  6. Gate through RiskGuard
  7. Notify operator
  8. Execute approved trades
  9. Monitor positions
  10. EOD summary

Usage:
    # One-shot signal scan (for cron / scheduler):
    python engine.py scan

    # Interactive mode (manual control):
    python engine.py interactive

    # Fetch data only (pre-market):
    python engine.py fetch

    # Show status:
    python engine.py status
"""

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pod_manager import PodManager
from pairs_engine import PairsEngine, PairSignal
from risk_guard import RiskGuard, PortfolioState
from notifier import Notifier


def load_config():
    """Load settings and initialize all components."""
    config_path = Path(__file__).parent.parent / "config"
    with open(config_path / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    return config_path, settings


class DirectPolygonProvider:
    """Fetches prices directly from Polygon REST API — no SQLite.
    Uses in-memory cache so each ticker is only fetched once per session."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: dict[str, np.ndarray] = {}

    def get_prices(self, ticker: str, days: int, **kw) -> np.ndarray:
        cache_key = f"{ticker}:{days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        import requests as req
        from_date = (date.today() - timedelta(days=days + 10)).isoformat()
        to_date = date.today().isoformat()

        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 5000}
        resp = req.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("resultsCount", 0) == 0:
            raise ValueError(f"No data for {ticker}")

        closes = np.array([bar["c"] for bar in data["results"]], dtype=float)
        result = closes[-days:] if len(closes) > days else closes
        self._cache[cache_key] = result
        return result


def init_data_provider(settings: dict, config_path: Path, cache_only: bool = False):
    """Initialize data provider — Polygon if key available, else mock.
    Uses SQLite-backed provider locally, direct API on cloud (no db file)."""
    api_key = os.environ.get("POLYGON_API_KEY")
    db_path = config_path.parent / settings["data"]["db_path"]

    if api_key:
        # Use SQLite provider if db exists (local), direct API if not (cloud/CI)
        if db_path.exists():
            from data_provider import PolygonDataProvider
            provider = PolygonDataProvider(api_key=api_key, db_path=str(db_path))
            if cache_only:
                class CacheOnlyWrapper:
                    def __init__(self, p): self._p = p
                    def get_prices(self, ticker, days, **kw):
                        return self._p.get_prices(ticker, days, cache_only=True)
                return CacheOnlyWrapper(provider)
            return provider
        else:
            print("  No local database — fetching directly from Polygon API")
            return DirectPolygonProvider(api_key)
    else:
        print("  No POLYGON_API_KEY set. Using mock data provider.")
        from pairs_engine import MockDataProvider
        mock = MockDataProvider(seed=42)
        # Create some cointegrated pairs for testing
        mock.create_cointegrated_pair("CROX", "CLF", theta=0.04, spread_vol=0.6)
        mock.create_cointegrated_pair("CAL", "STLD", theta=0.05, spread_vol=0.7)
        mock.create_cointegrated_pair("FIVE", "CENX", theta=0.03, spread_vol=0.5)
        mock.create_cointegrated_pair("VRT", "AI", theta=0.04, spread_vol=0.6)
        mock.create_cointegrated_pair("POWL", "SOUN", theta=0.05, spread_vol=0.5)
        mock.create_cointegrated_pair("RH", "DG", theta=0.04, spread_vol=0.5)
        mock.create_cointegrated_pair("BIRK", "DLTR", theta=0.05, spread_vol=0.6)
        return mock


def get_portfolio_state(settings: dict) -> PortfolioState:
    """
    Build portfolio state from Alpaca or from local tracking.
    Falls back to defaults if Alpaca isn't connected.
    """
    capital = settings.get("capital_base", 30000)

    alpaca_key = os.environ.get("ALPACA_API_KEY")
    if alpaca_key:
        try:
            from executor import AlpacaExecutor
            paper = settings.get("brokerage", {}).get("environment", "paper") == "paper"
            executor = AlpacaExecutor(paper=paper)
            acct = executor.get_account()
            positions = executor.get_positions()
            equity = float(acct["equity"])
            gross = sum(abs(p.market_value) for p in positions)
            long_val = sum(p.market_value for p in positions if p.side == "long")
            short_val = sum(abs(p.market_value) for p in positions if p.side == "short")
            net = long_val - short_val
            pnl = sum(p.unrealized_pnl for p in positions)
            return PortfolioState(
                capital=capital,
                peak_equity=max(equity, capital),
                current_equity=equity,
                gross_exposure=gross,
                net_exposure=net,
                open_pair_count=len(positions) // 2,
                daily_pnl=pnl,    # approximate
                weekly_pnl=pnl,   # would need historical tracking
                monthly_pnl=pnl,
            )
        except Exception as e:
            print(f"  Could not connect to Alpaca: {e}")

    # Default state (no positions)
    return PortfolioState(
        capital=capital,
        peak_equity=capital,
        current_equity=capital,
        gross_exposure=0,
        net_exposure=0,
        open_pair_count=0,
        daily_pnl=0,
        weekly_pnl=0,
        monthly_pnl=0,
    )


# ===========================================================================
# Commands
# ===========================================================================

def cmd_status():
    """Show current system status."""
    config_path, settings = load_config()
    pm = PodManager(config_path)
    data = init_data_provider(settings, config_path, cache_only=True)
    guard = RiskGuard(settings)
    portfolio = get_portfolio_state(settings)

    engine = PairsEngine(pm, settings, data)
    engine.discover_pairs()
    engine.update_spreads()

    print("=" * 60)
    print("THEMATIC CATALYST L/S ENGINE — STATUS")
    print("=" * 60)
    print()
    print(pm.summary())
    print()
    print(guard.summary(portfolio))
    print()
    print(engine.summary())


def cmd_fetch():
    """Fetch fresh price data for all tickers."""
    config_path, settings = load_config()
    pm = PodManager(config_path)

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Set POLYGON_API_KEY to fetch real data.")
        return

    from data_provider import PolygonDataProvider
    db_path = config_path.parent / settings["data"]["db_path"]
    provider = PolygonDataProvider(api_key=api_key, db_path=str(db_path))

    tickers = sorted(pm.all_tickers())
    print(f"Fetching {len(tickers)} tickers...")
    provider.bulk_fetch(tickers, days=90)

    stats = provider.cache_stats()
    print(f"Cache: {stats['tickers']} tickers, {stats['rows']} rows")
    print(f"Range: {stats['oldest_date']} to {stats['latest_date']}")


def init_executor(settings: dict):
    """Initialize Alpaca executor if keys are available."""
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    if not alpaca_key:
        return None
    try:
        from executor import AlpacaExecutor
        paper = settings.get("brokerage", {}).get("environment", "paper") == "paper"
        return AlpacaExecutor(paper=paper)
    except Exception as e:
        print(f"  Could not init executor: {e}")
        return None


def cmd_scan():
    """
    Run a full signal scan cycle with AUTO-EXECUTION.
    Flow: discover pairs -> generate signals -> gate through RiskGuard -> execute on Alpaca -> notify.
    No human approval needed. Risk rules are the gatekeeper.
    """
    config_path, settings = load_config()
    pm = PodManager(config_path)
    data = init_data_provider(settings, config_path, cache_only=True)
    guard = RiskGuard(settings)
    notifier = Notifier(method="auto")
    portfolio = get_portfolio_state(settings)
    executor = init_executor(settings)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Signal scan starting (auto-execute mode)...")

    # Discover / refresh pairs
    engine = PairsEngine(pm, settings, data)
    all_scores = engine.discover_pairs()
    valid_total = sum(len([s for s in scores if s.is_valid]) for scores in all_scores.values())
    kept = len(engine.valid_pairs)
    print(f"  Pairs: {valid_total} valid found, {kept} kept after trimming")

    # Update spreads — use live prices from Alpaca if available (intraday)
    live_prices = {}
    if executor:
        try:
            all_tickers = list(pm.all_tickers())
            live_prices = executor.get_live_prices(all_tickers)
            if live_prices:
                print(f"  Live prices: {len(live_prices)} tickers from Alpaca")
        except Exception as e:
            print(f"  Live prices unavailable: {e}")
    engine.update_spreads(live_prices=live_prices if live_prices else None)

    # Generate signals
    signals = engine.generate_signals()
    print(f"  Signals: {len(signals)} generated")

    if not signals:
        print("  No actionable signals at current z-score levels.")
        # Still check for circuit breaker on existing positions
        _check_circuit_breaker(guard, portfolio, executor, notifier)
        # EOD summary
        _send_eod_summary(portfolio, notifier)
        return

    # Gate each signal through RiskGuard, then auto-execute
    for sig in signals:
        is_near_close = datetime.now().hour >= 15 and datetime.now().minute >= 45
        stop_pct = 0.08  # 8% approximation for pair spread stop

        verdict = guard.check_signal(
            long_size=sig.long_size,
            short_size=sig.short_size,
            stop_distance_pct=stop_pct,
            portfolio=portfolio,
            pod_id=sig.pod_id,
            pair_last_tested=datetime.now(),
            is_near_close=is_near_close,
        )

        if not verdict.approved:
            print(f"  ✗ Signal blocked: {sig.long_ticker}/{sig.short_ticker} — {verdict.reason}")
            continue

        # Use adjusted sizes from RiskGuard
        long_size = verdict.adjusted_long_size or sig.long_size
        short_size = verdict.adjusted_short_size or sig.short_size

        pod = pm.get_pod(sig.pod_id)
        pod_name = pod.name if pod else sig.pod_id

        # --- AUTO-EXECUTE ---
        if executor:
            pair_id = f"{sig.pod_id}:{sig.long_ticker}-{sig.short_ticker}"

            if sig.action.startswith("ENTER"):
                print(f"  → Executing ENTRY: Long {sig.long_ticker} ${long_size:,.0f} / Short {sig.short_ticker} ${short_size:,.0f}")
                result = executor.enter_pair(
                    long_ticker=sig.long_ticker,
                    short_ticker=sig.short_ticker,
                    long_dollars=long_size,
                    short_dollars=short_size,
                    pair_id=pair_id,
                )
                if result.success:
                    print(f"  ✓ FILLED: {sig.long_ticker} {result.long_shares}sh @ ${result.long_fill_price:.2f}"
                          f" / {sig.short_ticker} {result.short_shares}sh @ ${result.short_fill_price:.2f}")
                    # Update portfolio state for subsequent signals
                    portfolio.gross_exposure += long_size + short_size
                    portfolio.open_pair_count += 1
                else:
                    print(f"  ✗ EXECUTION FAILED: {result.error}")

                # Notify what happened (confirmation, not request)
                action_result = "FILLED" if result.success else f"FAILED: {result.error}"
                msg = notifier.format_pair_signal(
                    action=sig.action,
                    long_ticker=sig.long_ticker,
                    short_ticker=sig.short_ticker,
                    pod_name=pod_name,
                    zscore=sig.zscore,
                    long_size=long_size,
                    short_size=short_size,
                    confidence=sig.confidence,
                    reason=f"{sig.reason} | RESULT: {action_result}",
                    risk_warnings=verdict.warnings,
                )
                notifier.send(msg)

            elif sig.action in ("EXIT", "STOP"):
                # Find current position sizes from Alpaca
                long_pos = executor.get_position(sig.long_ticker)
                short_pos = executor.get_position(sig.short_ticker)

                long_shares = long_pos.qty if long_pos else 0
                short_shares = short_pos.qty if short_pos else 0

                if long_shares > 0 or short_shares > 0:
                    print(f"  → Executing {sig.action}: Close {sig.long_ticker}/{sig.short_ticker}")
                    result = executor.exit_pair(
                        long_ticker=sig.long_ticker,
                        short_ticker=sig.short_ticker,
                        long_shares=long_shares,
                        short_shares=short_shares,
                        pair_id=pair_id,
                    )
                    if result.success:
                        print(f"  ✓ CLOSED: {sig.long_ticker}/{sig.short_ticker}")
                    else:
                        print(f"  ✗ EXIT FAILED: {result.error}")

                    msg = notifier.format_pair_signal(
                        action=sig.action,
                        long_ticker=sig.long_ticker,
                        short_ticker=sig.short_ticker,
                        pod_name=pod_name,
                        zscore=sig.zscore,
                        long_size=long_size,
                        short_size=short_size,
                        confidence=sig.confidence,
                        reason=f"{sig.reason} | {'CLOSED' if result.success else 'EXIT FAILED: ' + result.error}",
                        risk_warnings=verdict.warnings,
                    )
                    notifier.send(msg)
                else:
                    print(f"  ⊘ {sig.action} signal for {sig.long_ticker}/{sig.short_ticker} but no position found")
        else:
            # No executor — notify only
            msg = notifier.format_pair_signal(
                action=sig.action,
                long_ticker=sig.long_ticker,
                short_ticker=sig.short_ticker,
                pod_name=pod_name,
                zscore=sig.zscore,
                long_size=long_size,
                short_size=short_size,
                confidence=sig.confidence,
                reason=sig.reason + " | NO EXECUTOR — notification only",
                risk_warnings=verdict.warnings,
            )
            notifier.send(msg)
            print(f"  ⚠ No executor connected. Signal sent as notification only.")

    # Check circuit breaker after trades
    _check_circuit_breaker(guard, portfolio, executor, notifier)
    # EOD summary
    _send_eod_summary(portfolio, notifier)


def _check_circuit_breaker(guard, portfolio, executor, notifier):
    """Check if circuit breaker should trigger and flatten everything."""
    capital = portfolio.capital
    drawdown = (portfolio.peak_equity - portfolio.current_equity) / portfolio.peak_equity
    cb_threshold = guard.settings.get("hard_rules", {}).get("circuit_breaker_drawdown", 0.10)

    if drawdown >= cb_threshold and executor:
        print(f"  🚨 CIRCUIT BREAKER: Drawdown {drawdown:.1%} >= {cb_threshold:.0%}")
        results = executor.flatten_all()
        closed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        print(f"  Flattened: {closed} closed, {failed} failed")

        cooloff = guard.settings.get("hard_rules", {}).get("circuit_breaker_cooloff_hours", 48)
        msg = notifier.format_circuit_breaker(drawdown_pct=drawdown, cooloff_hours=cooloff)
        notifier.send(msg)


def _send_eod_summary(portfolio, notifier):
    """Send EOD summary if near market close."""
    if datetime.now().hour >= 15:
        msg = notifier.format_daily_summary(
            equity=portfolio.current_equity,
            daily_pnl=portfolio.daily_pnl,
            open_pairs=portfolio.open_pair_count,
            gross_exposure=portfolio.gross_exposure,
            net_exposure=portfolio.net_exposure,
        )
        notifier.send(msg)


def cmd_account():
    """Show Alpaca paper account status."""
    _, settings = load_config()
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    if not alpaca_key:
        print("Set ALPACA_API_KEY to connect to Alpaca.")
        return
    from executor import AlpacaExecutor
    paper = settings.get("brokerage", {}).get("environment", "paper") == "paper"
    ex = AlpacaExecutor(paper=paper)
    print(ex.account_summary())


def cmd_interactive():
    """Interactive mode — manual control."""
    config_path, settings = load_config()
    pm = PodManager(config_path)
    data = init_data_provider(settings, config_path, cache_only=True)
    guard = RiskGuard(settings)
    portfolio = get_portfolio_state(settings)
    engine = PairsEngine(pm, settings, data)

    print("=" * 60)
    print("THEMATIC CATALYST L/S ENGINE — INTERACTIVE MODE")
    print("=" * 60)
    print()
    print("Commands:")
    print("  discover  — run pair discovery")
    print("  scan      — generate signals")
    print("  status    — show current state")
    print("  risk      — show risk dashboard")
    print("  pod <id>  — show pod detail")
    print("  pairs     — show active pairs")
    print("  quit      — exit")
    print()

    while True:
        try:
            cmd = input("engine> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd:
            continue
        elif cmd == "quit":
            break
        elif cmd == "discover":
            all_scores = engine.discover_pairs()
            for pod_id, scores in all_scores.items():
                valid = [s for s in scores if s.is_valid]
                print(f"  [{pod_id}] {len(valid)} valid pairs")
            print(f"  Kept: {len(engine.valid_pairs)} after trimming")
        elif cmd == "scan":
            engine.update_spreads()
            signals = engine.generate_signals()
            if signals:
                for s in signals:
                    print(f"  {s.action}: {s.long_ticker}/{s.short_ticker} z={s.zscore:+.2f}")
            else:
                print("  No signals.")
        elif cmd == "status":
            print(pm.summary())
        elif cmd == "risk":
            print(guard.summary(portfolio))
        elif cmd.startswith("pod "):
            pod_id = cmd.split(None, 1)[1]
            print(pm.pod_detail(pod_id))
        elif cmd == "pairs":
            print(engine.summary())
        else:
            print(f"  Unknown command: {cmd}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python engine.py <command>")
        print("  scan        — run a signal scan cycle")
        print("  status      — show system status")
        print("  fetch       — fetch fresh price data from Polygon")
        print("  account     — show Alpaca paper account status")
        print("  interactive — manual control mode")
        sys.exit(0)

    command = sys.argv[1].lower()
    if command == "scan":
        cmd_scan()
    elif command == "status":
        cmd_status()
    elif command == "fetch":
        cmd_fetch()
    elif command == "account":
        cmd_account()
    elif command == "interactive":
        cmd_interactive()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
