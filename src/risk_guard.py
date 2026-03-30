"""
RiskGuard — the gatekeeper between signals and execution.

Every signal must pass through RiskGuard before it can become an order.
The hard rules are inviolable. No thesis, no conviction, no edge overrides them.

The five rules:
  1. Never risk more than 1-2% of capital on a single trade
  2. Never use leverage where a 5% correlated move is fatal
  3. Always assume a strategy can stop working
  4. Always assume a market can gap overnight
  5. Always assume "once in 10 years" happens every few years
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Optional

import yaml


@dataclass
class PortfolioState:
    """Snapshot of the current portfolio for risk calculations."""
    capital: float
    peak_equity: float
    current_equity: float
    gross_exposure: float       # sum of all |leg notionals|
    net_exposure: float         # sum of signed leg notionals
    open_pair_count: int
    daily_pnl: float            # today's realized + unrealized
    weekly_pnl: float           # rolling 5-day
    monthly_pnl: float          # rolling 21-day
    pair_losses_streak: dict[str, int] = field(default_factory=dict)  # pod_id -> consecutive losses
    last_circuit_breaker: Optional[datetime] = None


@dataclass
class RiskVerdict:
    """Result of a risk check. If not approved, the signal is blocked."""
    approved: bool
    reason: str
    adjusted_long_size: Optional[float] = None   # may reduce size
    adjusted_short_size: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


class RiskGuard:
    def __init__(self, settings: dict):
        self.settings = settings
        self.hard = settings.get("hard_rules", {})
        self.capital = settings.get("capital_base", 30000)

        # Rule 1: Per-trade risk
        self.max_risk_pct = self.hard.get("max_risk_per_trade_pct", 0.015)
        self.absolute_max_loss = self.hard.get("absolute_max_loss_per_pair", 600)

        # Rule 2: Leverage
        self.max_gross_ratio = self.hard.get("max_gross_to_capital_ratio", 1.20)

        # Rule 3: Strategy decay
        self.max_pair_age_days = self.hard.get("max_pair_age_without_retest_days", 7)
        self.max_consecutive_losses = self.hard.get("max_consecutive_pair_losses", 3)

        # Rule 4: Gap risk
        self.gap_multiplier = self.hard.get("gap_multiplier", 2.0)
        self.max_overnight_gross = self.hard.get("max_overnight_gross_exposure", 24000)
        self.reduce_before = self.hard.get("reduce_to_overnight_before", "15:45")

        # Rule 5: Tail risk
        self.tail_multiplier = self.hard.get("tail_risk_multiplier", 2.0)
        self.circuit_breaker_dd = self.hard.get("circuit_breaker_drawdown", 0.10)
        self.cooloff_hours = self.hard.get("circuit_breaker_cooloff_hours", 48)
        self.weekly_loss_limit = self.hard.get("weekly_loss_limit", 0.05)
        self.monthly_loss_limit = self.hard.get("monthly_loss_limit", 0.08)

    # ------------------------------------------------------------------
    # Main entry point: check a signal before execution
    # ------------------------------------------------------------------

    def check_signal(
        self,
        long_size: float,
        short_size: float,
        stop_distance_pct: float,
        portfolio: PortfolioState,
        pod_id: str,
        pair_last_tested: Optional[datetime] = None,
        is_near_close: bool = False,
    ) -> RiskVerdict:
        """
        Gate a pair trade signal through all hard rules.
        Returns a RiskVerdict — either approved (possibly with adjusted sizes)
        or blocked with a reason.
        """
        warnings = []

        # === RULE 5 CHECK FIRST: Circuit breaker / loss limits ===
        # These block ALL new trades, so check them before anything else.

        verdict = self._check_circuit_breaker(portfolio)
        if not verdict.approved:
            return verdict

        verdict = self._check_loss_limits(portfolio)
        if not verdict.approved:
            return verdict

        # === RULE 3: Strategy decay ===

        verdict = self._check_strategy_decay(pod_id, pair_last_tested, portfolio)
        if not verdict.approved:
            return verdict

        # === RULE 1: Per-trade risk ===

        pair_notional = long_size + short_size
        max_loss_from_capital = self.capital * self.max_risk_pct
        max_loss = min(max_loss_from_capital, self.absolute_max_loss)

        # Gap-adjusted risk (Rule 4): actual loss could be 2x stop distance
        gap_adjusted_stop = stop_distance_pct * self.gap_multiplier
        projected_loss = pair_notional * gap_adjusted_stop

        if projected_loss > max_loss:
            # Reduce size to fit within risk budget
            safe_notional = max_loss / gap_adjusted_stop
            ratio = safe_notional / pair_notional
            adjusted_long = long_size * ratio
            adjusted_short = short_size * ratio
            warnings.append(
                f"Rule 1+4: Sized down from ${pair_notional:,.0f} to ${safe_notional:,.0f} "
                f"(gap-adjusted loss ${projected_loss:,.0f} > max ${max_loss:,.0f})"
            )
            long_size = adjusted_long
            short_size = adjusted_short

        # === RULE 2: Leverage check ===

        new_gross = portfolio.gross_exposure + long_size + short_size
        max_gross = self.capital * self.max_gross_ratio

        if new_gross > max_gross:
            available = max_gross - portfolio.gross_exposure
            if available <= 0:
                return RiskVerdict(
                    approved=False,
                    reason=f"Rule 2: Gross exposure ${portfolio.gross_exposure:,.0f} "
                           f"already at limit ${max_gross:,.0f}. No room for new pairs.",
                )
            # Shrink to fit
            ratio = available / (long_size + short_size)
            long_size *= ratio
            short_size *= ratio
            warnings.append(
                f"Rule 2: Sized down to fit gross limit. "
                f"Available: ${available:,.0f}"
            )

        # Net exposure check
        pairs_risk = self.settings.get("pairs", {}).get("risk", {})
        max_net = pairs_risk.get("max_net_exposure", 3000)
        new_net = portfolio.net_exposure + (long_size - short_size)
        if abs(new_net) > max_net:
            warnings.append(
                f"Net exposure would be ${new_net:,.0f} (limit ±${max_net:,.0f}). "
                f"Consider rebalancing."
            )

        # === RULE 4: Overnight exposure ===

        if is_near_close:
            new_overnight_gross = portfolio.gross_exposure + long_size + short_size
            if new_overnight_gross > self.max_overnight_gross:
                return RiskVerdict(
                    approved=False,
                    reason=f"Rule 4: Near market close. New overnight gross "
                           f"${new_overnight_gross:,.0f} > limit ${self.max_overnight_gross:,.0f}. "
                           f"No new entries before close.",
                )

        # === TAIL RISK SANITY CHECK (Rule 5) ===
        # What if ALL open pairs move against us at 2x modeled risk simultaneously?

        tail_loss = self._estimate_tail_loss(portfolio, long_size + short_size, gap_adjusted_stop)
        tail_loss_pct = tail_loss / self.capital
        if tail_loss_pct > 0.15:  # 15% tail scenario = too concentrated
            warnings.append(
                f"Rule 5: Tail scenario loss ${tail_loss:,.0f} ({tail_loss_pct:.1%} of capital). "
                f"Portfolio may be too concentrated."
            )

        # All checks passed
        return RiskVerdict(
            approved=True,
            reason="All hard rules passed.",
            adjusted_long_size=long_size,
            adjusted_short_size=short_size,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    def _check_circuit_breaker(self, portfolio: PortfolioState) -> RiskVerdict:
        """Rule 5: Circuit breaker — drawdown from peak."""
        if portfolio.peak_equity <= 0:
            return RiskVerdict(approved=True, reason="")

        drawdown = (portfolio.peak_equity - portfolio.current_equity) / portfolio.peak_equity

        if drawdown >= self.circuit_breaker_dd:
            # Check if we're still in cooloff
            if portfolio.last_circuit_breaker:
                elapsed = datetime.now() - portfolio.last_circuit_breaker
                if elapsed < timedelta(hours=self.cooloff_hours):
                    remaining = timedelta(hours=self.cooloff_hours) - elapsed
                    return RiskVerdict(
                        approved=False,
                        reason=f"CIRCUIT BREAKER ACTIVE. Drawdown {drawdown:.1%} from peak. "
                               f"Cooloff: {remaining.total_seconds()/3600:.1f}h remaining. "
                               f"ALL positions should be flat.",
                    )

            return RiskVerdict(
                approved=False,
                reason=f"CIRCUIT BREAKER TRIGGERED. Drawdown {drawdown:.1%} >= {self.circuit_breaker_dd:.0%}. "
                       f"Flatten everything. {self.cooloff_hours}h cooloff starts now.",
            )

        return RiskVerdict(approved=True, reason="")

    def _check_loss_limits(self, portfolio: PortfolioState) -> RiskVerdict:
        """Rule 5: Daily, weekly, monthly loss limits."""
        daily_pct = abs(portfolio.daily_pnl) / self.capital if portfolio.daily_pnl < 0 else 0
        daily_limit = self.settings.get("risk", {}).get("daily_loss_limit", 0.025)
        if daily_pct >= daily_limit:
            return RiskVerdict(
                approved=False,
                reason=f"Daily loss limit hit: {daily_pct:.1%} >= {daily_limit:.1%}. "
                       f"No new trades until next session.",
            )

        weekly_pct = abs(portfolio.weekly_pnl) / self.capital if portfolio.weekly_pnl < 0 else 0
        if weekly_pct >= self.weekly_loss_limit:
            return RiskVerdict(
                approved=False,
                reason=f"Weekly loss limit hit: {weekly_pct:.1%} >= {self.weekly_loss_limit:.0%}. "
                       f"No new trades until next Monday.",
            )

        monthly_pct = abs(portfolio.monthly_pnl) / self.capital if portfolio.monthly_pnl < 0 else 0
        if monthly_pct >= self.monthly_loss_limit:
            return RiskVerdict(
                approved=False,
                reason=f"MONTHLY LOSS LIMIT HIT: {monthly_pct:.1%} >= {self.monthly_loss_limit:.0%}. "
                       f"Full stop. Review everything before resuming.",
            )

        return RiskVerdict(approved=True, reason="")

    def _check_strategy_decay(
        self,
        pod_id: str,
        pair_last_tested: Optional[datetime],
        portfolio: PortfolioState,
    ) -> RiskVerdict:
        """Rule 3: Has this pair's cointegration been validated recently?"""
        if pair_last_tested:
            age = (datetime.now() - pair_last_tested).days
            if age > self.max_pair_age_days:
                return RiskVerdict(
                    approved=False,
                    reason=f"Rule 3: Pair cointegration last tested {age}d ago "
                           f"(max {self.max_pair_age_days}d). Must retest before trading.",
                )

        # Check consecutive losses for this pod
        streak = portfolio.pair_losses_streak.get(pod_id, 0)
        if streak >= self.max_consecutive_losses:
            return RiskVerdict(
                approved=False,
                reason=f"Rule 3: Pod '{pod_id}' has {streak} consecutive pair losses "
                       f"(max {self.max_consecutive_losses}). Pod paused for review.",
            )

        return RiskVerdict(approved=True, reason="")

    def _estimate_tail_loss(
        self,
        portfolio: PortfolioState,
        new_pair_notional: float,
        gap_adjusted_stop: float,
    ) -> float:
        """
        Rule 5: Estimate worst-case loss if everything goes wrong simultaneously.
        Assumes all open positions hit gap-adjusted stops AND the new pair does too.
        Multiply by tail_risk_multiplier because reality is always worse than models.
        """
        # Existing exposure at gap-adjusted stop
        existing_risk = portfolio.gross_exposure * gap_adjusted_stop
        new_risk = new_pair_notional * gap_adjusted_stop
        total_risk = (existing_risk + new_risk) * self.tail_multiplier
        return total_risk

    # ------------------------------------------------------------------
    # Overnight risk check (run near market close)
    # ------------------------------------------------------------------

    def check_overnight_exposure(self, portfolio: PortfolioState) -> RiskVerdict:
        """
        Rule 4: Called near market close. Recommends position reduction
        if overnight gross exceeds the limit.
        """
        if portfolio.gross_exposure <= self.max_overnight_gross:
            return RiskVerdict(
                approved=True,
                reason=f"Overnight exposure OK: ${portfolio.gross_exposure:,.0f} "
                       f"<= ${self.max_overnight_gross:,.0f}",
            )

        excess = portfolio.gross_exposure - self.max_overnight_gross
        return RiskVerdict(
            approved=False,
            reason=f"Rule 4: Overnight gross ${portfolio.gross_exposure:,.0f} "
                   f"> limit ${self.max_overnight_gross:,.0f}. "
                   f"Reduce by ${excess:,.0f} before close.",
            warnings=[
                f"Consider closing the lowest-conviction pair(s) to reduce by ${excess:,.0f}."
            ],
        )

    # ------------------------------------------------------------------
    # Position sizing from risk budget (Rule 1 + Rule 4)
    # ------------------------------------------------------------------

    def max_pair_notional(self, stop_distance_pct: float) -> float:
        """
        Calculate the maximum pair notional allowed given a stop distance,
        accounting for gap risk. This is how you should size, not by
        picking a dollar amount and hoping the stop works.

        max_notional = max_loss / gap_adjusted_stop
        """
        max_loss = min(
            self.capital * self.max_risk_pct,
            self.absolute_max_loss,
        )
        gap_adjusted_stop = stop_distance_pct * self.gap_multiplier
        if gap_adjusted_stop <= 0:
            return 0
        return max_loss / gap_adjusted_stop

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, portfolio: PortfolioState) -> str:
        lines = ["=== RISK GUARD STATUS ==="]

        # Drawdown
        if portfolio.peak_equity > 0:
            dd = (portfolio.peak_equity - portfolio.current_equity) / portfolio.peak_equity
            dd_limit = self.circuit_breaker_dd
            dd_bar = "█" * int(dd / dd_limit * 20) + "░" * (20 - int(dd / dd_limit * 20))
            lines.append(f"  Drawdown:  [{dd_bar}] {dd:.1%} / {dd_limit:.0%}")

        # Gross exposure
        max_gross = self.capital * self.max_gross_ratio
        gross_pct = portfolio.gross_exposure / max_gross if max_gross > 0 else 0
        gross_bar = "█" * int(gross_pct * 20) + "░" * (20 - int(gross_pct * 20))
        lines.append(f"  Gross:     [{gross_bar}] ${portfolio.gross_exposure:,.0f} / ${max_gross:,.0f}")

        # Net exposure
        pairs_risk = self.settings.get("pairs", {}).get("risk", {})
        max_net = pairs_risk.get("max_net_exposure", 3000)
        net_pct = abs(portfolio.net_exposure) / max_net if max_net > 0 else 0
        net_bar = "█" * int(net_pct * 20) + "░" * (20 - int(net_pct * 20))
        lines.append(f"  Net:       [{net_bar}] ${portfolio.net_exposure:+,.0f} / ±${max_net:,.0f}")

        # Daily P&L
        daily_limit_pct = self.settings.get("risk", {}).get("daily_loss_limit", 0.025)
        daily_limit = self.capital * daily_limit_pct
        lines.append(f"  Daily P&L: ${portfolio.daily_pnl:+,.0f} (limit: -${daily_limit:,.0f})")

        # Risk budget per trade
        stop_example = 0.08  # 8% spread stop
        max_notional = self.max_pair_notional(stop_example)
        max_per_leg = max_notional / 2
        lines.append(
            f"  Risk budget: ${self.capital * self.max_risk_pct:,.0f}/trade → "
            f"max pair notional ${max_notional:,.0f} (${max_per_leg:,.0f}/leg) "
            f"@ {stop_example:.0%} stop"
        )

        # Overnight
        lines.append(f"  Overnight limit: ${self.max_overnight_gross:,.0f}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config"
    with open(config_path / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    guard = RiskGuard(settings)

    # Simulate a portfolio state
    portfolio = PortfolioState(
        capital=30000,
        peak_equity=30500,
        current_equity=29800,
        gross_exposure=20000,
        net_exposure=1200,
        open_pair_count=3,
        daily_pnl=-150,
        weekly_pnl=-400,
        monthly_pnl=-900,
    )

    print(guard.summary(portfolio))
    print()

    # Test a signal
    print("--- Testing a normal signal ---")
    verdict = guard.check_signal(
        long_size=2500, short_size=2500,
        stop_distance_pct=0.08,
        portfolio=portfolio,
        pod_id="tariff_scramble",
        pair_last_tested=datetime.now() - timedelta(days=3),
    )
    print(f"  Approved: {verdict.approved}")
    print(f"  Reason: {verdict.reason}")
    if verdict.adjusted_long_size:
        print(f"  Adjusted: L=${verdict.adjusted_long_size:,.0f} S=${verdict.adjusted_short_size:,.0f}")
    for w in verdict.warnings:
        print(f"  ⚠ {w}")

    # Test near close
    print("\n--- Testing near market close ---")
    verdict = guard.check_signal(
        long_size=2500, short_size=2500,
        stop_distance_pct=0.08,
        portfolio=PortfolioState(
            capital=30000, peak_equity=30000, current_equity=29900,
            gross_exposure=22000, net_exposure=500, open_pair_count=4,
            daily_pnl=-50, weekly_pnl=-100, monthly_pnl=-200,
        ),
        pod_id="ai_infra_vs_hype",
        pair_last_tested=datetime.now() - timedelta(days=1),
        is_near_close=True,
    )
    print(f"  Approved: {verdict.approved}")
    print(f"  Reason: {verdict.reason}")
    for w in verdict.warnings:
        print(f"  ⚠ {w}")

    # Test circuit breaker
    print("\n--- Testing circuit breaker ---")
    verdict = guard.check_signal(
        long_size=2500, short_size=2500,
        stop_distance_pct=0.08,
        portfolio=PortfolioState(
            capital=30000, peak_equity=32000, current_equity=28500,
            gross_exposure=15000, net_exposure=0, open_pair_count=2,
            daily_pnl=-800, weekly_pnl=-2000, monthly_pnl=-3500,
        ),
        pod_id="defense_upcycle",
        pair_last_tested=datetime.now(),
    )
    print(f"  Approved: {verdict.approved}")
    print(f"  Reason: {verdict.reason}")

    # Test stale pair
    print("\n--- Testing stale pair (no retest in 10 days) ---")
    verdict = guard.check_signal(
        long_size=2500, short_size=2500,
        stop_distance_pct=0.08,
        portfolio=portfolio,
        pod_id="tariff_scramble",
        pair_last_tested=datetime.now() - timedelta(days=10),
    )
    print(f"  Approved: {verdict.approved}")
    print(f"  Reason: {verdict.reason}")

    # Test consecutive losses
    print("\n--- Testing pod with 3 consecutive losses ---")
    bad_portfolio = PortfolioState(
        capital=30000, peak_equity=30000, current_equity=29500,
        gross_exposure=10000, net_exposure=0, open_pair_count=1,
        daily_pnl=-100, weekly_pnl=-500, monthly_pnl=-1000,
        pair_losses_streak={"k_shaped_consumer": 3},
    )
    verdict = guard.check_signal(
        long_size=2500, short_size=2500,
        stop_distance_pct=0.08,
        portfolio=bad_portfolio,
        pod_id="k_shaped_consumer",
        pair_last_tested=datetime.now(),
    )
    print(f"  Approved: {verdict.approved}")
    print(f"  Reason: {verdict.reason}")

    # Show risk budget at different stop distances
    print("\n--- Risk Budget by Stop Distance ---")
    for stop in [0.04, 0.06, 0.08, 0.10, 0.12]:
        gap_stop = stop * guard.gap_multiplier
        max_not = guard.max_pair_notional(stop)
        max_leg = max_not / 2
        print(f"  Stop {stop:.0%} (gap-adj {gap_stop:.0%}): "
              f"max pair ${max_not:,.0f} (${max_leg:,.0f}/leg)")
