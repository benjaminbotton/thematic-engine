"""
Pairs Engine — discovers cointegrated pairs within thematic pods and generates
spread-based trading signals.

The thesis selects the names. Statistics validate the relationship. The trade
is the spread. Dollar-neutral by default, with optional thesis-driven lean.

Data flow:
  PodManager.all_pair_candidates()
    -> discover_pairs()  [cointegration, scoring, filtering]
    -> update_spreads()  [refresh prices, z-scores, regime]
    -> generate_signals() [entry/exit/stop based on spread deviation]
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools import add_constant

import yaml

from pod_manager import Pod, PodManager, PodSpreadParams


# ---------------------------------------------------------------------------
# Data provider interface — anything that can fetch price arrays
# ---------------------------------------------------------------------------

class DataProvider(Protocol):
    def get_prices(self, ticker: str, days: int) -> np.ndarray:
        """Return array of closing prices, most recent last."""
        ...


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SpreadSeries:
    long_ticker: str
    short_ticker: str
    pod_id: str
    hedge_ratio: float
    spread_values: np.ndarray       # historical spread
    spread_mean: float
    spread_std: float
    current_zscore: float
    half_life: float
    hurst_exponent: float
    cointegration_pvalue: float
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def pair_id(self) -> str:
        return f"{self.pod_id}:{self.long_ticker}-{self.short_ticker}"


@dataclass
class PairScore:
    long_ticker: str
    short_ticker: str
    pod_id: str
    coint_pvalue: float
    correlation: float
    half_life: float
    hurst: float
    is_hinted: bool
    thesis_alignment: float     # 0-1 from pod confidence
    composite_score: float
    is_valid: bool
    rejection_reason: str = ""

    @property
    def pair_id(self) -> str:
        return f"{self.pod_id}:{self.long_ticker}-{self.short_ticker}"


@dataclass
class PairPosition:
    pair_id: str
    long_ticker: str
    short_ticker: str
    pod_id: str
    long_shares: int
    short_shares: int
    long_entry_price: float
    short_entry_price: float
    long_notional: float
    short_notional: float
    entry_zscore: float
    entry_time: datetime
    current_zscore: float = 0.0
    target_zscore: float = 0.5
    stop_zscore: float = 3.5
    lean_ratio: float = 0.50
    unrealized_pnl: float = 0.0
    status: str = "open"            # "open", "closing", "stopped"


@dataclass
class PairSignal:
    pair_id: str
    action: str     # "ENTER_LONG_SPREAD", "ENTER_SHORT_SPREAD", "EXIT", "STOP"
    zscore: float
    spread_value: float
    long_ticker: str
    short_ticker: str
    pod_id: str
    long_size: float    # dollar amount for long leg
    short_size: float   # dollar amount for short leg
    confidence: float   # 0-1
    reason: str


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def engle_granger_test(
    long_prices: np.ndarray,
    short_prices: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Engle-Granger cointegration test.
    Returns (p_value, hedge_ratio, residuals).
    """
    short_with_const = add_constant(short_prices)
    model = OLS(long_prices, short_with_const).fit()
    hedge_ratio = model.params[1]
    residuals = model.resid

    adf_result = adfuller(residuals, maxlag=10, autolag="AIC")
    p_value = adf_result[1]

    return p_value, hedge_ratio, residuals


def calculate_half_life(spread: np.ndarray) -> float:
    """
    Ornstein-Uhlenbeck half-life of mean reversion.
    Regress spread_change on spread_level.
    half_life = -ln(2) / coefficient.
    """
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    spread_lag_with_const = add_constant(spread_lag)
    model = OLS(spread_diff, spread_lag_with_const).fit()
    theta = model.params[1]

    if theta >= 0:
        return float("inf")  # not mean-reverting

    return -np.log(2) / theta


def calculate_hurst(spread: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent via variance ratio method.
    < 0.5 = mean-reverting, 0.5 = random walk, > 0.5 = trending.

    Uses the scaling of variance of increments: Var(X(t+tau) - X(t)) ~ tau^(2H).
    More robust than R/S on short series.
    """
    if len(spread) < 20:
        return 0.5

    lags = []
    variances = []
    for lag in range(2, min(max_lag + 1, len(spread) // 3)):
        diffs = spread[lag:] - spread[:-lag]
        if len(diffs) < 5:
            continue
        lags.append(lag)
        variances.append(np.var(diffs))

    if len(lags) < 3:
        return 0.5

    log_lags = np.log(np.array(lags, dtype=float))
    log_vars = np.log(np.array(variances, dtype=float))

    log_lags_with_const = add_constant(log_lags)
    model = OLS(log_vars, log_lags_with_const).fit()
    # Var ~ tau^(2H), so slope = 2H, H = slope/2
    return float(model.params[1] / 2.0)


def rolling_zscore(spread: np.ndarray, window: int) -> float:
    """Current z-score of spread using rolling mean/std."""
    if len(spread) < window:
        window = len(spread)
    recent = spread[-window:]
    mean = np.mean(recent)
    std = np.std(recent, ddof=1)
    if std < 1e-10:
        return 0.0
    return float((spread[-1] - mean) / std)


# ---------------------------------------------------------------------------
# Pairs Engine
# ---------------------------------------------------------------------------

CONFIDENCE_MAP = {"high": 1.0, "medium": 0.7, "low": 0.4, "stale": 0.2}
LEAN_MAP = {"high": 0.55, "medium": 0.52, "low": 0.50, "stale": 0.50}


class PairsEngine:
    def __init__(self, pod_manager: PodManager, settings: dict, data_provider: DataProvider):
        self.pm = pod_manager
        self.cfg = settings.get("pairs", {})
        self.data = data_provider

        # State
        self.valid_pairs: dict[str, SpreadSeries] = {}
        self.pair_scores: dict[str, PairScore] = {}
        self.positions: dict[str, PairPosition] = {}
        self._last_discovery: Optional[datetime] = None

        # Unpack config with defaults
        disc = self.cfg.get("discovery", {})
        self.lookback = disc.get("lookback_days", 60)
        self.min_corr = disc.get("min_correlation", 0.50)
        self.coint_pvalue = disc.get("cointegration_pvalue", 0.05)
        self.max_pairs_per_pod = disc.get("max_pairs_per_pod", 3)
        self.max_total_pairs = disc.get("max_total_pairs", 8)

        spread_cfg = self.cfg.get("spread", {})
        self.entry_z = spread_cfg.get("entry_zscore", 2.0)
        self.exit_z = spread_cfg.get("exit_zscore", 0.5)
        self.stop_z = spread_cfg.get("stop_zscore", 3.5)
        self.z_lookback = spread_cfg.get("zscore_lookback", 20)
        self.min_hl = spread_cfg.get("min_half_life", 3)
        self.max_hl = spread_cfg.get("max_half_life", 15)

        regime_cfg = self.cfg.get("regime", {})
        self.hurst_thresh = regime_cfg.get("hurst_threshold", 0.45)
        self.hurst_lookback = regime_cfg.get("hurst_lookback", 40)

        sizing_cfg = self.cfg.get("sizing", {})
        self.default_leg = sizing_cfg.get("default_leg_size", 3000)
        self.max_leg = sizing_cfg.get("max_leg_size", 4000)
        self.lean_enabled = sizing_cfg.get("lean_enabled", True)
        self.max_lean = sizing_cfg.get("max_lean_ratio", 0.55)

        risk_cfg = self.cfg.get("risk", {})
        self.max_concurrent = risk_cfg.get("max_concurrent_pairs", 5)
        self.max_gross = risk_cfg.get("max_gross_exposure", 45000)
        self.max_net = risk_cfg.get("max_net_exposure", 4000)
        self.max_pair_corr = risk_cfg.get("max_pair_correlation", 0.60)

    # ------------------------------------------------------------------
    # Pair discovery
    # ------------------------------------------------------------------

    def discover_pairs(self) -> dict[str, list[PairScore]]:
        """
        Test all pair candidates across active pods. Returns scored pairs
        grouped by pod_id. Updates self.valid_pairs and self.pair_scores.
        """
        all_scores: dict[str, list[PairScore]] = {}
        self.valid_pairs.clear()
        self.pair_scores.clear()

        for pod in self.pm.active_pods():
            pod_scores = []
            sp = pod.spread_params or PodSpreadParams()
            min_corr = sp.min_correlation
            coint_p = sp.cointegration_pvalue
            lookback = sp.lookback_days

            for long_entry, short_entry in pod.pair_candidates:
                lt, st = long_entry.ticker, short_entry.ticker

                # Skip excluded
                if pod.is_pair_excluded(lt, st):
                    continue

                # Fetch prices
                try:
                    long_prices = self.data.get_prices(lt, lookback)
                    short_prices = self.data.get_prices(st, lookback)
                except Exception:
                    continue

                if len(long_prices) < 30 or len(short_prices) < 30:
                    continue

                # Trim to same length
                n = min(len(long_prices), len(short_prices))
                long_prices = long_prices[-n:]
                short_prices = short_prices[-n:]

                # Pre-filter: correlation
                corr = float(np.corrcoef(long_prices, short_prices)[0, 1])
                if abs(corr) < min_corr:
                    pod_scores.append(PairScore(
                        long_ticker=lt, short_ticker=st, pod_id=pod.pod_id,
                        coint_pvalue=1.0, correlation=corr, half_life=0,
                        hurst=0.5, is_hinted=pod.is_pair_hinted(lt, st),
                        thesis_alignment=0, composite_score=0,
                        is_valid=False,
                        rejection_reason=f"correlation {corr:.2f} < {min_corr}",
                    ))
                    continue

                # Cointegration test
                try:
                    pvalue, hedge_ratio, residuals = engle_granger_test(long_prices, short_prices)
                except Exception:
                    continue

                if pvalue > coint_p:
                    pod_scores.append(PairScore(
                        long_ticker=lt, short_ticker=st, pod_id=pod.pod_id,
                        coint_pvalue=pvalue, correlation=corr, half_life=0,
                        hurst=0.5, is_hinted=pod.is_pair_hinted(lt, st),
                        thesis_alignment=0, composite_score=0,
                        is_valid=False,
                        rejection_reason=f"coint p={pvalue:.3f} > {coint_p}",
                    ))
                    continue

                # Half-life
                hl = calculate_half_life(residuals)
                if hl < self.min_hl or hl > self.max_hl:
                    pod_scores.append(PairScore(
                        long_ticker=lt, short_ticker=st, pod_id=pod.pod_id,
                        coint_pvalue=pvalue, correlation=corr, half_life=hl,
                        hurst=0.5, is_hinted=pod.is_pair_hinted(lt, st),
                        thesis_alignment=0, composite_score=0,
                        is_valid=False,
                        rejection_reason=f"half_life {hl:.1f}d outside [{self.min_hl}, {self.max_hl}]",
                    ))
                    continue

                # Hurst exponent
                hurst = calculate_hurst(residuals, max_lag=self.hurst_lookback)

                # Thesis alignment
                thesis_score = CONFIDENCE_MAP.get(pod.thesis.confidence, 0.5)

                # Composite score
                # Weights: coint 30%, half-life 25%, hurst 20%, thesis 15%, hint 10%
                coint_component = max(0, (coint_p - pvalue) / coint_p)  # 0-1, higher = better
                hl_ideal = (self.min_hl + self.max_hl) / 2
                hl_component = max(0, 1 - abs(hl - hl_ideal) / hl_ideal)
                hurst_component = max(0, (self.hurst_thresh - hurst) / self.hurst_thresh) if hurst < 0.5 else 0
                hint_bonus = 1.0 if pod.is_pair_hinted(lt, st) else 0.0

                composite = (
                    0.30 * coint_component
                    + 0.25 * hl_component
                    + 0.20 * hurst_component
                    + 0.15 * thesis_score
                    + 0.10 * hint_bonus
                )

                is_valid = hurst < 0.5  # must be mean-reverting

                score = PairScore(
                    long_ticker=lt, short_ticker=st, pod_id=pod.pod_id,
                    coint_pvalue=pvalue, correlation=corr, half_life=hl,
                    hurst=hurst, is_hinted=pod.is_pair_hinted(lt, st),
                    thesis_alignment=thesis_score, composite_score=composite,
                    is_valid=is_valid,
                    rejection_reason="" if is_valid else f"hurst {hurst:.2f} >= 0.5 (trending)",
                )
                pod_scores.append(score)

                if is_valid:
                    # Build SpreadSeries
                    spread = long_prices - hedge_ratio * short_prices
                    z = rolling_zscore(spread, self.z_lookback)
                    ss = SpreadSeries(
                        long_ticker=lt, short_ticker=st, pod_id=pod.pod_id,
                        hedge_ratio=hedge_ratio, spread_values=spread,
                        spread_mean=float(np.mean(spread[-self.z_lookback:])),
                        spread_std=float(np.std(spread[-self.z_lookback:], ddof=1)),
                        current_zscore=z, half_life=hl, hurst_exponent=hurst,
                        cointegration_pvalue=pvalue,
                    )
                    self.pair_scores[score.pair_id] = score
                    self.valid_pairs[ss.pair_id] = ss

            all_scores[pod.pod_id] = pod_scores

        # Trim to max_pairs_per_pod and max_total_pairs
        self._trim_pairs()
        self._last_discovery = datetime.now()
        return all_scores

    def _trim_pairs(self):
        """Keep top pairs per pod, then globally."""
        # Group by pod
        by_pod: dict[str, list[str]] = {}
        for pid, ss in self.valid_pairs.items():
            by_pod.setdefault(ss.pod_id, []).append(pid)

        # Keep top N per pod by composite score
        kept = []
        for pod_id, pair_ids in by_pod.items():
            scored = [(pid, self.pair_scores[pid].composite_score) for pid in pair_ids]
            scored.sort(key=lambda x: x[1], reverse=True)
            kept.extend([pid for pid, _ in scored[: self.max_pairs_per_pod]])

        # Trim globally — guarantee at least 1 pair per pod for diversification
        if len(kept) > self.max_total_pairs:
            # First: take the best pair from each pod
            guaranteed = []
            remaining = []
            pods_seen = set()
            all_scored = [(pid, self.pair_scores[pid].composite_score) for pid in kept]
            all_scored.sort(key=lambda x: x[1], reverse=True)
            for pid, score in all_scored:
                pod = self.valid_pairs[pid].pod_id
                if pod not in pods_seen:
                    guaranteed.append(pid)
                    pods_seen.add(pod)
                else:
                    remaining.append((pid, score))
            # Then: fill remaining slots by score
            remaining.sort(key=lambda x: x[1], reverse=True)
            slots_left = self.max_total_pairs - len(guaranteed)
            kept = guaranteed + [pid for pid, _ in remaining[:slots_left]]

        # Remove pairs not in kept
        kept_set = set(kept)
        self.valid_pairs = {k: v for k, v in self.valid_pairs.items() if k in kept_set}
        self.pair_scores = {k: v for k, v in self.pair_scores.items() if k in kept_set}

    # ------------------------------------------------------------------
    # Spread updates
    # ------------------------------------------------------------------

    def update_spreads(self, live_prices: dict[str, float] = None):
        """Refresh prices and z-scores for all valid pairs.

        If live_prices is provided (ticker -> current price), appends today's
        live price to the historical series for intraday z-score updates.
        """
        for pid, ss in list(self.valid_pairs.items()):
            try:
                long_prices = self.data.get_prices(ss.long_ticker, self.z_lookback + 5)
                short_prices = self.data.get_prices(ss.short_ticker, self.z_lookback + 5)
            except Exception:
                continue

            # Append live prices if available (intraday update)
            if live_prices:
                lp_live = live_prices.get(ss.long_ticker)
                sp_live = live_prices.get(ss.short_ticker)
                if lp_live and sp_live:
                    long_prices = np.append(long_prices, lp_live)
                    short_prices = np.append(short_prices, sp_live)

            n = min(len(long_prices), len(short_prices))
            long_prices = long_prices[-n:]
            short_prices = short_prices[-n:]

            spread = long_prices - ss.hedge_ratio * short_prices
            ss.spread_values = spread
            ss.spread_mean = float(np.mean(spread[-self.z_lookback:]))
            ss.spread_std = float(np.std(spread[-self.z_lookback:], ddof=1))
            ss.current_zscore = rolling_zscore(spread, self.z_lookback)
            ss.hurst_exponent = calculate_hurst(spread, max_lag=min(20, len(spread) // 2))
            ss.last_updated = datetime.now()

        # Also update open positions
        for pid, pos in self.positions.items():
            if pid in self.valid_pairs:
                pos.current_zscore = self.valid_pairs[pid].current_zscore

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self) -> list[PairSignal]:
        """
        Generate entry/exit/stop signals based on spread z-scores.

        ENTER_LONG_SPREAD: spread compressed (z < -entry_z), thesis says it widens
        ENTER_SHORT_SPREAD: spread expanded (z > +entry_z), thesis says it compresses
        EXIT: spread reverted (|z| < exit_z)
        STOP: spread diverged past stop (|z| > stop_z)
        """
        signals = []

        # Check exits and stops on open positions first
        for pid, pos in list(self.positions.items()):
            if pos.status != "open":
                continue
            ss = self.valid_pairs.get(pid)
            if not ss:
                continue

            z = ss.current_zscore

            # Stop loss
            if abs(z) > self.stop_z:
                signals.append(PairSignal(
                    pair_id=pid, action="STOP", zscore=z,
                    spread_value=float(ss.spread_values[-1]),
                    long_ticker=pos.long_ticker, short_ticker=pos.short_ticker,
                    pod_id=pos.pod_id,
                    long_size=pos.long_notional, short_size=pos.short_notional,
                    confidence=0.9,
                    reason=f"Spread diverged to {z:.2f}σ (stop at ±{self.stop_z})",
                ))
                continue

            # Take profit / exit
            if abs(z) < self.exit_z:
                signals.append(PairSignal(
                    pair_id=pid, action="EXIT", zscore=z,
                    spread_value=float(ss.spread_values[-1]),
                    long_ticker=pos.long_ticker, short_ticker=pos.short_ticker,
                    pod_id=pos.pod_id,
                    long_size=pos.long_notional, short_size=pos.short_notional,
                    confidence=0.8,
                    reason=f"Spread reverted to {z:.2f}σ (target ±{self.exit_z})",
                ))

        # Check entries on pairs not currently in a position
        open_pair_ids = {pid for pid, p in self.positions.items() if p.status == "open"}

        if len(open_pair_ids) >= self.max_concurrent:
            return signals  # at capacity

        for pid, ss in self.valid_pairs.items():
            if pid in open_pair_ids:
                continue

            z = ss.current_zscore

            # Regime filter — skip trending spreads
            if ss.hurst_exponent >= self.hurst_thresh:
                continue

            pod = self.pm.get_pod(ss.pod_id)
            if not pod:
                continue

            long_size, short_size = self._calculate_pair_size(ss, pod)

            # Long spread: spread is compressed (z < -entry), we expect it to widen
            if z < -self.entry_z:
                signals.append(PairSignal(
                    pair_id=pid, action="ENTER_LONG_SPREAD", zscore=z,
                    spread_value=float(ss.spread_values[-1]),
                    long_ticker=ss.long_ticker, short_ticker=ss.short_ticker,
                    pod_id=ss.pod_id,
                    long_size=long_size, short_size=short_size,
                    confidence=CONFIDENCE_MAP.get(pod.thesis.confidence, 0.5),
                    reason=(
                        f"Spread at {z:.2f}σ below mean. "
                        f"Half-life {ss.half_life:.1f}d. Hurst {ss.hurst_exponent:.2f}. "
                        f"Thesis: {pod.thesis.confidence}."
                    ),
                ))

            # Short spread: spread is expanded (z > +entry), we expect it to compress
            elif z > self.entry_z:
                signals.append(PairSignal(
                    pair_id=pid, action="ENTER_SHORT_SPREAD", zscore=z,
                    spread_value=float(ss.spread_values[-1]),
                    long_ticker=ss.long_ticker, short_ticker=ss.short_ticker,
                    pod_id=ss.pod_id,
                    long_size=short_size, short_size=long_size,  # flip for short spread
                    confidence=CONFIDENCE_MAP.get(pod.thesis.confidence, 0.5),
                    reason=(
                        f"Spread at +{z:.2f}σ above mean. "
                        f"Half-life {ss.half_life:.1f}d. Hurst {ss.hurst_exponent:.2f}. "
                        f"Thesis: {pod.thesis.confidence}."
                    ),
                ))

        return signals

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _calculate_pair_size(self, ss: SpreadSeries, pod: Pod) -> tuple[float, float]:
        """
        Returns (long_leg_dollars, short_leg_dollars).
        Dollar-neutral by default, with optional thesis lean.
        """
        base = float(self.default_leg)

        # Vol scaling
        vol_cfg = self.cfg.get("regime", {})
        if vol_cfg.get("vol_scaling", True) and ss.spread_std > 0:
            vol_lookback = vol_cfg.get("vol_lookback", 20)
            if len(ss.spread_values) >= vol_lookback * 2:
                recent_vol = float(np.std(ss.spread_values[-vol_lookback:], ddof=1))
                hist_vol = float(np.std(ss.spread_values[:-vol_lookback], ddof=1))
                if hist_vol > 0 and recent_vol / hist_vol > 1.5:
                    base *= vol_cfg.get("high_vol_size_multiplier", 0.7)

        # Cap
        base = min(base, float(self.max_leg))

        # Lean
        sp = pod.spread_params or PodSpreadParams()
        if self.lean_enabled and sp.lean_allowed:
            lean = LEAN_MAP.get(pod.thesis.confidence, 0.50)
            lean = min(lean, sp.max_lean_ratio)
            total = base * 2
            long_size = total * lean
            short_size = total * (1 - lean)
        else:
            long_size = base
            short_size = base

        # Cap each leg
        long_size = min(long_size, float(self.max_leg))
        short_size = min(short_size, float(self.max_leg))

        return long_size, short_size

    # ------------------------------------------------------------------
    # Portfolio risk
    # ------------------------------------------------------------------

    def check_portfolio_risk(self) -> dict:
        """Check aggregate risk across all open positions."""
        violations = {}

        open_positions = [p for p in self.positions.values() if p.status == "open"]

        if not open_positions:
            return violations

        total_long = sum(p.long_notional for p in open_positions)
        total_short = sum(p.short_notional for p in open_positions)
        gross = total_long + total_short
        net = total_long - total_short

        if gross > self.max_gross:
            violations["gross_exposure"] = f"${gross:,.0f} > ${self.max_gross:,.0f}"

        if abs(net) > self.max_net:
            violations["net_exposure"] = f"${net:,.0f} exceeds ±${self.max_net:,.0f}"

        if len(open_positions) > self.max_concurrent:
            violations["pair_count"] = f"{len(open_positions)} > {self.max_concurrent}"

        return violations

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Valid Pairs: {len(self.valid_pairs)} | "
            f"Open Positions: {len([p for p in self.positions.values() if p.status == 'open'])}",
            "",
        ]

        if not self.valid_pairs:
            lines.append("  No valid pairs discovered yet. Run discover_pairs() first.")
            return "\n".join(lines)

        # Group by pod
        by_pod: dict[str, list[SpreadSeries]] = {}
        for ss in self.valid_pairs.values():
            by_pod.setdefault(ss.pod_id, []).append(ss)

        for pod_id, pairs in sorted(by_pod.items()):
            pod = self.pm.get_pod(pod_id)
            pod_name = pod.name if pod else pod_id
            lines.append(f"  [{pod_id}] {pod_name}:")
            for ss in pairs:
                score = self.pair_scores.get(ss.pair_id)
                hint_flag = " ★" if score and score.is_hinted else ""
                z_display = f"z={ss.current_zscore:+.2f}"
                lines.append(
                    f"    {ss.long_ticker}/{ss.short_ticker} | "
                    f"{z_display} | hl={ss.half_life:.1f}d | "
                    f"H={ss.hurst_exponent:.2f} | "
                    f"score={score.composite_score:.3f}" if score else "score=N/A"
                    f"{hint_flag}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mock data provider for testing
# ---------------------------------------------------------------------------

class MockDataProvider:
    """Generates synthetic price data for testing pair discovery."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}

    def get_prices(self, ticker: str, days: int) -> np.ndarray:
        if ticker not in self._cache or len(self._cache[ticker]) < days:
            # Generate synthetic prices
            n = max(days, 120)
            returns = self.rng.normal(0.0005, 0.02, n)
            prices = 50 * np.exp(np.cumsum(returns))
            self._cache[ticker] = prices
        return self._cache[ticker][-days:]

    def create_cointegrated_pair(
        self, ticker_a: str, ticker_b: str, days: int = 120,
        theta: float = 0.08, spread_vol: float = 0.5,
    ):
        """
        Create two price series that are cointegrated.
        theta: mean-reversion speed (0.05-0.15 gives half-life ~5-14 days)
        spread_vol: noise in the spread process
        """
        n = days
        # Common stochastic trend (unit root)
        common = np.cumsum(self.rng.normal(0.0003, 0.015, n))
        # Mean-reverting spread (Ornstein-Uhlenbeck)
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = spread[i - 1] * (1 - theta) + self.rng.normal(0, spread_vol * 0.01)

        prices_a = 50 * np.exp(common + spread)
        prices_b = 40 * np.exp(common * 1.1)

        self._cache[ticker_a] = prices_a
        self._cache[ticker_b] = prices_b


# ---------------------------------------------------------------------------
# Main — demo with mock data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config"
    pm = PodManager(config_path)

    # Load settings
    with open(config_path / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    # Create mock data provider with some cointegrated pairs
    mock = MockDataProvider(seed=42)

    # Inject cointegrated relationships for hinted pairs
    # theta=0.04 -> half-life ~17d, theta=0.03 -> ~23d (OU half-life = ln2/theta)
    # For our 3-15d window, theta ~0.05-0.23 works
    # But the OLS-estimated half-life differs from theoretical, so tune empirically
    # Pod 1: tariff scramble
    mock.create_cointegrated_pair("CROX", "CLF", theta=0.04, spread_vol=0.6)
    mock.create_cointegrated_pair("SKX", "STLD", theta=0.05, spread_vol=0.7)
    mock.create_cointegrated_pair("FIVE", "CENX", theta=0.03, spread_vol=0.5)
    # Pod 3: AI infra vs hype
    mock.create_cointegrated_pair("VRT", "AI", theta=0.04, spread_vol=0.6)
    mock.create_cointegrated_pair("POWL", "SOUN", theta=0.05, spread_vol=0.5)
    # Pod 5: K-shaped consumer
    mock.create_cointegrated_pair("RH", "DG", theta=0.04, spread_vol=0.5)
    mock.create_cointegrated_pair("BIRK", "DLTR", theta=0.05, spread_vol=0.6)

    engine = PairsEngine(pm, settings, mock)

    print("=== PAIR DISCOVERY ===")
    all_scores = engine.discover_pairs()

    for pod_id, scores in all_scores.items():
        valid = [s for s in scores if s.is_valid]
        invalid = [s for s in scores if not s.is_valid]
        print(f"\n[{pod_id}] {len(valid)} valid, {len(invalid)} rejected")
        for s in sorted(valid, key=lambda x: x.composite_score, reverse=True)[:5]:
            hint = " ★ HINTED" if s.is_hinted else ""
            print(
                f"  ✓ {s.long_ticker}/{s.short_ticker} "
                f"score={s.composite_score:.3f} "
                f"coint_p={s.coint_pvalue:.3f} "
                f"hl={s.half_life:.1f}d "
                f"H={s.hurst:.2f}"
                f"{hint}"
            )
        # Show a few rejections
        for s in invalid[:3]:
            print(f"  ✗ {s.long_ticker}/{s.short_ticker} — {s.rejection_reason}")

    print("\n\n=== ENGINE SUMMARY ===")
    print(engine.summary())

    print("\n\n=== SIGNAL GENERATION ===")
    engine.update_spreads()
    signals = engine.generate_signals()
    if signals:
        for sig in signals:
            print(
                f"  {sig.action}: {sig.long_ticker}/{sig.short_ticker} "
                f"| z={sig.zscore:+.2f} | "
                f"long=${sig.long_size:,.0f} short=${sig.short_size:,.0f} "
                f"| {sig.reason}"
            )
    else:
        print("  No signals at current z-score levels.")

    print("\n\n=== RISK CHECK ===")
    violations = engine.check_portfolio_risk()
    if violations:
        for k, v in violations.items():
            print(f"  ⚠ {k}: {v}")
    else:
        print("  All clear — no risk violations.")
