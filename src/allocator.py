"""
Dynamic Pod Allocator — momentum-weighted capital allocation.

Pods that win get more capital. Pods that lose get sized down or killed.
Uses exponential decay so recent performance matters more than old.

The allocator replaces equal-weight pod allocation with a feedback loop:
  pod wins → score goes up → bigger positions → compounds winners
  pod loses → score goes down → smaller positions → limits damage
"""

from dataclasses import dataclass, field
from datetime import datetime
import math


@dataclass
class PodPerformance:
    pod_id: str
    score: float = 1.0              # momentum score (1.0 = neutral)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    recent_pnl: list = field(default_factory=list)  # last N trade P&Ls
    consecutive_losses: int = 0
    disabled_until: int = -1         # day_idx when re-enabled (-1 = active)
    last_trade_date: str = ""

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.5

    @property
    def recent_win_rate(self) -> float:
        if not self.recent_pnl:
            return 0.5
        wins = sum(1 for p in self.recent_pnl if p > 0)
        return wins / len(self.recent_pnl)

    @property
    def is_hot(self) -> bool:
        """Pod is on a winning streak."""
        return self.recent_win_rate >= 0.6 and len(self.recent_pnl) >= 3

    @property
    def is_cold(self) -> bool:
        """Pod is bleeding."""
        return self.recent_win_rate <= 0.3 and len(self.recent_pnl) >= 3


class DynamicAllocator:
    """
    Manages capital allocation across pods based on performance momentum.

    Key mechanics:
    - Each pod starts with score=1.0 (equal weight)
    - Wins multiply score by win_boost, losses by loss_decay
    - Capital is allocated proportional to normalized scores
    - Floor and ceiling prevent total concentration
    - Pods with 3+ consecutive losses get temporarily disabled
    """

    def __init__(
        self,
        pod_ids: list[str],
        capital: float = 100000,
        win_boost: float = 1.15,        # score *= this on win
        loss_decay: float = 0.80,       # score *= this on loss
        min_allocation: float = 0.05,   # 5% floor per pod
        max_allocation: float = 0.45,   # 45% ceiling per pod
        max_recent_trades: int = 8,     # lookback for recent performance
        disable_after_consecutive: int = 3,  # consecutive losses to disable
        disable_duration: int = 15,     # trading days to disable
    ):
        self.capital = capital
        self.win_boost = win_boost
        self.loss_decay = loss_decay
        self.min_alloc = min_allocation
        self.max_alloc = max_allocation
        self.max_recent = max_recent_trades
        self.disable_after = disable_after_consecutive
        self.disable_duration = disable_duration

        self.pods = {pid: PodPerformance(pod_id=pid) for pid in pod_ids}
        self._allocations_cache = None

    def record_trade(self, pod_id: str, pnl: float, trade_date: str, day_idx: int):
        """Record a completed trade and update pod scoring."""
        pod = self.pods.get(pod_id)
        if not pod:
            return

        pod.total_trades += 1
        pod.total_pnl += pnl
        pod.last_trade_date = trade_date
        pod.recent_pnl.append(pnl)

        # Keep only recent trades
        if len(pod.recent_pnl) > self.max_recent:
            pod.recent_pnl = pod.recent_pnl[-self.max_recent:]

        if pnl > 0:
            pod.wins += 1
            pod.consecutive_losses = 0
            pod.score *= self.win_boost
        else:
            pod.losses += 1
            pod.consecutive_losses += 1
            pod.score *= self.loss_decay

            # Disable pod after too many consecutive losses
            if pod.consecutive_losses >= self.disable_after:
                pod.disabled_until = day_idx + self.disable_duration

        # Clamp score to prevent extreme values
        pod.score = max(0.1, min(5.0, pod.score))

        # Invalidate allocation cache
        self._allocations_cache = None

    def is_pod_active(self, pod_id: str, day_idx: int) -> bool:
        """Check if a pod is currently active (not disabled)."""
        pod = self.pods.get(pod_id)
        if not pod:
            return False

        if pod.disabled_until >= 0 and day_idx < pod.disabled_until:
            return False

        # Re-enable if cooloff is over
        if pod.disabled_until >= 0 and day_idx >= pod.disabled_until:
            pod.disabled_until = -1
            pod.consecutive_losses = 0
            pod.score = max(pod.score, 0.5)  # give it a chance but not full weight

        return True

    def get_allocations(self, day_idx: int) -> dict[str, float]:
        """Get capital allocation fraction for each active pod."""
        if self._allocations_cache:
            return self._allocations_cache

        active_pods = {pid: pod for pid, pod in self.pods.items()
                       if self.is_pod_active(pid, day_idx)}

        if not active_pods:
            return {}

        # Normalize scores to get raw weights
        total_score = sum(p.score for p in active_pods.values())
        if total_score <= 0:
            # Fallback to equal weight
            n = len(active_pods)
            return {pid: 1.0 / n for pid in active_pods}

        raw_weights = {pid: p.score / total_score for pid, p in active_pods.items()}

        # Apply floor and ceiling
        allocations = {}
        for pid, weight in raw_weights.items():
            allocations[pid] = max(self.min_alloc, min(self.max_alloc, weight))

        # Re-normalize to sum to 1.0
        total = sum(allocations.values())
        allocations = {pid: w / total for pid, w in allocations.items()}

        self._allocations_cache = allocations
        return allocations

    def get_leg_size(self, pod_id: str, default_size: float, day_idx: int) -> float:
        """Get sized leg amount for a pod based on its allocation."""
        allocs = self.get_allocations(day_idx)
        pod_weight = allocs.get(pod_id, 0)
        if pod_weight <= 0:
            return 0

        n_active = len(allocs)
        equal_weight = 1.0 / n_active if n_active > 0 else 0.2

        # Scale relative to equal weight
        # If pod has 2x its equal share, size is 2x default
        multiplier = pod_weight / equal_weight if equal_weight > 0 else 1.0
        multiplier = max(0.3, min(2.5, multiplier))  # clamp

        return default_size * multiplier

    def summary(self, day_idx: int) -> str:
        """Human-readable allocation summary."""
        allocs = self.get_allocations(day_idx)
        lines = ["POD ALLOCATIONS (momentum-weighted):"]
        lines.append(f"  {'Pod':<30s} {'Score':>6s} {'Alloc':>6s} {'W/L':>8s} {'WR':>5s} {'P&L':>10s} {'Status':>10s}")
        lines.append("  " + "-" * 80)

        for pid in sorted(self.pods.keys()):
            pod = self.pods[pid]
            alloc = allocs.get(pid, 0)
            active = self.is_pod_active(pid, day_idx)
            status = "ACTIVE" if active else f"OFF til {pod.disabled_until}"
            hot_cold = " HOT" if pod.is_hot else " COLD" if pod.is_cold else ""

            lines.append(
                f"  {pid:<30s} {pod.score:>6.2f} {alloc:>5.0%} "
                f"{pod.wins:>3d}/{pod.losses:<3d} {pod.win_rate:>4.0%} "
                f"${pod.total_pnl:>+9,.0f} {status}{hot_cold}"
            )

        return "\n".join(lines)
