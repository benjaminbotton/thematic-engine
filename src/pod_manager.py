"""
Pod Manager — loads, validates, and manages thematic pod definitions.

Pods are living, event-driven containers. They track:
  - A core thesis with long/short sides and confidence level
  - Invalidation triggers (what would kill the thesis)
  - Watch events (awareness markers, not rigid date triggers)
  - An evolution log (how the thesis has shifted over time)
  - Watchlists with per-ticker status (active/paused/removed)
  - Pair hints, exclusions, and spread parameters for the pairs engine
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class WatchlistEntry:
    ticker: str
    name: str
    thesis: str
    side: str       # "LONG" or "SHORT"
    status: str     # "active", "paused", "removed"
    added: str      # date string


@dataclass
class WatchEvent:
    event: str
    expected: str   # date, quarter, "ongoing", "rolling", etc.
    status: str     # "upcoming", "active", "resolved", "invalidated"
    impact: str
    notes: str = ""


@dataclass
class EvolutionEntry:
    date: str
    note: str


@dataclass
class Thesis:
    core: str
    long: str
    short: str
    last_validated: str
    confidence: str     # "high", "medium", "low", "stale"
    invalidation_triggers: list[str] = field(default_factory=list)

    @property
    def days_since_validated(self) -> int:
        try:
            validated = date.fromisoformat(self.last_validated)
            return (date.today() - validated).days
        except ValueError:
            return -1

    @property
    def is_stale(self) -> bool:
        days = self.days_since_validated
        return days > 14 or self.confidence == "stale"


# --- Pairs support ---

@dataclass
class PairHint:
    long_ticker: str
    short_ticker: str
    rationale: str


@dataclass
class PodSpreadParams:
    min_correlation: float = 0.50
    cointegration_pvalue: float = 0.05
    lookback_days: int = 60
    lean_allowed: bool = True
    max_lean_ratio: float = 0.55


@dataclass
class Pod:
    pod_id: str
    name: str
    status: str         # ACTIVE or RETIRED
    created_date: str
    thesis: Thesis
    watch_events: list[WatchEvent]
    evolution_log: list[EvolutionEntry]
    watchlist: list[WatchlistEntry]
    pair_hints: list[PairHint] = field(default_factory=list)
    pair_excludes: list[tuple[str, str]] = field(default_factory=list)
    spread_params: Optional[PodSpreadParams] = None

    @property
    def active_watchlist(self) -> list[WatchlistEntry]:
        return [w for w in self.watchlist if w.status == "active"]

    @property
    def long_watchlist(self) -> list[WatchlistEntry]:
        return [w for w in self.active_watchlist if w.side == "LONG"]

    @property
    def short_watchlist(self) -> list[WatchlistEntry]:
        return [w for w in self.active_watchlist if w.side == "SHORT"]

    @property
    def all_tickers(self) -> list[str]:
        return [w.ticker for w in self.active_watchlist]

    @property
    def long_tickers(self) -> list[str]:
        return [w.ticker for w in self.long_watchlist]

    @property
    def short_tickers(self) -> list[str]:
        return [w.ticker for w in self.short_watchlist]

    @property
    def active_events(self) -> list[WatchEvent]:
        return [e for e in self.watch_events if e.status in ("active", "upcoming")]

    @property
    def needs_review(self) -> bool:
        return self.thesis.is_stale

    @property
    def pair_candidates(self) -> list[tuple[WatchlistEntry, WatchlistEntry]]:
        """All possible long x short combinations from active watchlist."""
        return [(l, s) for l in self.long_watchlist for s in self.short_watchlist]

    @property
    def excluded_pairs(self) -> set[tuple[str, str]]:
        return set(self.pair_excludes)

    @property
    def hinted_pairs(self) -> set[tuple[str, str]]:
        return {(h.long_ticker, h.short_ticker) for h in self.pair_hints}

    def is_pair_excluded(self, long_ticker: str, short_ticker: str) -> bool:
        return (long_ticker, short_ticker) in self.excluded_pairs

    def is_pair_hinted(self, long_ticker: str, short_ticker: str) -> bool:
        return (long_ticker, short_ticker) in self.hinted_pairs

    def get_hint_rationale(self, long_ticker: str, short_ticker: str) -> str:
        for h in self.pair_hints:
            if h.long_ticker == long_ticker and h.short_ticker == short_ticker:
                return h.rationale
        return ""


class PodManager:
    def __init__(self, config_dir: str | Path):
        self.config_dir = Path(config_dir)
        self.pods_dir = self.config_dir / "pods"
        self.pods: dict[str, Pod] = {}
        self._load_all()

    def _load_all(self):
        if not self.pods_dir.exists():
            raise FileNotFoundError(f"Pods directory not found: {self.pods_dir}")

        for yaml_file in sorted(self.pods_dir.glob("*.yaml")):
            pod = self._parse_pod(yaml_file)
            self.pods[pod.pod_id] = pod

    def _parse_pod(self, path: Path) -> Pod:
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse thesis
        thesis_data = data.get("thesis", {})
        thesis = Thesis(
            core=thesis_data.get("core", ""),
            long=thesis_data.get("long", ""),
            short=thesis_data.get("short", ""),
            last_validated=thesis_data.get("last_validated", ""),
            confidence=thesis_data.get("confidence", "medium"),
            invalidation_triggers=thesis_data.get("invalidation_triggers", []),
        )

        # Parse watch events
        watch_events = [
            WatchEvent(
                event=e["event"],
                expected=e.get("expected", ""),
                status=e.get("status", "upcoming"),
                impact=e.get("impact", ""),
                notes=e.get("notes", ""),
            )
            for e in data.get("watch_events", [])
        ]

        # Parse evolution log
        evolution_log = [
            EvolutionEntry(date=e["date"], note=e["note"])
            for e in data.get("evolution_log", [])
        ]

        # Parse watchlist
        watchlist = []
        for entry in data.get("watchlist", {}).get("long", []):
            watchlist.append(WatchlistEntry(
                ticker=entry["ticker"],
                name=entry["name"],
                thesis=entry["thesis"],
                side="LONG",
                status=entry.get("status", "active"),
                added=entry.get("added", ""),
            ))
        for entry in data.get("watchlist", {}).get("short", []):
            watchlist.append(WatchlistEntry(
                ticker=entry["ticker"],
                name=entry["name"],
                thesis=entry["thesis"],
                side="SHORT",
                status=entry.get("status", "active"),
                added=entry.get("added", ""),
            ))

        # Parse pairs config (optional)
        pairs_data = data.get("pairs", {})
        pair_hints = [
            PairHint(h["long"], h["short"], h.get("rationale", ""))
            for h in pairs_data.get("hints", [])
        ]
        pair_excludes = [
            (e["long"], e["short"])
            for e in pairs_data.get("exclude", [])
        ]
        spread_params_data = pairs_data.get("spread_params")
        spread_params = (
            PodSpreadParams(**spread_params_data)
            if spread_params_data else None
        )

        return Pod(
            pod_id=data["pod_id"],
            name=data["name"],
            status=data["status"],
            created_date=data["created_date"],
            thesis=thesis,
            watch_events=watch_events,
            evolution_log=evolution_log,
            watchlist=watchlist,
            pair_hints=pair_hints,
            pair_excludes=pair_excludes,
            spread_params=spread_params,
        )

    def get_pod(self, pod_id: str) -> Optional[Pod]:
        return self.pods.get(pod_id)

    def active_pods(self) -> list[Pod]:
        return [p for p in self.pods.values() if p.status == "ACTIVE"]

    def stale_pods(self) -> list[Pod]:
        return [p for p in self.active_pods() if p.needs_review]

    def all_tickers(self) -> set[str]:
        tickers = set()
        for pod in self.active_pods():
            tickers.update(pod.all_tickers)
        return tickers

    def all_pair_candidates(self) -> dict[str, list[tuple[WatchlistEntry, WatchlistEntry]]]:
        """Returns {pod_id: [(long_entry, short_entry), ...]} for all active pods."""
        return {
            pod.pod_id: pod.pair_candidates
            for pod in self.active_pods()
        }

    def summary(self) -> str:
        lines = []
        for pod in self.active_pods():
            stale_flag = " ⚠ NEEDS REVIEW" if pod.needs_review else ""
            events_active = len(pod.active_events)
            n_candidates = len(pod.pair_candidates)
            n_excluded = len(pod.pair_excludes)
            n_hinted = len(pod.pair_hints)
            lines.append(
                f"  {pod.pod_id}: {pod.name} "
                f"({len(pod.long_tickers)}L / {len(pod.short_tickers)}S) "
                f"| pairs: {n_candidates} candidates, {n_hinted} hinted, {n_excluded} excluded "
                f"| confidence: {pod.thesis.confidence}"
                f"{stale_flag}"
            )
        total = self.all_tickers()
        stale = self.stale_pods()
        header = f"Active Pods: {len(self.active_pods())} | Unique Tickers: {len(total)}"
        if stale:
            header += f" | ⚠ {len(stale)} pod(s) need thesis review"
        return header + "\n" + "\n".join(lines)

    def pod_detail(self, pod_id: str) -> str:
        pod = self.get_pod(pod_id)
        if not pod:
            return f"Pod '{pod_id}' not found."

        lines = [
            f"=== {pod.name} ===",
            f"Status: {pod.status} | Confidence: {pod.thesis.confidence}",
            f"Last validated: {pod.thesis.last_validated} "
            f"({pod.thesis.days_since_validated}d ago)",
            "",
            f"CORE THESIS: {pod.thesis.core.strip()}",
            "",
            "INVALIDATION TRIGGERS:",
        ]
        for trigger in pod.thesis.invalidation_triggers:
            lines.append(f"  ✗ {trigger}")

        lines.append("")
        lines.append("WATCH EVENTS:")
        for e in pod.watch_events:
            status_icon = {"active": "●", "upcoming": "○", "resolved": "✓", "invalidated": "✗"}.get(e.status, "?")
            lines.append(f"  {status_icon} [{e.status}] {e.event} (expected: {e.expected})")
            if e.notes:
                lines.append(f"    └ {e.notes}")

        lines.append("")
        lines.append(f"LONGS ({len(pod.long_tickers)} active):")
        for w in pod.long_watchlist:
            lines.append(f"  {w.ticker:6s} {w.name} — {w.thesis}")

        lines.append(f"SHORTS ({len(pod.short_tickers)} active):")
        for w in pod.short_watchlist:
            lines.append(f"  {w.ticker:6s} {w.name} — {w.thesis}")

        # Pairs info
        if pod.pair_hints:
            lines.append("")
            lines.append(f"PAIR HINTS ({len(pod.pair_hints)}):")
            for h in pod.pair_hints:
                lines.append(f"  {h.long_ticker}/{h.short_ticker} — {h.rationale}")

        if pod.pair_excludes:
            lines.append(f"PAIR EXCLUSIONS ({len(pod.pair_excludes)}):")
            for long_t, short_t in pod.pair_excludes:
                lines.append(f"  {long_t}/{short_t}")

        if pod.spread_params:
            sp = pod.spread_params
            lines.append(f"SPREAD PARAMS: corr>{sp.min_correlation} | "
                         f"coint p<{sp.cointegration_pvalue} | "
                         f"lookback={sp.lookback_days}d | "
                         f"lean={'yes' if sp.lean_allowed else 'no'} "
                         f"(max {sp.max_lean_ratio})")

        n_candidates = len(pod.pair_candidates)
        n_after_excludes = n_candidates - len(pod.pair_excludes)
        lines.append(f"PAIR CANDIDATES: {n_candidates} total, {n_after_excludes} after exclusions")

        if pod.evolution_log:
            lines.append("")
            lines.append("EVOLUTION LOG:")
            for e in pod.evolution_log:
                lines.append(f"  [{e.date}] {e.note}")

        return "\n".join(lines)


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config"
    pm = PodManager(config_path)
    print(pm.summary())
    print()

    # Show detail for first pod only
    pod = pm.active_pods()[0]
    print(pm.pod_detail(pod.pod_id))
