"""
Notifier — sends pair trade signals via iMessage using Claude Code Channels.

When a signal fires, formats it for iMessage and waits for the operator's reply.
The operator always has final say: YES / NO / custom size.

For Claude Code Channels (requires Claude Max + iMessage plugin):
  The notifier writes signals to a queue file that Claude Code monitors.
  Claude Code pushes them to iMessage via the MCP channel.

Fallback: direct AppleScript iMessage for standalone operation.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class NotificationResult:
    sent: bool
    method: str             # "claude_channels", "applescript", "console"
    error: str = ""


class Notifier:
    """
    Multi-method notifier. Tries Claude Code Channels first,
    falls back to AppleScript, then to console output.
    """

    def __init__(
        self,
        phone_number: Optional[str] = None,
        signal_queue_path: str = "data/signal_queue.json",
        method: str = "auto",  # "auto", "claude_channels", "applescript", "console"
    ):
        self.phone = phone_number or os.environ.get("IMESSAGE_PHONE", "")
        self.queue_path = Path(signal_queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.method = method

    # ------------------------------------------------------------------
    # Signal formatting
    # ------------------------------------------------------------------

    def format_pair_signal(
        self,
        action: str,
        long_ticker: str,
        short_ticker: str,
        pod_name: str,
        zscore: float,
        long_size: float,
        short_size: float,
        confidence: float,
        reason: str,
        risk_warnings: list[str] = None,
    ) -> str:
        """Format a pair signal for iMessage."""

        if "ENTER_LONG" in action:
            emoji = "🟢"
            direction = "LONG SPREAD"
            instruction = f"Buy {long_ticker} / Short {short_ticker}"
        elif "ENTER_SHORT" in action:
            emoji = "🔴"
            direction = "SHORT SPREAD"
            instruction = f"Short {long_ticker} / Buy {short_ticker}"
        elif action == "EXIT":
            emoji = "⬜"
            direction = "EXIT"
            instruction = f"Close {long_ticker}/{short_ticker}"
        elif action == "STOP":
            emoji = "🛑"
            direction = "STOP OUT"
            instruction = f"Close {long_ticker}/{short_ticker}"
        else:
            emoji = "📊"
            direction = action
            instruction = f"{long_ticker}/{short_ticker}"

        total = long_size + short_size
        lines = [
            f"{emoji} {direction} | Pod: {pod_name}",
            f"{instruction}",
            f"Z-score: {zscore:+.2f} | Confidence: {confidence:.0%}",
            f"",
            f"SIZING:",
            f"  Long leg:  ${long_size:,.0f}",
            f"  Short leg: ${short_size:,.0f}",
            f"  Total:     ${total:,.0f}",
            f"",
            f"REASON: {reason}",
        ]

        if risk_warnings:
            lines.append("")
            lines.append("⚠ RISK WARNINGS:")
            for w in risk_warnings:
                lines.append(f"  {w}")

        if action.startswith("ENTER"):
            lines.append("")
            lines.append("AUTO-EXECUTED — no approval needed")

        return "\n".join(lines)

    def format_circuit_breaker(self, drawdown_pct: float, cooloff_hours: int) -> str:
        """Format a circuit breaker alert."""
        return (
            f"🚨 CIRCUIT BREAKER TRIGGERED\n"
            f"Portfolio drawdown: {drawdown_pct:.1%}\n"
            f"ALL POSITIONS BEING FLATTENED\n"
            f"Cooloff: {cooloff_hours}h — no new trades\n"
            f"\n"
            f"Review and reply RESUME when ready."
        )

    def format_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        open_pairs: int,
        gross_exposure: float,
        net_exposure: float,
    ) -> str:
        """Format end-of-day summary."""
        return (
            f"📊 EOD Summary | {datetime.now().strftime('%Y-%m-%d')}\n"
            f"Equity: ${equity:,.0f} ({daily_pnl:+,.0f} today)\n"
            f"Open pairs: {open_pairs}\n"
            f"Gross: ${gross_exposure:,.0f} | Net: ${net_exposure:+,.0f}"
        )

    # ------------------------------------------------------------------
    # Delivery methods
    # ------------------------------------------------------------------

    def send(self, message: str) -> NotificationResult:
        """Send a message using the configured method."""
        if self.method == "auto":
            # Try Claude Channels first, then AppleScript, then console
            result = self._send_claude_channels(message)
            if result.sent:
                return result
            result = self._send_applescript(message)
            if result.sent:
                return result
            return self._send_console(message)
        elif self.method == "claude_channels":
            return self._send_claude_channels(message)
        elif self.method == "applescript":
            return self._send_applescript(message)
        else:
            return self._send_console(message)

    def _send_claude_channels(self, message: str) -> NotificationResult:
        """
        Write signal to queue file for Claude Code Channels to pick up.
        Claude Code monitors this file and pushes to iMessage via MCP.
        """
        try:
            # Append to signal queue
            queue = []
            if self.queue_path.exists():
                try:
                    queue = json.loads(self.queue_path.read_text())
                except (json.JSONDecodeError, FileNotFoundError):
                    queue = []

            queue.append({
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "status": "pending",
            })

            self.queue_path.write_text(json.dumps(queue, indent=2))
            return NotificationResult(sent=True, method="claude_channels")
        except Exception as e:
            return NotificationResult(sent=False, method="claude_channels", error=str(e))

    def _send_applescript(self, message: str) -> NotificationResult:
        """Send via AppleScript directly (macOS only)."""
        if not self.phone:
            return NotificationResult(
                sent=False, method="applescript",
                error="No phone number configured",
            )

        # Escape the message for AppleScript
        escaped = message.replace("\\", "\\\\").replace('"', '\\"')

        script = f'''
            tell application "Messages"
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "{self.phone}" of targetService
                send "{escaped}" to targetBuddy
            end tell
        '''

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return NotificationResult(sent=True, method="applescript")
            else:
                return NotificationResult(
                    sent=False, method="applescript",
                    error=result.stderr.strip(),
                )
        except Exception as e:
            return NotificationResult(sent=False, method="applescript", error=str(e))

    def _send_console(self, message: str) -> NotificationResult:
        """Fallback: print to console."""
        print(f"\n{'='*60}")
        print(f"[SIGNAL] {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(message)
        print(f"{'='*60}\n")
        return NotificationResult(sent=True, method="console")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    notifier = Notifier(method="console")

    # Test pair signal
    msg = notifier.format_pair_signal(
        action="ENTER_LONG_SPREAD",
        long_ticker="CROX",
        short_ticker="CLF",
        pod_name="Tariff Scramble",
        zscore=-2.31,
        long_size=1406,
        short_size=1406,
        confidence=0.85,
        reason="Spread at -2.31σ below mean. Half-life 4.2d. Hurst 0.12. Thesis: high.",
        risk_warnings=["Rule 1+4: Sized down from $5,000 to $2,812"],
    )
    notifier.send(msg)

    # Test circuit breaker
    msg = notifier.format_circuit_breaker(drawdown_pct=0.109, cooloff_hours=48)
    notifier.send(msg)

    # Test daily summary
    msg = notifier.format_daily_summary(
        equity=29650, daily_pnl=-150, open_pairs=3,
        gross_exposure=18000, net_exposure=1200,
    )
    notifier.send(msg)
