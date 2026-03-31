"""
Thematic Catalyst L/S Engine — Dashboard

Run locally:   streamlit run src/dashboard.py
Run with auth:  streamlit run src/dashboard.py -- --password yourpassword
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load env from .env file (local) or st.secrets (Streamlit Cloud)
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
else:
    # On Streamlit Cloud — copy secrets to env vars for downstream modules
    for key in ["POLYGON_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
        try:
            os.environ[key] = st.secrets[key]
        except Exception:
            pass

import yaml
import requests as req_lib
from pod_manager import PodManager
from pairs_engine import PairsEngine
from risk_guard import RiskGuard

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Thematic L/S Engine", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Password protection
# ---------------------------------------------------------------------------
def get_secret(key, default=""):
    """Read from Streamlit secrets (cloud) or os.environ (local)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

DASHBOARD_PASSWORD = get_secret("DASHBOARD_PASSWORD")

if DASHBOARD_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Thematic Catalyst L/S Engine")
        st.caption("Enter password to access the dashboard")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary"):
            if password == DASHBOARD_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

# ---------------------------------------------------------------------------
# Plotly dark template
# ---------------------------------------------------------------------------
DARK = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, sans-serif", size=12, color="#8b949e"),
    title_font=dict(size=14, color="#f0f6fc"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#3b82f6", "#22c55e", "#ef4444", "#f59e0b", "#8b5cf6", "#06b6d4"],
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#1a1b26", bordercolor="#30363d", font=dict(color="#e2e8f0", size=12)),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", linecolor="#21262d",
               tickfont=dict(color="#8b949e", size=11), showgrid=True),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d", linecolor="#21262d",
               tickfont=dict(color="#8b949e", size=11), showgrid=True, side="right"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e", size=11)),
    margin=dict(l=10, r=10, t=35, b=10),
))
pio.templates["dark"] = DARK
pio.templates.default = "dark"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_zscore_chart(z: float):
    """Compact horizontal z-score bar."""
    if abs(z) >= 3.5: dot_color = "#ef4444"
    elif abs(z) >= 2.0: dot_color = "#22c55e"
    elif abs(z) <= 0.5: dot_color = "#f59e0b"
    else: dot_color = "#3b82f6"

    fig = go.Figure()
    zones = [
        (-4, -3.5, "rgba(239,68,68,0.15)"), (-3.5, -2, "rgba(34,197,94,0.12)"),
        (-2, -0.5, "rgba(59,130,246,0.05)"), (-0.5, 0.5, "rgba(245,158,11,0.08)"),
        (0.5, 2, "rgba(59,130,246,0.05)"), (2, 3.5, "rgba(34,197,94,0.12)"),
        (3.5, 4, "rgba(239,68,68,0.15)"),
    ]
    for x0, x1, color in zones:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1,
                      fillcolor=color, line_width=0, layer="below")
    for xval, color in [(-2, "#22c55e"), (2, "#22c55e"), (-3.5, "#ef4444"), (3.5, "#ef4444"), (0, "#30363d")]:
        fig.add_vline(x=xval, line_color=color, line_width=1, line_dash="dot", opacity=0.5)

    fig.add_trace(go.Scatter(
        x=[z], y=[0.5], mode="markers",
        marker=dict(size=16, color=dot_color, line=dict(width=2, color=dot_color)),
        hovertemplate=f"z = {z:+.2f}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(
        height=55, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[-4, 4], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True),
    )
    return fig


class CloudDataProvider:
    """Fetches prices directly from Polygon REST API — no SQLite needed.
    Uses Streamlit's @st.cache_data to cache API responses in memory."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_prices(self, ticker: str, days: int, **kw) -> np.ndarray:
        """Fetch closing prices from Polygon. Cached by Streamlit."""
        return _fetch_polygon_prices(self.api_key, ticker, days)

    def cache_stats(self):
        return {"tickers": "cloud", "rows": "live", "oldest_date": "N/A", "latest_date": "live"}


@st.cache_data(ttl=300)
def _fetch_polygon_prices(api_key: str, ticker: str, days: int) -> np.ndarray:
    """Fetch daily closes from Polygon — cached 5 min by Streamlit."""
    from datetime import date, timedelta
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=days + 10)).isoformat()

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
    params = {"apiKey": api_key, "adjusted": "true", "sort": "asc", "limit": 5000}

    resp = req_lib.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("resultsCount", 0) == 0:
        raise ValueError(f"No data for {ticker}")

    closes = np.array([bar["c"] for bar in data["results"]], dtype=float)
    return closes[-days:] if len(closes) > days else closes


def get_spread_history(provider, long_t, short_t, days=60):
    try:
        lp = provider.get_prices(long_t, days)
        sp = provider.get_prices(short_t, days)
        n = min(len(lp), len(sp))
        return np.log(lp[-n:]) - np.log(sp[-n:])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------
@st.cache_resource(ttl=300)
def load_system():
    config_path = ROOT / "config"
    with open(config_path / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    pm = PodManager(config_path)

    api_key = get_secret("POLYGON_API_KEY")
    provider = CloudDataProvider(api_key)

    engine = PairsEngine(pm, settings, provider)
    engine.discover_pairs()
    engine.update_spreads()
    guard = RiskGuard(settings)
    return settings, pm, provider, engine, guard


@st.cache_resource(ttl=60)
def load_alpaca():
    try:
        from executor import AlpacaExecutor
        ex = AlpacaExecutor(paper=True)
        acct = ex.get_account()
        positions = ex.get_positions()
        return acct, positions, ex, None
    except Exception as e:
        return None, [], None, str(e)


@st.cache_data(ttl=60)
def load_order_history():
    """Fetch all order history from Alpaca."""
    try:
        alpaca_key = get_secret("ALPACA_API_KEY")
        alpaca_secret = get_secret("ALPACA_SECRET_KEY")
        if not alpaca_key:
            return []
        headers = {
            "APCA-API-KEY-ID": alpaca_key,
            "APCA-API-SECRET-KEY": alpaca_secret,
        }
        orders = []
        url = "https://paper-api.alpaca.markets/v2/orders?status=all&limit=500&direction=desc"
        resp = req_lib.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            for o in resp.json():
                filled_price = float(o.get("filled_avg_price", 0) or 0)
                filled_qty = int(float(o.get("filled_qty", 0) or 0))
                orders.append({
                    "Time": o["created_at"][:19].replace("T", " "),
                    "Side": o["side"].upper(),
                    "Ticker": o["symbol"],
                    "Qty": int(float(o.get("qty", 0))),
                    "Filled": filled_qty,
                    "Price": f"${filled_price:.2f}" if filled_price > 0 else "—",
                    "Notional": f"${filled_price * filled_qty:,.0f}" if filled_price > 0 else "—",
                    "Status": o["status"].upper(),
                    "Type": o.get("type", "market").upper(),
                })
        return orders
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
settings, pm, provider, engine, guard = load_system()
acct, positions, executor, alpaca_err = load_alpaca()
signals = engine.generate_signals()
capital = settings.get("capital_base", 100000)

if acct:
    equity = float(acct["equity"])
    cash = float(acct["cash"])
else:
    equity, cash = capital, capital

pnl = equity - capital
pnl_pct = (pnl / capital) * 100
gross = sum(abs(p.market_value) for p in positions) if positions else 0
net = sum(p.market_value for p in positions) if positions else 0
drawdown = max(0, (capital - equity) / capital) * 100
hard_rules = settings.get("hard_rules", {})
sorted_pairs = sorted(engine.valid_pairs.values(),
                       key=lambda x: min(abs(x.current_zscore - 2.0), abs(x.current_zscore + 2.0)))


# ===========================================================================
# HEADER
# ===========================================================================
col_t, col_s, col_b = st.columns([4, 3, 1])
with col_t:
    st.title("Thematic Catalyst L/S Engine")
    st.caption(f"Paper Trading  |  {datetime.now().strftime('%b %d, %Y %I:%M %p PT')}")
with col_b:
    st.write("")
    if st.button("Refresh", type="primary", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# --- Thesis blurb ---
with st.container(border=True):
    st.markdown("""
**What this system does:** This is a fully automated pairs trading engine that finds stocks whose prices move together,
waits for them to diverge, then bets on convergence. It's **market-neutral** — it doesn't care if the market goes up
or down, only that paired stocks revert to their historical relationship.

**How it finds pairs:** Five thematic "pods" — tariff policy, Fed leadership transition, AI infrastructure vs. hype,
defense spending, and the K-shaped consumer — each identify stocks on opposite sides of a macro catalyst.
The engine tests every long/short combination for statistical cointegration (a mathematical proof that two price series are linked).

**How it trades:** Every 30 minutes during market hours (6:30 AM - 1 PM PT), the engine scans all active pairs.
When a spread hits 2 standard deviations from its mean, both legs are automatically executed on Alpaca.
When it reverts to 0.5 sigma, the position is closed. All trades are gated through 5 hard risk rules that cannot be overridden.

**Current status:** {len(engine.valid_pairs)} active pairs being monitored. {len(signals)} signal{'s' if len(signals) != 1 else ''} right now.
${equity:,.0f} equity on a ${capital:,.0f} base.
""")

st.divider()


# ===========================================================================
# TABS
# ===========================================================================
tab_overview, tab_pairs, tab_pods, tab_log, tab_config = st.tabs(["Overview", "Pairs & Spreads", "Thematic Pods", "Trade Log", "Config & Risk"])


# ===========================================================================
# TAB 1: Overview
# ===========================================================================
with tab_overview:
    # KPIs
    st.subheader("Portfolio")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", f"${equity:,.0f}", f"{pnl:+,.0f} ({pnl_pct:+.1f}%)" if pnl != 0 else None)
    c2.metric("Cash", f"${cash:,.0f}")
    c3.metric("Positions", f"{len(positions)}")
    c4.metric("Gross Exposure", f"${gross:,.0f}", f"{gross/capital*100:.0f}% of capital" if gross > 0 else "0%")
    c5.metric("Drawdown", f"{drawdown:.1f}%", f"Limit: {hard_rules.get('circuit_breaker_drawdown', 0.10)*100:.0f}%",
              delta_color="inverse")

    with st.expander("What do these numbers mean?"):
        st.markdown("""
- **Equity** — Total account value. This is what you'd have if you closed every position right now.
- **Cash** — Money not tied up in trades. Used to open new pairs.
- **Positions** — Individual stock positions. Each pair trade creates 2 (one long, one short).
- **Gross Exposure** — Total dollars at work (ignoring direction). Higher = more capital deployed. Capped at 120% of capital to limit risk.
- **Drawdown** — How far below the peak the account has fallen. At 10%, the circuit breaker fires and closes everything.
""")

    st.divider()

    # Signals
    st.subheader("Signal Status")
    if signals:
        for sig in signals:
            st.success(f"**SIGNAL: {sig.action}** — Long **{sig.long_ticker}** / Short **{sig.short_ticker}** at z={sig.zscore:+.2f}")
            st.caption(sig.reason)
    else:
        closest = None
        closest_dist = 999
        for ss in sorted_pairs:
            z = ss.current_zscore
            dist = min(abs(z - 2.0), abs(z + 2.0))
            if dist < closest_dist:
                closest_dist = dist
                closest = ss

        st.info("No signals at current z-score levels. The engine scans every 30 minutes during market hours.")
        if closest:
            tickers = closest.pair_id.split(":")[1].replace("-", " / ")
            st.caption(f"Nearest to entry: **{tickers}** at z={closest.current_zscore:+.2f} ({closest_dist:.2f} from trigger)")

    with st.expander("What is a signal?"):
        st.markdown("""
A **signal** fires when the price spread between two paired stocks stretches far enough from its historical average
that we expect it to snap back. Think of it like a rubber band — the further it stretches (higher z-score), the
stronger the pull back to normal.

- At **z = +/-2.0**, the system enters a trade (buys the underperforming stock, shorts the outperforming one).
- At **z = +/-0.5**, the system exits (spread has reverted, take profit).
- At **z = +/-3.5**, the system stops out (relationship may be permanently broken, cut losses).

Trades execute automatically on Alpaca. No approval needed — the 5 risk rules are the gatekeeper.
""")

    st.divider()

    # Open positions
    if positions:
        st.subheader("Open Positions")
        pos_data = []
        for p in positions:
            pnl_pct_pos = (p.unrealized_pnl / (p.avg_entry * p.qty)) * 100 if p.qty else 0
            pos_data.append({
                "Side": p.side.upper(), "Ticker": p.ticker, "Qty": p.qty,
                "Entry": f"${p.avg_entry:.2f}", "Current": f"${p.current_price:.2f}",
                "P&L $": f"${p.unrealized_pnl:+,.2f}", "P&L %": f"{pnl_pct_pos:+.1f}%",
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    else:
        st.subheader("Open Positions")
        st.caption("No positions open. The engine will automatically enter trades when z-scores hit entry thresholds.")

    st.divider()

    # Quick z-score summary table
    st.subheader("Active Pairs Summary")
    st.caption("All monitored pairs ranked by proximity to trade entry")

    pair_table = []
    for ss in sorted_pairs:
        tickers = ss.pair_id.split(":")[1]
        long_t, short_t = tickers.split("-")
        pod_obj = pm.get_pod(ss.pod_id)
        z = ss.current_zscore
        dist = min(abs(z - 2.0), abs(z + 2.0))
        pair_table.append({
            "Long": long_t, "Short": short_t,
            "Pod": pod_obj.name if pod_obj else ss.pod_id,
            "Z-Score": f"{z:+.2f}",
            "To Entry": f"{dist:.2f}",
            "Half-Life": f"{ss.half_life:.1f}d",
            "Hurst": f"{ss.hurst_exponent:.2f}",
        })
    st.dataframe(pd.DataFrame(pair_table), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2: Pairs & Spreads
# ===========================================================================
with tab_pairs:
    st.subheader("Z-Score Monitor")
    st.caption(f"{len(engine.valid_pairs)} active pairs — sorted by proximity to entry")

    with st.expander("How to read these"):
        st.markdown("""
Each card shows a **pair trade** — two stocks whose prices normally move together.

The **z-score bar** shows where the spread currently sits:
- **Center (yellow)** = pair is in sync, no trade
- **Green zones (+/-2.0)** = spread has stretched enough to enter a trade
- **Red zones (+/-3.5)** = spread has gone too far, cut losses

**Half-life** = how many days the spread typically takes to revert halfway to normal. Shorter = faster profits.

**Hurst exponent** = below 0.5 means the spread tends to snap back (good for us). Above 0.5 means it trends (bad).
""")

    leg1, leg2, leg3, leg4 = st.columns(4)
    leg1.caption(":green[Green] = Entry zone (z > 2.0)")
    leg2.caption(":orange[Yellow] = Exit zone (z < 0.5)")
    leg3.caption(":red[Red] = Stop loss (z > 3.5)")
    leg4.caption(":blue[Blue dot] = Current position")

    left_col, right_col = st.columns(2)
    for i, ss in enumerate(sorted_pairs):
        tickers = ss.pair_id.split(":")[1]
        long_t, short_t = tickers.split("-")
        pod_obj = pm.get_pod(ss.pod_id)
        pod_name = pod_obj.name if pod_obj else ss.pod_id
        z = ss.current_zscore
        dist = min(abs(z - 2.0), abs(z + 2.0))

        col = left_col if i % 2 == 0 else right_col
        with col:
            with st.container(border=True):
                h1, h2 = st.columns([3, 1])
                with h1:
                    st.markdown(f"**{long_t} / {short_t}**")
                    st.caption(pod_name)
                with h2:
                    if abs(z) >= 2.0: st.markdown(f":green[**z = {z:+.2f}**]")
                    elif abs(z) >= 1.5: st.markdown(f"**z = {z:+.2f}**")
                    else: st.markdown(f"z = {z:+.2f}")

                st.plotly_chart(make_zscore_chart(z), use_container_width=True, key=f"zbar_{ss.pair_id}")

                f1, f2, f3 = st.columns(3)
                f1.caption(f"Half-life: {ss.half_life:.1f}d")
                f2.caption(f"Hurst: {ss.hurst_exponent:.2f}")
                f3.caption(f"{dist:.2f} to entry")

    st.divider()

    # Spread charts
    st.subheader("Spread History")
    st.caption("Select pairs to view historical spread and z-score evolution")

    with st.expander("How to read these charts"):
        st.markdown("""
**Top chart (Log Spread)** — The price ratio between the two stocks over time. The dashed line is the rolling average.
When the blue line moves away from the dashed line, the spread is "stretching."

**Bottom chart (Z-Score)** — How many standard deviations the spread is from its mean:
- Crossing **green lines** (+/-2.0) = entry signal (spread is stretched, trade it)
- Crossing **yellow lines** (+/-0.5) = exit signal (spread has reverted, take profit)
- Crossing **red lines** (+/-3.5) = stop loss (relationship broken, cut losses)
""")

    pair_options = [ss.pair_id for ss in sorted_pairs]
    if pair_options:
        selected = st.multiselect(
            "Select pairs", pair_options, default=pair_options[:2],
            format_func=lambda x: x.split(":")[1].replace("-", " / ") + f"  ({x.split(':')[0]})",
        )
        for pair_id in selected:
            ss = engine.valid_pairs[pair_id]
            tickers = pair_id.split(":")[1]
            long_t, short_t = tickers.split("-")

            spread = get_spread_history(provider, long_t, short_t, days=60)
            if spread is None or len(spread) < 20:
                st.warning(f"Insufficient data for {tickers}")
                continue

            window = settings.get("pairs", {}).get("spread", {}).get("zscore_lookback", 20)
            s = pd.Series(spread)
            mean = s.rolling(window).mean()
            std = s.rolling(window).std()
            z_series = (s - mean) / std

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08)
            fig.add_trace(go.Scatter(y=spread, mode="lines", name="Spread",
                                      line=dict(color="#3b82f6", width=2),
                                      fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"), row=1, col=1)
            fig.add_trace(go.Scatter(y=mean.values, mode="lines", name="Mean",
                                      line=dict(color="#8b949e", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(y=z_series.values, mode="lines", name="Z-Score",
                                      line=dict(color="#3b82f6", width=2),
                                      fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"), row=2, col=1)

            for level, color, dash in [
                (2.0, "#22c55e", "dot"), (-2.0, "#22c55e", "dot"),
                (0.5, "#f59e0b", "dot"), (-0.5, "#f59e0b", "dot"),
                (3.5, "#ef4444", "dash"), (-3.5, "#ef4444", "dash"),
            ]:
                fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1, opacity=0.6, row=2, col=1)

            fig.update_layout(height=400, showlegend=False,
                              title=dict(text=f"{long_t} / {short_t}", font=dict(size=14)))
            fig.update_yaxes(title_text="Log Spread", row=1, col=1)
            fig.update_yaxes(title_text="Z-Score", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True, key=f"spread_{pair_id}")


# ===========================================================================
# TAB 3: Thematic Pods
# ===========================================================================
with tab_pods:
    st.subheader("Thematic Pods")
    st.caption("Each pod is a macro-catalyst thesis that generates pair trade candidates")

    with st.expander("What are pods?"):
        st.markdown("""
A **pod** groups stocks around a single macro theme — like tariff policy shifts or the Fed chair transition.

Each pod has a **long side** (stocks that benefit from the catalyst) and a **short side** (stocks that get hurt).
The engine tests all possible long/short combinations to find pairs that are **cointegrated** — meaning their prices
are mathematically linked and tend to revert to a stable relationship.

**Confidence** = how strong the thesis is right now based on unfolding events.
""")

    for pod in pm.active_pods():
        valid_in_pod = [ss for ss in engine.valid_pairs.values() if ss.pod_id == pod.pod_id]
        confidence = pod.thesis.confidence
        conf_display = {"high": ":green[High]", "medium": ":orange[Medium]",
                        "low": ":red[Low]", "stale": "Stale"}.get(confidence, confidence)

        with st.expander(f"{pod.name}  |  Confidence: {confidence}  |  {len(valid_in_pod)} active pairs"):
            st.markdown(f"**Core thesis:** {pod.thesis.core.strip()}")
            st.markdown(f"**Long thesis:** {pod.thesis.long.strip()}")
            st.markdown(f"**Short thesis:** {pod.thesis.short.strip()}")

            col_l, col_s = st.columns(2)
            with col_l:
                st.markdown("**Long watchlist:**")
                for entry in pod.long_watchlist:
                    status_icon = "active" if entry.status == "active" else entry.status
                    st.caption(f"**{entry.ticker}** — {entry.name}: {entry.thesis}")
            with col_s:
                st.markdown("**Short watchlist:**")
                for entry in pod.short_watchlist:
                    st.caption(f"**{entry.ticker}** — {entry.name}: {entry.thesis}")

            if valid_in_pod:
                st.markdown("**Active pairs from this pod:**")
                for ss in valid_in_pod:
                    t = ss.pair_id.split(":")[1].replace("-", " / ")
                    z = ss.current_zscore
                    st.caption(f"{t}  |  z={z:+.2f}  |  half-life={ss.half_life:.1f}d  |  hurst={ss.hurst_exponent:.2f}")


# ===========================================================================
# TAB 4: Trade Log
# ===========================================================================
with tab_log:
    st.subheader("Trade Log")
    st.caption("Every order submitted to Alpaca paper trading — most recent first")

    orders = load_order_history()
    if orders:
        # Summary stats
        filled_orders = [o for o in orders if o["Status"] == "FILLED"]
        buys = [o for o in filled_orders if o["Side"] == "BUY"]
        sells = [o for o in filled_orders if o["Side"] == "SELL"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Orders", len(orders))
        c2.metric("Filled", len(filled_orders))
        c3.metric("Buy Orders", len(buys))
        c4.metric("Sell/Short Orders", len(sells))

        st.divider()

        # Filter
        status_filter = st.selectbox("Filter by status", ["All", "FILLED", "CANCELED", "ACCEPTED", "REJECTED"], index=0)
        ticker_filter = st.text_input("Filter by ticker", placeholder="e.g. VRT")

        filtered = orders
        if status_filter != "All":
            filtered = [o for o in filtered if o["Status"] == status_filter]
        if ticker_filter:
            filtered = [o for o in filtered if ticker_filter.upper() in o["Ticker"].upper()]

        if filtered:
            st.dataframe(
                pd.DataFrame(filtered),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time": st.column_config.TextColumn("Time", width="medium"),
                    "Side": st.column_config.TextColumn("Side", width="small"),
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Qty": st.column_config.NumberColumn("Qty", width="small"),
                    "Filled": st.column_config.NumberColumn("Filled", width="small"),
                    "Price": st.column_config.TextColumn("Avg Price", width="small"),
                    "Notional": st.column_config.TextColumn("Notional", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                },
            )
            st.caption(f"Showing {len(filtered)} of {len(orders)} orders")
        else:
            st.info("No orders match the filter")
    else:
        st.info("No order history available. Orders will appear here once the engine starts trading.")

    with st.expander("How to read the trade log"):
        st.markdown("""
- **BUY** = opening a long position (betting the stock goes up) or closing a short position
- **SELL** = opening a short position (betting it goes down) or closing a long position
- Pairs always appear as **two orders at the same time** — one buy and one sell. That's the long/short pair.
- **FILLED** = order completed successfully at the shown price
- **CANCELED** = order was canceled (usually because the other leg of the pair failed)
- **Notional** = total dollar value of the trade (price × quantity)
""")


# ===========================================================================
# TAB 5: Config & Risk
# ===========================================================================
with tab_config:
    st.subheader("Risk Rules")
    st.caption("These 5 rules are non-negotiable. No signal, no thesis, no conviction overrides them.")

    max_gross_ratio = hard_rules.get("max_gross_to_capital_ratio", 1.20)
    risk_data = [
        {"#": "1", "Rule": "Max Risk Per Trade",
         "Limit": f"{hard_rules.get('max_risk_per_trade_pct', 0.015)*100:.1f}% = ${capital * hard_rules.get('max_risk_per_trade_pct', 0.015):,.0f}",
         "Why": "One bad trade can't ruin the account"},
        {"#": "2", "Rule": "Max Gross Exposure",
         "Limit": f"${capital * max_gross_ratio:,.0f} ({max_gross_ratio:.0%})",
         "Why": "Limits damage from a 5% correlated move"},
        {"#": "3", "Rule": "Strategy Failure Detection",
         "Limit": f"Retest every {hard_rules.get('max_pair_age_without_retest_days', 7)}d, auto-retire on failure",
         "Why": "Strategies stop working. Detect early."},
        {"#": "4", "Rule": "Overnight Gap Protection",
         "Limit": f"${hard_rules.get('max_overnight_gross_exposure', 80000):,.0f} overnight, {hard_rules.get('gap_multiplier', 2.0)}x gap multiplier",
         "Why": "Markets can gap overnight. Size for the gap."},
        {"#": "5", "Rule": "Tail Risk / Circuit Breaker",
         "Limit": f"{hard_rules.get('circuit_breaker_drawdown', 0.10)*100:.0f}% drawdown = flatten everything",
         "Why": "'Once in 10 years' happens every few years."},
    ]
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    with st.expander("Explain these rules in plain English"):
        st.markdown("""
1. **Max Risk Per Trade** — The most you can lose on any single pair before it's force-closed. At $100K with 1.5% max risk, that's $1,500 per pair. This means even 5 consecutive losing pairs only costs $7,500 (7.5% of capital).

2. **Max Gross Exposure** — Total dollars at work can't exceed 120% of capital. This means a 5% move across all positions loses at most 6% of capital. Painful but survivable.

3. **Strategy Failure** — Every week, the engine re-runs the cointegration test on all pairs. If a pair no longer passes, it's automatically retired. Strategies decay — this catches it early.

4. **Gap Protection** — Overnight positions are reduced to $80K. The gap multiplier means position sizes assume the loss could be 2x the stop-loss distance (because markets can open far from where they closed).

5. **Circuit Breaker** — If the account drops 10% from its peak, every position is immediately closed. Then a 48-hour cooldown. This is the nuclear option that prevents catastrophic drawdown.
""")

    st.divider()

    st.subheader("Trading Parameters")
    col_p1, col_p2 = st.columns(2)

    spread_cfg = settings.get("pairs", {}).get("spread", {})
    sizing_cfg = settings.get("pairs", {}).get("sizing", {})
    disc_cfg = settings.get("pairs", {}).get("discovery", {})

    with col_p1:
        with st.container(border=True):
            st.markdown("**Signal Thresholds**")
            st.caption(f"Entry: +/-{spread_cfg.get('entry_zscore', 2.0)} sigma")
            st.caption(f"Exit: +/-{spread_cfg.get('exit_zscore', 0.5)} sigma")
            st.caption(f"Stop: +/-{spread_cfg.get('stop_zscore', 3.5)} sigma")
            st.caption(f"Z-score window: {spread_cfg.get('zscore_lookback', 20)} days")
            st.caption(f"Half-life range: {spread_cfg.get('min_half_life', 2)}-{spread_cfg.get('max_half_life', 15)} days")

    with col_p2:
        with st.container(border=True):
            st.markdown("**Sizing & Limits**")
            st.caption(f"Default leg size: ${sizing_cfg.get('default_leg_size', 8000):,}")
            st.caption(f"Max leg size: ${sizing_cfg.get('max_leg_size', 10000):,}")
            st.caption(f"Max pairs total: {disc_cfg.get('max_total_pairs', 15)}")
            st.caption(f"Max per pod: {disc_cfg.get('max_pairs_per_pod', 4)}")
            st.caption(f"Lean ratio: {sizing_cfg.get('max_lean_ratio', 0.55):.0%} / {1-sizing_cfg.get('max_lean_ratio', 0.55):.0%} max")

    st.divider()

    st.subheader("Data Source")
    st.caption("Prices fetched live from Polygon.io REST API (daily bars, adjusted). Cached in-memory for 5 minutes per ticker.")
    st.caption("Trade execution runs on a separate machine via cron (every 30 min, 6:30 AM - 1 PM PT weekdays).")
    st.caption("This dashboard is read-only — refreshing it will never trigger trades.")
