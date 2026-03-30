"""
LLM Pod Generator — uses Claude API to create new thematic pods from news.

Live mode: Calls Claude API with current financial headlines to identify
emerging catalysts and generate pod YAML files.

Backtest mode: Uses pre-defined quarterly catalysts that were active in 2025
to simulate what the LLM would have generated at each point in time.

The key insight: static pods decay. The tariff thesis that works in Q1
may be dead by Q3. The system needs to continuously generate new pods
from fresh catalysts to maintain edge.
"""

import os
import yaml
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class GeneratedPod:
    pod_id: str
    name: str
    thesis_core: str
    thesis_long: str
    thesis_short: str
    confidence: str
    long_tickers: list[dict]   # [{ticker, name, thesis}, ...]
    short_tickers: list[dict]
    pair_hints: list[dict]     # [{long, short, rationale}, ...]
    catalyst_date: str         # when this catalyst emerged
    expiry_date: str           # when to re-evaluate (typically 60-90 days)


# =========================================================================
# BACKTEST MODE — pre-defined catalysts for 2025
# =========================================================================

CATALYSTS_2025 = {
    # Q1 2025: Trump inauguration, tariff threats, AI capex boom
    "2025-01-02": [
        GeneratedPod(
            pod_id="trump_tariff_threat",
            name="Trump Tariff Threat",
            thesis_core="Incoming Trump administration threatening broad tariffs on China, Mexico, Canada. Import-heavy companies vulnerable, domestic producers benefit.",
            thesis_long="Domestic manufacturers and reshoring beneficiaries that gain pricing power from tariff protection.",
            thesis_short="Import-dependent retailers and manufacturers with heavy China/Mexico supply chains facing margin compression.",
            confidence="high",
            long_tickers=[
                {"ticker": "NUE", "name": "Nucor", "thesis": "Domestic steel, tariff protection"},
                {"ticker": "CLF", "name": "Cleveland-Cliffs", "thesis": "Domestic steel/auto steel"},
                {"ticker": "STLD", "name": "Steel Dynamics", "thesis": "Domestic steel producer"},
                {"ticker": "CMC", "name": "Commercial Metals", "thesis": "Domestic rebar/steel"},
                {"ticker": "RS", "name": "Reliance Steel", "thesis": "Metals distribution, domestic"},
                {"ticker": "ATKR", "name": "Atkore", "thesis": "Steel conduit, domestic mfg"},
                {"ticker": "CENX", "name": "Century Aluminum", "thesis": "Domestic aluminum"},
                {"ticker": "KALU", "name": "Kaiser Aluminum", "thesis": "Specialty aluminum"},
            ],
            short_tickers=[
                {"ticker": "FIVE", "name": "Five Below", "thesis": "Heavy China imports"},
                {"ticker": "CROX", "name": "Crocs", "thesis": "Vietnam/China manufacturing"},
                {"ticker": "SONO", "name": "Sonos", "thesis": "China electronics mfg"},
                {"ticker": "SN", "name": "SharkNinja", "thesis": "China appliance mfg"},
                {"ticker": "FOXF", "name": "Fox Factory", "thesis": "Taiwan/China components"},
                {"ticker": "COLM", "name": "Columbia Sportswear", "thesis": "Asia apparel sourcing"},
                {"ticker": "DORM", "name": "Dorman Products", "thesis": "Auto parts importer"},
                {"ticker": "CAL", "name": "Caleres", "thesis": "Footwear, China/Vietnam"},
            ],
            pair_hints=[
                {"long": "NUE", "short": "FIVE", "rationale": "Steel (protection) vs retail (import cost)"},
                {"long": "CLF", "short": "CROX", "rationale": "Steel vs footwear importer"},
                {"long": "STLD", "short": "SN", "rationale": "Domestic steel vs China appliances"},
            ],
            catalyst_date="2025-01-02",
            expiry_date="2025-04-30",
        ),
        GeneratedPod(
            pod_id="ai_capex_surge",
            name="AI Capex Surge",
            thesis_core="Hyperscalers committing $200B+ to AI infrastructure in 2025. Physical infra (power, cooling, data centers) has real revenue. App-layer AI is mostly narrative.",
            thesis_long="Picks-and-shovels: power infrastructure, cooling, electrical contractors with data center backlogs.",
            thesis_short="AI application companies burning cash with no path to profitability. Valuation = narrative.",
            confidence="high",
            long_tickers=[
                {"ticker": "VRT", "name": "Vertiv", "thesis": "Data center cooling/power"},
                {"ticker": "POWL", "name": "Powell Industries", "thesis": "Switchgear for data centers"},
                {"ticker": "GEV", "name": "GE Vernova", "thesis": "Gas turbines for DC power"},
                {"ticker": "ETN", "name": "Eaton", "thesis": "DC power distribution"},
                {"ticker": "EME", "name": "EMCOR", "thesis": "Electrical construction"},
                {"ticker": "MOD", "name": "Modine Mfg", "thesis": "Thermal management"},
                {"ticker": "MTZ", "name": "MasTec", "thesis": "Infra construction"},
                {"ticker": "DLR", "name": "Digital Realty", "thesis": "Data center REIT"},
            ],
            short_tickers=[
                {"ticker": "AI", "name": "C3.ai", "thesis": "Enterprise AI, neg FCF"},
                {"ticker": "BBAI", "name": "BigBear.ai", "thesis": "Minimal revenue"},
                {"ticker": "SOUN", "name": "SoundHound", "thesis": "Tiny revenue vs valuation"},
                {"ticker": "UPST", "name": "Upstart", "thesis": "AI lending, stretched"},
                {"ticker": "PATH", "name": "UiPath", "thesis": "RPA-to-AI pivot stalling"},
                {"ticker": "PLTR", "name": "Palantir", "thesis": "Extreme narrative premium"},
                {"ticker": "DOCN", "name": "DigitalOcean", "thesis": "AI premium on small cloud"},
                {"ticker": "INTA", "name": "Intapp", "thesis": "AI premium without AI revenue"},
            ],
            pair_hints=[
                {"long": "VRT", "short": "AI", "rationale": "Real DC revenue vs AI-washing"},
                {"long": "POWL", "short": "SOUN", "rationale": "Record backlog vs tiny revenue"},
                {"long": "EME", "short": "BBAI", "rationale": "Real buildout vs narrative"},
            ],
            catalyst_date="2025-01-02",
            expiry_date="2025-06-30",
        ),
    ],
    # Q2 2025: Tariffs actually imposed, defense spending surge, rate uncertainty
    "2025-04-01": [
        GeneratedPod(
            pod_id="defense_surge_2025",
            name="Defense Spending Acceleration",
            thesis_core="NATO countries accelerating defense spending. US budget up 5%. Ukraine/Taiwan tensions driving urgency. Sub-tier suppliers with growing backlogs vs commercial aero facing crowding out.",
            thesis_long="Mid-cap defense suppliers with multi-year backlogs: drones, munitions, cyber, subsystems.",
            thesis_short="Commercial aerospace suppliers facing defense prioritization and production slowdowns.",
            confidence="high",
            long_tickers=[
                {"ticker": "KTOS", "name": "Kratos Defense", "thesis": "Tactical UAV programs"},
                {"ticker": "AVAV", "name": "AeroVironment", "thesis": "Small UAS, loitering munitions"},
                {"ticker": "AXON", "name": "Axon Enterprise", "thesis": "Law enforcement/military tech"},
                {"ticker": "PANW", "name": "Palo Alto Networks", "thesis": "Cyber defense"},
                {"ticker": "LDOS", "name": "Leidos", "thesis": "Defense IT, growing backlog"},
                {"ticker": "BWXT", "name": "BWX Technologies", "thesis": "Nuclear propulsion"},
                {"ticker": "MRCY", "name": "Mercury Systems", "thesis": "Defense electronics"},
                {"ticker": "RCAT", "name": "Red Cat Holdings", "thesis": "Military drones"},
            ],
            short_tickers=[
                {"ticker": "HXL", "name": "Hexcel", "thesis": "Commercial aero composites slowdown"},
                {"ticker": "HWM", "name": "Howmet Aerospace", "thesis": "Commercial aero cycle risk"},
                {"ticker": "TXT", "name": "Textron", "thesis": "Diversified, commercial drag"},
                {"ticker": "ATRO", "name": "Astronics", "thesis": "Commercial aero lighting"},
                {"ticker": "SAIC", "name": "SAIC", "thesis": "IT services recompete cycle"},
                {"ticker": "PSN", "name": "Parsons", "thesis": "Engineering contract completions"},
                {"ticker": "ESAB", "name": "ESAB Corp", "thesis": "Industrial capex crowding-out"},
                {"ticker": "VSAT", "name": "Viasat", "thesis": "Satellite comms integration risk"},
            ],
            pair_hints=[
                {"long": "KTOS", "short": "PSN", "rationale": "Drone growth vs engineering completions"},
                {"long": "AVAV", "short": "HXL", "rationale": "Munitions demand vs commercial aero"},
                {"long": "PANW", "short": "SAIC", "rationale": "Cyber growth vs IT recompete"},
            ],
            catalyst_date="2025-04-01",
            expiry_date="2025-09-30",
        ),
        GeneratedPod(
            pod_id="consumer_bifurcation",
            name="Consumer Bifurcation",
            thesis_core="High-income consumers benefiting from asset appreciation and tax cuts. Low-income crushed by cumulative inflation. Luxury vs discount spread widening.",
            thesis_long="Premium consumer brands, wealth management, luxury home improvement.",
            thesis_short="Dollar stores, subprime lenders, value QSR — stressed consumer base.",
            confidence="high",
            long_tickers=[
                {"ticker": "RH", "name": "RH", "thesis": "Luxury home, affluent customer"},
                {"ticker": "BIRK", "name": "Birkenstock", "thesis": "Premium footwear"},
                {"ticker": "TPR", "name": "Tapestry", "thesis": "Coach/Kate Spade"},
                {"ticker": "LPLA", "name": "LPL Financial", "thesis": "Wealth mgmt AUM growth"},
                {"ticker": "EVR", "name": "Evercore", "thesis": "Wealth mgmt + advisory"},
                {"ticker": "POOL", "name": "Pool Corp", "thesis": "High-end home improvement"},
                {"ticker": "TREX", "name": "Trex", "thesis": "Premium decking"},
                {"ticker": "CPRI", "name": "Capri Holdings", "thesis": "Versace/MK luxury"},
            ],
            short_tickers=[
                {"ticker": "DG", "name": "Dollar General", "thesis": "Stressed consumer"},
                {"ticker": "DLTR", "name": "Dollar Tree", "thesis": "Low-income traffic decline"},
                {"ticker": "CACC", "name": "Credit Acceptance", "thesis": "Subprime auto, delinquencies"},
                {"ticker": "ALLY", "name": "Ally Financial", "thesis": "Auto lending, credit risk"},
                {"ticker": "AFRM", "name": "Affirm", "thesis": "BNPL subprime exposure"},
                {"ticker": "QSR", "name": "Restaurant Brands", "thesis": "BK/Popeyes low-income traffic"},
                {"ticker": "WEN", "name": "Wendy's", "thesis": "QSR value customer pullback"},
                {"ticker": "JACK", "name": "Jack in the Box", "thesis": "Value QSR, stressed base"},
            ],
            pair_hints=[
                {"long": "RH", "short": "DG", "rationale": "Luxury vs dollar store — purest K spread"},
                {"long": "LPLA", "short": "AFRM", "rationale": "Wealth mgmt vs BNPL subprime"},
                {"long": "POOL", "short": "JACK", "rationale": "High-end home vs value QSR"},
            ],
            catalyst_date="2025-04-01",
            expiry_date="2025-09-30",
        ),
    ],
    # Q3 2025: Market correction fears, rate cut expectations, reshoring momentum
    "2025-07-01": [
        GeneratedPod(
            pod_id="rate_sensitivity_q3",
            name="Rate Cut Positioning",
            thesis_core="Fed signaling potential cuts in H2 2025. Regional banks benefit from NIM expansion. Mortgage originators already priced for cuts — vulnerable if delayed.",
            thesis_long="Regional banks with asset-sensitive balance sheets that benefit either way.",
            thesis_short="Mortgage originators and fintechs overextended on rate cut expectations.",
            confidence="medium",
            long_tickers=[
                {"ticker": "PNFP", "name": "Pinnacle Financial", "thesis": "Regional bank, NIM expansion"},
                {"ticker": "FNB", "name": "F.N.B. Corp", "thesis": "Asset-sensitive bank"},
                {"ticker": "UMBF", "name": "UMB Financial", "thesis": "Yield curve steepening"},
                {"ticker": "GBCI", "name": "Glacier Bancorp", "thesis": "Rate-sensitive mortgage book"},
                {"ticker": "ONB", "name": "Old National", "thesis": "Regional bank, scale"},
                {"ticker": "SFBS", "name": "ServisFirst", "thesis": "High-growth regional"},
                {"ticker": "WTFC", "name": "Wintrust", "thesis": "Chicago regional, asset-sensitive"},
                {"ticker": "CADE", "name": "Cadence Bank", "thesis": "Southeast bank, NIM"},
            ],
            short_tickers=[
                {"ticker": "RKT", "name": "Rocket Companies", "thesis": "Priced for refi boom"},
                {"ticker": "UWMC", "name": "UWM Holdings", "thesis": "Wholesale mortgage, overextended"},
                {"ticker": "SOFI", "name": "SoFi", "thesis": "Rate-dependent valuation"},
                {"ticker": "LC", "name": "LendingClub", "thesis": "Credit deterioration"},
                {"ticker": "OPEN", "name": "Opendoor", "thesis": "Priced in rate cuts"},
                {"ticker": "TREE", "name": "LendingTree", "thesis": "Volume-dependent refi"},
                {"ticker": "LMND", "name": "Lemonade", "thesis": "Rate-dependent growth"},
                {"ticker": "AGNC", "name": "AGNC Investment", "thesis": "Spread compression risk"},
            ],
            pair_hints=[
                {"long": "PNFP", "short": "RKT", "rationale": "Bank NIM vs mortgage volume"},
                {"long": "FNB", "short": "SOFI", "rationale": "Traditional bank vs fintech"},
                {"long": "UMBF", "short": "UWMC", "rationale": "Regional bank vs wholesale mtg"},
            ],
            catalyst_date="2025-07-01",
            expiry_date="2025-12-31",
        ),
    ],
}


def get_backtest_pods(as_of_date: str) -> list[GeneratedPod]:
    """Get all pods that should be active as of a given date."""
    active = []
    for launch_date, pods in CATALYSTS_2025.items():
        if launch_date <= as_of_date:
            for pod in pods:
                if pod.expiry_date >= as_of_date:
                    active.append(pod)
    return active


# =========================================================================
# LIVE MODE — Claude API pod generation
# =========================================================================

SYSTEM_PROMPT = """You are a quantitative analyst identifying thematic catalysts for pairs trading.

Given financial news headlines, identify ONE emerging macro catalyst that creates clear winners and losers.

Output a JSON object with:
{
  "pod_id": "snake_case_name",
  "name": "Human Readable Name",
  "thesis_core": "2-3 sentence thesis",
  "thesis_long": "Why the long side benefits",
  "thesis_short": "Why the short side gets hurt",
  "confidence": "high/medium/low",
  "long_tickers": [{"ticker": "XXX", "name": "Company Name", "thesis": "Why this stock benefits"}],
  "short_tickers": [{"ticker": "XXX", "name": "Company Name", "thesis": "Why this stock gets hurt"}],
  "pair_hints": [{"long": "XXX", "short": "YYY", "rationale": "Why these pair well"}]
}

Requirements:
- 6-10 tickers per side
- Tickers must be US-listed, liquid (>$500M market cap), real companies
- Long and short sides must be exposed to the SAME catalyst from opposite sides
- The pair should have a reasonable chance of cointegrating (same sector or shared macro driver)
- Focus on mid-caps ($1B-$15B) — less analyst coverage = more edge
"""


def generate_pod_from_news(headlines: list[str], api_key: Optional[str] = None) -> Optional[GeneratedPod]:
    """Call Claude API to generate a pod from news headlines."""
    if anthropic is None:
        raise ImportError("pip install anthropic")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=key)

    prompt = "Here are today's financial headlines:\n\n"
    for h in headlines:
        prompt += f"- {h}\n"
    prompt += "\nIdentify the strongest emerging catalyst for a pairs trade and output the JSON."

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    text = response.content[0].text
    # Extract JSON from response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0:
        return None

    data = json.loads(text[start:end])

    return GeneratedPod(
        pod_id=data["pod_id"],
        name=data["name"],
        thesis_core=data["thesis_core"],
        thesis_long=data["thesis_long"],
        thesis_short=data["thesis_short"],
        confidence=data.get("confidence", "medium"),
        long_tickers=data["long_tickers"],
        short_tickers=data["short_tickers"],
        pair_hints=data.get("pair_hints", []),
        catalyst_date=date.today().isoformat(),
        expiry_date=(date.today().replace(month=date.today().month + 3) if date.today().month <= 9
                     else date.today().replace(year=date.today().year + 1, month=date.today().month - 9)).isoformat(),
    )
