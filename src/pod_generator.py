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
    # =====================================================================
    # 2024 CATALYSTS
    # =====================================================================

    # Q1 2024: AI boom accelerates (NVDA blowout earnings), Fed rate cut hopes, CRE stress
    "2024-01-02": [
        GeneratedPod(
            pod_id="ai_infrastructure_2024",
            name="AI Infrastructure Boom 2024",
            thesis_core="NVDA earnings blowout confirms AI capex cycle. Hyperscalers accelerating spend. Power/cooling infrastructure is the bottleneck — real revenue, real backlogs.",
            thesis_long="Power infrastructure, data center cooling, electrical contractors benefiting from AI buildout.",
            thesis_short="Overhyped AI application companies with no revenue path. Pure narrative plays.",
            confidence="high",
            long_tickers=[
                {"ticker": "VRT", "name": "Vertiv", "thesis": "Data center cooling, direct AI beneficiary"},
                {"ticker": "POWL", "name": "Powell Industries", "thesis": "Switchgear for data centers"},
                {"ticker": "ETN", "name": "Eaton", "thesis": "Power distribution, DC buildout"},
                {"ticker": "EME", "name": "EMCOR", "thesis": "Electrical construction, DC projects"},
                {"ticker": "MOD", "name": "Modine Mfg", "thesis": "Thermal management for DCs"},
                {"ticker": "GEV", "name": "GE Vernova", "thesis": "Gas turbines for DC power"},
                {"ticker": "MTZ", "name": "MasTec", "thesis": "Infrastructure construction"},
                {"ticker": "DLR", "name": "Digital Realty", "thesis": "Data center REIT"},
            ],
            short_tickers=[
                {"ticker": "AI", "name": "C3.ai", "thesis": "Enterprise AI, negative FCF"},
                {"ticker": "BBAI", "name": "BigBear.ai", "thesis": "Minimal revenue, narrative"},
                {"ticker": "SOUN", "name": "SoundHound", "thesis": "Tiny revenue vs valuation"},
                {"ticker": "PATH", "name": "UiPath", "thesis": "RPA declining, AI pivot failing"},
                {"ticker": "UPST", "name": "Upstart", "thesis": "AI lending, credit risk"},
                {"ticker": "DOCN", "name": "DigitalOcean", "thesis": "Small cloud, AI premium"},
                {"ticker": "INTA", "name": "Intapp", "thesis": "AI premium without AI rev"},
                {"ticker": "PRCT", "name": "PROCEPT BioRobotics", "thesis": "AI-adjacent, priced for perfection"},
            ],
            pair_hints=[
                {"long": "VRT", "short": "AI", "rationale": "Real DC revenue vs AI-washing"},
                {"long": "POWL", "short": "SOUN", "rationale": "Record backlog vs tiny revenue"},
            ],
            catalyst_date="2024-01-02",
            expiry_date="2024-06-30",
        ),
        GeneratedPod(
            pod_id="rate_cut_hope_2024",
            name="Rate Cut Euphoria 2024",
            thesis_core="Market pricing in 6-7 Fed rate cuts in 2024. Rate-sensitive sectors rallying on hope. But sticky inflation may delay cuts, punishing the over-optimistic.",
            thesis_long="Regional banks with asset-sensitive balance sheets that benefit from steepening.",
            thesis_short="Mortgage originators and fintechs already priced for aggressive cut cycle.",
            confidence="medium",
            long_tickers=[
                {"ticker": "PNFP", "name": "Pinnacle Financial", "thesis": "Regional bank, NIM expansion"},
                {"ticker": "FNB", "name": "F.N.B. Corp", "thesis": "Asset-sensitive bank"},
                {"ticker": "UMBF", "name": "UMB Financial", "thesis": "Yield curve steepening"},
                {"ticker": "GBCI", "name": "Glacier Bancorp", "thesis": "Community bank"},
                {"ticker": "ONB", "name": "Old National", "thesis": "Regional bank, scale"},
                {"ticker": "WTFC", "name": "Wintrust", "thesis": "Asset-sensitive, deposit-funded"},
                {"ticker": "SFBS", "name": "ServisFirst", "thesis": "High-growth regional"},
                {"ticker": "HWC", "name": "Hancock Whitney", "thesis": "Gulf South regional"},
            ],
            short_tickers=[
                {"ticker": "RKT", "name": "Rocket Companies", "thesis": "Priced for refi boom"},
                {"ticker": "UWMC", "name": "UWM Holdings", "thesis": "Wholesale mortgage"},
                {"ticker": "SOFI", "name": "SoFi", "thesis": "Rate-dependent valuation"},
                {"ticker": "LC", "name": "LendingClub", "thesis": "Consumer lending risk"},
                {"ticker": "OPEN", "name": "Opendoor", "thesis": "Priced in rate cuts"},
                {"ticker": "TREE", "name": "LendingTree", "thesis": "Refi-dependent volume"},
                {"ticker": "LMND", "name": "Lemonade", "thesis": "Rate-dependent growth"},
                {"ticker": "AGNC", "name": "AGNC Investment", "thesis": "Spread compression"},
            ],
            pair_hints=[
                {"long": "PNFP", "short": "RKT", "rationale": "Bank NIM vs mortgage volume"},
                {"long": "FNB", "short": "SOFI", "rationale": "Traditional bank vs fintech"},
            ],
            catalyst_date="2024-01-02",
            expiry_date="2024-06-30",
        ),
    ],

    # Q2 2024: Rate cuts fade, defense spending surge, consumer stress
    "2024-04-01": [
        GeneratedPod(
            pod_id="defense_geopolitical_2024",
            name="Defense Geopolitical Surge 2024",
            thesis_core="Ukraine aid packages, Israel-Hamas conflict, Taiwan tensions. Defense budgets expanding globally. Sub-tier suppliers with backlogs vs commercial aero facing delays.",
            thesis_long="Mid-cap defense: drones, munitions, cyber, subsystems with growing backlogs.",
            thesis_short="Commercial aerospace suppliers facing Boeing production issues and defense crowding-out.",
            confidence="high",
            long_tickers=[
                {"ticker": "KTOS", "name": "Kratos Defense", "thesis": "Tactical UAV programs"},
                {"ticker": "AVAV", "name": "AeroVironment", "thesis": "Small UAS, loitering munitions"},
                {"ticker": "AXON", "name": "Axon Enterprise", "thesis": "Law enforcement/military tech"},
                {"ticker": "PANW", "name": "Palo Alto Networks", "thesis": "Cyber defense"},
                {"ticker": "LDOS", "name": "Leidos", "thesis": "Defense IT, growing backlog"},
                {"ticker": "BWXT", "name": "BWX Technologies", "thesis": "Nuclear propulsion"},
                {"ticker": "MRCY", "name": "Mercury Systems", "thesis": "Defense electronics"},
                {"ticker": "TDG", "name": "TransDigm", "thesis": "Proprietary aero components"},
            ],
            short_tickers=[
                {"ticker": "HXL", "name": "Hexcel", "thesis": "Commercial aero composites"},
                {"ticker": "HWM", "name": "Howmet Aerospace", "thesis": "Commercial aero cycle risk"},
                {"ticker": "TXT", "name": "Textron", "thesis": "Diversified, commercial drag"},
                {"ticker": "ATRO", "name": "Astronics", "thesis": "Commercial aero lighting"},
                {"ticker": "SAIC", "name": "SAIC", "thesis": "IT services recompete"},
                {"ticker": "PSN", "name": "Parsons", "thesis": "Engineering completions"},
                {"ticker": "ESAB", "name": "ESAB Corp", "thesis": "Industrial crowding-out"},
                {"ticker": "VSAT", "name": "Viasat", "thesis": "Satellite integration risk"},
            ],
            pair_hints=[
                {"long": "KTOS", "short": "PSN", "rationale": "Drone growth vs completions"},
                {"long": "AVAV", "short": "HXL", "rationale": "Munitions vs commercial aero"},
            ],
            catalyst_date="2024-04-01",
            expiry_date="2024-09-30",
        ),
        GeneratedPod(
            pod_id="consumer_stress_2024",
            name="Consumer Stress Divergence 2024",
            thesis_core="Cumulative inflation crushing low-income consumers. Affluent consumers benefit from stock market highs and home equity. K-shaped recovery widening.",
            thesis_long="Luxury consumer, wealth management, premium home improvement.",
            thesis_short="Dollar stores, subprime lenders, value QSR chains with stressed customers.",
            confidence="high",
            long_tickers=[
                {"ticker": "RH", "name": "RH", "thesis": "Luxury home furnishings"},
                {"ticker": "BIRK", "name": "Birkenstock", "thesis": "Premium footwear"},
                {"ticker": "TPR", "name": "Tapestry", "thesis": "Coach/Kate Spade luxury"},
                {"ticker": "LPLA", "name": "LPL Financial", "thesis": "Wealth mgmt AUM growth"},
                {"ticker": "POOL", "name": "Pool Corp", "thesis": "High-end home improvement"},
                {"ticker": "TREX", "name": "Trex", "thesis": "Premium decking"},
                {"ticker": "EVR", "name": "Evercore", "thesis": "Wealth advisory"},
                {"ticker": "CPRI", "name": "Capri Holdings", "thesis": "Versace/MK luxury"},
            ],
            short_tickers=[
                {"ticker": "DG", "name": "Dollar General", "thesis": "Stressed consumer base"},
                {"ticker": "DLTR", "name": "Dollar Tree", "thesis": "Low-income traffic decline"},
                {"ticker": "CACC", "name": "Credit Acceptance", "thesis": "Subprime auto delinquencies"},
                {"ticker": "ALLY", "name": "Ally Financial", "thesis": "Auto lending credit risk"},
                {"ticker": "AFRM", "name": "Affirm", "thesis": "BNPL subprime exposure"},
                {"ticker": "QSR", "name": "Restaurant Brands", "thesis": "BK low-income traffic"},
                {"ticker": "WEN", "name": "Wendy's", "thesis": "QSR value customer pullback"},
                {"ticker": "JACK", "name": "Jack in the Box", "thesis": "Value QSR stressed"},
            ],
            pair_hints=[
                {"long": "RH", "short": "DG", "rationale": "Luxury vs dollar store"},
                {"long": "POOL", "short": "JACK", "rationale": "Premium home vs value QSR"},
            ],
            catalyst_date="2024-04-01",
            expiry_date="2024-09-30",
        ),
    ],

    # Q3 2024: First rate cut (Sept), election uncertainty (Biden out, Harris in)
    "2024-07-01": [
        GeneratedPod(
            pod_id="rate_cut_reality_2024",
            name="Rate Cut Reality Q3 2024",
            thesis_core="Fed finally cutting in September. But only 25bps — not the 50bps the market wanted. Rate-sensitive names already rallied. The cut is priced in. Sell the news on mortgage/BNPL, buy banks that actually benefit from steepening.",
            thesis_long="Regional banks with asset-sensitive balance sheets benefiting from actual curve steepening.",
            thesis_short="Mortgage/fintech names that rallied on cut expectations — now vulnerable to 'sell the news'.",
            confidence="medium",
            long_tickers=[
                {"ticker": "PNFP", "name": "Pinnacle Financial", "thesis": "NIM expansion"},
                {"ticker": "FNB", "name": "F.N.B. Corp", "thesis": "Asset-sensitive"},
                {"ticker": "UMBF", "name": "UMB Financial", "thesis": "Curve steepening"},
                {"ticker": "SFBS", "name": "ServisFirst", "thesis": "High-growth regional"},
                {"ticker": "WTFC", "name": "Wintrust", "thesis": "Chicago regional"},
                {"ticker": "CADE", "name": "Cadence Bank", "thesis": "Southeast NIM"},
                {"ticker": "BOKF", "name": "BOK Financial", "thesis": "Energy/Midwest bank"},
                {"ticker": "ONB", "name": "Old National", "thesis": "Scale benefits"},
            ],
            short_tickers=[
                {"ticker": "RKT", "name": "Rocket Companies", "thesis": "Refi boom priced in"},
                {"ticker": "UWMC", "name": "UWM Holdings", "thesis": "Sell the news"},
                {"ticker": "SOFI", "name": "SoFi", "thesis": "Rate-dependent valuation"},
                {"ticker": "OPEN", "name": "Opendoor", "thesis": "iBuying priced in cuts"},
                {"ticker": "LC", "name": "LendingClub", "thesis": "Credit risk persists"},
                {"ticker": "TREE", "name": "LendingTree", "thesis": "Volume not materializing"},
                {"ticker": "LMND", "name": "Lemonade", "thesis": "Growth model needs low rates"},
                {"ticker": "AGNC", "name": "AGNC Investment", "thesis": "Spread risk remains"},
            ],
            pair_hints=[
                {"long": "PNFP", "short": "RKT", "rationale": "Real NIM vs priced-in refi"},
                {"long": "SFBS", "short": "SOFI", "rationale": "Growing bank vs stretched fintech"},
            ],
            catalyst_date="2024-07-01",
            expiry_date="2024-12-31",
        ),
    ],

    # Q4 2024: Trump wins, tariff fears, crypto/fintech rally, defense rally
    "2024-10-01": [
        GeneratedPod(
            pod_id="trump_election_2024",
            name="Trump Election Trade 2024",
            thesis_core="Trump victory = tariff threats, deregulation, reshoring push. Domestic manufacturers benefit, importers face margin risk. Market rotating into 'Trump trades'.",
            thesis_long="Domestic steel/aluminum, reshoring beneficiaries, energy independence plays.",
            thesis_short="Import-heavy retailers and manufacturers facing tariff uncertainty.",
            confidence="high",
            long_tickers=[
                {"ticker": "NUE", "name": "Nucor", "thesis": "Domestic steel, tariff protection"},
                {"ticker": "CLF", "name": "Cleveland-Cliffs", "thesis": "Auto steel, domestic"},
                {"ticker": "STLD", "name": "Steel Dynamics", "thesis": "Domestic steel"},
                {"ticker": "CMC", "name": "Commercial Metals", "thesis": "Domestic rebar/steel"},
                {"ticker": "CENX", "name": "Century Aluminum", "thesis": "Domestic aluminum"},
                {"ticker": "RS", "name": "Reliance Steel", "thesis": "Metals distribution"},
                {"ticker": "ATKR", "name": "Atkore", "thesis": "Steel conduit domestic"},
                {"ticker": "KALU", "name": "Kaiser Aluminum", "thesis": "Specialty aluminum"},
            ],
            short_tickers=[
                {"ticker": "FIVE", "name": "Five Below", "thesis": "Heavy China imports"},
                {"ticker": "CROX", "name": "Crocs", "thesis": "Vietnam/China mfg"},
                {"ticker": "SONO", "name": "Sonos", "thesis": "China electronics"},
                {"ticker": "SN", "name": "SharkNinja", "thesis": "China appliances"},
                {"ticker": "FOXF", "name": "Fox Factory", "thesis": "Asia components"},
                {"ticker": "COLM", "name": "Columbia", "thesis": "Asia sourcing"},
                {"ticker": "DORM", "name": "Dorman", "thesis": "Auto parts import"},
                {"ticker": "CAL", "name": "Caleres", "thesis": "Footwear imports"},
            ],
            pair_hints=[
                {"long": "NUE", "short": "FIVE", "rationale": "Steel protection vs import cost"},
                {"long": "CENX", "short": "CROX", "rationale": "Aluminum tariff vs footwear import"},
            ],
            catalyst_date="2024-10-01",
            expiry_date="2025-03-31",
        ),
    ],

    # =====================================================================
    # 2025 CATALYSTS (existing)
    # =====================================================================

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

    # Q1 2026: SCOTUS IEEPA ruling, tariff unwinding, Fed chair transition
    "2026-01-02": [
        GeneratedPod(
            pod_id="tariff_unwind_2026",
            name="Tariff Regime Unwind 2026",
            thesis_core="SCOTUS IEEPA ruling invalidating tariffs. Importers get refunds + margin relief. Domestic producers lose protection. Section 122 replacement expiring July.",
            thesis_long="Import-heavy companies owed tariff refunds. Input cost declines expand margins.",
            thesis_short="Domestic producers that benefited from tariff protection facing renewed import competition.",
            confidence="high",
            long_tickers=[
                {"ticker": "FIVE", "name": "Five Below", "thesis": "Heavy China imports, refund claimant"},
                {"ticker": "CROX", "name": "Crocs", "thesis": "Footwear importer, margin relief"},
                {"ticker": "COLM", "name": "Columbia", "thesis": "Asia apparel, refund beneficiary"},
                {"ticker": "SN", "name": "SharkNinja", "thesis": "China appliances, margin expansion"},
                {"ticker": "SONO", "name": "Sonos", "thesis": "China electronics, cost relief"},
                {"ticker": "DORM", "name": "Dorman Products", "thesis": "Auto parts, import relief"},
                {"ticker": "FOXF", "name": "Fox Factory", "thesis": "Asia components, cost decline"},
                {"ticker": "CAL", "name": "Caleres", "thesis": "Footwear, tariff refund"},
            ],
            short_tickers=[
                {"ticker": "CLF", "name": "Cleveland-Cliffs", "thesis": "Loses tariff protection"},
                {"ticker": "NUE", "name": "Nucor", "thesis": "Domestic steel, margin compression"},
                {"ticker": "CMC", "name": "Commercial Metals", "thesis": "Tariff protection unwinding"},
                {"ticker": "CENX", "name": "Century Aluminum", "thesis": "Domestic aluminum, tariff loss"},
                {"ticker": "STLD", "name": "Steel Dynamics", "thesis": "Elevated margins at risk"},
                {"ticker": "KALU", "name": "Kaiser Aluminum", "thesis": "Specialty aluminum, protection loss"},
                {"ticker": "RS", "name": "Reliance Steel", "thesis": "Metals distribution margin risk"},
                {"ticker": "ATKR", "name": "Atkore", "thesis": "Steel conduit, import competition"},
            ],
            pair_hints=[
                {"long": "CROX", "short": "CLF", "rationale": "Footwear refund vs steel protection loss"},
                {"long": "SN", "short": "ATKR", "rationale": "China appliance relief vs steel conduit"},
            ],
            catalyst_date="2026-01-02",
            expiry_date="2026-06-30",
        ),
        GeneratedPod(
            pod_id="ai_infra_2026",
            name="AI Infra vs Hype 2026",
            thesis_core="AI capex >$500B projected. Physical infrastructure has real revenue. Application-layer AI still mostly narrative. Spread widening each earnings cycle.",
            thesis_long="Power infra, cooling, data center REITs with real backlogs.",
            thesis_short="AI-washing companies burning cash, narrative-driven valuations.",
            confidence="high",
            long_tickers=[
                {"ticker": "VRT", "name": "Vertiv", "thesis": "DC cooling/power"},
                {"ticker": "POWL", "name": "Powell Industries", "thesis": "Switchgear, record backlog"},
                {"ticker": "GEV", "name": "GE Vernova", "thesis": "Gas turbines for DCs"},
                {"ticker": "ETN", "name": "Eaton", "thesis": "DC power distribution"},
                {"ticker": "EME", "name": "EMCOR", "thesis": "Electrical construction"},
                {"ticker": "MOD", "name": "Modine", "thesis": "Thermal management"},
                {"ticker": "MTZ", "name": "MasTec", "thesis": "Infra construction"},
                {"ticker": "IRM", "name": "Iron Mountain", "thesis": "DC expansion"},
            ],
            short_tickers=[
                {"ticker": "AI", "name": "C3.ai", "thesis": "Negative FCF, AI-washing"},
                {"ticker": "BBAI", "name": "BigBear.ai", "thesis": "Minimal revenue"},
                {"ticker": "SOUN", "name": "SoundHound", "thesis": "Tiny revenue vs cap"},
                {"ticker": "PLTR", "name": "Palantir", "thesis": "Extreme narrative premium"},
                {"ticker": "PATH", "name": "UiPath", "thesis": "Growth decelerating"},
                {"ticker": "UPST", "name": "Upstart", "thesis": "AI lending, stretched"},
                {"ticker": "DOCN", "name": "DigitalOcean", "thesis": "AI premium on small cloud"},
                {"ticker": "RVTY", "name": "Revvity", "thesis": "AI-adjacent narrative"},
            ],
            pair_hints=[
                {"long": "VRT", "short": "AI", "rationale": "Real DC revenue vs AI-washing"},
                {"long": "POWL", "short": "SOUN", "rationale": "Backlog vs tiny revenue"},
            ],
            catalyst_date="2026-01-02",
            expiry_date="2026-06-30",
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
