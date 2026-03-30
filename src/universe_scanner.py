"""
Universe Scanner — scans S&P 500 sectors for cointegrated pairs.

No thematic thesis required. Pure statistical arbitrage.
Pairs are discovered by scanning within sectors (financials vs financials,
industrials vs industrials, etc.) since same-sector stocks are more likely
to cointegrate.

This runs alongside thematic pods to increase signal flow.
"""

# Top ~200 liquid mid/large-caps organized by sector for pair scanning
# Focus on $2B-$50B market cap — enough liquidity, less analyst coverage

SECTOR_UNIVERSE = {
    "financials": [
        "PNFP", "FNB", "UMBF", "GBCI", "ONB", "SFBS", "WTFC", "CADE", "HWC", "BOKF",
        "ALLY", "SOFI", "LC", "RKT", "UWMC", "CACC", "AFRM", "LPLA", "EVR", "AGNC",
        "NLY", "TREE", "LMND", "OPEN",
    ],
    "industrials_defense": [
        "KTOS", "AVAV", "AXON", "LDOS", "BWXT", "MRCY", "TDG", "PSN", "SAIC", "TXT",
        "HXL", "HWM", "ESAB", "CW", "VSAT", "EME", "MTZ", "ATRO",
    ],
    "technology": [
        "PANW", "VRT", "POWL", "GEV", "ETN", "MOD", "DLR", "IRM",
        "AI", "BBAI", "SOUN", "PLTR", "PATH", "UPST", "DOCN", "INTA", "PRCT", "RVTY",
    ],
    "consumer_discretionary": [
        "RH", "BIRK", "TPR", "CPRI", "CROX", "CAL", "COLM", "FIVE",
        "DG", "DLTR", "POOL", "TREX", "QSR", "WEN", "JACK", "DENN",
        "PTON", "PLNT", "SN", "SONO", "FOXF",
    ],
    "materials_metals": [
        "CLF", "NUE", "STLD", "CMC", "CENX", "KALU", "RS", "ATKR",
        "ZEUS", "MTRN", "ATI",
    ],
    "consumer_staples_food": [
        "CASY", "WOOF", "DORM",
    ],
    "energy_utilities": [
        "GEV", "HUBB",
    ],
}


def get_all_universe_tickers() -> set[str]:
    """Get all tickers across all sectors."""
    tickers = set()
    for sector_tickers in SECTOR_UNIVERSE.values():
        tickers.update(sector_tickers)
    return tickers


def get_sector_pair_candidates() -> list[tuple[str, str, str]]:
    """
    Generate all within-sector pair candidates.
    Returns list of (ticker_a, ticker_b, sector).
    Only pairs within the same sector — cross-sector pairs rarely cointegrate.
    """
    candidates = []
    for sector, tickers in SECTOR_UNIVERSE.items():
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                candidates.append((tickers[i], tickers[j], sector))
    return candidates


def get_sector_for_ticker(ticker: str) -> str:
    """Look up which sector a ticker belongs to."""
    for sector, tickers in SECTOR_UNIVERSE.items():
        if ticker in tickers:
            return sector
    return "unknown"
