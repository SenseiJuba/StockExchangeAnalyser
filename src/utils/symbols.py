# Common stock symbols - S&P 500 + NASDAQ 100 + popular stocks
# This avoids runtime HTTP requests to Wikipedia/NASDAQ

SP500_TOP = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B", "UNH", "XOM",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT", "CSCO", "ABT", "ACN",
    "DHR", "NEE", "LIN", "ADBE", "TXN", "PM", "NKE", "UNP", "RTX", "ORCL",
    "CRM", "AMD", "INTC", "IBM", "QCOM", "HON", "LOW", "CAT", "BA", "GE",
    "AMGN", "SBUX", "INTU", "GS", "BLK", "AXP", "MDLZ", "ISRG", "DE", "ADI",
    "GILD", "BKNG", "SYK", "VRTX", "MMC", "REGN", "PLD", "CB", "TMUS", "CVS",
    "SCHW", "MO", "ZTS", "SO", "DUK", "BDX", "CI", "CME", "PNC", "TJX",
    "CL", "EOG", "ITW", "SLB", "MCO", "USB", "FDX", "EMR", "NOC", "BSX",
    "APD", "GD", "WM", "ORLY", "HUM", "SHW", "CCI", "PSA", "FCX", "LRCX",
]

NASDAQ100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "COST",
    "ASML", "PEP", "CSCO", "AZN", "ADBE", "NFLX", "AMD", "TMUS", "TXN", "INTC",
    "CMCSA", "INTU", "QCOM", "AMGN", "HON", "AMAT", "ISRG", "SBUX", "BKNG", "VRTX",
    "GILD", "ADI", "MDLZ", "ADP", "REGN", "LRCX", "PANW", "MU", "PYPL", "SNPS",
    "KLAC", "CDNS", "MELI", "MNST", "ORLY", "MAR", "CHTR", "ABNB", "CTAS", "FTNT",
    "MRVL", "CSX", "KDP", "NXPI", "ADSK", "PCAR", "KHC", "DXCM", "AEP", "WDAY",
    "PAYX", "CPRT", "EXC", "LULU", "ROST", "MCHP", "ODFL", "IDXX", "EA", "FAST",
    "VRSK", "CTSH", "XEL", "GEHC", "CSGP", "BKR", "FANG", "DDOG", "ANSS", "TEAM",
    "ZS", "ILMN", "ALGN", "BIIB", "ENPH", "SIRI", "LCID", "RIVN", "WBD", "JD",
]

POPULAR_CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD",
]

POPULAR_ETF = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM",
    "GLD", "SLV", "USO", "TLT", "HYG", "LQD", "XLF", "XLE", "XLK", "XLV",
    "ARKK", "ARKG", "VNQ", "SCHD", "VIG", "VYM", "BND", "AGG",
]

POPULAR_INTL = [
    "TSM", "BABA", "NIO", "SONY", "TM", "SAP", "SHOP", "SE", "MELI", "NU",
    "BP", "SHEL", "RIO", "VALE", "UBS", "CS", "HSBC", "BHP", "LFC", "HDB",
]


def get_all_symbols():
    """Get combined list of all symbols, deduplicated"""
    all_syms = set(SP500_TOP + NASDAQ100 + POPULAR_CRYPTO + POPULAR_ETF + POPULAR_INTL)
    return sorted(list(all_syms))


def get_symbols_by_category():
    """Get symbols organized by category"""
    return {
        "S&P 500": SP500_TOP,
        "NASDAQ 100": NASDAQ100,
        "Crypto": POPULAR_CRYPTO,
        "ETFs": POPULAR_ETF,
        "International": POPULAR_INTL,
    }
