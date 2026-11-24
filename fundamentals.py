import pandas as pd
import requests
import argparse
import io, time
import yfinance as yf

import edgar
from pathlib import Path

CACHE = Path(__file__).parent / "fundamentals.csv"
WEIGHTS = {"value": 0.4, "quality":0.4, "growth":0.2}
MAX_AGE = 24 * 60 * 60

HIST_COLS = ["histYears", "avgNetIncome", "avgMargin", "marginStd", "revCagr"]


def history_metrics(tk: yf.Ticker) -> dict:
    """Summarize the last few years of annual results (yfinance gives ~4).

    One great year can make a cyclical company look cheap, profitable and fast-growing
    all at once. Averaging over several years shows what a *normal* year looks like.
    """
    try:
        fin = tk.income_stmt  # rows = line items, columns = fiscal years (newest first)
    except Exception:
        return {}
    if fin is None or fin.empty:
        return {}

    def line(*names):
        for n in names:
            if n in fin.index:
                return fin.loc[n]
        return None

    rev = line("Total Revenue", "Operating Revenue")
    ni = line("Net Income", "Net Income Common Stockholders")
    if rev is None or ni is None:
        return {}

    years = pd.DataFrame({"rev": rev, "ni": ni}).dropna().sort_index()  # oldest first
    years = years[years.rev > 0]
    n = len(years)
    if n < 3:  # too little history to say anything about cycles
        return {"histYears": n}

    margins = years.ni / years.rev
    return {
        "histYears": n,
        "avgNetIncome": years.ni.mean(),
        "avgMargin": margins.mean(),
        "marginStd": margins.std(),
        "revCagr": (years.rev.iloc[-1] / years.rev.iloc[0]) ** (1 / (n - 1)) - 1,
    }


def get_data() -> pd.DataFrame:
    # 1. Get the S&P 500 list from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S&P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    sp500 = pd.read_html(io.StringIO(resp.text))[0]
    tickers = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()

    # 2. Pull fundamentals for each ticker
    FIELDS = ["sector", "marketCap", "trailingPE", "freeCashflow", "profitMargins",
            "returnOnEquity", "debtToEquity", "revenueGrowth", "dividendYield"]

    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            tk = yf.Ticker(t)
            info = tk.info
            rows.append({"ticker": t, **{f: info.get(f) for f in FIELDS}, **history_metrics(tk)})
        except Exception as e:
            print(f"skip {t}: {e}")

        print(f"[{i}/{len(tickers)}] {t}")
        time.sleep(0.5)

    # 3. Cache it
    df = pd.DataFrame(rows)
    for c in HIST_COLS:  # make sure the columns exist even if no ticker had history
        if c not in df.columns:
            df[c] = float("nan")
    df.to_csv(CACHE, index=False)
    return df


def apply_edgar_history(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Swap Yahoo's ~4-year history for SEC EDGAR's longer one where available."""
    df = df.copy()
    df["histSource"] = "Yahoo"
    if not edgar.ANNUAL_FILE.exists():
        print("Tip: run `python edgar.py` to use 8+ years of SEC history instead of Yahoo's ~4.\n")
        return df

    hist = edgar.history_asof(edgar.load_annual(), years=years).reindex(df.ticker)
    use = (hist.histYears >= 3).to_numpy()  # only replace when EDGAR has enough years
    for c in HIST_COLS:
        df.loc[use, c] = hist[c].to_numpy()[use]
    df.loc[use, "histSource"] = "EDGAR"
    print(f"History: EDGAR (up to {years} yrs) for {use.sum()} stocks, Yahoo for {(~use).sum()}\n")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["sector"])
    df = df[df.profitMargins > 0]
    df = df[df.marketCap > 2e9]
    df = df[(df.freeCashflow > 0) | df.freeCashflow.isna()]
    df = df[(df.debtToEquity < 200) | df.debtToEquity.isna()]
    df = df.copy()
    df["fcf_yield"] = df.freeCashflow / df.marketCap
    df["earn_yield"] = 1/df.trailingPE

    # Cycle-adjusted versions. Fall back to the current-year number when a stock
    # doesn't have enough history.
    df["norm_earn_yield"] = (df.avgNetIncome / df.marketCap).fillna(df.earn_yield)
    df["base_margin"] = df.avgMargin.fillna(df.profitMargins)
    df["growth_basis"] = df.revCagr.fillna(df.revenueGrowth)
    # How far today's margin is above its multi-year average: 1.0 = normal year,
    # 2.0 = earning twice the usual margin (possible cyclical peak).
    df["peak_ratio"] = (df.profitMargins / df.avgMargin).where(df.avgMargin > 0)
    # Profitable now but lost money on average over the period = the strongest
    # possible peak signal, so rank it as worse than any finite ratio.
    df.loc[df.avgMargin <= 0, "peak_ratio"] = float("inf")

    return df

def pick(df: pd.DataFrame, n=15, max_per_sector=3) -> pd.DataFrame:
    return df.groupby("sector").head(max_per_sector).head(n)

def score(df: pd.DataFrame, weights: dict = WEIGHTS) -> pd.DataFrame:
    df = df.copy()
    df["roe_rank"] = df.groupby("sector")["returnOnEquity"].rank(pct=True)
    df["fcf_rank"] = df.groupby("sector")["fcf_yield"].rank(pct=True)
    df["earn_rank"] = df.groupby("sector")["norm_earn_yield"].rank(pct=True)
    df["margin_rank"] = df.groupby("sector")["base_margin"].rank(pct=True)
    df["growth_rank"] = df.groupby("sector")["growth_basis"].rank(pct=True)
    # ascending=False: LOWER is better here, so the steadiest companies get the top ranks
    # Missing history counts as neutral (0.5) - otherwise a stock with no history
    # would skip these checks entirely and score *better* than one that was checked.
    df["steady_rank"] = df.groupby("sector")["marginStd"].rank(pct=True, ascending=False).fillna(0.5)
    df["peak_rank"] = df.groupby("sector")["peak_ratio"].rank(pct=True, ascending=False).fillna(0.5)

    df["value"] = df[["fcf_rank", "earn_rank"]].mean(axis=1)
    df["quality"] = df[["roe_rank", "margin_rank", "steady_rank", "peak_rank"]].mean(axis=1)
    df["growth"] = df["growth_rank"]

    df["score"] = sum(df[f] * w for f, w in weights.items())

    return df.sort_values("score", ascending=False)

def fmt(val, spec: str, suffix: str = "") -> str:
    """Format a number, or show 'n/a' if it's missing (e.g. banks have no FCF)."""
    return "n/a" if pd.isna(val) else f"{val:{spec}}{suffix}"


def top_pct(rank) -> str:
    """Turn a 0-1 sector percentile into 'top X% of sector'."""
    if pd.isna(rank):
        return ""
    return f" (top {max(1, round((1 - rank) * 100))}% of sector)"


def explain(row: pd.Series) -> str:
    lines = [f"{row.ticker}  ({row.sector})  score {row.score:.2f}"]
    lines.append(
        f"  - Value:   FCF yield {fmt(row.fcf_yield, '.1%')}{top_pct(row.fcf_rank)}, "
        f"P/E {fmt(row.trailingPE, '.1f')}, "
        f"normalized P/E {fmt(1 / row.norm_earn_yield if row.norm_earn_yield > 0 else float('nan'), '.1f')}"
        f"{top_pct(row.earn_rank)}"
    )
    lines.append(
        f"  - Quality: ROE {fmt(row.returnOnEquity, '.0%')}{top_pct(row.roe_rank)}, "
        f"margin {fmt(row.profitMargins, '.0%')} now / {fmt(row.avgMargin, '.0%')} avg{top_pct(row.margin_rank)}"
    )
    lines.append(
        f"  - Growth:  revenue {fmt(row.revenueGrowth, '+.0%')} latest, "
        f"{fmt(row.revCagr, '+.0%', '/yr')} multi-year{top_pct(row.growth_rank)}"
    )
    lines.append(f"  - History: {fmt(row.histYears, '.0f')} yrs from {row.histSource}")
    if pd.isna(row.avgMargin):
        lines.append("  - Note: not enough history to check for a cyclical peak")
    elif row.avgMargin <= 0:
        lines.append(
            f"  - WARNING: lost money on average over {int(row.histYears)} yrs but is profitable now "
            "- earnings may be at a cyclical peak"
        )
    elif not pd.isna(row.peak_ratio) and row.peak_ratio >= 1.5:
        lines.append(
            f"  - WARNING: margin is {row.peak_ratio:.1f}x its {int(row.histYears)}-yr average "
            "- earnings may be at a cyclical peak"
        )
    # dividendYield is already in percent units (1.69 means 1.69%)
    if not pd.isna(row.dividendYield) and row.dividendYield > 0:
        lines.append(f"  - Dividend: {row.dividendYield:.2f}%")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank S&P 500 stocks by value, quality and growth.")
    parser.add_argument("--weights", nargs=3, type=float, metavar=("VALUE", "QUALITY", "GROWTH"),
                        default=[WEIGHTS["value"], WEIGHTS["quality"], WEIGHTS["growth"]],
                        help="factor weights (rescaled to sum to 1), default: 0.4 0.4 0.2")
    parser.add_argument("--top", type=int, default=15, help="how many stocks to show (default: 15)")
    parser.add_argument("--max-per-sector", type=int, default=3,
                        help="cap on picks from any one sector (default: 3)")
    parser.add_argument("--sector", action="append",
                        help='only include this sector; repeat for several, e.g. --sector Technology --sector Healthcare')
    parser.add_argument("--refresh", action="store_true", help="re-download data even if the cache is fresh")
    parser.add_argument("--history-years", type=int, default=8,
                        help="years of SEC history to average over (default: 8)")
    args = parser.parse_args()

    if any(w < 0 for w in args.weights) or sum(args.weights) == 0:
        parser.error("weights must be non-negative and not all zero")
    return args


def main():
    args = parse_args()

    cache_fresh = CACHE.exists() and time.time() - CACHE.stat().st_mtime < MAX_AGE
    if cache_fresh and not args.refresh:
        df = pd.read_csv(CACHE)
        if "avgNetIncome" not in df.columns:
            print("Cached data has no multi-year history yet - re-downloading (takes ~15 min)...")
            df = get_data()
    else:
        df = get_data()

    df = apply_edgar_history(df, args.history_years)
    df = clean_data(df)

    if args.sector:
        wanted = {s.lower() for s in args.sector}
        df = df[df.sector.str.lower().isin(wanted)]
        if df.empty:
            print("No stocks matched those sectors. Valid sectors are:")
            print("  " + "\n  ".join(sorted(pd.read_csv(CACHE).sector.dropna().unique())))
            return

    total = sum(args.weights)
    weights = dict(zip(["value", "quality", "growth"], (w / total for w in args.weights)))
    print("Weights: " + ", ".join(f"{k} {v:.0%}" for k, v in weights.items()) + "\n")

    df = score(df, weights)
    picks = pick(df, n=args.top, max_per_sector=args.max_per_sector)
    for _, row in picks.iterrows():
        print(explain(row))
        print()


if __name__ == "__main__":
    main()
