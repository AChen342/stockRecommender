"""Backtest the screener: would following it in past years have beaten the S&P 500?

For each rebalance date (default: April 1 of every year, after most annual reports are out):
  1. Rebuild every stock's numbers using ONLY SEC filings published before that date.
  2. Run the same clean_data -> score -> pick steps as fundamentals.py.
  3. "Buy" the picks in equal amounts, hold one year, repeat.

Compared against:
  - SPY (the S&P 500 index fund)
  - Every eligible stock, equal-weighted. This matters: our stock list is TODAY's
    S&P 500, which leaves out companies that failed (survivorship bias) and makes
    everything look better. If the screener only matches this line, it has no real edge.

Usage:
    python backtest.py
    python backtest.py --start 2015 --top 20 --weights 0.6 0.3 0.1 --show-picks
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

import edgar
import fundamentals as F

HERE = Path(__file__).parent
PRICE_DIR = HERE / "price_cache"
RESULTS_FILE = HERE / "backtest_results.csv"
PICKS_FILE = HERE / "backtest_picks.csv"
BENCHMARK = "SPY"


# ---------------------------------------------------------------- prices

def load_prices(tickers: list, start: pd.Timestamp, refresh: bool = False) -> dict:
    """Daily prices for every ticker, cached in price_cache/.

    close     - adjusted for splits only  -> used for market cap (price x shares)
    adj_close - adjusted for splits AND dividends -> used for returns
    splits    - split ratio on each split date (0 = no split)
    """
    PRICE_DIR.mkdir(exist_ok=True)
    files = {k: PRICE_DIR / f"{k}.csv" for k in ("close", "adj_close", "splits")}
    cached = all(f.exists() for f in files.values())
    if cached and not refresh and time.time() - files["close"].stat().st_mtime < F.MAX_AGE:
        frames = {k: pd.read_csv(f, index_col=0, parse_dates=True) for k, f in files.items()}
        if set(tickers) <= set(frames["close"].columns) and frames["close"].index.min() <= start:
            return frames

    print(f"Downloading daily prices for {len(tickers)} tickers (1-3 minutes)...")
    data = yf.download(tickers, start=start - pd.Timedelta(days=10), auto_adjust=False,
                       actions=True, progress=True, threads=True)
    fields = set(data.columns.get_level_values(0))
    frames = {
        "close": data["Close"],
        "adj_close": data["Adj Close"],
        "splits": data["Stock Splits"].fillna(0) if "Stock Splits" in fields
                  else pd.DataFrame(0.0, index=data.index, columns=data["Close"].columns),
    }
    for k, f in files.items():
        frames[k].to_csv(f)
    return frames


def trading_day(index: pd.DatetimeIndex, date: pd.Timestamp):
    """First trading day on or after `date` (None if it's in the future)."""
    later = index[index >= date]
    return later[0] if len(later) else None


def split_factor(splits: pd.Series, after: pd.Timestamp) -> float:
    """How many of today's shares one share at `after` became (e.g. 40 for NVDA 2021).

    Yahoo's prices are rewritten to today's share basis after every split, but SEC
    share counts are as reported at the time - this puts them on the same basis.
    """
    s = splits[(splits.index > after) & (splits > 0)]
    return float(s.prod()) if len(s) else 1.0


# ---------------------------------------------------------------- point-in-time snapshot

def snapshot(annual: pd.DataFrame, prices: dict, sectors: pd.Series, date: pd.Timestamp,
             history_years: int) -> pd.DataFrame:
    """Build the same columns fundamentals.csv has, as investors would have seen them on `date`."""
    day = trading_day(prices["close"].index, date)
    known = annual[annual.filed <= date].sort_values("period_end")
    known = known[known.ticker.isin(sectors.index)]

    g = known.groupby("ticker")
    latest = g.tail(1).set_index("ticker")
    prior = g.tail(2).groupby("ticker").head(1).set_index("ticker").reindex(latest.index)
    prior.loc[(g.size().reindex(latest.index) < 2).to_numpy()] = float("nan")  # no earlier year

    # Skip companies whose latest annual report is very old (stopped filing / not yet public).
    latest = latest[latest.period_end >= date - pd.Timedelta(days=550)]
    prior = prior.reindex(latest.index)

    rows = []
    for t, r in latest.iterrows():
        price = prices["close"].at[day, t] if t in prices["close"].columns else float("nan")
        if pd.isna(price) or pd.isna(r.diluted_shares) or r.diluted_shares <= 0:
            continue  # not trading yet, or can't compute market cap
        mcap = price * r.diluted_shares * split_factor(prices["splits"][t], r.filed)

        gap = (r.period_end - prior.at[t, "period_end"]).days if pd.notna(prior.at[t, "period_end"]) else None
        rev_growth = (r.revenue / prior.at[t, "revenue"] - 1
                      if gap and 300 <= gap <= 430 and prior.at[t, "revenue"] > 0 else float("nan"))
        rows.append({
            "ticker": t,
            "sector": sectors[t],
            "marketCap": mcap,
            "trailingPE": mcap / r.net_income if r.net_income > 0 else float("nan"),
            "freeCashflow": r.op_cash_flow - r.capex if pd.notna(r.capex) else float("nan"),
            "profitMargins": r.net_income / r.revenue if r.revenue > 0 else float("nan"),
            "returnOnEquity": r.net_income / r.equity if r.equity > 0 else float("nan"),
            "debtToEquity": r.lt_debt / r.equity * 100 if r.equity > 0 else float("nan"),  # percent, like Yahoo
            "revenueGrowth": rev_growth,
            "dividendYield": float("nan"),
            "period_end": r.period_end,
        })
    snap = pd.DataFrame(rows)
    if snap.empty:
        return snap

    # Share classes of one company (GOOG/GOOGL) have identical filings - keep one.
    snap = snap.drop_duplicates(subset=["period_end", "marketCap"]).drop(columns="period_end")

    hist = edgar.history_asof(annual, date, history_years).reindex(snap.ticker)
    for c in F.HIST_COLS:
        snap[c] = hist[c].to_numpy()
    snap["histSource"] = "EDGAR"
    return snap


# ---------------------------------------------------------------- returns

def period_return(adj: pd.DataFrame, tickers: list, start, end) -> float:
    """Equal-weight return from start to end. A stock with no price at `end`
    (acquired/delisted) is valued at its last available price."""
    rets = []
    for t in tickers:
        p0 = adj.at[start, t]
        hist = adj[t].loc[:end].dropna()
        if pd.isna(p0) or hist.empty:
            continue
        rets.append(hist.iloc[-1] / p0 - 1)
    return sum(rets) / len(rets) if rets else float("nan")


def run(args) -> pd.DataFrame:
    if not edgar.ANNUAL_FILE.exists():
        raise SystemExit("Run `python edgar.py` first - the backtest needs SEC history.")
    annual = edgar.load_annual()
    fund = pd.read_csv(F.CACHE)
    sectors = fund.dropna(subset=["sector"]).drop_duplicates("ticker").set_index("ticker").sector

    first = pd.Timestamp(year=args.start, month=args.month, day=1)
    prices = load_prices(sorted(sectors.index) + [BENCHMARK], first, args.refresh_prices)
    adj, idx = prices["adj_close"], prices["adj_close"].index
    last_day = idx[-1]

    total = sum(args.weights)
    weights = dict(zip(["value", "quality", "growth"], (w / total for w in args.weights)))

    dates = []
    d = first
    while d <= last_day:
        dates.append(d)
        d = pd.Timestamp(year=d.year + 1, month=args.month, day=1)

    results, all_picks = [], []
    for i, d in enumerate(dates):
        start = trading_day(idx, d)
        end = trading_day(idx, dates[i + 1]) if i + 1 < len(dates) else last_day
        if end is None:
            end = last_day
        if start is None or end <= start:
            continue

        snap = snapshot(annual, prices, sectors, d, args.history_years)
        if snap.empty:
            continue
        eligible = F.clean_data(snap)
        ranked = F.score(eligible, weights)
        picks = F.pick(ranked, n=args.top, max_per_sector=args.max_per_sector)

        partial = i == len(dates) - 1  # the last period runs only up to today
        results.append({
            "start": start.date(), "end": end.date(), "partial": partial,
            "universe_size": len(eligible),
            "screener": period_return(adj, picks.ticker.tolist(), start, end),
            "all_eligible": period_return(adj, eligible.ticker.tolist(), start, end),
            BENCHMARK: period_return(adj, [BENCHMARK], start, end),
        })
        for rank, (_, p) in enumerate(picks.iterrows(), 1):
            all_picks.append({"start": start.date(), "rank": rank, "ticker": p.ticker,
                              "sector": p.sector, "score": round(p.score, 3),
                              "return": period_return(adj, [p.ticker], start, end)})
        if args.show_picks:
            print(f"{start.date()}: {', '.join(picks.ticker)}")

    data_check(annual, prices, sectors, fund, last_day, args.history_years)
    res = pd.DataFrame(results)
    res.to_csv(RESULTS_FILE, index=False)
    pd.DataFrame(all_picks).to_csv(PICKS_FILE, index=False)
    return res


def data_check(annual, prices, sectors, fund, day, history_years) -> None:
    """Compare today's market caps built from SEC shares x price against Yahoo's.

    If these disagree a lot, share counts or split handling are off and the
    backtest's valuations can't be trusted.
    """
    if "marketCap" not in fund.columns:
        return
    snap = snapshot(annual, prices, sectors, day, history_years).set_index("ticker")
    yahoo = fund.drop_duplicates("ticker").set_index("ticker").marketCap
    ratio = (snap.marketCap / yahoo.reindex(snap.index)).dropna()
    off = ratio[(ratio < 0.67) | (ratio > 1.5)].sort_values()
    print(f"\nData check: SEC-based market cap vs Yahoo for {len(ratio)} stocks - "
          f"median ratio {ratio.median():.2f} (1.00 = perfect), {len(off)} off by more than 1.5x")
    if len(off):
        print("  Off: " + ", ".join(f"{t} {r:.2f}x" for t, r in off.head(15).items()))


# ---------------------------------------------------------------- report

def report(res: pd.DataFrame, args) -> None:
    cols = ["screener", "all_eligible", BENCHMARK]
    names = {"screener": "Screener", "all_eligible": "All eligible (eq-wt)", BENCHMARK: "SPY"}
    print(f"\nTop {args.top}, max {args.max_per_sector}/sector, weights {args.weights}, "
          f"{args.history_years}-yr history, rebalanced every {pd.Timestamp(2000, args.month, 1):%B}\n")
    print(f"{'Period':<25}{'Stocks':>7}" + "".join(f"{names[c]:>22}" for c in cols))
    for _, r in res.iterrows():
        label = f"{r.start} -> {r.end}" + (" *" if r.partial else "")
        print(f"{label:<25}{r.universe_size:>7}" + "".join(f"{r[c]:>22.1%}" for c in cols))

    years = (pd.Timestamp(res.end.iloc[-1]) - pd.Timestamp(res.start.iloc[0])).days / 365.25
    print("-" * (32 + 22 * len(cols)))
    growth = {c: (1 + res[c]).prod() for c in cols}
    print(f"{'$10,000 grew to':<32}" + "".join(f"{'$' + format(10000 * growth[c], ',.0f'):>22}" for c in cols))
    print(f"{'Yearly average (CAGR)':<32}" + "".join(f"{growth[c] ** (1 / years) - 1:>22.1%}" for c in cols))
    full = res[~res.partial]
    print(f"{'Worst full year':<32}" + "".join(f"{full[c].min():>22.1%}" for c in cols))
    print(f"\nScreener beat SPY in {(full.screener > full[BENCHMARK]).sum()} of {len(full)} full years, "
          f"and beat all-eligible in {(full.screener > full.all_eligible).sum()} of {len(full)}.")
    if res.partial.any():
        print("* = partial period (rebalance date not reached yet)")
    print("\nCaveats: today's S&P 500 list only (survivorship bias), no trading costs or taxes,"
          f"\nonly {len(full)} full years. Details saved to {RESULTS_FILE.name} and {PICKS_FILE.name}.")


def parse_args():
    p = argparse.ArgumentParser(description="Backtest the stock screener against SPY.")
    p.add_argument("--start", type=int, default=2013, help="first year (default: 2013)")
    p.add_argument("--month", type=int, default=4, help="rebalance month, 1-12 (default: 4 = April)")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--max-per-sector", type=int, default=3)
    p.add_argument("--weights", nargs=3, type=float, default=[0.4, 0.4, 0.2],
                   metavar=("VALUE", "QUALITY", "GROWTH"))
    p.add_argument("--history-years", type=int, default=8)
    p.add_argument("--show-picks", action="store_true", help="print each year's picks")
    p.add_argument("--refresh-prices", action="store_true", help="re-download prices")
    args = p.parse_args()
    if not 1 <= args.month <= 12:
        p.error("--month must be 1-12")
    if any(w < 0 for w in args.weights) or sum(args.weights) == 0:
        p.error("weights must be non-negative and not all zero")
    return args


if __name__ == "__main__":
    args = parse_args()
    report(run(args), args)
