"""Download 10+ years of annual financials from SEC EDGAR, point-in-time.

Every value is saved with the date it was FIRST filed with the SEC. That lets the
rest of the program ask "what did investors actually know on date X?" - which is
what makes an honest backtest possible (no peeking at numbers from the future).

Usage:
    python edgar.py            # download anything not cached yet, then rebuild the table
    python edgar.py --refresh  # re-download everything (do this after new 10-Ks come out)
"""
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
RAW_DIR = HERE / "edgar_cache"            # one small CSV per ticker
ANNUAL_FILE = HERE / "edgar_annual.csv"   # everything combined
FUNDAMENTALS = HERE / "fundamentals.csv"  # source of the ticker list


def load_env(path: Path = HERE / ".env") -> dict:
    """Read KEY=value lines from a .env file (kept out of Git so private info stays private)."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


# The SEC requires a User-Agent with a name and contact email, and blocks requests without one.
# Set it in .env as:  SEC_CONTACT_EMAIL=you@example.com
CONTACT_EMAIL = os.environ.get("SEC_CONTACT_EMAIL") or load_env().get("SEC_CONTACT_EMAIL", "")

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
REQUEST_PAUSE = 0.15  # SEC allows max 10 requests/second

# Tickers whose SEC mapping points at a new entity with no filing history yet.
# XOM now maps to "ExxonMobil Holdings Corporation" (a new holding company, no 10-Ks),
# while the years of history live under the original Exxon Mobil Corp (CIK 34088).
CIK_OVERRIDES = {"XOM": 34088}

# field -> (XBRL concept names in priority order, unit, kind)
# Companies don't all use the same names, and many switched names over time
# (e.g. SalesRevenueNet -> RevenueFromContractWithCustomer... after a 2018 rule change),
# so for each year we take the highest-priority name that has a value.
CONCEPTS = {
    "revenue": ([
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
        "RegulatedAndUnregulatedOperatingRevenue",  # utilities (e.g. DTE)
        "RegulatedOperatingRevenue",
        "UtilityRevenue",
        "OperatingLeasesIncomeStatementLeaseRevenue",  # REITs: rent
        "OperatingLeaseLeaseIncome",
    ], "USD", "duration"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD", "duration"),
    "op_cash_flow": ([
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ], "USD", "duration"),
    "capex": ([
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ], "USD", "duration"),
    "equity": ([
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ], "USD", "instant"),
    "lt_debt": (["LongTermDebt", "LongTermDebtNoncurrent"], "USD", "instant"),
    "diluted_shares": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", "duration"),
}
# For revenue, take the LARGEST value reported under any of the names instead of the first.
# Banks, REITs and insurers report only fee income as "RevenueFromContractWithCustomer...",
# with interest, rent and premiums under other names - the total is the biggest one.
TAKE_MAX = {"revenue"}
# Revenue built by adding two concepts. Banks' main income (net interest income) isn't
# filed under any "revenue" name, so their revenue = net interest income + fee income.
SUMMED_REVENUE = [("InterestIncomeExpenseNet", "NoninterestIncome")]
CORE = ["revenue", "net_income"]  # a fiscal year needs at least one of these to be kept


# ---------------------------------------------------------------- downloading

def make_session() -> requests.Session:
    if "@" not in CONTACT_EMAIL:
        raise SystemExit('Add SEC_CONTACT_EMAIL=you@example.com to a .env file next to edgar.py - the SEC requires one.')
    s = requests.Session()
    s.headers["User-Agent"] = f"stockRecommender personal project {CONTACT_EMAIL}"
    return s


def get_json(s: requests.Session, url: str, tries: int = 4):
    for attempt in range(tries):
        r = s.get(url, timeout=60)
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):  # rate-limited or SEC hiccup: back off
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def ticker_to_cik(s: requests.Session) -> dict:
    """The SEC identifies companies by CIK number, not ticker."""
    data = get_json(s, TICKER_URL)
    ciks = {row["ticker"].upper().replace(".", "-"): int(row["cik_str"]) for row in data.values()}
    return {**ciks, **CIK_OVERRIDES}


# ---------------------------------------------------------------- parsing

def first_reported(gaap: dict, concept: str, unit: str, kind: str) -> dict:
    """{period_end: (filed, value)} for one concept, using the earliest 10-K that reported it."""
    first_seen = {}
    for f in gaap.get(concept, {}).get("units", {}).get(unit, []):
        if not str(f.get("form", "")).startswith("10-K"):  # annual reports only
            continue
        end, filed = f.get("end"), f.get("filed")
        if not end or not filed:
            continue
        if kind == "duration":  # must cover a full year, not a quarter
            if not f.get("start"):
                continue
            days = (pd.Timestamp(end) - pd.Timestamp(f["start"])).days
            if not 330 <= days <= 400:
                continue
        if end not in first_seen or filed < first_seen[end][0]:
            first_seen[end] = (filed, f["val"])
    return first_seen


def extract(facts: dict) -> pd.DataFrame:
    """Turn one company's raw SEC facts into one row per fiscal year.

    Each 10-K repeats the previous years' numbers, so the same value shows up
    several times with different filing dates. We keep the EARLIEST filing,
    since that's when the number first became public.
    """
    gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
    values = {}  # field -> {period_end: (value, filed)}

    for field, (concepts, unit, kind) in CONCEPTS.items():
        # 1) For each concept, the earliest-filed annual value per fiscal year end
        per_concept = [first_reported(gaap, c, unit, kind) for c in concepts]
        if field == "revenue":
            for a, b in SUMMED_REVENUE:
                fa, fb = first_reported(gaap, a, unit, kind), first_reported(gaap, b, unit, kind)
                per_concept.append({e: (max(fa[e][0], fb[e][0]), fa[e][1] + fb[e][1])
                                    for e in fa.keys() & fb.keys()})

        # 2) Combine the concepts: highest priority wins, or the largest value for TAKE_MAX fields
        combined = {}
        for first_seen in per_concept:
            for end, (filed, val) in first_seen.items():
                if end not in combined:
                    combined[end] = (val, filed)
                elif field in TAKE_MAX and val > combined[end][0]:
                    combined[end] = (val, filed)
        values[field] = combined

    year_ends = sorted(set().union(*(values[f] for f in CORE)))
    rows = []
    for end in year_ends:
        row = {"period_end": end}
        filed_dates = []
        for field in CONCEPTS:
            if end in values[field]:
                row[field], filed = values[field][end]
                filed_dates.append(filed)
        # Conservative: the year counts as "known" once ALL of its numbers were public.
        row["filed"] = max(filed_dates)
        rows.append(row)
    return pd.DataFrame(rows, columns=["period_end", "filed", *CONCEPTS])


# ---------------------------------------------------------------- main download

def download(refresh: bool = False, only: list | None = None) -> pd.DataFrame:
    s = make_session()
    RAW_DIR.mkdir(exist_ok=True)
    tickers = pd.read_csv(FUNDAMENTALS).ticker.tolist()
    if only:  # re-download just these tickers
        tickers = [t.upper().replace(".", "-") for t in only]
        refresh = True
    cik_map = ticker_to_cik(s)

    for i, t in enumerate(tickers, 1):
        out = RAW_DIR / f"{t}.csv"
        if out.exists() and not refresh:
            continue  # already have it - makes interrupted runs resumable
        cik = cik_map.get(t)
        if cik is None:
            print(f"[{i}/{len(tickers)}] {t}: no SEC CIK found, skipping")
            continue
        try:
            df = extract(get_json(s, FACTS_URL.format(cik=cik)))
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: failed ({e}), will retry next run")
            continue
        df.insert(0, "ticker", t)
        df.to_csv(out, index=False)
        span = f"{df.period_end.min()[:4]}-{df.period_end.max()[:4]}" if len(df) else "no data"
        print(f"[{i}/{len(tickers)}] {t}: {len(df)} years ({span})")
        time.sleep(REQUEST_PAUSE)

    return build_annual()


def build_annual() -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(RAW_DIR.glob("*.csv"))]
    parts = [p for p in parts if len(p)]
    annual = pd.concat(parts, ignore_index=True)
    annual.to_csv(ANNUAL_FILE, index=False)
    print(f"\nSaved {len(annual)} company-years for {annual.ticker.nunique()} tickers to {ANNUAL_FILE.name}")
    return annual


# ---------------------------------------------------------------- using the data

def load_annual() -> pd.DataFrame:
    return pd.read_csv(ANNUAL_FILE, parse_dates=["period_end", "filed"])


def history_asof(annual: pd.DataFrame, as_of=None, years: int = 8) -> pd.DataFrame:
    """Multi-year summary per ticker using ONLY reports filed on or before `as_of`.

    Returns the same columns fundamentals.py's Yahoo-based history uses:
    histYears, avgNetIncome, avgMargin, marginStd, revCagr (indexed by ticker).
    """
    as_of = pd.Timestamp.today() if as_of is None else pd.Timestamp(as_of)
    known = annual[annual.filed <= as_of].dropna(subset=["revenue", "net_income"])
    known = known[known.revenue > 0].sort_values("period_end")
    known = known.groupby("ticker").tail(years).copy()
    known["margin"] = known.net_income / known.revenue

    g = known.groupby("ticker")
    out = pd.DataFrame({
        "histYears": g.size(),
        "avgNetIncome": g.net_income.mean(),
        "avgMargin": g.margin.mean(),
        "marginStd": g.margin.std(),
    })
    first, last = g.first(), g.last()
    span_years = (last.period_end - first.period_end).dt.days / 365.25
    out["revCagr"] = ((last.revenue / first.revenue) ** (1 / span_years) - 1).where(span_years > 0)
    out.loc[out.histYears < 3, ["avgNetIncome", "avgMargin", "marginStd", "revCagr"]] = float("nan")
    return out


def inspect(ticker: str) -> None:
    """Debugging aid: list every revenue / income-like concept a company reports in 10-Ks.

    Use it when a company comes back with no data, to see which names it uses.
    """
    s = make_session()
    ticker = ticker.upper().replace(".", "-")
    cik = ticker_to_cik(s).get(ticker)
    if cik is None:
        raise SystemExit(f"No SEC CIK for {ticker}")
    facts = get_json(s, FACTS_URL.format(cik=cik)) or {}
    lines = [f"{ticker} (CIK {cik}) - {facts.get('entityName', '?')}"]
    for taxonomy, concepts in facts.get("facts", {}).items():
        for name, body in sorted(concepts.items()):
            if not any(k in name for k in ("Revenue", "Sales", "NetIncome", "ProfitLoss")):
                continue
            for unit, rows in body.get("units", {}).items():
                annual = [r for r in rows if str(r.get("form", "")).startswith("10-K")]
                if annual:
                    ends = sorted(r["end"] for r in annual)
                    lines.append(f"  {taxonomy}:{name} [{unit}] {len(annual)} 10-K values, {ends[0]} .. {ends[-1]}")
    text = "\n".join(lines)
    print(text)
    out = HERE / f"edgar_inspect_{ticker}.txt"
    out.write_text(text)
    print(f"\n(saved to {out.name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download annual financials from SEC EDGAR.")
    parser.add_argument("--refresh", action="store_true", help="re-download every company")
    parser.add_argument("--inspect", metavar="TICKER", help="list the revenue/income names one company uses")
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="re-download only these tickers, e.g. --tickers DTE XOM")
    args = parser.parse_args()
    if args.inspect:
        inspect(args.inspect)
    else:
        download(args.refresh, args.tickers)
