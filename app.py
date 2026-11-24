"""Interactive dashboard for the stock screener.

Run with:   streamlit run app.py
It reads the cached data files - run fundamentals.py (and edgar.py / backtest.py) first.
"""
import argparse
import contextlib
import io
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import backtest as B
import edgar
import fundamentals as F

st.set_page_config(page_title="Stock Screener", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data (cached)

@st.cache_data
def load_screen_data(history_years: int, mtime: float) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):  # hide the console tips
        df = F.apply_edgar_history(pd.read_csv(F.CACHE), history_years)
    return F.clean_data(df)


@st.cache_data
def load_annual(mtime: float) -> pd.DataFrame:
    return edgar.load_annual()


@st.cache_data
def load_backtest(mtime: float):
    res = pd.read_csv(B.RESULTS_FILE, parse_dates=["start", "end"])
    picks = pd.read_csv(B.PICKS_FILE) if B.PICKS_FILE.exists() else pd.DataFrame()
    return res, picks


def mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def warning_label(row) -> str:
    if pd.isna(row.avgMargin):
        return "Short history"
    if row.avgMargin <= 0 or (pd.notna(row.peak_ratio) and row.peak_ratio >= 1.5):
        return "Possible cyclical peak"
    return ""


# ---------------------------------------------------------------- sidebar

if not F.CACHE.exists():
    st.error("No data yet. Run `python fundamentals.py` once to download it, then refresh this page.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    st.caption("Factor weights (rescaled to add up to 100%)")
    w_value = st.slider("Value", 0.0, 1.0, F.WEIGHTS["value"], 0.05,
                        help="Cheap vs. sector peers: free-cash-flow yield and normalized P/E")
    w_quality = st.slider("Quality", 0.0, 1.0, F.WEIGHTS["quality"], 0.05,
                          help="ROE, average margin, steady margins, not at a cyclical peak")
    w_growth = st.slider("Growth", 0.0, 1.0, F.WEIGHTS["growth"], 0.05,
                         help="Multi-year revenue growth")
    total = w_value + w_quality + w_growth
    if total == 0:
        st.warning("At least one weight must be above zero.")
        st.stop()
    weights = {"value": w_value / total, "quality": w_quality / total, "growth": w_growth / total}
    st.caption(" · ".join(f"{k} {v:.0%}" for k, v in weights.items()))

    st.divider()
    top_n = st.number_input("Number of picks", 5, 50, 15)
    max_per_sector = st.number_input("Max per sector", 1, 15, 3)
    history_years = st.slider("Years of SEC history to average", 3, 15, 8)

data = load_screen_data(history_years, mtime(F.CACHE) + mtime(edgar.ANNUAL_FILE))
all_sectors = sorted(data.sector.unique())
with st.sidebar:
    sectors = st.multiselect("Sectors", all_sectors, placeholder="All sectors")
    st.divider()
    st.caption(f"Data from {datetime.fromtimestamp(mtime(F.CACHE)):%b %d, %Y %I:%M %p}. "
               "Run `python fundamentals.py --refresh` for newer numbers.")

filtered = data[data.sector.isin(sectors)] if sectors else data
ranked = F.score(filtered, weights)
ranked["warning"] = ranked.apply(warning_label, axis=1)
picks = F.pick(ranked, n=int(top_n), max_per_sector=int(max_per_sector))


# ---------------------------------------------------------------- header

st.title("Stock Screener")
st.caption("Ranks S&P 500 stocks on value, quality and growth within their sector. "
           "A starting point for research, not a buy list. The backtest found no reliable edge over "
           "picking from the same list at random. Not financial advice.")

tab_list, tab_stock, tab_bt = st.tabs(["Shortlist", "Stock detail", "Backtest"])


# ---------------------------------------------------------------- shortlist

with tab_list:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks passing filters", len(filtered))
    c2.metric("Picks", len(picks))
    c3.metric("Sectors in picks", picks.sector.nunique())
    c4.metric("Flagged picks", int((picks.warning != "").sum()),
              help="Short history or possible cyclical peak")

    table = pd.DataFrame({
        "Ticker": picks.ticker,
        "Sector": picks.sector,
        "Score": picks.score,
        "Value": picks.value,
        "Quality": picks.quality,
        "Growth": picks.growth,
        "P/E": picks.trailingPE,
        "Normalized P/E": 1 / picks.norm_earn_yield.where(picks.norm_earn_yield > 0),
        "FCF yield %": picks.fcf_yield * 100,
        "ROE %": picks.returnOnEquity * 100,
        "Margin now %": picks.profitMargins * 100,
        "Margin avg %": picks.avgMargin * 100,
        "Revenue growth %/yr": picks.revCagr * 100,
        "Warning": picks.warning,
    })
    score_col = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=1, format="%.2f")
    pct = lambda label: st.column_config.NumberColumn(label, format="%.1f%%")
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Score": score_col("Score"), "Value": score_col("Value"),
            "Quality": score_col("Quality"), "Growth": score_col("Growth"),
            "P/E": st.column_config.NumberColumn(format="%.1f"),
            "Normalized P/E": st.column_config.NumberColumn(format="%.1f",
                help="Price vs. average profit over the history window"),
            "FCF yield %": pct("FCF yield"), "ROE %": pct("ROE"),
            "Margin now %": pct("Margin now"), "Margin avg %": pct("Margin avg"),
            "Revenue growth %/yr": pct("Revenue growth/yr"),
        },
    )
    st.caption("Value, Quality and Growth are percentile ranks within each sector (1.00 = best in sector).")

    st.download_button("Download shortlist (CSV)", table.to_csv(index=False),
                       file_name="shortlist.csv", mime="text/csv")


# ---------------------------------------------------------------- stock detail

with tab_stock:
    options = list(picks.ticker) + [t for t in ranked.ticker if t not in set(picks.ticker)]
    ticker = st.selectbox("Stock (picks first, then everything else by score)", options)
    row = ranked[ranked.ticker == ticker].iloc[0]
    rank = int((ranked.ticker == ticker).to_numpy().nonzero()[0][0]) + 1

    left, right = st.columns([2, 3])
    with left:
        st.subheader(f"{ticker} · #{rank} of {len(ranked)}")
        st.caption(row.sector)
        m1, m2, m3 = st.columns(3)
        m1.metric("Value", f"{row.value:.2f}")
        m2.metric("Quality", f"{row.quality:.2f}")
        m3.metric("Growth", f"{row.growth:.2f}")
        if row.warning:
            st.warning(row.warning)
        st.code(F.explain(row), language=None, wrap_lines=True)

    with right:
        if edgar.ANNUAL_FILE.exists():
            annual = load_annual(mtime(edgar.ANNUAL_FILE))
            hist = annual[annual.ticker == ticker].sort_values("period_end").tail(15)
            hist = hist.dropna(subset=["revenue", "net_income"])
            if len(hist):
                long = pd.DataFrame({
                    "Fiscal year": hist.period_end.dt.year.astype(str).tolist() * 2,
                    "Billions $": list(hist.revenue / 1e9) + list(hist.net_income / 1e9),
                    "Line": ["Revenue"] * len(hist) + ["Net income"] * len(hist),
                })
                money = alt.Chart(long).mark_bar().encode(
                    x=alt.X("Fiscal year:N", title=None),
                    xOffset="Line:N",
                    y=alt.Y("Billions $:Q"),
                    color=alt.Color("Line:N", legend=alt.Legend(orient="top", title=None),
                                    scale=alt.Scale(range=["#4C78A8", "#9ECAE9"])),
                    tooltip=["Fiscal year", "Line", alt.Tooltip("Billions $:Q", format=",.2f")],
                ).properties(height=260, title="SEC annual reports")
                margins = pd.DataFrame({"Fiscal year": hist.period_end.dt.year.astype(str),
                                        "Net margin": hist.net_income / hist.revenue})
                margin_chart = alt.Chart(margins).mark_line(point=True, color="#4C78A8").encode(
                    x=alt.X("Fiscal year:N", title=None),
                    y=alt.Y("Net margin:Q", axis=alt.Axis(format="%")),
                    tooltip=["Fiscal year", alt.Tooltip("Net margin:Q", format=".1%")],
                ).properties(height=180, title="Net margin by year - spikes mean cyclical peaks")
                st.altair_chart(money, width="stretch")
                st.altair_chart(margin_chart, width="stretch")
            else:
                st.info("No SEC history for this stock.")
        else:
            st.info("Run `python edgar.py` to see multi-year SEC history here.")


# ---------------------------------------------------------------- backtest

with tab_bt:
    st.caption("Each April, the screener is rebuilt using only SEC filings published before that date, "
               "its picks are held for a year, and the result is compared to SPY and to owning every "
               "eligible stock equally. That last line is the honest benchmark: the stock list is today's "
               "S&P 500, which leaves out companies that failed, so it flatters everything.")

    if st.button("Run backtest with the current settings", type="primary"):
        args = argparse.Namespace(start=2013, month=4, top=int(top_n), max_per_sector=int(max_per_sector),
                                  weights=[w_value, w_quality, w_growth], history_years=history_years,
                                  show_picks=False, refresh_prices=False)
        with st.spinner("Running... (downloads prices first if the cache is over a day old: 1-3 min)"):
            with contextlib.redirect_stdout(io.StringIO()):
                B.run(args)
        st.session_state["bt_settings"] = (f"weights {weights['value']:.0%}/{weights['quality']:.0%}/"
                                           f"{weights['growth']:.0%}, top {top_n}, max {max_per_sector}/sector, "
                                           f"{history_years}-yr history")

    if not B.RESULTS_FILE.exists():
        st.info("No backtest yet. Click the button above, or run `python backtest.py`.")
    else:
        res, bt_picks = load_backtest(mtime(B.RESULTS_FILE))
        st.caption("Showing: " + st.session_state.get(
            "bt_settings", "the last backtest run (settings as used then)"))
        names = {"screener": "Screener", "all_eligible": "All eligible (equal weight)", "SPY": "SPY"}
        full = res[~res.partial]
        years = (res.end.iloc[-1] - res.start.iloc[0]).days / 365.25

        cols = st.columns(3)
        for col, (key, name) in zip(cols, names.items()):
            growth = (1 + res[key]).prod()
            beat = f"beat SPY {(full[key] > full.SPY).sum()}/{len(full)} yrs" if key != "SPY" else "benchmark"
            col.metric(name, f"${10000 * growth:,.0f}")
            col.caption(f"{growth ** (1 / years) - 1:.1%} per year · {beat}")

        curve = [{"Date": res.start.iloc[0], "Strategy": names[k], "Value": 10000.0} for k in names]
        for k in names:
            value = 10000.0
            for _, r in res.iterrows():
                value *= 1 + r[k]
                curve.append({"Date": r.end, "Strategy": names[k], "Value": value})
        chart = alt.Chart(pd.DataFrame(curve)).mark_line(point=True).encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title="Growth of $10,000", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Strategy:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(domain=list(names.values()),
                                            range=["#E45756", "#9D9D9D", "#4C78A8"])),
            tooltip=["Strategy", alt.Tooltip("Date:T"), alt.Tooltip("Value:Q", format="$,.0f")],
        ).properties(height=340)
        st.altair_chart(chart, width="stretch")

        yearly = pd.DataFrame({
            "Period": [f"{s:%Y-%m-%d} → {e:%Y-%m-%d}" + (" (partial)" if p else "")
                       for s, e, p in zip(res.start, res.end, res.partial)],
            "Stocks eligible": res.universe_size,
            **{names[k] + " %": res[k] * 100 for k in names},
        })
        st.dataframe(yearly, hide_index=True, width="stretch",
                     column_config={names[k] + " %": st.column_config.NumberColumn(names[k], format="%.1f%%")
                                    for k in names})

        if len(bt_picks):
            with st.expander("Picks by year"):
                year = st.selectbox("Rebalance date", sorted(bt_picks.start.unique(), reverse=True))
                yp = bt_picks[bt_picks.start == year].copy()
                yp["return"] = yp["return"] * 100
                st.dataframe(yp[["rank", "ticker", "sector", "score", "return"]], hide_index=True,
                             width="stretch",
                             column_config={"return": st.column_config.NumberColumn("Return", format="%.1f%%")})
