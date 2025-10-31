#!/usr/bin/env python3
"""
Main screener script.

- Uses TradingView scanner (tradingview_screener) to find premarket movers.
- Filters by relative volume and other basic criteria.
- Uses the Finnhub client (finnhub-python) to check company news within the last N hours.
- Loads FINNHUB_API_KEY from environment (supports .env via python-dotenv).
- Exposes two output columns:
    - "Makes News (24H)" (boolean)
    - "News URL" (string with the first matching article URL or empty)
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import rookiepy
from tradingview_screener.query import Query
from tradingview_screener.column import col

import pandas as pd
import pytz
import finnhub
from dotenv import load_dotenv

# Load .env from the script directory if present
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
AVG_VOL_PERIOD = "60d"
MARKET = "america"
MIN_CLOSE_PRICE = 2
MAX_CLOSE_PRICE = 20
MIN_PREMARKET_CHANGE = 10
MAX_FLOAT_SHARES = 20_000_000
MIN_RELATIVE_VOLUME_MULTIPLIER = 5.0

# Timezone and news window configuration
MARKET_TIMEZONE = "America/New_York"
NEWS_CHECK_HOURS = 24
TIME_BUFFER_MINUTES = 30  # small buffer to account for data latency

# Column names used by the TradingView response
TICKER_COL = "name"
PRICE_COL = "close"
CHANGE_COL = "premarket_change"
FLOAT_COL = "float_shares_outstanding"
VOL_COL = "volume"
AVG_VOL_COL_NAME = f"average_volume_{AVG_VOL_PERIOD}_calc"
PREV_AVG_VOL_COL = f"{AVG_VOL_COL_NAME}|1"

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_session_cookies() -> dict:
    """Load TradingView cookies using rookiepy (Firefox). Returns a dict."""
    try:
        cookies_list = rookiepy.firefox([".tradingview.com"])
        return {
            c.get("name"): c.get("value")
            for c in cookies_list
            if c and c.get("name") in ("sessionid", "tv_auth")
        }
    except Exception as exc:
        logger.debug("Failed to load rookiepy cookies: %s", exc)
        return {}


def format_number(series: pd.Series, decimal_places: int = 0) -> pd.Series:
    """Format numeric Pandas Series into comma-separated strings while preserving NaN."""
    fmt_string = f"{{:,.{decimal_places}f}}"

    def _fmt(x):
        try:
            if pd.isna(x):
                return ""
            return fmt_string.format(x)
        except Exception:
            return ""

    return series.map(_fmt)


def _normalize_ticker_for_finnhub(symbol: str) -> str:
    """Normalize tickers to the symbol Finnhub expects (strip exchange prefix)."""
    if not symbol or not isinstance(symbol, str):
        return symbol
    return symbol.split(":")[-1].strip()


def get_finnhub_client() -> Optional[finnhub.Client]:
    """Create a finnhub.Client using FINNHUB_API_KEY from environment, or None if missing."""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        logger.warning(
            "FINNHUB_API_KEY not set in environment; news checks will be disabled."
        )
        return None
    try:
        client = finnhub.Client(api_key=api_key)
        return client
    except Exception as exc:
        logger.warning("Failed to initialize Finnhub client: %s", exc)
        return None


def check_for_news_finnhub(
    ticker: str, cutoff_time_utc: datetime, client: Optional[finnhub.Client]
) -> Tuple[bool, str]:
    """
    Check Finnhub for company news for `ticker` and return (makes_news_bool, news_url_or_empty).

    - ticker: incoming ticker string (may include exchange prefix).
    - cutoff_time_utc: timezone-aware UTC datetime cutoff (articles after this return True).
    - client: finnhub.Client instance or None.

    Returns:
    - (True, url) if at least one article exists with timestamp > cutoff_time_utc
      where url is the first matching article's URL.
    - (False, "") if no articles or client is None or any error occurs.
    """
    if client is None:
        return False, ""

    symbol = _normalize_ticker_for_finnhub(ticker)
    if not symbol:
        return False, ""

    try:
        from_date = cutoff_time_utc.date().isoformat()
        to_date = datetime.now(timezone.utc).date().isoformat()
        # finnhub-python expects (_from, to) args; named to be explicit
        articles = client.company_news(symbol, _from=from_date, to=to_date)
        if not articles:
            return False, ""

        cutoff_unix = int(cutoff_time_utc.timestamp())

        # Return the first article after cutoff with a URL
        for art in articles:
            art_ts = art.get("datetime")
            art_url = art.get("url") or art.get("news_url") or ""
            if art_ts and art_ts > cutoff_unix:
                return True, art_url or ""
        return False, ""
    except Exception as exc:
        logger.debug("Finnhub news check failed for %s: %s", symbol, exc)
        return False, ""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # compute market-aware cutoff for news checking
    market_tz = pytz.timezone(MARKET_TIMEZONE)
    current_market_time = datetime.now(market_tz)

    news_cutoff_local = current_market_time - timedelta(hours=NEWS_CHECK_HOURS)
    # subtract buffer in UTC (to include slightly older items if desired)
    news_cutoff_utc = news_cutoff_local.astimezone(pytz.utc) - timedelta(
        minutes=TIME_BUFFER_MINUTES
    )

    logger.info("Timezone configured to: %s", MARKET_TIMEZONE)
    logger.info(
        "Checking for news published after: %s (includes %dm buffer)",
        news_cutoff_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        TIME_BUFFER_MINUTES,
    )

    logger.info("Loading TradingView cookies...")
    session_cookies = get_session_cookies()

    # initialize Finnhub client once
    fh_client = get_finnhub_client()

    try:
        # Execute TradingView scanner query
        total_count, df = (
            Query()
            .set_markets(MARKET)
            .select(
                TICKER_COL,
                PRICE_COL,
                CHANGE_COL,
                FLOAT_COL,
                VOL_COL,
                AVG_VOL_COL_NAME,
                PREV_AVG_VOL_COL,
            )
            .where(
                col(PRICE_COL).between(MIN_CLOSE_PRICE, MAX_CLOSE_PRICE),
                col(CHANGE_COL) > MIN_PREMARKET_CHANGE,
                col(FLOAT_COL) < MAX_FLOAT_SHARES,
            )
            .get_scanner_data(cookies=session_cookies)
        )

        logger.info("Initial match count: %s", total_count)

        if df.empty:
            logger.info("No stocks matched the basic criteria.")
            return

        # compute relative volume and filter
        df["Volume Ratio"] = df[VOL_COL] / df[PREV_AVG_VOL_COL].replace(0, pd.NA)
        filtered_df = df[df["Volume Ratio"] > MIN_RELATIVE_VOLUME_MULTIPLIER].copy()
        logger.info(
            "Final results filtered for %.1fx Relative Volume: %d",
            MIN_RELATIVE_VOLUME_MULTIPLIER,
            len(filtered_df),
        )

        if filtered_df.empty:
            logger.info("No stocks after relative volume filter.")
            return

        # Run Finnhub news check for each ticker and return tuple (bool, url)
        logger.info("Running Finnhub news check (last %d hours)...", NEWS_CHECK_HOURS)
        news_series = filtered_df[TICKER_COL].apply(
            lambda sym: check_for_news_finnhub(sym, news_cutoff_utc, fh_client)
        )

        # news_series is a Series of tuples -> create two columns from it
        news_list = list(news_series.tolist())  # list of (bool, url)
        if news_list:
            news_df = pd.DataFrame(
                news_list,
                index=filtered_df.index,
                columns=["Makes News (24H)", "News URL"],
            )
            filtered_df = pd.concat([filtered_df, news_df], axis=1)
        else:
            filtered_df["Makes News (24H)"] = False
            filtered_df["News URL"] = ""

        # Ensure Makes News is boolean and News URL is string
        filtered_df["Makes News (24H)"] = filtered_df["Makes News (24H)"].astype(bool)
        filtered_df["News URL"] = filtered_df["News URL"].fillna("")

        # Formatting for display (keep numeric columns available, display columns are formatted)
        filtered_df["Current Vol"] = format_number(filtered_df[VOL_COL], 0)
        filtered_df["Avg Vol 60d"] = format_number(filtered_df[AVG_VOL_COL_NAME], 0)
        filtered_df["Float"] = format_number(filtered_df[FLOAT_COL], 0)
        filtered_df["Rel Vol 60d (x)"] = format_number(filtered_df["Volume Ratio"], 2)
        filtered_df["Change from Close (%)"] = format_number(filtered_df[CHANGE_COL], 2)
        filtered_df["Price"] = (
            filtered_df[PRICE_COL]
            .apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            .astype(str)
        )

        final_columns = [
            TICKER_COL,
            "Price",
            "Change from Close (%)",
            "Float",
            "Makes News (24H)",
            # "News URL",  # Removed from table output
            "Rel Vol 60d (x)",
            "Current Vol",
            "Avg Vol 60d",
        ]

        display_df = filtered_df[final_columns].rename(columns={TICKER_COL: "Ticker"})
        # Sort by 'Change from Close (%)' descending (Z to A)
        display_df = display_df.sort_values(by="Change from Close (%)", ascending=False)

        logger.info("--- Final Results (Limited to Top 10) ---")
        # Ensure Price column always shows two decimals for Markdown export
        display_df["Price"] = display_df["Price"].apply(
            lambda x: f"{float(x):.2f}" if x not in ("", None) else ""
        )
        try:
            table_md = display_df.head(10).to_markdown(index=False, floatfmt=".2f")

        except Exception:
            table_md = display_df.head(10).to_string(index=False)

        # Prepare formatted article links after the table
        links_md = "\n--- Article Links ---\n"
        for _, row in filtered_df.head(10).iterrows():
            ticker = row[TICKER_COL]
            url = row["News URL"]
            if url:
                links_md += f"- [{ticker}]({url})\n"

        # Export to Markdown file
        export_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        export_filename = f"{export_date} screener wt.md"
        with open(export_filename, "w") as f:
            f.write("# Screener Results\n\n")
            f.write(table_md)
            f.write("\n\n")
            f.write(links_md)

        # Also print to console for visibility
        print(table_md)
        print(links_md)

    except Exception as exc:
        logger.exception("An error occurred during the query: %s", exc)


if __name__ == "__main__":
    main()
