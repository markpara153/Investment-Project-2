#!/usr/bin/env python3

"""
nasdaq_quant_basic.py

A beginner-friendly quantitative analysis script for NASDAQ using Yahoo Finance data (via yfinance).
- Fetches historical data (defaults to QQQ as a liquid NASDAQ-100 proxy with real volume).
- Computes rolling volatility, moving averages, and simple z-scores for volume/volatility.
- Produces a simple Buy / Hold / Sell signal and a 1–5 "star" score (heuristic).
- Saves plots (PNG) and a CSV of computed indicators to ./output

USAGE (from your terminal):
    python nasdaq_quant_basic.py --ticker QQQ --years 5
    python nasdaq_quant_basic.py --ticker NQ=F --years 3     # Nasdaq-100 futures (volume behaves differently)
    python nasdaq_quant_basic.py --ticker ^IXIC --years 5    # NASDAQ Composite Index (volume may be less reliable)

REQUIREMENTS (install these once inside your project/venv):
    pip install yfinance pandas numpy matplotlib
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def fetch_data(ticker: str, years: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=365 * years + 30)
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Try a different ticker or date range.")
    # Ensure columns we expect exist
    required_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        # yfinance sometimes names columns differently when auto_adjust=True; fill what we can
        for col in list(missing):
            if col == "Adj Close" and "Close" in df.columns:
                df["Adj Close"] = df["Close"]
                missing.remove("Adj Close")
        missing = required_cols - set(df.columns)
        if missing:
            raise RuntimeError(f"Data for {ticker} is missing columns: {missing}")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Adj Close"].pct_change()
    out["SMA50"] = out["Adj Close"].rolling(50).mean()
    out["SMA200"] = out["Adj Close"].rolling(200).mean()
    # 30-day annualized volatility (close-to-close)
    out["Vol30"] = out["Return"].rolling(30).std() * np.sqrt(252)
    # 30-day average volume
    out["Vol_SMA30"] = out["Volume"].rolling(30).mean()
    # Z-scores over the last N years for volatility and volume
    vol_mean, vol_std = out["Vol30"].mean(skipna=True), out["Vol30"].std(skipna=True)
    vol_std = vol_std if vol_std and not np.isnan(vol_std) else 1.0
    out["Vol_Z"] = (out["Vol30"] - vol_mean) / vol_std

    volu_mean, volu_std = out["Vol_SMA30"].mean(skipna=True), out["Vol_SMA30"].std(skipna=True)
    volu_std = volu_std if volu_std and not np.isnan(volu_std) else 1.0
    out["Volume_Z"] = (out["Vol_SMA30"] - volu_mean) / volu_std
    return out


def simple_signal_and_stars(df: pd.DataFrame) -> tuple[str, int, dict]:
    """Return a naive Buy/Hold/Sell and 1–5 star heuristic with component details."""
    last = df.dropna().iloc[-1]
    price = float(last["Adj Close"])
    sma50 = float(last["SMA50"])
    sma200 = float(last["SMA200"])
    vol_z = float(last["Vol_Z"])
    volume_z = float(last["Volume_Z"])

    # Trend context
    above_200 = price > sma200 if not np.isnan(sma200) else False
    slope_cross = sma50 > sma200 if (not np.isnan(sma50) and not np.isnan(sma200)) else False

    # Base decision (very simple, purely technical/regime-based)
    if above_200 and vol_z <= 0.5:
        decision = "BUY"
    elif (not above_200) and vol_z > 0.5:
        decision = "SELL"
    else:
        decision = "HOLD"

    # Star score (1–5): start at 3, then adjust
    stars = 3
    if above_200:
        stars += 1
    if slope_cross:
        stars += 1
    if vol_z > 1.0:
        stars -= 1
    if (volume_z > 1.0) and (not above_200):
        stars -= 1
    stars = int(max(1, min(5, stars)))

    components = {
        "price": price,
        "SMA50": sma50,
        "SMA200": sma200,
        "Vol_Z": vol_z,
        "Volume_Z": volume_z,
        "above_200": above_200,
        "slope_cross_50_over_200": slope_cross,
    }
    return decision, stars, components


def make_plots(df: pd.DataFrame, ticker: str, outdir: str) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    saved = []

    # 1) Price with 50/200 SMAs
    plt.figure(figsize=(10, 5))
    df["Adj Close"].plot(label="Adj Close")
    df["SMA50"].plot(label="SMA50")
    df["SMA200"].plot(label="SMA200")
    plt.title(f"{ticker} Price with 50/200 SMAs")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    f1 = os.path.join(outdir, f"{ticker}_price_ma.png")
    plt.tight_layout()
    plt.savefig(f1, dpi=150)
    plt.close()
    saved.append(f1)

    # 2) Volume with 30D SMA
    plt.figure(figsize=(10, 5))
    df["Volume"].plot(kind="line", label="Daily Volume")
    df["Vol_SMA30"].plot(label="Volume 30D SMA")
    plt.title(f"{ticker} Volume (Daily) & 30D Avg")
    plt.xlabel("Date")
    plt.ylabel("Shares / Contracts")
    plt.legend()
    f2 = os.path.join(outdir, f"{ticker}_volume.png")
    plt.tight_layout()
    plt.savefig(f2, dpi=150)
    plt.close()
    saved.append(f2)

    # 3) Rolling volatility (annualized)
    plt.figure(figsize=(10, 5))
    df["Vol30"].plot(label="30D Ann. Volatility")
    plt.title(f"{ticker} 30-Day Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (σ, annualized)")
    plt.legend()
    f3 = os.path.join(outdir, f"{ticker}_volatility.png")
    plt.tight_layout()
    plt.savefig(f3, dpi=150)
    plt.close()
    saved.append(f3)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Basic NASDAQ quantitative analysis with volume, volatility, and a simple star rating.")
    parser.add_argument("--ticker", type=str, default="QQQ", help="Ticker to download (default: QQQ). Examples: QQQ, ^IXIC, NQ=F")
    parser.add_argument("--years", type=int, default=5, help="Number of years of history to fetch (default: 5)")
    args = parser.parse_args()

    ticker = args.ticker.strip()
    years = max(2, int(args.years))  # need enough history for MAs

    print(f"Fetching {years} years of data for {ticker}...")
    df = fetch_data(ticker, years)
    df = add_indicators(df)

    decision, stars, comp = simple_signal_and_stars(df)

    os.makedirs("output", exist_ok=True)
    indicators_csv = os.path.join("output", f"{ticker}_indicators.csv")
    df.to_csv(indicators_csv, index=True)

    plot_files = make_plots(df, ticker, "output")

    # Console summary
    print("\n=== SUMMARY ===")
    print(f"Ticker: {ticker}")
    print(f"Last date: {df.dropna().index[-1].date()}")
    print(f"Price: {comp['price']:.2f}")
    print(f"SMA50: {comp['SMA50']:.2f} | SMA200: {comp['SMA200']:.2f}")
    print(f"Vol_Z: {comp['Vol_Z']:.2f} (negative = calmer than average)")
    print(f"Volume_Z: {comp['Volume_Z']:.2f} (positive = higher than average)")
    print(f"Signal: {decision}")
    print(f"Stars: {'★' * stars}{'☆' * (5 - stars)}  ({stars}/5)")
    print("\nFiles saved:")
    print(f" - Indicators CSV: {indicators_csv}")
    for p in plot_files:
        print(f" - Plot: {p}")

    # Also save a tiny text report
    report_path = os.path.join("output", f"{ticker}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BASIC NASDAQ QUANT REPORT\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Price: {comp['price']:.2f}\n")
        f.write(f"SMA50: {comp['SMA50']:.2f} | SMA200: {comp['SMA200']:.2f}\n")
        f.write(f"Vol_Z: {comp['Vol_Z']:.2f}\n")
        f.write(f"Volume_Z: {comp['Volume_Z']:.2f}\n")
        f.write(f"Trend above 200D?: {comp['above_200']}\n")
        f.write(f"SMA50 > SMA200?: {comp['slope_cross_50_over_200']}\n")
        f.write(f"\nSignal: {decision}\n")
        f.write(f"Stars: {stars}/5\n")
        f.write("\nNotes:\n- Heuristic only. Not investment advice.\n- Consider validating with fundamentals/valuation for a robust Morningstar-like rating.\n")
    print(f" - Report: {report_path}")


if __name__ == "__main__":
    main()
