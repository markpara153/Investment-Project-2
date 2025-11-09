"""
nasdaq_start.py — minimal NASDAQ quantitative script.
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def fetch_data(ticker: str, years: int) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=365 * years + 30)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        repair=True
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Try a different ticker or range.")
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["Return"] = df["Adj Close"].pct_change()
    df["SMA50"] = df["Adj Close"].rolling(50).mean()
    df["SMA200"] = df["Adj Close"].rolling(200).mean()
    df["Vol30"] = df["Return"].rolling(30).std() * np.sqrt(252)
    df["Vol_SMA30"] = df["Volume"].rolling(30).mean()
    vstd = df["Vol30"].std(skipna=True) or 1.0
    df["Vol_Z"] = (df["Vol30"] - df["Vol30"].mean(skipna=True)) / vstd
    vsstd = df["Vol_SMA30"].std(skipna=True) or 1.0
    df["Volume_Z"] = (df["Vol_SMA30"] - df["Vol_SMA30"].mean(skipna=True)) / vsstd
    return df


def rate_signal(df: pd.DataFrame):
    last = df.dropna().iloc[-1]
    price = float(last["Adj Close"])
    sma50 = float(last["SMA50"])
    sma200 = float(last["SMA200"])
    vol_z = float(last["Vol_Z"])
    volu_z = float(last["Volume_Z"])
    above200 = price > sma200
    slopecross = sma50 > sma200
    if above200 and vol_z <= 0.5:
        signal = "BUY"
    elif not above200 and vol_z > 0.5:
        signal = "SELL"
    else:
        signal = "HOLD"
    stars = 3
    if above200:
        stars += 1
    if slopecross:
        stars += 1
    if vol_z > 1.0:
        stars -= 1
    if volu_z > 1.0 and not above200:
        stars -= 1
    stars = max(1, min(5, int(stars)))
    return signal, stars


def make_plots(df: pd.DataFrame, ticker: str):
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(10, 5))
    df["Adj Close"].plot(label="Adj Close")
    df["SMA50"].plot(label="SMA50")
    df["SMA200"].plot(label="SMA200")
    plt.legend()
    plt.title(f"{ticker} Price + 50/200-Day SMAs")
    plt.tight_layout()
    plt.savefig(f"output/{ticker}_price.png", dpi=150)
    plt.close()
    plt.figure(figsize=(10, 5))
    df["Vol30"].plot(label="30D Annualized Volatility")
    plt.legend()
    plt.title(f"{ticker} 30-Day Annualized Volatility")
    plt.tight_layout()
    plt.savefig(f"output/{ticker}_vol.png", dpi=150)
    plt.close()


def run(ticker="QQQ", years=5):
    print(f"Fetching {years} years of data for {ticker}…")
    df = fetch_data(ticker, years)
    df = add_indicators(df)
    signal, stars = rate_signal(df)
    make_plots(df, ticker)
    print("\n=== NASDAQ Quant Summary ===")
    print(f"Ticker: {ticker}")
    print(f"Signal: {signal}")
    print(f"Star Rating: {'★'*stars}{'☆'*(5-stars)} ({stars}/5)")
    print("Charts saved in ./output")


if __name__ == "__main__":
    run("NQ=F", 3)
