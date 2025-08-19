import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tick_list = ['SPY', 'TLT', 'DBC', 'TIP', 'GLD', 'XLP', 'XLE', 'VNQ', 'XLU', 'XLV']
Tickers = yf.Tickers(tick_list).history(start="2025-01-01", end="2025-08-17", interval="1mo")

def view_market(df):
    for ticker in df["Close"]:
        data = df["Close"][ticker]
        plt.plot(data/data.max(), label=f"{ticker}")
    plt.legend()
    plt.show()


view_market(Tickers)
