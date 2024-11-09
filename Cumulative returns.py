from collections import defaultdict
import yfinance as yf

def cumRet(start, end, tickers):
    """Given the best performing industry etfs, which ones are most likely to sucseed for the next various time frames?"""
    prices = yf.Tickers(tickers).history(start=start, end=end, interval='1d')
    changes = {ticker: np.log(prices["Close"][ticker][-1]/prices["Close"][ticker][0]) for ticker in tickers}

    return sorted(changes.items(), key=lambda x:x[1])

tickers = ["VOX", "XLU", "VNQ", "QQQ", "GDX", "XLI", "XLV", "XLF", "XLE", "XLP", "XLY"]

# rets = cumRet('2024-9-01', '2024-11-30', tickers=["VOX", "XLU", "VNQ", "QQQ", "GDX", "XLI", "XLV", "XLF", "XLE", "XLP", "XLY"])
# print(rets)

def multiCumRet(tickers, years, months):
    cumRetDict = {}
    for year in years:
        for i, month in enumerate(months):
            if i+1 == len(months):
                break
            cumRetDict[f"{year}-{month}-01_{year}-{months[i+1]}-01"] = cumRet(f"{year}-{month}-01", f"{year}-{months[i+1]}-01", tickers)
            
    return cumRetDict

print(multiCumRet(tickers, ["2023"], ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]))
