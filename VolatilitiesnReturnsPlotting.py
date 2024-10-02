import matplotlib.pyplot as plt

industry_etfs = {
    "telecom":['VOX', 'FCOM', 'NXTG', 'IXP', 'RSPC'],
    "utilities": ['XLU', 'PAVE', 'VPU', 'IGF', 'IFRA'],
    "technology": ['QQQ', 'VGT', 'XLK', 'IYW', 'SMH', 'IGV'],
    "realestate": ['VNQ', 'SCHH', 'XLRE', 'IYR', 'VNQI', 'REET'],
    "materials": ['GDX', 'GUNR', 'XLB', 'GDXJ', 'GNR', 'VAW'],
    "industrials": ['XLI', 'ITA', 'VIS', 'PPA', 'XAR', 'PHO', 'JETS', 'FXR', 'FIW', 'XTN'],
    "healthcare": ['XLV', 'VHT', 'IBB', 'XBI', 'IHI', 'IXJ'],
    "financials": ['XLF', 'VFH', 'KRE', 'FAS', 'IYF', 'BIZD', 'KIE', 'IAI', 'PSP', 'KCE'],
    "energy": ['XLE', 'VDE', 'AMLP', 'XOP', 'ICLN', 'OIH', 'MLPX', 'FCG'],
    "consumerstaples": ['XLP', 'VDC', 'IEV', 'IYK', 'EATV', 'FSTA'],
    "consumerdiscretionary": ['XLY', 'XLC', 'VCR', 'FXD', 'FDIS', 'XRT', 'PEJ'],
    "RegionalBanks": ['IAT'],
    "Gold": ['GLD'],
}

"""Import the Industry etf price closings data csv"""
CLOSINGS = pd.read_csv("IND_ETF_PRICE_CLOSINGS.csv")
ETF_RETURNS = pd.read_csv("ETF_RETURNS.csv")


"""Make a composite etf for all the etfs that are positively correlated with each other"""
CLOSINGS.index = CLOSINGS['Date']
del CLOSINGS['Date']
C = CLOSINGS.dropna()

def SortCorr(ind, corr_df):
    """Sorts the other ind etfs based on correlations from h to l."""
    return corr_df[ind].sort_values()

"""Matrix of etf % changes"""
def perc_mat(price_mat):
    mat = pd.DataFrame()
    for ticker in price_mat:
        perc_changes = np.log(price_mat[ticker]/price_mat[ticker].shift()).dropna()
        mat[ticker] = perc_changes
    
    return mat


ticker = "XLK"
start = 3000
fig, ax1 = plt.subplots()

# Plot stock prices on ax1
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price', color='tab:blue')
ax1.plot(CLOSINGS.index[start:], CLOSINGS[ticker][start:], color='tab:blue', label='Stock Price')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for the returns
ax2 = ax1.twinx()
ax2.set_ylabel('Stock Returns', color='tab:orange')
ax2.plot(CLOSINGS.index[start:], np.abs(perc_mat(CLOSINGS)[ticker][start-1:]), color='tab:orange', label='Stock Returns')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Title and show
plt.title(f"{ticker} Prices and Returns")
fig.tight_layout()
plt.show()
