"""Import the Industry etf price closings data csv"""
CLOSINGS = pd.read_csv("IND_ETF_PRICE_CLOSINGS.csv")
ETF_RETURNS = pd.read_csv("ETF_RETURNS.csv")


"""Make a composite etf for all the etfs that are positively correlated with each other"""
CLOSINGS.index = CLOSINGS['Date']
del CLOSINGS['Date']
C = CLOSINGS.dropna().corr()

def SortCorr(ind, corr_df):
    """Sorts the other ind etfs based on correlations from h to l."""
    return corr_df[ind].sort_values()

print(SortCorr('VOX', C)) 
