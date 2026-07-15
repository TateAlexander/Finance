import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from functools import lru_cache


def fit_autocorr_adjusted_trend_segment(close_values):
    """
    Fit one AR(1)-adjusted trend segment:

        Close_t = beta0 + beta1*t + error_t

    using feasible GLS / Cochrane-Orcutt.
    """

    close_values = np.asarray(close_values, dtype=float)
    n = len(close_values)

    if n < 5:
        raise ValueError("Need at least 5 observations.")

    t = np.arange(n)

    # Step 1: OLS
    X = sm.add_constant(t)
    ols_model = sm.OLS(close_values, X).fit()
    resid = ols_model.resid

    # Step 2: estimate rho
    e_t = resid[1:]
    e_lag = resid[:-1]

    denom = np.sum(e_lag ** 2)

    if denom == 0:
        rho_hat = 0.0
    else:
        rho_hat = np.sum(e_t * e_lag) / denom

    rho_hat = np.clip(rho_hat, -0.99, 0.99)

    # Step 3: transform data
    y_star = close_values[1:] - rho_hat * close_values[:-1]
    t_star = t[1:] - rho_hat * t[:-1]

    X_star = pd.DataFrame({
        "const_star": np.full(len(t_star), 1 - rho_hat),
        "t_star": t_star
    })

    fgls_model = sm.OLS(y_star, X_star).fit()

    beta0 = fgls_model.params["const_star"]
    beta1 = fgls_model.params["t_star"]

    fitted = beta0 + beta1 * t
    residuals_original_scale = close_values - fitted

    # Use transformed residual SSE for objective scoring
    ssr_transformed = float(fgls_model.ssr)

    # Also keep original-scale SSE for diagnostics
    ssr_original = float(np.sum(residuals_original_scale ** 2))

    return {
        "n": n,
        "rho_hat": rho_hat,
        "beta0": beta0,
        "beta1": beta1,
        "fitted": fitted,
        "resid_original": residuals_original_scale,
        "ssr_transformed": ssr_transformed,
        "ssr_original": ssr_original,
        "ols_model": ols_model,
        "fgls_model": fgls_model
    }


def optimal_segmented_autocorr_trends(
    ticker,
    start=None,
    end=None,
    period=None,
    interval="1d",
    min_segment_len=20,
    penalty_strength=1.0,
    candidate_step=1,
    plot=True
):
    """
    Finds objectively chosen segmented AR(1)-adjusted trend lines.

    The objective is approximately:

        total_score = sum(segment fit error) + BIC penalty for more segments

    Higher penalty_strength = fewer segments.
    Lower penalty_strength = more segments.

    Parameters
    ----------
    ticker : str
        Stock ticker, e.g. "AM", "AAPL".
    start : str
        Start date, e.g. "2024-01-01".
    end : str
        End date, e.g. "2025-01-01".
    period : str
        yfinance period, e.g. "1y", "6mo".
        Use period OR start/end.
    interval : str
        yfinance interval, e.g. "1d", "1wk".
    min_segment_len : int
        Minimum number of bars per trend segment.
    penalty_strength : float
        Multiplier on BIC penalty.
        0.5 = more breaks, 1.0 = BIC default, 2.0 = fewer breaks.
    candidate_step : int
        Only consider breakpoints every candidate_step bars.
        Use 1 for most accurate, 3 or 5 for faster.
    plot : bool
        Whether to plot segmented trend lines.

    Returns
    -------
    result : dict
        Contains data, segment table, and objective score.
    """

    # --------------------------------------------------
    # Download data
    # --------------------------------------------------
    if period is not None:
        raw = yf.Ticker(ticker).history(period=period, interval=interval)
    else:
        raw = yf.Ticker(ticker).history(start=start, end=end, interval=interval)

    if raw.empty:
        raise ValueError("No data returned. Check ticker, dates, period, or interval.")

    df = raw[["Close"]].dropna().copy()
    df["global_t"] = np.arange(len(df))

    y = df["Close"].to_numpy()
    n = len(y)

    if n < 2 * min_segment_len:
        raise ValueError("Not enough data for segmentation. Lower min_segment_len or use more data.")

    # BIC-style penalty.
    # Each new segment estimates beta0, beta1, rho.
    params_per_segment = 3
    penalty_per_segment = penalty_strength * params_per_segment * np.log(n)

    # --------------------------------------------------
    # Cached segment fitting
    # --------------------------------------------------
    @lru_cache(maxsize=None)
    def segment_fit_cached(a, b):
        """
        Fit segment y[a:b].
        """
        values = y[a:b]
        fit = fit_autocorr_adjusted_trend_segment(values)

        n_eff = max(fit["n"] - 1, 1)
        ssr = max(fit["ssr_transformed"], 1e-12)

        # Negative log-likelihood style fit cost
        fit_cost = n_eff * np.log(ssr / n_eff)

        # Penalize adding a segment
        total_cost = fit_cost + penalty_per_segment

        return total_cost, fit

    # --------------------------------------------------
    # Dynamic programming
    # dp[j] = best score for y[0:j]
    # --------------------------------------------------
    dp = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=int)

    dp[0] = 0.0

    for j in range(min_segment_len, n + 1):

        # Candidate previous breakpoints
        possible_starts = range(0, j - min_segment_len + 1, candidate_step)

        for i in possible_starts:
            seg_len = j - i

            if seg_len < min_segment_len:
                continue

            if i != 0 and dp[i] == np.inf:
                continue

            try:
                seg_cost, _ = segment_fit_cached(i, j)
            except Exception:
                continue

            candidate_score = dp[i] + seg_cost

            if candidate_score < dp[j]:
                dp[j] = candidate_score
                prev[j] = i

    if prev[n] == -1:
        raise RuntimeError("No valid segmentation found. Try lowering min_segment_len.")

    # --------------------------------------------------
    # Backtrack chosen segments
    # --------------------------------------------------
    segments = []

    j = n
    while j > 0:
        i = prev[j]
        segments.append((i, j))
        j = i

    segments = segments[::-1]

    # --------------------------------------------------
    # Build output dataframe
    # --------------------------------------------------
    df["segment_id"] = np.nan
    df["segment_trend"] = np.nan
    df["segment_resid"] = np.nan

    segment_rows = []

    for segment_id, (a, b) in enumerate(segments, start=1):
        _, fit = segment_fit_cached(a, b)

        df.iloc[a:b, df.columns.get_loc("segment_id")] = segment_id
        df.iloc[a:b, df.columns.get_loc("segment_trend")] = fit["fitted"]
        df.iloc[a:b, df.columns.get_loc("segment_resid")] = fit["resid_original"]

        segment_rows.append({
            "segment_id": segment_id,
            "start_date": df.index[a],
            "end_date": df.index[b - 1],
            "start_idx": a,
            "end_idx": b - 1,
            "n": b - a,
            "rho_hat": fit["rho_hat"],
            "beta0": fit["beta0"],
            "beta1_slope_per_bar": fit["beta1"],
            "trend_start": fit["fitted"][0],
            "trend_end": fit["fitted"][-1],
            "trend_change": fit["fitted"][-1] - fit["fitted"][0],
            "ssr_transformed": fit["ssr_transformed"],
            "ssr_original": fit["ssr_original"]
        })

    segment_table = pd.DataFrame(segment_rows)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    if plot:
        plt.figure(figsize=(14, 7))

        plt.plot(
            df.index,
            df["Close"],
            label="Actual Close",
            linewidth=1.5
        )

        for _, row in segment_table.iterrows():
            sid = int(row["segment_id"])
            mask = df["segment_id"] == sid

            plt.plot(
                df.index[mask],
                df.loc[mask, "segment_trend"],
                linewidth=3,
                label=f"Seg {sid}: slope={row['beta1_slope_per_bar']:.4f}"
            )

            if sid > 1:
                plt.axvline(row["start_date"], linestyle="--", linewidth=1)

        plt.title(
            f"{ticker}: Objective Segmented AR(1)-Adjusted Trends\n"
            f"min_segment_len={min_segment_len}, penalty_strength={penalty_strength}"
        )
        plt.xlabel("Date")
        plt.ylabel("Close price")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df["segment_resid"], label="Segment residuals")
        plt.axhline(0, linewidth=1)
        plt.title(f"{ticker}: Residuals Around Objective Segmented Trends")
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("----- Objective Segmentation Summary -----")
    print(f"Ticker: {ticker}")
    print(f"Number of observations: {n}")
    print(f"Number of segments: {len(segment_table)}")
    print(f"Final objective score: {dp[n]:.4f}")
    print(f"Penalty per segment: {penalty_per_segment:.4f}")
    print()

    print(segment_table[[
        "segment_id",
        "start_date",
        "end_date",
        "n",
        "rho_hat",
        "beta1_slope_per_bar",
        "trend_change",
        "ssr_original"
    ]])

    return {
        "ticker": ticker,
        "data": df,
        "segments": segment_table,
        "objective_score": dp[n],
        "penalty_per_segment": penalty_per_segment,
        "penalty_strength": penalty_strength,
        "min_segment_len": min_segment_len
    }

result = optimal_segmented_autocorr_trends(
  ticker="AEP",
  start="2024-01-01",
  end="2025-01-01",
  interval="1d",
  min_segment_len=10,
  penalty_strength=1.0,
  candidate_step=1,
  plot=True
)
#----------------------------------------------------Code that puts autocorr trend, interval splits, vasieck based model together, H0: k = 0 rejection confirmation and 
#  Non-parametric estimation of variance together is below:

