import numpy as np
from scipy.interpolate import PchipInterpolator
from collections import defaultdict


def map_dividends_to_steps(dt, N, dividends=None):
    """
    Map continuous dividend times to discrete binomial tree steps.

    Parameters
    ----------
    dt : float
        Time step size (T / N).
    N : int
        Total number of time steps.
    dividends : list of (float, float), optional
        List of (time, amount) for discrete cash dividends.
        If None or empty, no dividends are applied.

    Returns
    -------
    dict
        Mapping {step_index: total_dividend} where step_index is the
        integer time step at or just after the dividend date. If multiple
        dividends fall on the same step, amounts are summed.
    """
    
    dividend_steps = defaultdict(float)

    if not dividends:
        return {}

    for t, D in dividends:
        if D <= 0:
            continue
        step = int(np.floor(t / dt))
        if step <= N:   # ignore dividends beyond maturity
            dividend_steps[step] += D

    return dict(dividend_steps)
            

def interpolate_post_dividend(S_nodes, values, D):
    """
    Apply Vellekoop dividend adjustment by interpolating option values 
    after a discrete cash dividend.

    Parameters
    ----------
    S_nodes : array_like
        Stock prices at the current tree step (before dividend).
    values : array_like
        Option values corresponding to S_nodes at time t+.
    D : float
        Dividend amount to subtract from stock prices.

    Returns
    -------
    ndarray
        Adjusted option values at time t- corresponding to original S_nodes.

    Notes
    -----
    Implements:
        V(t-, S-) = V(t+, max(S- - D, 0))
    Uses monotone PCHIP interpolation to evaluate values at shifted stock 
    prices S- - D. Negative stock levels are floored at zero.
    """
    S_nodes = np.asarray(S_nodes, dtype=float)
    values = np.asarray(values, dtype=float)

    if S_nodes.shape != values.shape:
        raise ValueError("S_nodes and values must have the same shape.")

    # Shift stock levels after dividend
    S_after = np.maximum(S_nodes - D, 0.0)

    # Interpolate option values at shifted stock levels
    cs = PchipInterpolator(S_nodes, values, extrapolate=True)
    adjusted_values = cs(S_after)

    return adjusted_values


def pp_inv(z, n):
    """ Peizer-Pratt Inversion of z """
    sgn = 1.0 if z >= 0 else -1.0
    return 0.5 + 0.5 * sgn * np.sqrt(1.0 - np.exp( -1 * (z / ((n + 1 / 3) + (0.1 / (n + 1))))**2 * (n + 1 / 6)))


def american_option_leisen_reimer_with_cash_dividends(N, S0, K, T, sigma, r, dividends, option_type):
    if N % 2 == 0: N += 1
    dt = T / N
    sdt = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5*sigma**2) * T) / sdt
    d2 = d1 - sdt
    p = pp_inv(d2, N)
    pNot = pp_inv(d1, N)
    R = np.exp((r) * dt) 
    df = np.exp(-r * dt) 
    u = R * pNot / p
    d = (R - p * u) / (1 - p)
    
    # option payoff at maturity
    stockGrid = S0 * d**N * (u / d)**np.arange(N + 1)
    values = np.maximum(stockGrid - K, 0.0) if option_type == "call" else np.maximum(K - stockGrid, 0.0)
    
    # convert dividend times to step indices
    dividend_steps = map_dividends_to_steps(dt, N, dividends)

    # Backward induction from column j = n - 2 to 0
    for i in range(N - 1, -1, -1):
        # standard backward step
        values = df * (p * values[1:] + (1-p) * values[:-1])

        # stock prices at this time step
        indices = np.arange(i + 1)        # 0..j
        stockGrid = S0 * (u ** indices) * (d ** (i - indices))

        # dividend adjustment (if a dividend occurs here)
        if i in dividend_steps:
            D = dividend_steps[i]
            values = interpolate_post_dividend(stockGrid, values, D)
            
        values = np.maximum(values, stockGrid - K) if option_type == "call" else np.maximum(values, K - stockGrid)

    return values[0]



S0 = 100
K = 100
sigma = 0.3
r = 0.05
T = 1.0
N = 200
option_type = "call"
dividends = [(0.1, 7.0)]

price = american_option_leisen_reimer_with_cash_dividends(N, S0, K, T, sigma, r, dividends, option_type)














