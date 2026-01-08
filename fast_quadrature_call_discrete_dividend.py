import numpy as np
from scipy.special import ndtr


def fejer_quadrature(N: int):
    if N < 1:
        raise ValueError("N must be >= 1")

    k = np.arange(1, N+1)
    theta = (k - 0.5) * np.pi / N
    x = np.cos(theta)

    p = (N - 1) // 2
    w = np.full(N, 2.0 / N)

    if p > 0:
        h = np.arange(1, p+1)[:, None]
        denom = 4*h*h - 1
        add = np.cos(2*h*theta) / denom
        w -= (4.0 / N) * add.sum(axis=0)

    return x, w


def price_american_call_cash_dividends(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    div_times,      # iterable of dividend times tD in (0, T); strictly increasing
    div_amounts,    # iterable of cash dividends D at those times
    N: int,         # Clenshaw–Curtis order
    Xi: float       # width in std devs for integration bounds
) -> float:
    """
    American call with *cash* dividends under GBM with constant r, sigma.
    Spectral collocation (Clenshaw–Curtis) rollback with the corrected dividend indexing & exercise timing.

    Returns: price at t=0.
    """
    # --- input checks
    S0 = float(S0); K = float(K); r = float(r); sigma = float(sigma); T = float(T)
    tD = np.asarray(div_times, dtype=float)
    D  = np.asarray(div_amounts, dtype=float)
    
    m = len(tD) 
    timeGrid = np.concatenate(([0.0], tD, [T]))
    tau = np.diff(timeGrid)

    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Invalid S0, K, T, or sigma.")
    if (tD.size != D.size):
        raise ValueError("div_times and div_amounts must have same length.")
    if np.any(tD <= 0) or np.any(tD >= T) or np.any(np.diff(tD) <= 0):
        raise ValueError("div_times must be strictly increasing and lie in (0, T).")
    if N < 16:
        raise ValueError("N too small; use at least 16 (e.g., 256–2048).")
        
    if m == 0:
        c1 = sigma * np.sqrt(T)
        c2 = K * np.exp(-r * T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / c1
        d2 = d1 - c1
        return S0 * ndtr(d1) - c2 * ndtr(d2)

    # --- precompute
    sigma_sq = sigma * sigma
    twoPiVar = 2.0 * np.pi * sigma_sq
    mu = r - 0.5 * sigma_sq

    # Integration range [L, U] for cum-div stock
    U = S0 * np.exp(mu * T + Xi * sigma * np.sqrt(T))
    L = max(S0 * np.exp(mu * T - Xi * sigma * np.sqrt(T)), min(D)) # clamp lower bound by min dividend to avoid negative post-div S

    # Fejer nodes & weights in [-1, 1]
    xs, ws = fejer_quadrature(N)
    
    # get stock grid
    stockGrid = 0.5 * (U - L) * xs + 0.5 * (U + L)
    
    # Black Scholes Price at the last Ex-dividend date
    c1 = (r + 0.5 * sigma_sq) * tau[-1]
    c2 = sigma * np.sqrt(tau[-1])
    c3 = K * np.exp(-r * tau[-1])
    
    S_plus = stockGrid - D[-1] # Ex dividend prices
    d1 = (np.log(S_plus / K) + c1) / c2
    d2 = d1 - c2
    V = S_plus * ndtr(d1) - c3 * ndtr(d2)

    # --- rollback across dividend dates
    for k in range(m - 1, 0, -1):        
        if D[k] > K * (1.0 - np.exp(-r * tau[k])):          
            exercise = np.maximum(S_plus + D[k] - K, 0.0) # Early exercise check just before dividend k
            V = np.maximum(V, exercise)
            
        S_plus = stockGrid - D[k-1]        
        V_back = np.empty_like(stockGrid)
        for i in range(N):
            log_ratio = np.log(stockGrid / S_plus[i])
            expo = np.exp(-((log_ratio - mu * tau[k]) ** 2) / (2.0 * sigma_sq * tau[k]))
            kernel = (V / stockGrid) * (expo / np.sqrt(twoPiVar * tau[k]))
            V_back[i] = np.dot(kernel, ws)
        V = np.exp(-r * tau[k]) * 0.5 * (U - L) * V_back

    # --- Handle first dividend date
    if D[0] > K * (1.0 - np.exp(-r * tau[0])):
        V = np.maximum(V, stockGrid - K)

    # --- Final projection back to t0 
    log_ratio = np.log(stockGrid / S0)
    expo = np.exp(-((log_ratio - mu * tau[0]) ** 2) / (2.0 * sigma_sq * tau[0]))
    kernel = (V / stockGrid) * (expo / np.sqrt(twoPiVar * tau[0]))
    V0 = np.exp(-r * tau[0]) * 0.5 * (U - L) * np.dot(kernel, ws)
    
    return V0


# ------------------------ example ------------------------
S0, K, r, sigma, T = 100.0, 70.0, 0.05, 0.2, 1.0
div_times = [0.25, 0.75]
div_amounts = [5.0] * len(div_times)
price = price_american_call_cash_dividends(S0, K, r, sigma, T, div_times, div_amounts, N=1024, Xi=6.5)
print(f"American call with cash dividends = {price:.6f}")

