from scipy.special import ndtr
import math


def BJERKSUNDSTENSLANDCALL(S0, K, r, q, sigma, T):
    b = r - q  # drift rate
    term1 = sigma * sigma
    term2 = b / term1
    term3 = sigma * math.sqrt(T)
    
    # Beta (related to early exercise boundary)
    beta = (0.5 - term2) + math.sqrt((term2 - 0.5)**2 + 2.0 * r / term1)

    def phiFunc(S, T, gamma, H, X):
        lamda = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * term1
        kappa = 2.0 * b / term1 + (2.0 * gamma - 1.0)
        term4 = (b + (gamma - 0.5) * sigma * sigma) * T
        d1 = - (math.log(S / H) + term4) / term3
        d2 = - (math.log(X**2 / (S * H)) + term4) / term3
        return math.exp(lamda * T) * S**gamma * (ndtr(d1) - (X / S)**kappa * ndtr(d2))

    B_0 = max(K, r * K / q)
    B_inf = beta * K / (beta - 1.0)
    h = -(b * T + 2.0 * term3) * (K**2 / ((B_inf - B_0) * B_0))
    X = B_0 + (B_inf - B_0) * (1.0 - math.exp(h))  # early exercise boundary
    alpha = (X - K) * X**-beta

    call = (
        alpha * S0**beta
        - alpha * phiFunc(S0, T, beta, X, X)
        + phiFunc(S0, T, 1.0, X, X)
        - phiFunc(S0, T, 1.0, K, X)
        - K * phiFunc(S0, T, 0.0, X, X)
        + K * phiFunc(S0, T, 0.0, K, X)
    )
    return call


def BJERKSUNDSTENSLAND(S0, K, r, q, sigma, T, optionType):
    if optionType not in ["CALL", "PUT"]:
        raise ValueError("Invalid Option Type")
    
    if optionType=="PUT":
        # infer call price using put call symmetry C(S,K,r,q,T) = P(K,S,q,r,T)
        S0, K = K, S0
        r, q = q, r
    price = BJERKSUNDSTENSLANDCALL(S0, K, r, q, sigma, T)
    return price

r = 0.05
q = 0.07
sigma = 0.3
K = 100.0
S0 = 100.0
T = 1.0
optionType = "CALL"

price = BJERKSUNDSTENSLAND(S0, K, r, q, sigma, T, optionType)

