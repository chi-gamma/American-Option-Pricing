import numpy as np



def pp_inv(z, n):
    """ Peizer-Pratt Inversion of z """
    sgn = 1.0 if z >= 0 else -1.0
    return 0.5 + 0.5 * sgn * np.sqrt(1.0 - np.exp( -1 * (z / ((n + 1 / 3) + (0.1 / (n + 1))))**2 * (n + 1 / 6)))


def priceAmericanPut_LR(n, S0, K, tenor, sigma, rate, div):
    dt = tenor / n
    sdt = sigma * np.sqrt(tenor)
    d1 = (np.log(S0 / K) + (rate - div + 0.5*sigma**2) * tenor) / sdt
    d2 = d1 - sdt
    p = pp_inv(d2, n)
    q = 1 - p
    pNot = pp_inv(d1, n)
    R = np.exp((rate - div) * dt) 
    df = np.exp(-rate * dt) 
    u = R * pNot / p
    d = (R - p * u) / (1 - p)
    h = (n - 1) // 2  # maximum node index with non-zero exercise value 
    payoff = np.zeros(h + 2)
    
    # Terminal payoff (American put, only up to h)
    for j in range(h + 1):
        payoff[j] = max(K - S0 * d**(n-j) * u**j, 0.0)

    # Penultimate column: find the first node where early exercise is optimal
    opt_ex_index = 0 
    opt_ex_flag = False
    temp = np.empty(h + 1)
    for i in range(h, -1, -1):
        exercise_value = K - S0 * d**(n - 1 - i) * u**i
        hold_value = df * (q * payoff[i] + p * payoff[i + 1])
        if exercise_value > hold_value:
            opt_ex_index = i
            opt_ex_flag = True
            temp[i] = exercise_value
            break
        else:
            temp[i] = hold_value
    payoff[i : -1] = temp[i : ]
    
    # Backward induction from column j = n - 2 to 0
    for j in range(n - 2, -1, -1):
        max_ind = min(h, j) + 1
        start = 0
        if opt_ex_flag:
            # Check current boundary node
            exercise_value = K - S0 * d**(j - opt_ex_index) * u**opt_ex_index
            hold_value = df * (q * payoff[opt_ex_index] + p * payoff[opt_ex_index + 1])
            if exercise_value > hold_value:
                # Boundary index unchanged
                payoff[opt_ex_index] = exercise_value
                start = opt_ex_index + 1   
                
                if opt_ex_index == max_ind - 1:
                    # All nodes are optimal ⇒ price is exercise at t=0
                    return max(K - S0, 0.0)
                
            elif opt_ex_index != 0:
                # Boundary index moves one step up
                payoff[opt_ex_index] = hold_value
                opt_ex_index -= 1
                payoff[opt_ex_index] = K - S0 * d**(j - opt_ex_index) * u**opt_ex_index
                start = opt_ex_index + 2
                
                if opt_ex_index == max_ind - 1:
                    # All nodes are optimal ⇒ price is exercise at t=0
                    return max(K - S0, 0.0)
            else:
                # Boundary vanishes
                opt_ex_flag = False
                
        if start < max_ind:
            payoff[start:max_ind] = df * (
                q * payoff[start:max_ind] + p * payoff[start + 1:max_ind + 1]
            )
            
    # # Option price at time 0
    price = payoff[0]
    return price




def priceAmerican_LR(n, S0, K, tenor, sigma, rate, div, optionType):
    if n < 2: raise ValueError("Steps must be >= 2")
    n = n + 1 if n % 2 == 0 else n
    if optionType not in ["CALL", "PUT"]:
        raise ValueError("Invalid Option Type")
    
    if optionType=="CALL":
        # infer call price using put call symmetry C(S,K,r,q,T) = P(K,S,q,r,T)
        S0, K = K, S0
        rate, div = div, rate
    result = priceAmericanPut_LR(n, S0, K, tenor, sigma, rate, div)
    return result


S0 = 100
K = 100
sigma = 0.3
rate = 0.05
div = 0.07
tenor = 1.0
n = 75
optionType = "PUT"


price = priceAmerican_LR(n, S0, K, tenor, sigma, rate, div, optionType)











