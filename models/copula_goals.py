import numpy as np
from scipy.stats import poisson

def frank_copula(u, v, theta):
    """Frank copula for symmetric tail dependence."""
    if abs(theta) < 1e-6:
        return u * v
    num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
    den = np.exp(-theta) - 1
    
    # Add small epsilon to avoid log(0)
    arg = 1 + num/den
    if arg <= 0:
        return 0.0
    return -1/theta * np.log(arg)

def bivariate_score_matrix(lam_h, lam_a, theta=2.0, max_g=8):
    """
    Scoreline matrix using Frank copula to model H/A goal correlation.
    More realistic than independent Poisson for international football.
    """
    g = np.arange(max_g + 1)
    F_h = poisson.cdf(g, lam_h)   # CDF
    F_a = poisson.cdf(g, lam_a)
    
    matrix = np.zeros((max_g+1, max_g+1))
    for i in range(max_g+1):
        for j in range(max_g+1):
            u = F_h[i]; v = F_a[j]
            # Copula density approximation
            if i == 0:
                F_h_prev = 0.0
            else:
                F_h_prev = F_h[i-1]
            if j == 0:
                F_a_prev = 0.0
            else:
                F_a_prev = F_a[j-1]

            C11 = frank_copula(u, v, theta)
            C10 = frank_copula(F_h_prev, v, theta)
            C01 = frank_copula(u, F_a_prev, theta)
            C00 = frank_copula(F_h_prev, F_a_prev, theta)
            matrix[i,j] = max(0, C11 - C10 - C01 + C00)

    total = matrix.sum()
    return matrix / total if total > 0 else matrix
