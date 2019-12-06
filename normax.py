# Methods for calculating the expected value of the maximum of n standard normal RVs

import numpy as np
import scipy
from scipy.stats import norm

# Brute force simulation
def sim(n, trials):
    xs = np.random.randn(trials, n)
    maxes = np.amax(xs, axis=1)
    return np.mean(maxes), np.std(maxes, ddof=1) / np.sqrt(trials)

# Numerical integral method 1
# (Same as Methods S1 Equation 28.)
def normax_pdf(n, x):
    return n * np.power(scipy.stats.norm.cdf(x), n-1) * scipy.stats.norm.pdf(x)

def integ1(n):
    def integrand(x):
        return x * normax_pdf(n, x)
    mean, err = scipy.integrate.quad(integrand, -10, 10)

    return mean

# Numerical integral method 2
def integrand2(n, x):
    return np.where(x > 0,
        1-norm.cdf(x)**n,
        -norm.cdf(x)**n)

def integ2(n):
    mean, err = scipy.integrate.quad(lambda x: integrand2(n, x), -10, 10)
    return mean

# Karavani et al. Equation 1
# Proportionality constant from Methods S1 Equation 36
def calc_eqn_1(n):
    return 1.09 * np.sqrt(np.log(n))

# Karavani et al. Methods S1 Equation 32
def calc_eqn_s1_32(n):
    if n == 1: return 0
    em_const = 0.5772156649
    return scipy.stats.norm.ppf(1-1/n) + em_const / (n * scipy.stats.norm.pdf(scipy.stats.norm.ppf(1-1/n)))
