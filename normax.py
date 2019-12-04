import numpy as np
import scipy
from scipy.stats import norm

def sim(n, trials):
    xs = np.random.randn(trials, n)
    maxes = np.amax(xs, axis=1)
    return np.mean(maxes)

def normax_pdf(n, x):
    return n * np.power(scipy.stats.norm.cdf(x), n-1) * scipy.stats.norm.pdf(x)

def integ1(n):
    def integrand(x):
        return x * normax_pdf(n, x)
    mean, err = scipy.integrate.quad(integrand, -10, 10)

    return mean

def integrand2(n, x):
    return np.where(x > 0,
        1-norm.cdf(x)**n,
        -norm.cdf(x)**n)

def integ2(n):
    mean, err = scipy.integrate.quad(lambda x: integrand2(n, x), -10, 10)
    return mean
