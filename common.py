import numpy as np
from scipy.special import binom
import pandas as pd

import pytest

def calculate_item_selection_gain_brute(k, x, y):
    assert k == 2
    # k = size of tuple
    # x = predicted values
    # y = actual values

    n = len(x)

    subtotal_gain = 0.
    num_pairs = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                a,b = i,j
                num_pairs += 1
                if x[b] > x[a]:
                    a,b = b,a
                cur_gain = (y[a] - y[b]) / 2
                subtotal_gain += cur_gain

    return subtotal_gain / num_pairs

def calculate_item_selection_gain_naive(k, x, y):
    # k = size of tuple
    # x = predicted values
    # y = actual values

    n = len(x)

    a = np.recarray((n,), dtype=[('x', float), ('y', float)])
    a['x'] = x
    a['y'] = y
    a = np.sort(a, order='x')
    x = a['x']
    y = a['y']

    benefit = 0.
    for idx in range(n):
        term = y[idx]
        i = idx+1
        benefit += y[idx] * binom(i-1, k-1)
    benefit /= binom(n, k)

    cost = np.mean(y)

    return benefit - cost

def calculate_item_selection_gain(k, x, y):
    # k = size of tuple
    # x = predicted values
    # y = actual values

    n = len(x)
    assert n >= k

    a = np.recarray((n,), dtype=[('x', float), ('y', float)])
    a['x'] = x
    a['y'] = y
    a = np.sort(a, order='x')
    x = a['x']
    y = a['y']

    benefit = 0.
    for idx in range(n):
        term = y[idx] * (k / n)
        i = idx+1
        for j in range(1, k):
            term *= (i-j) / (n-j)
        benefit += term

    cost = np.mean(y)

    return benefit - cost

def hockey_stick_pmf_unstable(n, k):
    pmf = np.zeros(shape=(n,), dtype=np.float)
    for idx in range(n):
        i = idx+1
        pmf[idx] = binom(i-1, k-1) / binom(n, k)
    return pmf

def hockey_stick_pmf(n, k):
    assert n >= k

    pmf = np.zeros(shape=(n,), dtype=np.float)

    pmf[n-1] = k/n

    for idx in reversed(range(n-1)):
        i = idx+1

        pmf[idx] = pmf[idx+1] * (i - (k-1)) / i

    return pmf

def calculate_item_selection_gain_fast(k, x, y):
    # k = size of tuple
    # x = predicted values
    # y = actual values

    n = len(x)
    assert n >= k

    sorted_idxs = np.argsort(x)
    y = y[sorted_idxs]

    benefit = hockey_stick_pmf(n, k).dot(y)
    cost = np.mean(y)

    return benefit - cost

def test_basic():
    fns = [
        calculate_item_selection_gain_brute,
        calculate_item_selection_gain_naive,
        calculate_item_selection_gain,
        calculate_item_selection_gain_fast,
    ]

    for fn in fns:
        y = np.array([0,2])
        x = y
        assert fn(2, x, y) == 1

        y = np.array([2,0])
        x = y
        assert fn(2, x, y) == 1

        y = np.array([0,0,6])
        x = y
        assert fn(2, x, y) == 2

        # (1/3)(3-1.5 + 6-3 + 6-4.5)
        # = (1/3)(6)
        # = 2
        y = np.array([0,3,6])
        x = y
        assert fn(2, x, y) == 2

        y = np.array([0,2])
        x = np.array([2,0])
        assert fn(2, x, y) == -1

######

HEIGHT_MEAN = 173
HEIGHT_SD = 5.6

def cm_of_sd(phen_sd):
    return HEIGHT_MEAN + HEIGHT_SD * phen_sd

HEIGHT_H2 = 0.8
HEIGHT_PGS_R2 = 0.243

######

def calculate_gain(df, k, fids):
    total_gain = 0.
    for fid in fids:
        family_df = df.loc[df['FID'] == fid]

        fid_gain = calculate_item_selection_gain_fast(k, family_df['predicted'].values, family_df['measured'].values)
        total_gain += fid_gain / HEIGHT_SD
    avg_gain = total_gain / len(fids)
    return avg_gain

######

def variance_of_permuted_dot_simulation(u, v, trials):
    dots = np.zeros(shape=(trials,), dtype=np.float)
    for i in range(trials):
        np.random.shuffle(v)
        dots[i] = np.dot(u, v)
    return dots.var()

def variance_of_permuted_dot(u, v):
    n = len(u)
    assert len(v) == n

    uhat = u - u.mean()
    vhat = v - v.mean()

    return np.dot(uhat, uhat) * np.dot(vhat, vhat) / (n-1)

######

full_df = pd.read_csv('~/tpc/embryo-pgs-selection/analysis/data/large-fam_height_permitted-subset.csv')
all_fids = set(full_df['FID'])
kids_df = full_df.loc[full_df['IID'] < 91]

def fids_for_min_family_size(k):
    tmp = kids_df.groupby('FID').count()['measured'] >= k
    return np.array(tmp[tmp].index)

def kids_df_for_min_family_size(k):
    return kids_df.loc[(kids_df.groupby('FID').count()['measured'][kids_df['FID']] >= k).values]

def kids_df_of_fid(fid):
    return kids_df.loc[kids_df['FID'] == fid]

######

def null_gain_exact_variance(k, fids, fn=variance_of_permuted_dot):
    ret = 0

    for fid in fids:
        family_df = kids_df.loc[kids_df['FID'] == fid]

        n = len(family_df)

        u = hockey_stick_pmf(n, k)
        v = family_df.measured.values / HEIGHT_SD

        fid_gain_var = fn(u, v)

        ret += fid_gain_var

    return ret / (len(fids)**2)
