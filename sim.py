import numpy as np
import pandas as pd
from scipy.special import binom

# Population variance explained by the height PGS
pop_r2 = 0.243

oracle = True
if oracle: print('ORACLE MODE ENABLED')

def calculate_item_selection_gain_brute(k, x, y):
    # k = size of tuple
    # x = predicted values
    # y = actual values

    n = len(x)
    if oracle: x = y

    subtotal_gain = 0.
    num_pairs = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                num_pairs += 1
                if x[j] > x[i]:
                    i,j = j,i
                cur_gain = (y[i] - y[j]) / 2
                subtotal_gain += cur_gain

    return subtotal_gain / num_pairs

def calculate_item_selection_gain_naive(k, x, y):
    # k = size of tuple
    # x = predicted values
    # y = actual values

    n = len(x)
    
    if oracle: x = y
    a = np.stack([x,y])
    print(f' a = {a}')
    a = np.sort(a)
    x = a[0]
    y = a[1]
    
    benefit = 0.
    for idx in range(n):
        term = y[idx]
        i = idx+1
        benefit += y[idx] * binom(i-1, k-1)
    benefit /= binom(n, k)
    
    cost = np.mean(y)
    
    return benefit - cost

def main():
    df = pd.read_csv('~/tpc/embryo-pgs-selection/analysis/data/large-fam_height_permitted-subset.csv')

    fids = set(df['FID'])
    print(f'Number of individuals: {len(df)}')
    print(f'Number of families: {len(fids)}')

    kids_indic = df['IID'] < 91
    kids_df = df.loc[kids_indic]
    print(f'Number of kids: {len(kids_df)}')

    total_gain = 0.
    for fid in fids:
        family_df = kids_df.loc[kids_df['FID'] == fid]
        
        fid_gain = calculate_item_selection_gain_brute(2, family_df['predicted'].values, family_df['measured'].values)
        print(f'  fid_gain = {fid_gain}')
        total_gain += fid_gain
    avg_gain = total_gain / len(fids)
    print(f'Average gain:  {avg_gain}')

main()
