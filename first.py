import pandas as pd

df = pd.read_csv('~/tpc/embryo-pgs-selection/analysis/data/large-fam_height_permitted-subset.csv')
print(df)
print(df.describe())
