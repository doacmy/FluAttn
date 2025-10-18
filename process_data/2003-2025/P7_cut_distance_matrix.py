import pandas as pd
import numpy as np

df = pd.read_csv("data/prd/2003-2025/2003_2025_distance_matrix.csv", index_col=0)



row_years = df.index.astype(str).str[-4:].astype(int)
col_years = df.columns.astype(str).str[-4:].astype(int)

row_mask = (row_years >= 2007) & (row_years <= 2019)
col_mask = (col_years >= 2007) & (col_years <= 2019)

cdf = df.loc[row_mask, col_mask]

# Convert to 3-column long format using upper triangle (exclude diagonal)
upper_mask = pd.DataFrame(
    np.triu(np.ones(cdf.shape, dtype=bool), k=1),
    index=cdf.index,
    columns=cdf.columns,
)
short_df = cdf.where(upper_mask).stack().reset_index()
short_df.columns = ["test", "ref", "distance"]

v1_year = pd.to_numeric(short_df["test"].astype(str).str.extract(r"(\d{4})$")[0], errors="coerce")
v2_year = pd.to_numeric(short_df["ref"].astype(str).str.extract(r"(\d{4})$")[0], errors="coerce")
train_mask = (v1_year >= 2007) & (v1_year <= 2017) & (v2_year >= 2007) & (v2_year <= 2017)
train_df = short_df.loc[train_mask].reset_index(drop=True)

val_mask = (
    ((v1_year == 2018) & (v2_year >= 2007) & (v2_year <= 2017))
    | ((v2_year == 2018) & (v1_year >= 2007) & (v1_year <= 2017))
)
val_df = short_df.loc[val_mask].reset_index(drop=True)

test_mask = (
    ((v1_year == 2019) & (v2_year >= 2007) & (v2_year <= 2017))
    | ((v2_year == 2019) & (v1_year >= 2007) & (v1_year <= 2017))
)
test_df = short_df.loc[test_mask].reset_index(drop=True)

seq_df = pd.read_csv('data/prd/2003-2025/sequences.csv')
seq_df = seq_df[~seq_df['HA1_Sequence'].astype(str).str.contains('X', na=False)].copy()

seq_map = seq_df.set_index('Virus')['HA1_Sequence']

train_df['S1'] = train_df['test'].map(seq_map)
train_df['S2'] = train_df['ref'].map(seq_map)
train_df = train_df.dropna(subset=['S1', 'S2']).reset_index(drop=True)
train_df = train_df[['S1', 'S2', 'distance']].reset_index(drop=True)


val_df['S1'] = val_df['test'].map(seq_map)
val_df['S2'] = val_df['ref'].map(seq_map)
val_df = val_df.dropna(subset=['S1', 'S2']).reset_index(drop=True)
val_df = val_df[['S1', 'S2', 'distance']].reset_index(drop=True)

test_df['S1'] = test_df['test'].map(seq_map)
test_df['S2'] = test_df['ref'].map(seq_map)
test_df = test_df.dropna(subset=['S1', 'S2']).reset_index(drop=True)
test_df = test_df[['S1', 'S2', 'distance']].reset_index(drop=True)

train_out_path = 'data/time_series/train.csv'
val_out_path = 'data/time_series/val.csv'
test_out_path = 'data/time_series/test.csv'

train_df.to_csv(train_out_path, index=False)
val_df.to_csv(val_out_path, index=False)
test_df.to_csv(test_out_path, index=False)
