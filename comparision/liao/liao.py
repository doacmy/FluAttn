import random
import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def Calculate_X_Y(groups, pairs, virus_names, seqs_var):

    aa_to_group = {}
    for group, aa_list in groups.items():
        for aa in aa_list:
            aa_to_group[aa] = group

    m = seqs_var.shape[1]

    N = len(pairs)
    X = np.zeros((N, m))

    for j in range(m):
        for idx, (k, l) in enumerate(pairs):
            aa_k = seqs_var[k, j]
            aa_l = seqs_var[l, j]

            g1 = aa_to_group.get(aa_k, 'other')
            g2 = aa_to_group.get(aa_l, 'other')
            X[idx, j] = 1 if g1 != g2 else 0

    distance_df = pd.read_csv(distance_matrix_path, index_col=0)
        
    Y = []
    for i, j in pairs:
        virus_i = virus_names[i]
        virus_j = virus_names[j]
        dist = distance_df.loc[virus_i, virus_j]
        Y.append(dist)


    Y = np.array(Y)
    return X, Y

def Calculate_X_Y_from_df(groups, df: pd.DataFrame):
    """
    Build X and Y directly from a DataFrame with columns:
    - 'S1': sequence 1 (string)
    - 'S2': sequence 2 (string)
    - 'distance': antigenic distance (float)

    For each position j, X[j] = 1 if amino-acid groups (per 'groups') differ
    between S1[j] and S2[j], else 0. Unknown residues fall back to 'other'.
    """
    required_cols = {"S1", "S2", "distance"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Map amino-acids to group label
    aa_to_group = {}
    for group, aa_list in groups.items():
        for aa in aa_list:
            aa_to_group[aa] = group

    N = len(df)
    if N == 0:
        return np.zeros((0, 0)), np.array([])

    # Infer sequence length from first row and validate consistency
    first_s1 = str(df.iloc[0]["S1"]).strip()
    first_s2 = str(df.iloc[0]["S2"]).strip()
    if len(first_s1) != len(first_s2):
        raise ValueError("S1 and S2 in the first row have different lengths")
    m = len(first_s1)

    X = np.zeros((N, m))
    Y = np.zeros(N)

    for idx, row in df.iterrows():
        s1 = str(row["S1"]).strip()
        s2 = str(row["S2"]).strip()

        if len(s1) != m or len(s2) != m:
            raise ValueError(
                f"Inconsistent sequence lengths at row {idx}: len(S1)={len(s1)}, len(S2)={len(s2)}, expected {m}"
            )

        for j in range(m):
            aa_k = s1[j]
            aa_l = s2[j]
            g1 = aa_to_group.get(aa_k, 'other')
            g2 = aa_to_group.get(aa_l, 'other')
            X[idx, j] = 1 if g1 != g2 else 0

        Y[idx] = float(row["distance"])

    return X, Y


gm1 = {
    "nonpolar": ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'],
    "polar": ['C', 'N', 'Q', 'S', 'T', 'Y'],
    "charged": ['D', 'E', 'H', 'K', 'R']
}

gm2 = {
    "non-polar aliphatic" : ['A', 'G', 'I', 'L', 'M', 'V'],
    "non-polar aromatic" : ['F', 'P', 'W'],
    "polar": ['C', 'N', 'Q', 'S', 'T', 'Y'],
    "charged": ['D', 'E', 'H', 'K', 'R']
}

gm3 = {
    "non-polar": ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'],
    "polar": ['C', 'N', 'Q', 'S', 'T', 'Y'],
    "positively charged": ['H', 'K', 'R'],
    "negatively charged": ['D', 'E']
}

gm4 = {
    "non-polar aliphatic": ['A', 'G', 'I', 'L', 'M', 'V'],
    "non-polar aromatic": ['F', 'P', 'W'],
    "polar": ['C', 'N', 'Q', 'S', 'T', 'Y'],
    "positively charged": ['H', 'K', 'R'],
    "negatively charged": ['D', 'E']
}

gm5 = {
    "non-polar aliphatic": ['A', 'I', 'L', 'M', 'P', 'V'],
    "non-polar aromatic": ['F', 'W', 'Y'],
    "polar": ['N', 'Q', 'S', 'T'],
    "positively charged": ['H', 'K', 'R'],
    "negatively charged": ['D', 'E'],
    "C": ['C'],
    "G": ['G']
}

gm6 = {
    "non-polar aliphatic": ['A', 'I', 'L', 'M', 'P', 'V'],
    "non-polar aromatic": ['F', 'W', 'Y'],
    "polar": ['N', 'Q', 'S', 'T'],
    "charged": ['D', 'E', 'H', 'K', 'R'],
    "C": ['C'],
    "G": ['G']
}

if __name__ == "__main__":

    test_year = 2026

    train_csv = f"data/all_time/H3N2/train.csv"
    val_csv = f"data/all_time/H3N2/val.csv"
    test_csv = f"data/all_time/H3N2/test.csv"



    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)
    df_test = pd.read_csv(test_csv)

    i = 1
    gs = [gm1, gm2, gm3, gm4, gm5, gm6]
    for gm in gs:
        X_train, Y_train = Calculate_X_Y_from_df(gm, df_train)
        X_test, Y_test = Calculate_X_Y_from_df(gm, df_test)

        model = LinearRegression()
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        print(f"Group: gm_{i} RMSE: {rmse:.4f}, R^2: {r2:.4f}, MAE: {mae:.4f}")

        

        if i == 4:
            result_path = 'time_result/liao.csv'
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            row = {"year": test_year, "RMSE": rmse, "MAE": mae, "R2": r2}
            df_row = pd.DataFrame([row])
            write_header = not os.path.exists(result_path) or os.stat(result_path).st_size == 0
            df_row.to_csv(result_path, mode='a', index=False, header=write_header)
        
        i += 1


# 1963-2002
# Group: gm_1 RMSE: 1.6251, R^2: 0.9164, MAE: 1.2319
# Group: gm_2 RMSE: 1.6043, R^2: 0.9185, MAE: 1.2186
# Group: gm_3 RMSE: 1.5876, R^2: 0.9202, MAE: 1.2115
# Group: gm_4 RMSE: 1.5623, R^2: 0.9227, MAE: 1.1986
# Group: gm_5 RMSE: 1.5728, R^2: 0.9217, MAE: 1.2016
# Group: gm_6 RMSE: 1.6293, R^2: 0.9159, MAE: 1.2362



# 2003-2025
# Group: gm_1 RMSE: 1.6788, R^2: 0.2798, MAE: 1.3301
# Group: gm_2 RMSE: 1.6778, R^2: 0.2807, MAE: 1.3293
# Group: gm_3 RMSE: 1.6746, R^2: 0.2834, MAE: 1.3257
# Group: gm_4 RMSE: 1.6749, R^2: 0.2831, MAE: 1.3259
# Group: gm_5 RMSE: 1.6668, R^2: 0.2900, MAE: 1.3196
# Group: gm_6 RMSE: 1.6689, R^2: 0.2883, MAE: 1.3224
