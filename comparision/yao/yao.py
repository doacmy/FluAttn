import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_aaindex(matrix_path):
    matrix_df = pd.read_csv(matrix_path)

    matrix_df.set_index('Unnamed: 0', inplace=True)
    A = matrix_df.values
    aa_list = matrix_df.index.tolist()

    D = np.zeros_like(A)
    for i in range(len(aa_list)):
        for j in range(len(aa_list)):
            D[i, j] = A[i, i] + A[j, j] - 2 * A[i, j]

    aa_idx = {aa: idx for idx, aa in enumerate(aa_list)}
    return D, aa_idx


def extract_variable_sites(sequences_path):
    sequences_df = pd.read_csv(sequences_path).head(200)
    seqs = sequences_df['HA1_sequence'].apply(list).to_list()
    seqs_array = np.array(seqs)
    m_full = seqs_array.shape[1]

    variable_sites = []
    for col in range(m_full):
        residues = set(seqs_array[:, col])
        residues.discard('-')
        if len(residues) > 1:
            variable_sites.append(col)

    seqs_var = seqs_array[:, variable_sites]

    virus_names = sequences_df['short_name'].tolist()
    return virus_names, seqs_var


def split_data(n, test_size=0.2):
    pairs = list(combinations(range(n), 2))
    random.seed(random_state)
    random.shuffle(pairs)
    
    N = len(pairs)
    test_count = int(N * test_size)
    
    test_pairs = pairs[:test_count]
    train_pairs = pairs[test_count:]
    
    return train_pairs, test_pairs


def Calculate_X_Y(pairs, virus_names, seqs_var, matrix_path):
    D, aa_idx = load_aaindex(matrix_path)

    m = seqs_var.shape[1]

    N = len(pairs)
    X = np.zeros((N, m))

    for j in range(m):
        for idx, (k, l) in enumerate(pairs):
            aa_k = seqs_var[k, j]
            aa_l = seqs_var[l, j]
            try:
                dist = D[aa_idx[aa_k], aa_idx[aa_l]]
            except KeyError:
                dist = np.nan
            X[idx, j] = dist

    distance_df = pd.read_csv(distance_matrix_path, index_col=0)
    
    Y = []
    for i, j in pairs:
        virus_i = virus_names[i]
        virus_j = virus_names[j]
        dist = distance_df.loc[virus_i, virus_j]
        Y.append(dist)


    Y = np.array(Y)
    return X, Y

def Calculate_X_Y_from_df(df: pd.DataFrame, matrix_path: str, selected_sites: Optional[list] = None):
    """
    Build X and Y directly from a DataFrame with columns:
    - 'S1': sequence 1 (string)
    - 'S2': sequence 2 (string)
    - 'distance': antigenic distance (float)

    For each position j, feature is AAIndex-based distance between S1[j] and S2[j].
    Unknown residues (e.g., '-', 'X') are set to NaN and imputed later.

    If selected_sites is provided, only compute features for those 0-based positions.
    """
    required_cols = {"S1", "S2", "distance"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    D, aa_idx = load_aaindex(matrix_path)

    N = len(df)
    if N == 0:
        return np.zeros((0, 0)), np.array([])

    s1_0 = str(df.iloc[0]["S1"]).strip()
    s2_0 = str(df.iloc[0]["S2"]).strip()
    if len(s1_0) != len(s2_0):
        raise ValueError("S1 and S2 in the first row have different lengths")

    m_full = len(s1_0)
    if selected_sites is None:
        sites = list(range(m_full))
    else:
        sites = list(selected_sites)

    m = len(sites)
    X = np.zeros((N, m))
    Y = np.zeros(N)

    for idx, row in df.iterrows():
        s1 = str(row["S1"]).strip()
        s2 = str(row["S2"]).strip()
        if len(s1) != m_full or len(s2) != m_full:
            raise ValueError(
                f"Inconsistent sequence lengths at row {idx}: len(S1)={len(s1)}, len(S2)={len(s2)}, expected {m_full}"
            )
        for j_pos, col in enumerate(sites):
            aa_k = s1[col]
            aa_l = s2[col]
            try:
                dist = D[aa_idx[aa_k], aa_idx[aa_l]]
            except KeyError:
                dist = np.nan
            X[idx, j_pos] = dist

        Y[idx] = float(row["distance"])

    return X, Y

def calculate_residual_matrix(X_main, X_aux):
    N, m = X_main.shape
    residual_X_aux = np.zeros_like(X_aux)
    
    for j in range(m):
        x_main = X_main[:, j].reshape(-1, 1)
        x_aux = X_aux[:, j]
        
        reg = LinearRegression().fit(x_main, x_aux)
        x_aux_pred = reg.predict(x_main)
        
        residual = x_aux - x_aux_pred
        
        residual_X_aux[:, j] = residual
    
    return residual_X_aux


def find_main_matrix(df_train: pd.DataFrame):
    rec_df = pd.read_csv(main_record_path)
    matrix_list = rec_df['matrix_name'].tolist()

    min_rmse = float('inf')
    best_matrix = None
    best_model = None
    for filename in os.listdir(matrix_dir):
        if filename.endswith('.csv'):

            if filename in matrix_list:
                continue


            print(f"Matrix: {filename}")

            matrix_path = os.path.join(matrix_dir, filename)
            
            X, Y = Calculate_X_Y_from_df(df_train, matrix_path)

            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

            rmse_list = []

            fold = 1
            for train_index, test_index in kf.split(X_imputed):
                X_train, X_test = X_imputed[train_index], X_imputed[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                model = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1]//4, random_state=random_state, n_jobs=8)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_list.append(rmse)

                fold += 1
            rmse_mean = np.mean(rmse_list)
            if rmse_mean < min_rmse:
                min_rmse = rmse_mean
                best_matrix = filename
                best_model = model

            print(f"RMSE: {rmse_mean:.4f}")
            with open(main_record_path, 'a') as f:
                f.write(f"{filename},{rmse_mean:.4f}\n")

    print(f"Best matrix: {best_matrix} with RMSE: {min_rmse:.4f}")
    return best_matrix, min_rmse, best_model

def find_first_auxiliary_matrix(df_train: pd.DataFrame, main_matrix_name: str):
    main_matrix_path = os.path.join(matrix_dir, main_matrix_name)
    rec_df = pd.read_csv(aux_0_record_path)
    X_main, Y = Calculate_X_Y_from_df(df_train, main_matrix_path)

    min_rmse = float('inf')
    best_matrix = None
    best_model = None
    for filename in os.listdir(matrix_dir):
        if filename.endswith('.csv'):
            if filename == main_matrix_name:
                continue
            if ((rec_df["main_matrix_name"] == main_matrix_name) & (rec_df["aux_matrix_name"] == filename)).any():
                continue   
            print(f"AUX Matrix: {filename}")

            aux_matrix_path = os.path.join(matrix_dir, filename)
            
            X_aux, Y = Calculate_X_Y_from_df(df_train, aux_matrix_path)

            X_main = imputer.fit_transform(X_main)
            X_aux = imputer.fit_transform(X_aux)
            residual_X_aux = calculate_residual_matrix(X_main, X_aux)
            X_final = np.concatenate((X_main, residual_X_aux), axis=1)

            
            X_imputed = imputer.fit_transform(X_final)

            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

            rmse_list = []

            fold = 1
            for train_index, test_index in kf.split(X_imputed):
                X_train, X_test = X_imputed[train_index], X_imputed[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                model = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1]//4, random_state=random_state, n_jobs=8)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_list.append(rmse)

                fold += 1
            rmse_mean = np.mean(rmse_list)
            if rmse_mean < min_rmse:
                min_rmse = rmse_mean
                best_matrix = filename
                best_model = model

            print(f"First AUX RMSE: {rmse_mean:.4f}")
            with open(aux_0_record_path, 'a') as f:
                f.write(f"{main_matrix_name},{filename},{rmse_mean:.4f}\n")
    if best_matrix is None:
        return None, None, None
    print(f"Best first auxiliary matrix: {best_matrix} with RMSE: {min_rmse:.4f}")
    return best_matrix, min_rmse, best_model

def find_second_auxiliary_matrix(df_train: pd.DataFrame, main_matrix_name: str, first_aux_name: str):
    imputer = SimpleImputer(strategy="mean")
    main_matrix_path = os.path.join(matrix_dir, main_matrix_name)
    first_aux_matrix = os.path.join(matrix_dir, first_aux_name)
    rec_df = pd.read_csv(aux_1_record_path)
    X_main, Y = Calculate_X_Y_from_df(df_train, main_matrix_path)
    X_main = imputer.fit_transform(X_main)
    X_aux_0, Y = Calculate_X_Y_from_df(df_train, first_aux_matrix)
    X_aux_0 = imputer.fit_transform(X_aux_0)

    first_residual_X_aux = calculate_residual_matrix(X_main, X_aux_0)
    X_final = np.concatenate((X_main, first_residual_X_aux), axis=1)

    min_rmse = float('inf')
    best_matrix = None
    best_model = None
    for filename in os.listdir(matrix_dir):
        if filename.endswith('.csv'):
            if filename == main_matrix_name or filename == first_aux_name:
                continue
            if ((rec_df["main_matrix_name"] == main_matrix_name) & (rec_df["aux_0_matrix_name"] == first_aux_name) & (rec_df["aux_1_matrix_name"] == filename)).any():
                continue   
            print(f"AUX Matrix: {filename}")

            aux_1_matrix_path = os.path.join(matrix_dir, filename)
            
            X_aux_1, Y = Calculate_X_Y_from_df(df_train, aux_1_matrix_path)
            X_aux_1 = imputer.fit_transform(X_aux_1)

            second_residual_X_aux = calculate_residual_matrix(X_main, X_aux_1)
            second_residual_X_aux = imputer.fit_transform(second_residual_X_aux)
            
            X_final = np.concatenate((X_final, second_residual_X_aux), axis=1)
            X_imputed = imputer.fit_transform(X_final)

            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

            rmse_list = []

            fold = 1
            for train_index, test_index in kf.split(X_imputed):
                X_train, X_test = X_imputed[train_index], X_imputed[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                model = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1]//4, random_state=random_state, n_jobs=8)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_list.append(rmse)

                fold += 1
            rmse_mean = np.mean(rmse_list)
            if rmse_mean < min_rmse:
                min_rmse = rmse_mean
                best_matrix = filename
                best_model = model

            print(f"Second AUX RMSE: {rmse_mean:.4f}")
            with open(aux_1_record_path, 'a') as f:
                f.write(f"{main_matrix_name},{first_aux_name},{filename},{rmse_mean:.4f}\n")
    if best_matrix is None:
        return None, None, None
    print(f"Best second auxiliary matrix: {best_matrix} with RMSE: {min_rmse:.4f}")
    return best_matrix, min_rmse, best_model

def get_final_model(main_matrix_name: str, first_aux_name: str, second_aux_name: str, df_train: pd.DataFrame, df_test: pd.DataFrame):

    imputer = SimpleImputer(strategy="mean")

    main_matrix_path = os.path.join(matrix_dir, main_matrix_name)
    first_aux_path = os.path.join(matrix_dir, first_aux_name)      
    second_aux_path = os.path.join(matrix_dir, second_aux_name)  

    # Train on all sites first
    X, Y = Calculate_X_Y_from_df(df_train, main_matrix_path)
    X_aux_0, _ = Calculate_X_Y_from_df(df_train, first_aux_path)
    X_aux_1, _ = Calculate_X_Y_from_df(df_train, second_aux_path)

    X = imputer.fit_transform(X)
    X_aux_0 = imputer.fit_transform(X_aux_0)
    X_aux_1 = imputer.fit_transform(X_aux_1)

    residual_X_aux_0 = calculate_residual_matrix(X, X_aux_0)
    X_final = np.concatenate((X, residual_X_aux_0), axis=1)

    residual_X_aux_1 = calculate_residual_matrix(X, X_aux_1)
    X_final = np.concatenate((X_final, residual_X_aux_1), axis=1)

    model = RandomForestRegressor(n_estimators=500, max_features=X_final.shape[1]//4, random_state=random_state, n_jobs=8)
    model.fit(X_final, Y)

    # Select top 85 sites by averaging importances across main + two residuals (3 features per site)
    importances = model.feature_importances_

    m = X.shape[1]
    k = 3

    importances_reshaped = importances.reshape(m, k)
    site_importance = np.mean(importances_reshaped, axis=1)

    site_importance_df = pd.DataFrame({
        "site": range(0, m),
        "importance": site_importance
    }).sort_values(by="importance", ascending=False)

    top85_features = site_importance_df.head(85)['site'].tolist()

    # Retrain model on selected sites
    X, Y = Calculate_X_Y_from_df(df_train, main_matrix_path, selected_sites=top85_features)
    X_aux_0, _ = Calculate_X_Y_from_df(df_train, first_aux_path, selected_sites=top85_features)
    X_aux_1, _ = Calculate_X_Y_from_df(df_train, second_aux_path, selected_sites=top85_features)

    X = imputer.fit_transform(X)
    X_aux_0 = imputer.fit_transform(X_aux_0)
    X_aux_1 = imputer.fit_transform(X_aux_1)

    residual_X_aux_0 = calculate_residual_matrix(X, X_aux_0)
    X_final = np.concatenate((X, residual_X_aux_0), axis=1)

    residual_X_aux_1 = calculate_residual_matrix(X, X_aux_1)
    X_final = np.concatenate((X_final, residual_X_aux_1), axis=1)

    model = RandomForestRegressor(n_estimators=500, max_features=X_final.shape[1]//4, random_state=random_state, n_jobs=8)
    model.fit(X_final, Y)

    # Evaluate on test set
    X_test, Y_test = Calculate_X_Y_from_df(df_test, main_matrix_path, selected_sites=top85_features)
    X_test_aux_0, _ = Calculate_X_Y_from_df(df_test, first_aux_path, selected_sites=top85_features)
    X_test_aux_1, _ = Calculate_X_Y_from_df(df_test, second_aux_path, selected_sites=top85_features)

    X_test = imputer.fit_transform(X_test)
    X_test_aux_0 = imputer.fit_transform(X_test_aux_0)
    X_test_aux_1 = imputer.fit_transform(X_test_aux_1)

    residual_X_aux_0 = calculate_residual_matrix(X_test, X_test_aux_0)
    X_test_final = np.concatenate((X_test, residual_X_aux_0), axis=1)

    residual_X_aux_1 = calculate_residual_matrix(X_test, X_test_aux_1)
    X_test_final = np.concatenate((X_test_final, residual_X_aux_1), axis=1)

    Y_pred = model.predict(X_test_final)

    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)

    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")

    result_path = 'time_result/yao.csv'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    row = {"year": test_year, "RMSE": rmse, "MAE": mae, "R2": r2}
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(result_path) or os.stat(result_path).st_size == 0
    df_row.to_csv(result_path, mode='a', index=False, header=write_header)


    




if __name__ == "__main__":
    for test_year in range(2022,2025):
        random_state=42
        imputer = SimpleImputer(strategy="mean")

        train_csv = f"data/time_series/{test_year}/train.csv"
        val_csv = f"data/time_series/{test_year}/val.csv"
        test_csv = f"data/time_series/{test_year}/test.csv"

        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        df_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)
        df_test = pd.read_csv(test_csv)

        matrix_dir = 'data/prd/AAIndex/'

        os.makedirs(os.path.dirname(f"comparision/yao/{test_year}/"), exist_ok=True)

        main_record_path = f"comparision/yao/{test_year}/main_record.csv"
        aux_0_record_path = f"comparision/yao/{test_year}/aux_0_record.csv"
        aux_1_record_path = f"comparision/yao/{test_year}/aux_1_record.csv"

        if not Path(main_record_path).exists():
            df = pd.DataFrame(columns=['matrix_name','rmse'])
            df.to_csv(Path(main_record_path), index=False)

        if not Path(aux_0_record_path).exists():
            df = pd.DataFrame(columns=['main_matrix_name','aux_matrix_name','rmse'])
            df.to_csv(Path(aux_0_record_path), index=False)

        if not Path(aux_1_record_path).exists():
            df = pd.DataFrame(columns=['main_matrix_name','aux_0_matrix_name','aux_1_matrix_name','rmse'])
            df.to_csv(Path(aux_1_record_path), index=False)

        # Step 1: 寻找主矩阵（10 折交叉验证）
        find_main_matrix(df_train)

        # Step 2: 寻找第一辅助矩阵（基于 Step 1 的结果池）
        df = pd.read_csv(main_record_path)
        top15 = df.sort_values(by="rmse").head(15)
        matrix_names = top15['matrix_name'].tolist()

        for matrix in matrix_names:
            print(f"Processing matrix: {matrix}")
            find_first_auxiliary_matrix(df_train, matrix)

        # Step 3: 寻找第二辅助矩阵（从最佳 first aux 继续）
        df = pd.read_csv(aux_0_record_path)
        top = df.sort_values(by="rmse").head(1)
        main_matrix_name = top['main_matrix_name'].values[0]
        first_aux_matrix = top['aux_matrix_name'].values[0]

        find_second_auxiliary_matrix(df_train, main_matrix_name, first_aux_matrix)

        # Step 4: 获得最终模型并在测试集评估
        df = pd.read_csv(aux_1_record_path)
        top = df.sort_values(by="rmse").head(1)
        main_matrix_name = top['main_matrix_name'].values[0]
        first_aux_name = top['aux_0_matrix_name'].values[0]
        second_aux_name = top['aux_1_matrix_name'].values[0]
        get_final_model(main_matrix_name, first_aux_name, second_aux_name, df_train, df_test)

