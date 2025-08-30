import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
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


def find_main_matrix(train_pairs, virus_names, seqs_var):
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
            
            X, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, matrix_path)

            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)

            rmse_list = []

            fold = 1
            for train_index, test_index in kf.split(X_imputed):
                X_train, X_test = X_imputed[train_index], X_imputed[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                model = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1]//4, random_state=random_state, n_jobs=-1)
                
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

def find_first_auxiliary_matrix(train_pairs, virus_names, seqs_var, main_matrix_name):
    main_matrix_path = os.path.join(matrix_dir, main_matrix_name)
    rec_df = pd.read_csv(aux_0_record_path)
    X_main, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, main_matrix_path)

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
            
            X_aux, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, aux_matrix_path)

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

                model = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1]//4, random_state=random_state, n_jobs=-1)
                
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

def find_second_auxiliary_matrix(train_pairs, virus_names, seqs_var, main_matrix_name, first_aux_name):
    imputer = SimpleImputer(strategy="mean")
    main_matrix_path = os.path.join(matrix_dir, main_matrix_name)
    first_aux_matrix = os.path.join(matrix_dir, first_aux_name)
    rec_df = pd.read_csv(aux_1_record_path)
    X_main, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, main_matrix_path)
    X_main = imputer.fit_transform(X_main)
    X_aux_0, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, first_aux_matrix)
    X_aux_0 = imputer.fit_transform(X_aux_0)

    first_residual_X_aux = calculate_residual_matrix(X_main, X_aux_0)
    X_final = np.concatenate((X_main, first_residual_X_aux), axis=1)

    min_rmse = float('inf')
    best_matrix = None
    best_model = None
    for filename in os.listdir(matrix_dir):
        if filename.endswith('.csv'):
            if filename == main_matrix_name:
                continue
            if ((rec_df["main_matrix_name"] == main_matrix_name) & (rec_df["aux_0_matrix_name"] == first_aux_name) & (rec_df["aux_1_matrix_name"] == filename)).any():
                continue   
            print(f"AUX Matrix: {filename}")

            aux_1_matrix_path = os.path.join(matrix_dir, filename)
            
            X_aux_1, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, aux_1_matrix_path)
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

                model = RandomForestRegressor(n_estimators=500, max_features=X_train.shape[1]//4, random_state=random_state, n_jobs=-1)
                
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

def get_final_model(main_matrix_name, first_aux_name, second_aux_name, train_pairs, test_pairs, virus_names, seqs_var):

    imputer = SimpleImputer(strategy="mean")

    main_matrix_path = os.path.join(matrix_dir, main_matrix_name)
    first_aux_path = os.path.join(matrix_dir, first_aux_name)      
    second_aux_path = os.path.join(matrix_dir, second_aux_name)  

    X, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, main_matrix_path)
    X_aux_0, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, first_aux_path)
    X_aux_1, Y = Calculate_X_Y(train_pairs, virus_names, seqs_var, second_aux_path)

    X = imputer.fit_transform(X)
    X_aux_0 = imputer.fit_transform(X_aux_0)
    X_aux_1 = imputer.fit_transform(X_aux_1)

    residual_X_aux_0 = calculate_residual_matrix(X, X_aux_0)
    X_final = np.concatenate((X, residual_X_aux_0), axis=1)

    residual_X_aux_1 = calculate_residual_matrix(X, X_aux_1)
    X_final = np.concatenate((X_final, residual_X_aux_1), axis=1)

    model = RandomForestRegressor(n_estimators=500, max_features=X_final.shape[1]//4, random_state=random_state, n_jobs=-1)
    model.fit(X_final, Y)

    # 计算前85个重要性评分最高的特征
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

    top_seqs_var = seqs_var[:, top85_features]

    X, Y = Calculate_X_Y(train_pairs, virus_names, top_seqs_var, main_matrix_path)
    X_aux_0, Y = Calculate_X_Y(train_pairs, virus_names, top_seqs_var, first_aux_path)
    X_aux_1, Y = Calculate_X_Y(train_pairs, virus_names, top_seqs_var, second_aux_path)

    X = imputer.fit_transform(X)
    X_aux_0 = imputer.fit_transform(X_aux_0)
    X_aux_1 = imputer.fit_transform(X_aux_1)

    residual_X_aux_0 = calculate_residual_matrix(X, X_aux_0)
    X_final = np.concatenate((X, residual_X_aux_0), axis=1)

    residual_X_aux_1 = calculate_residual_matrix(X, X_aux_1)
    X_final = np.concatenate((X_final, residual_X_aux_1), axis=1)

    model = RandomForestRegressor(n_estimators=500, max_features=X_final.shape[1]//4, random_state=random_state, n_jobs=-1)
    model.fit(X_final, Y)

    X, Y = Calculate_X_Y(test_pairs, virus_names, top_seqs_var, main_matrix_path)
    X_aux_0, Y = Calculate_X_Y(test_pairs, virus_names, top_seqs_var, first_aux_path)
    X_aux_1, Y = Calculate_X_Y(test_pairs, virus_names, top_seqs_var, second_aux_path)

    X = imputer.fit_transform(X)
    X_aux_0 = imputer.fit_transform(X_aux_0)
    X_aux_1 = imputer.fit_transform(X_aux_1)

    residual_X_aux_0 = calculate_residual_matrix(X, X_aux_0)
    X_final = np.concatenate((X, residual_X_aux_0), axis=1)

    residual_X_aux_1 = calculate_residual_matrix(X, X_aux_1)
    X_final = np.concatenate((X_final, residual_X_aux_1), axis=1)


    Y_pred = model.predict(X_final)

    rmse = np.sqrt(mean_squared_error(Y, Y_pred))
    r2 = r2_score(Y, Y_pred)
    mae = mean_absolute_error(Y, Y_pred)

    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")


    




if __name__ == "__main__":
    data_set = "2003-2025"

    random_state=42
    imputer = SimpleImputer(strategy="mean")

    if data_set == "1963-2002":
        sequences_path = 'data/prd/1963-2002/sequences.csv'
        distance_matrix_path = 'data/prd/1963-2002/distance_matrix.csv'
    else:
        sequences_path = "data/prd/2003-2025/final_sequences.csv"
        distance_matrix_path = "data/prd/2003-2025/2003_2025_distance_matrix.csv"

    matrix_dir = 'data/prd/AAIndex/'
    main_record_path = f"comparision/yao/record/{data_set}/main_record.csv"
    aux_0_record_path = f"comparision/yao/record/{data_set}/aux_0_record.csv"
    aux_1_record_path = f"comparision/yao/record/{data_set}/aux_1_record.csv"


    if not Path(main_record_path).exists():
        df = pd.DataFrame(columns=['matrix_name','rmse'])
        df.to_csv(Path(main_record_path), index=False)

    if not Path(aux_0_record_path).exists():
        df = pd.DataFrame(columns=['main_matrix_name','aux_matrix_name','rmse'])
        df.to_csv(Path(aux_0_record_path), index=False)

    if not Path(aux_1_record_path).exists():
        df = pd.DataFrame(columns=['main_matrix_name','aux_0_matrix_name','aux_1_matrix_name','rmse'])
        df.to_csv(Path(aux_1_record_path), index=False)

    virus_names, seqs_var = extract_variable_sites(sequences_path)
    train_pairs, test_pairs = split_data(len(virus_names), test_size=0.2)

    # Step 1: 寻找主矩阵
    find_main_matrix(train_pairs, virus_names, seqs_var)

    # Step 2: 寻找第一辅助矩阵
    df = pd.read_csv(main_record_path)
    top15 = df.sort_values(by="rmse").head(15)
    matrix_names = top15['matrix_name'].tolist()

    for matrix in matrix_names:
        print(f"Processing matrix: {matrix}")
        find_first_auxiliary_matrix(train_pairs, virus_names, seqs_var, matrix)

    # Step 3: 寻找第二辅助矩阵
    df = pd.read_csv(aux_0_record_path)
    top = df.sort_values(by="rmse").head(1)
    main_matrix_name = top['main_matrix_name'].values[0]
    first_aux_matrix = top['aux_matrix_name'].values[0]

    find_second_auxiliary_matrix(train_pairs, virus_names, seqs_var, main_matrix_name, first_aux_matrix)

    # Step 4: 获得最终模型
    df = pd.read_csv(aux_1_record_path)
    top = df.sort_values(by="rmse").head(1)
    main_matrix_name = top['main_matrix_name'].values[0]
    first_aux_name = top['aux_0_matrix_name'].values[0]
    second_aux_name = top['aux_1_matrix_name'].values[0]
    get_final_model(main_matrix_name, first_aux_name, second_aux_name, train_pairs, test_pairs, virus_names, seqs_var)


# 1963-2002
# RMSE: 1.1537, R2: 0.9578, MAE: 0.9007

# 2003-2025
# RMSE: 1.6553, R2: 0.5440, MAE: 1.2802