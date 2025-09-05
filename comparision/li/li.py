import numpy as np
import random
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


# Kyte-Doolittle hydrophobicity scale (KYTJ820101)
hydrophobicity = {
    'A': 1.8,
    'R': -4.5,
    'N': -3.5,
    'D': -3.5,
    'C': 2.5,
    'Q': -3.5,
    'E': -3.5,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'L': 3.8,
    'K': -3.9,
    'M': 1.9,
    'F': 2.8,
    'P': -1.6,
    'S': -0.8,
    'T': -0.7,
    'W': -0.9,
    'Y': -1.3,
    'V': 4.2
}

# Bigelow Residue volume (BIGC670101)
volume = {
    'A': 52.6,
    'R': 109.1,
    'N': 75.7,
    'D': 68.4,
    'C': 68.3,
    'Q': 89.7,
    'E': 84.7,
    'G': 36.3,
    'H': 91.9,
    'I': 102.0,
    'L': 102.0,
    'K': 105.1,
    'M': 97.7,
    'F': 113.9,
    'P': 73.6,
    'S': 54.9,
    'T': 71.2,
    'W': 135.4,
    'Y': 116.2,
    'V': 85.1
}

charge = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Zimmerman Polarity (ZIMJ680103)
polarity = {
    'A': 0.00,
    'R': 52.00,
    'N': 3.38,
    'D': 49.70,
    'C': 1.48,
    'Q': 3.53,
    'E': 49.90,
    'G': 0.00,
    'H': 51.60,
    'I': 0.13,
    'L': 0.13,
    'K': 49.50,
    'M': 1.43,
    'F': 0.35,
    'P': 1.58,
    'S': 1.67,
    'T': 1.66,
    'W': 2.10,
    'Y': 1.61,
    'V': 0.13
}


# Substitutions near the receptor binding site determine major antigenic change during influenza virus evolution
crucial_positions = [145, 155, 156, 158, 159, 189, 193]

# Structural identification of the antibody-binding sites of Hong Kong influenza haemagglutinin and their involvement in antigenic variation.
epitope_sites = {
    "A": [133, 137, 140, 141, 142, 143, 144, 145, 146],
    "B": [155, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196],
    "C": [52, 53, 54, 275, 277, 278],
    "D": [201, 205, 207, 217, 220, 226, 242],
    "E": [122, 126]
}

# ========== 判断 N-糖基化位点 ==========
def has_glycosylation(seq):
    positions = []
    for i in range(len(seq)-2):
        if seq[i] == 'N' and seq[i+1] != 'P' and seq[i+2] in ['S','T']:
            positions.append(i)
    return positions


def split_data(n, test_size=0.2):
    pairs = list(combinations(range(n), 2))
    random.seed(random_state)
    random.shuffle(pairs)
    
    N = len(pairs)
    test_count = int(N * test_size)
    
    test_pairs = pairs[:test_count]
    train_pairs = pairs[test_count:]
    
    return train_pairs, test_pairs

def extract_variable_sites(sequences_path):
    sequences_df = pd.read_csv(sequences_path)
    seqs = sequences_df['HA1_sequence'].apply(list).to_list()
    seqs_var = np.array(seqs)

    virus_names = sequences_df['short_name'].tolist()
    return virus_names, seqs_var

def Calculate_X_Y(pairs, virus_names, seqs_var, distance_matrix_path):
    distance_df = pd.read_csv(distance_matrix_path, index_col=0)

    N = len(pairs)
    L = seqs_var.shape[1]
    Y = []

    X = np.zeros((N, L + 12))
    for idx, (i, j) in enumerate(pairs):
        seq1 = seqs_var[i]
        seq2 = seqs_var[j]

        ### ----- 1. 计算二进制突变向量 -----
        binary_diff = np.array([0 if aa1 == aa2 else 1 for aa1, aa2 in zip(seq1, seq2)])
        X[idx, :L] = binary_diff

        ### ----- 2. Feature 1: 替换数量 -----
        mutations = np.where(binary_diff == 1)[0]
        X[idx, L + 0] = len(mutations)

        ### ----- 3. Feature 2: N-糖基化位点差异 -----
        gly1 = set(has_glycosylation(seq1))
        gly2 = set(has_glycosylation(seq2))
        X[idx, L + 1] = 1 if gly1 != gly2 else 0

        ### ----- 4. Feature 3: 关键位点是否突变 -----
        key_mut = any(seq1[pos-1] != seq2[pos-1] for pos in crucial_positions if pos-1 < len(seq1))
        X[idx, L + 2] = 1 if key_mut else 0

        ### ----- 5. Feature 4-8: 表位 A-E 是否有突变 -----
        for epi_idx, epi in enumerate(["A", "B", "C", "D", "E"]):
            epi_sites = epitope_sites[epi]
            epi_mut = any(seq1[pos-1] != seq2[pos-1] for pos in epi_sites if pos-1 < len(seq1))
            X[idx, L + 3 + epi_idx] = 1 if epi_mut else 0

        ### ----- 6. Feature 9-12: 物理化学属性差异 -----
        for feat_idx, prop in enumerate([hydrophobicity, volume, charge, polarity]):
            diffs = []
            for k in mutations:
                aa1 = seq1[k]
                aa2 = seq2[k]
                val1 = prop.get(aa1, 0)
                val2 = prop.get(aa2, 0)
                diffs.append(abs(val1 - val2))

            if len(diffs) == 0:
                mean_diff = 0
            elif len(diffs) < 3:
                mean_diff = np.mean(diffs)
            else:
                mean_diff = np.mean(sorted(diffs, reverse=True)[:3])

            X[idx, L + 8 + feat_idx] = mean_diff

        ### ----- 7. Y: 抗原距离 -----
        virus_i = virus_names[i]
        virus_j = virus_names[j]
        dist = distance_df.loc[virus_i, virus_j]
        Y.append(dist)

    Y = np.array(Y)
    return X, Y



def train_xgboost_with_cv(X_train, Y_train):
    
    param_grid = {
        'max_depth': [6, 7, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [200, 300, 400],
        'gamma': [0.05, 0.1, 0.2]
    }

    model = xgb.XGBRegressor(
        booster='gbtree',
        objective='reg:squarederror',
        random_state=42,
        n_jobs=4
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=4
    )

    grid_search.fit(X_train, Y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def plot_true_vs_predicted_MDS(distance_matrix_path, virus_names, test_pairs, Y_pred, data_set):
    distance_df = pd.read_csv(distance_matrix_path, index_col=0)

    test_virus_indices = set([idx for pair in test_pairs for idx in pair])
    test_virus_names = [virus_names[idx] for idx in test_virus_indices]

    # ========= 1. 构造真实距离矩阵 =========
    distance_sub_df = distance_df.loc[test_virus_names, test_virus_names]
    true_matrix = distance_sub_df.values

    # ========= 2. 构造预测距离矩阵 =========
    pred_matrix = true_matrix.copy()

    # 创建名称到index映射
    name_to_idx = {name: i for i, name in enumerate(test_virus_names)}

    for (pair_idx, (i, j)) in enumerate(test_pairs):
        name_i = virus_names[i]
        name_j = virus_names[j]
        if name_i in name_to_idx and name_j in name_to_idx:
            idx_i = name_to_idx[name_i]
            idx_j = name_to_idx[name_j]
            pred_matrix[idx_i, idx_j] = Y_pred[pair_idx]
            pred_matrix[idx_j, idx_i] = Y_pred[pair_idx]

    # ========= 3. MDS 可视化 =========
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.suptitle("True vs Predicted MDS", fontsize=16)

    for ax, matrix, subtitle in zip(axes, [true_matrix, pred_matrix], ["True Distances", "Predicted Distances"]):
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(matrix)

        ax.scatter(coords[:, 0], coords[:, 1], c='skyblue', s=50, edgecolors='k')

        # 标注病毒名
        for idx, name in enumerate(test_virus_names):
            ax.text(coords[idx, 0]+0.05, coords[idx, 1]+0.05, name, fontsize=7)

        ax.set_title(subtitle)
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"comparison/li/plot/mds_plot_{data_set}.png")


if __name__ == "__main__":
    random_state = 42
    data_set = "2003-2025" # "1963-2002" or "2003-2025"
    if data_set == "1963-2002":
        sequences_path = "data/prd/1963-2002/sequences.csv"
        distance_matrix_path = "data/prd/1963-2002/distance_matrix.csv"
    else:
        sequences_path = "data/prd/2003-2025/final_sequences.csv"
        distance_matrix_path = "data/prd/2003-2025/2003_2025_distance_matrix.csv"

    virus_names, seqs_var = extract_variable_sites(sequences_path)
    train_pairs, test_pairs = split_data(len(virus_names), test_size=0.2)

    X_train, Y_train = Calculate_X_Y(train_pairs, virus_names, seqs_var, distance_matrix_path)
    X_test, Y_test = Calculate_X_Y(test_pairs, virus_names, seqs_var, distance_matrix_path)

    model = train_xgboost_with_cv(X_train, Y_train)
    Y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    print(f"RMSE: {rmse:.4f}, R^2: {r2:.4f}, MAE: {mae:.4f}")

    # plot_true_vs_predicted_MDS(distance_matrix_path, virus_names, test_pairs, Y_pred, data_set)

# 1963-2002
# Best parameters: {'gamma': 0.05, 'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 400}
# RMSE: 0.7768, R^2: 0.9809, MAE: 0.5720

# 2003-2025
# Best parameters: {'gamma': 0.1, 'learning_rate': 0.2, 'max_depth': 8, 'n_estimators': 400}
# RMSE: 1.0576, R^2: 0.7142, MAE: 0.7949

