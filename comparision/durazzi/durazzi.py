# 预训练模型ProtBERT的下载
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="Rostlab/prot_bert", local_dir="./prot_bert", local_dir_use_symlinks=False)
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations

# ProtBERT 序列嵌入函数
def get_protbert_embedding(seq):
    seq = ' '.join(list(seq))
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
    tokens = tokenizer(seq, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[0][0].numpy()

# 划分训练/测试病毒索引
def split_indices(n, test_size=0.2):
    indices = list(range(n))
    random.seed(random_state)
    random.shuffle(indices)
    test_count = int(n * test_size)
    return indices[test_count:], indices[:test_count]

# 构造输入X和目标Y
def Calculate_X_Y_by_index(indices, virus_names, seqs, coordinates_path):
    coord_df = pd.read_csv(coordinates_path)
    coord_map = {row['short_name']: (row['MDS1'], row['MDS2']) for _, row in coord_df.iterrows()}
    
    X, Y = [], []
    for i in tqdm(indices, desc="Generating embeddings"):
        name = virus_names[i]
        if name not in coord_map:
            continue
        emb = get_protbert_embedding(seqs[i])
        X.append(emb)
        Y.append(coord_map[name])
    return np.array(X), np.array(Y)

def compare_pairwise_distance(Y_pred, test_indices, virus_names, distance_matrix_path):
    distance_df = pd.read_csv(distance_matrix_path, index_col=0)
    virus_list = distance_df.columns.tolist()
    distance_matrix = distance_df.values

    test_viruses = [virus_names[i] for i in test_indices]
    true_dists, pred_dists = [], []

    for i, j in combinations(range(len(test_viruses)), 2):
        v1, v2 = test_viruses[i], test_viruses[j]
        if v1 in virus_list and v2 in virus_list:
            idx1, idx2 = virus_list.index(v1), virus_list.index(v2)
            d_true = distance_matrix[idx1, idx2]
            d_pred = np.linalg.norm(Y_pred[i] - Y_pred[j])
            true_dists.append(d_true)
            pred_dists.append(d_pred)
    
    rmse = np.sqrt(mean_squared_error(true_dists, pred_dists))
    r2 = r2_score(true_dists, pred_dists)
    mae = mean_absolute_error(true_dists, pred_dists)

    print(f"Distance RMSE: {rmse:.4f}, R^2: {r2:.4f}, MAE: {mae:.4f}")



if __name__ == "__main__":
    random_state = 42
    data_set = "1963-2002" # "1963-2002" or "2003-2025"
    if data_set == "1963-2002":
        sequences_path = "data/prd/1963-2002/sequences.csv"
        coordinates_path = "data/prd/1963-2002/antigen_coordinates.csv"
        distance_matrix_path = "data/prd/1963-2002/distance_matrix.csv"
    else:
        sequences_path = "data/prd/2003-2025/final_sequences.csv"
        coordinates_path = "data/prd/2003-2025/2003_2025_antigen_coordinates.csv"
        distance_matrix_path = "data/prd/2003-2025/2003_2025_distance_matrix.csv"

    random.seed(random_state)

    tokenizer = BertTokenizer.from_pretrained("comparision/durazzi/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("comparision/durazzi/prot_bert")
    model.eval()

    seq_df = pd.read_csv(sequences_path)
    virus_names = seq_df['short_name'].tolist()
    seqs = seq_df['HA1_sequence'].tolist()

    train_idx, test_idx = split_indices(len(virus_names), test_size=0.2)

    X_train, Y_train = Calculate_X_Y_by_index(train_idx, virus_names, seqs, coordinates_path)
    X_test, Y_test = Calculate_X_Y_by_index(test_idx, virus_names, seqs, coordinates_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, Y_train)

    Y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    print(f"Coordinates RMSE: {rmse:.4f}, R^2: {r2:.4f}, MAE: {mae:.4f}")

    compare_pairwise_distance(Y_pred, test_idx, virus_names, distance_matrix_path)



# 1963-2002
# Coordinates RMSE: 1.6011, R^2: 0.8809, MAE: 1.1138
# Distance RMSE: 2.1924, R^2: 0.8689, MAE: 1.6807

# 2003-2025
# Coordinates RMSE: 1.7277, R^2: 0.2096, MAE: 1.2838
# Distance RMSE: 2.1760, R^2: -0.2107, MAE: 1.6601