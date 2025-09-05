import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json

from itertools import product
import pandas as pd
import os

aa_list = list("ARNDCQEGHILKMFPSTWYV")
aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

def rank_aaindex_matrices_by_attention(model, matrix_names):
    softmax_attn_weights = F.softmax(model.attn_weights, dim=-1)  # (n_heads, n_props)
    fusion_weights = F.softmax(model.head_weights, dim=0)  # (n_heads,)
    final_weights = torch.sum(softmax_attn_weights * fusion_weights.view(-1, 1), dim=0)  # (n_props,)
    final_weights = final_weights.detach().cpu().numpy()
    scores = list(zip(matrix_names, final_weights.tolist()))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores_sorted


class PairDataset(Dataset):
    def __init__(self, X_path, y_path, shape, y_scaler=None):
        self.lazy = config["use_lazy_memmap"]
        self.shape = shape
        self.y_scaler = y_scaler

        if self.lazy:
            self.X = np.memmap(X_path, dtype=np.float32, mode='r', shape=shape)
            self.y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(shape[0],))
        else:
            self.X = np.load(X_path)
            self.y = np.load(y_path)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].copy() if self.lazy else self.X[idx]
        y = self.y[idx]
        if self.y_scaler:
            y = (y - self.y_scaler['mean']) / self.y_scaler['std']
        return torch.from_numpy(x).float(), torch.tensor(y).float()


class WeightedMultiHeadAttentionMLP(nn.Module):
    def __init__(self, seq_len, n_props, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.n_props = n_props

        # 每个注意力头共享属性权重
        self.attn_weights = nn.Parameter(torch.randn(n_heads, n_props))  # (n_heads, n_props)

        # 多头融合权重
        self.head_weights = nn.Parameter(torch.randn(n_heads))

        # 输出MLP网络
        self.net = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B = x.size(0)
        head_outputs = []

        for i in range(self.n_heads):
            attn = F.softmax(self.attn_weights[i], dim=-1)  # (P,)
            attn = attn.unsqueeze(0).expand(self.seq_len, -1)  # 广播到 (L, P)
            x_head = x * attn.unsqueeze(0)  # (B, L, P)
            x_head = x_head.sum(dim=-1)  # (B, L)
            head_outputs.append(x_head)

        heads_stack = torch.stack(head_outputs, dim=0).permute(1, 0, 2)  # (B, H, L)
        fusion_weights = F.softmax(self.head_weights, dim=0)  # (H,)
        x_fused = torch.sum(heads_stack * fusion_weights.view(1, -1, 1), dim=1)  # (B, L)
        return self.net(x_fused).squeeze(-1)


def load_aaindex_tensor(config):
    with open(config["json_path"], "r") as f:
        json_dict = json.load(f)

    matrix_dict = {k: np.array(v) for k, v in json_dict.items()}

    matrix_list = []
    matrix_names = []
    for fname, matrix in matrix_dict.items():    
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val > min_val:
            matrix = matrix / (max_val - min_val)
        else:
            matrix = np.zeros_like(matrix)

        matrix_list.append(matrix)
        matrix_names.append(fname)

    prop_matrix = np.stack(matrix_list)
    return prop_matrix, matrix_names

def split_data(n, test_size=0.2):
    from itertools import combinations
    pairs = list(combinations(range(n), 2))
    random.shuffle(pairs)
    cut = int(len(pairs) * test_size)
    return pairs[cut:], pairs[:cut]

def build_X_Y_fast(pairs, seq_indices, dist_mat, prop_matrix):
    n_props = prop_matrix.shape[0]
    L = len(seq_indices[0])
    n_pairs = len(pairs)

    X = np.zeros((n_pairs, L, n_props), dtype=np.float32)
    Y = np.empty(n_pairs, dtype=np.float32)

    for k, (i, j) in enumerate(pairs):
        si = seq_indices[i]
        sj = seq_indices[j]

        valid_mask = (si >= 0) & (sj >= 0)
        scores = np.zeros((L, n_props), dtype=np.float32)

        for m in range(n_props):
            mat = prop_matrix[m]
            scores[valid_mask, m] = mat[si[valid_mask], sj[valid_mask]]

        X[k] = scores
        Y[k] = dist_mat[i, j]
    return X, Y


def build_large_X_Y_memmap(pairs, seq_indices, dist_mat, prop_matrix, prefix, batch_size=10000):
    n_pairs = len(pairs)
    L = len(seq_indices[0])
    n_props = len(prop_matrix)

    X_mem = np.memmap(f"{prefix}_X.npy", dtype=np.float32, mode='w+', shape=(n_pairs, L, n_props))
    Y_mem = np.memmap(f"{prefix}_Y.npy", dtype=np.float32, mode='w+', shape=(n_pairs,))

    for i in range(0, n_pairs, batch_size):
        print(f"Building batch {i}/{n_pairs}")
        sub_pairs = pairs[i:i+batch_size]
        X_sub, Y_sub = build_X_Y_fast(sub_pairs, seq_indices, dist_mat, prop_matrix)
        end = i + len(sub_pairs)
        X_mem[i:end] = X_sub
        Y_mem[i:end] = Y_sub

    X_mem.flush()
    Y_mem.flush()

    return f"{prefix}_X.npy", f"{prefix}_Y.npy", (n_pairs, L, n_props)

def get_data(seq_csv, dist_csv, json_path, test_size=0.2, selected_names=None):
    seq_df = pd.read_csv(seq_csv)
    dist_df = pd.read_csv(dist_csv, index_col=0)

    seq_df = seq_df[['short_name', 'HA1_sequence']].set_index('short_name')
    common = seq_df.index.intersection(dist_df.index)
    seq_df = seq_df.loc[common]
    dist_df = dist_df.loc[common, common]

    seqs_array = np.array(seq_df['HA1_sequence'].apply(list).to_list())
    variable_sites = [i for i in range(seqs_array.shape[1]) if len(set(seqs_array[:, i]) - {'-'}) > 1]
    seqs_var = seqs_array[:, variable_sites]
    seqs = ["".join(row) for row in seqs_var]
    dist_mat = dist_df.values

    prop_matrix, matrix_names = load_aaindex_tensor(config)

    n = len(seqs)
    train_p, test_p = split_data(n, test_size)

    seq_indices = np.full((len(seqs), len(seqs[0])), -1, dtype=np.int32)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            seq_indices[i, j] = aa_to_idx.get(aa, -1)

    if config.get("use_lazy_memmap", True):
        train_X_path, train_Y_path, train_shape = build_large_X_Y_memmap(train_p, seq_indices, dist_mat, prop_matrix, prefix="train")
        test_X_path, test_Y_path, test_shape = build_large_X_Y_memmap(test_p, seq_indices, dist_mat, prop_matrix, prefix="test")
    else:
        X_train, Y_train = build_X_Y_fast(train_p, seq_indices, dist_mat, prop_matrix)
        X_test, Y_test = build_X_Y_fast(test_p, seq_indices, dist_mat, prop_matrix)

        train_X_path, train_Y_path = "train_X.npy", "train_Y.npy"
        test_X_path, test_Y_path = "test_X.npy", "test_Y.npy"
        np.save(train_X_path, X_train)
        np.save(train_Y_path, Y_train)
        np.save(test_X_path, X_test)
        np.save(test_Y_path, Y_test)
        train_shape = X_train.shape
        test_shape = X_test.shape

    return train_X_path, train_Y_path, test_X_path, test_Y_path, matrix_names, prop_matrix, train_shape, test_shape



def evaluate(model, dl, device, y_scaler=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    if y_scaler:
        preds = preds * y_scaler['std'] + y_scaler['mean']
        trues = trues * y_scaler['std'] + y_scaler['mean']
    return {
        'RMSE': np.sqrt(mean_squared_error(trues, preds)),
        'MAE': mean_absolute_error(trues, preds),
        'R2': r2_score(trues, preds),
        'Pearson_r': pearsonr(trues, preds)[0]
    }

def train_loop(model, train_dl, val_dl, device, epochs=200, lr=1e-3, weight_decay=1e-4, patience=15, y_scaler=None):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val_rmse = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
        val_metrics = evaluate(model, val_dl, device, y_scaler)
        val_rmse = val_metrics['RMSE']
        scheduler.step(val_rmse)
        print(f"[Epoch {epoch:03d}] Val RMSE={val_rmse:.4f}, Pearson={val_metrics['Pearson_r']:.3f}")
        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
    model.load_state_dict(best_state)
    return model

def main(config, X_train_path, y_train_path, X_test_path, y_test_path, prop_names, train_shape, test_shape):
    y_scaler = None
    if config["standardize_y"]:

        if config["use_lazy_memmap"]:
            y_train_array = np.memmap(y_train_path, dtype=np.float32, mode='r')
        else:
            y_train_array = np.load(y_train_path)
        y_scaler = {'mean': y_train_array.mean(), 'std': y_train_array.std()}

    train_ds = PairDataset(X_train_path, y_train_path, train_shape, y_scaler)
    test_ds  = PairDataset(X_test_path, y_test_path, test_shape, y_scaler)

    val_size = max(1, int(0.1 * len(train_ds)))
    train_ds, val_ds = random_split(
        train_ds,
        [len(train_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(config["random_state"])
    )

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=8, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=config["batch_size"], num_workers=8, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WeightedMultiHeadAttentionMLP(
        seq_len=train_shape[1],
        n_props=train_shape[2],
        n_heads=config["n_heads"]
    ).to(device)

    model = train_loop(
        model, train_dl, val_dl, device,
        epochs=config["epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        patience=config["patience"],
        y_scaler=y_scaler
    )

    test_metrics = evaluate(model, test_dl, device, y_scaler)

    print("\n=== Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k:10}: {v:.4f}")

    scores = rank_aaindex_matrices_by_attention(model, prop_names)
    print(f"\nTop-{config['top_n']} property importances:")
    for name, score in scores[:config['top_n']]:
        print(f"{name:15}: {score:.4f}")
    
    return model, scores, prop_names, test_metrics

def slice_memmap_features(X_path, Y_path, shape, indices, prefix):
    n, L, P = shape
    new_P = len(indices)

    new_X_path = f"{prefix}_topn_X.npy"
    new_Y_path = Y_path

    if config["use_lazy_memmap"]:
        X_old = np.memmap(X_path, dtype=np.float32, mode='r', shape=(n, L, P))
        X_new = np.memmap(new_X_path, dtype=np.float32, mode='w+', shape=(n, L, new_P))
    else:
        X_old = np.load(X_path)
        X_new = np.empty((n, L, new_P), dtype=np.float32)

    for i in range(n):
        X_new[i] = X_old[i][:, indices]

    if config["use_lazy_memmap"]:
        X_new.flush()
    else:
        np.save(new_X_path, X_new)
    
    return new_X_path, new_Y_path, (n, L, new_P)


def auto_select_and_retrain(config):
    print("=== Step 0: Load full data once ===")
    X_train_path, y_train_path, X_test_path, y_test_path, prop_names, props, train_shape, test_shape = get_data(
        config["seq_path"],
        config["dist_path"],
        config["json_path"],
        test_size=config["test_size"],
        selected_names=None
    )

    print("=== Step 1: Full property training ===")
    model, scores, _, test_metrics = main(config, X_train_path, y_train_path, X_test_path, y_test_path, prop_names, train_shape, test_shape)
    scores_wo_diff = [(name, s) for name, s in scores]
    top_n_props = [name for name, _ in scores_wo_diff[:config["top_n"]]]
    print(f"\nTop-{config['top_n']} selected AAIndex properties:")
    for name in top_n_props:
        print(f" - {name}")

    print("\n=== Step 2: Filter features and retrain ===")
    prop_indices = [prop_names.index(name) for name in top_n_props]



    X_train_topn_path, y_train_topn_path, train_topn_shape = slice_memmap_features(
        X_train_path, y_train_path, train_shape, prop_indices, prefix="train"
    )
    X_test_topn_path, y_test_topn_path, test_topn_shape = slice_memmap_features(
        X_test_path, y_test_path, test_shape, prop_indices, prefix="test"
    )

    config_topn = config.copy()
    config_topn["n_heads"] = config_topn["n_retrain_heads"]

    model_topn, scores_topn, _, test_metrics_topn = main(
        config_topn,
        X_train_topn_path, y_train_topn_path,
        X_test_topn_path, y_test_topn_path,
        top_n_props,
        train_topn_shape, test_topn_shape
    )

    return model_topn, scores_topn, top_n_props, test_metrics_topn


if __name__ == "__main__":
    
    data_set = "2003-2025" # "1963-2002" or "2003-2025"

    config = {
        # 文件路径
        "json_path": "data/prd/aaindex2_dicts.json",

        # 是否启用懒加载 memmap
        "use_lazy_memmap": False,

        # 模型与训练参数
        "batch_size": 256,
        "n_heads": 4,
        "n_retrain_heads": 1,
        "epochs": 200,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,

        # 数据处理参数
        "test_size": 0.2,
        "standardize_y": True,
        "corr_threshold": 0.8,
        "random_state": 42,

        # Top-N 属性选择
        "top_n": 5
    }

    if data_set == "1963-2002":
        config["seq_path"] = "data/prd/1963-2002/sequences.csv"
        config["dist_path"] = "data/prd/1963-2002/distance_matrix.csv"
        config["out_path"] = "data/prd/1963-2002/prop2.csv"

    else:
        config["seq_path"] = "data/prd/2003-2025/final_sequences.csv"
        config["dist_path"] = "data/prd/2003-2025/2003_2025_distance_matrix.csv"
        config["out_path"] = "data/prd/2003-2025/prop2.csv"

    torch.manual_seed(config["random_state"])
    np.random.seed(config["random_state"])
    random.seed(config["random_state"])


    model_topn, scores_topn, top_n_props, test_metrics_topn = auto_select_and_retrain(config)

    scores_df = pd.DataFrame(scores_topn, columns=['prop_name', 'weight'])
    scores_df.to_csv(config["out_path"], index=False)




