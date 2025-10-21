import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
import copy

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
    def __init__(self, X, y, y_scaler=None):
        self.X = X
        self.y = y
        self.y_scaler = y_scaler

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
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


def build_X_Y_from_time_series(df: pd.DataFrame, prop_matrix: np.ndarray):
    """
    将包含列 S1, S2, distance 的 DataFrame 转换为 (X, Y)。
    - X: (N, L, P) 其中 L 为序列长度，P 为属性矩阵数量
    - Y: (N,)
    属性矩阵为 (P, 20, 20)，按位取 mat[aa_i, aa_j]。
    """
    df = df.reset_index(drop=True)
    n_props = prop_matrix.shape[0]
    L = len(str(df.iloc[0]['S1']))

    X = np.zeros((len(df), L, n_props), dtype=np.float32)
    Y = df['distance'].astype(float).values.astype(np.float32)

    for idx, row in df.iterrows():
        s1 = str(row['S1'])
        s2 = str(row['S2'])
        si = np.fromiter((aa_to_idx.get(c, -1) for c in s1), dtype=np.int32)
        sj = np.fromiter((aa_to_idx.get(c, -1) for c in s2), dtype=np.int32)
        valid = (si >= 0) & (sj >= 0)
        scores = np.zeros((L, n_props), dtype=np.float32)
        if valid.any():
            for m in range(n_props):
                mat = prop_matrix[m]
                scores[valid, m] = mat[si[valid], sj[valid]]
        X[idx] = scores

    return X, Y



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
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
    model.load_state_dict(best_state)
    return model

def main(config, X_train, y_train, X_test, y_test, prop_names, X_val=None, y_val=None):
    y_scaler = None
    if config["standardize_y"]:
        y_scaler = {'mean': y_train.mean(), 'std': y_train.std()}

    if X_val is None or y_val is None:
        raise ValueError("Validation data (X_val, y_val) is required. No random split fallback.")

    train_ds = PairDataset(X_train, y_train, y_scaler)
    val_ds   = PairDataset(X_val,   y_val,   y_scaler)
    test_ds  = PairDataset(X_test,  y_test,  y_scaler)

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=8, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=config["batch_size"], num_workers=8, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WeightedMultiHeadAttentionMLP(
        seq_len=X_train.shape[1],
        n_props=X_train.shape[2],
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

    print(f"\n=== {test_year}: Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k:10}: {v:.4f}")

    scores = rank_aaindex_matrices_by_attention(model, prop_names)
    print(f"\nTop-{config['top_n']} property importances:")
    for name, score in scores[:config['top_n']]:
        print(f"{name:15}: {score:.4f}")
    
    return model, scores, prop_names, test_metrics

def slice_features(X, Y, indices):
    return X[:, :, indices], Y


def auto_select_and_retrain(config):
    print("=== Step 0: Load full data once (time_series CSV: train/val/test) ===")
    # 直接从 data/time_series/{train,val,test}.csv 读取 S1,S2,distance
    df_train = pd.read_csv(config["train_csv"])  # 需包含列: S1, S2, distance
    df_val   = pd.read_csv(config["val_csv"])    # 需包含列: S1, S2, distance

    if not df_val.empty:
        rs = config.get("random_state", None)
        keep_val = df_val.sample(frac=0.2, random_state=rs)
        move_to_train = df_val.drop(keep_val.index)
        if not move_to_train.empty:
            df_train = pd.concat([df_train, move_to_train], ignore_index=True)
        df_val = keep_val.reset_index(drop=True)

    df_test  = pd.read_csv(config["test_csv"])   # 需包含列: S1, S2, distance

    prop_matrix, prop_names = load_aaindex_tensor(config)
    X_train, y_train = build_X_Y_from_time_series(df_train, prop_matrix)
    X_val,   y_val   = build_X_Y_from_time_series(df_val,   prop_matrix)
    X_test,  y_test  = build_X_Y_from_time_series(df_test,  prop_matrix)

    print("=== Step 1: Full property training ===")
    model, scores, _, test_metrics = main(
        config,
        X_train, y_train,
        X_test,  y_test,
        prop_names,
        X_val=X_val, y_val=y_val
    )
    scores_wo_diff = [(name, s) for name, s in scores]
    top_n_props = [name for name, _ in scores_wo_diff[:config["top_n"]]]
    print(f"\nTop-{config['top_n']} selected AAIndex properties:")
    for name in top_n_props:
        print(f" - {name}")

    print("\n=== Step 2: Filter features and retrain ===")
    prop_indices = [prop_names.index(name) for name in top_n_props]



    X_train_topn, y_train_topn = slice_features(X_train, y_train, prop_indices)
    X_val_topn,   y_val_topn   = slice_features(X_val,   y_val,   prop_indices)
    X_test_topn,  y_test_topn  = slice_features(X_test,  y_test,  prop_indices)

    config_topn = config.copy()
    config_topn["n_heads"] = config_topn["n_retrain_heads"]

    model_topn, scores_topn, _, test_metrics_topn = main(
        config_topn,
        X_train_topn, y_train_topn,
        X_test_topn,  y_test_topn,
        top_n_props,
        X_val=X_val_topn, y_val=y_val_topn
    )

    return model_topn, scores_topn, top_n_props, test_metrics_topn


if __name__ == "__main__":


    for test_year in range(2022,2025):
    
        config = {
            # 文件路径
            "json_path": "data/prd/aaindex2_dicts.json",
            "train_csv": f"data/time_series/{test_year}/train.csv",
            "val_csv":   f"data/time_series/{test_year}/val.csv",
            "test_csv":  f"data/time_series/{test_year}/test.csv",
            "out_path":  f"fluAttn/prop/{test_year}/",

            # 模型与训练参数
            "batch_size": 256,
            "n_heads": 4,
            "n_retrain_heads": 1,
            "epochs": 200,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 15,

            # 数据处理参数
            "standardize_y": True,
            "corr_threshold": 0.8,
            "random_state": 42,

            # Top-N 属性选择
            "top_n": 5
        }

        torch.manual_seed(config["random_state"])
        np.random.seed(config["random_state"])
        random.seed(config["random_state"])

        model_topn, scores_topn, top_n_props, test_metrics_topn = auto_select_and_retrain(config)

        os.makedirs(os.path.dirname(config["out_path"]), exist_ok=True)
        scores_df = pd.DataFrame(scores_topn, columns=['prop_name', 'weight'])
        scores_df.to_csv(config["out_path"] + 'prop2.csv', index=False)

        result_path = os.path.join('time_result', 'find2.csv')
        result_row = {
            'year': test_year,
            **test_metrics_topn
        }
        result_df = pd.DataFrame([result_row], columns=['year', 'RMSE', 'MAE', 'R2', 'Pearson_r'])
        write_header = not os.path.exists(result_path)
        result_df.to_csv(result_path, mode='a', header=write_header, index=False)
