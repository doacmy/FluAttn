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
from itertools import combinations
from itertools import product
import pandas as pd
import os
import copy

aa_list = list("ARNDCQEGHILKMFPSTWYV")
aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}


def extract_average_attention_weights(attn_weights, head_weights, prop_names):
    """
    计算最终的属性权重，结合多头注意力和融合权重，并返回排序后的属性权重。
    """
    # 对每个注意力头的属性权重进行 softmax 归一化
    softmax_attn_weights = F.softmax(attn_weights, dim=-1)  # (n_heads, n_props)

    # 对多头融合权重进行 softmax 归一化
    fusion_weights = F.softmax(head_weights, dim=0)  # (n_heads,)

    # 计算最终的属性权重
    final_weights = torch.sum(softmax_attn_weights * fusion_weights.view(-1, 1), dim=0)  # (n_props,)

    # 转换为 CPU 并排序
    final_weights = final_weights.detach().cpu().numpy()
    scores = list(zip(prop_names, final_weights.tolist()))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores_sorted


def remove_highly_correlated_props(raw_props, threshold=0.95):
    prop_names = list(raw_props.keys())
    aa_list = list(next(iter(raw_props.values())).keys())
    
    # 构建属性矩阵 (n_props, 20)
    prop_matrix = np.array([
        [raw_props[prop][aa] for aa in aa_list] for prop in prop_names
    ])
    
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(prop_matrix)
    keep = []
    removed = set()
    
    for i in range(len(prop_names)):
        if prop_names[i] in removed:
            continue
        keep.append(prop_names[i])
        for j in range(i + 1, len(prop_names)):
            if abs(corr_matrix[i, j]) > threshold:
                removed.add(prop_names[j])
    
    filtered_props = {k: raw_props[k] for k in keep}
    return filtered_props

def load_standardized_aaindex_props(json_path, corr_threshold=0.95):
    with open(json_path, "r") as f:
        raw_props = json.load(f)

    filtered_props = remove_highly_correlated_props(raw_props, threshold=corr_threshold)
    props = [standardize_property_dict(d) for d in filtered_props.values()]
    prop_names = list(filtered_props.keys())
    return props, prop_names

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


def standardize_property_dict(prop_dict):
    vals = np.array(list(prop_dict.values()))
    mean, std = vals.mean(), vals.std()
    return {aa: (v - mean) / std for aa, v in prop_dict.items()}


def build_X_Y_from_time_series(df: pd.DataFrame, prop_matrix: np.ndarray):
    """
    将包含列 S1, S2, distance 的 DataFrame 转换为 (X, Y)。
    - X: (N, L, P) 其中 L 为序列长度，P 为属性数量（AAIndex1 归一化后的属性向量）
    - Y: (N,)
    对于无效氨基酸（不在 20 标准氨基酸中）的位置，特征置 0。
    """
    df = df.reset_index(drop=True)
    n_props = prop_matrix.shape[0]
    # 使用首行的 S1 长度作为 L（通常 HA1 序列长度一致）
    L = len(str(df.iloc[0]['S1'])) if len(df) > 0 else 0

    X = np.zeros((len(df), L, n_props), dtype=np.float32)
    Y = df['distance'].astype(float).values.astype(np.float32) if 'distance' in df.columns else np.zeros((len(df),), dtype=np.float32)

    for idx, row in df.iterrows():
        s1 = str(row['S1'])
        s2 = str(row['S2'])
        # 若长度不一致，使用较短长度以避免越界
        cur_L = min(len(s1), len(s2), L)
        si = np.fromiter((aa_to_idx.get(c, -1) for c in s1[:cur_L]), dtype=np.int32)
        sj = np.fromiter((aa_to_idx.get(c, -1) for c in s2[:cur_L]), dtype=np.int32)

        # 有效位掩码
        valid = (si >= 0) & (sj >= 0)
        if cur_L < L:
            # 对齐到 L 的长度
            pad_len = L - cur_L
            si = np.pad(si, (0, pad_len), constant_values=-1)
            sj = np.pad(sj, (0, pad_len), constant_values=-1)
            valid = (si >= 0) & (sj >= 0)

        # 通过属性向量计算 |prop(aa_i) - prop(aa_j)|
        si_clip = si.clip(min=0)
        sj_clip = sj.clip(min=0)
        pi = prop_matrix[:, si_clip]  # (P, L)
        pj = prop_matrix[:, sj_clip]  # (P, L)
        prop_diff = np.abs(pi - pj)   # (P, L)
        prop_diff[:, ~valid] = 0
        X[idx] = prop_diff.T          # (L, P)

    return X, Y

def build_prop_matrix(props):
    """
    将属性列表（每个为 {aa: value} 的字典）转换为 (P, 20) 的矩阵。
    """
    prop_matrix = np.zeros((len(props), 20), dtype=np.float32)
    for p, prop in enumerate(props):
        for aa, idx in aa_to_idx.items():
            prop_matrix[p, idx] = prop.get(aa, 0.0)
    return prop_matrix


def get_time_series_datasets(train_csv, val_csv, test_csv, json_path, corr_threshold):
    """
    从 time_series CSV 读取数据，计算基于 AAIndex1 的按位属性差，返回 (X_train, y_train, X_val, y_val, X_test, y_test, prop_names, props)。
    CSV 需包含列：S1, S2, distance。
    """
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)

    if not df_val.empty:
        rs = config.get("random_state", None)
        keep_val = df_val.sample(frac=0.2, random_state=rs)
        move_to_train = df_val.drop(keep_val.index)
        if not move_to_train.empty:
            df_train = pd.concat([df_train, move_to_train], ignore_index=True)
        df_val = keep_val.reset_index(drop=True)


    df_test  = pd.read_csv(test_csv)

    props, prop_names = load_standardized_aaindex_props(json_path, corr_threshold=corr_threshold)
    prop_matrix = build_prop_matrix(props)

    X_train, y_train = build_X_Y_from_time_series(df_train, prop_matrix)
    X_val,   y_val   = build_X_Y_from_time_series(df_val,   prop_matrix)
    X_test,  y_test  = build_X_Y_from_time_series(df_test,  prop_matrix)

    return X_train, y_train, X_val, y_val, X_test, y_test, prop_names, props


def get_test_data_from_time_series(test_csv, json_path, corr_threshold):
    """
    保留兼容接口：从 test.csv 直接读取并构建 (X_test, y_test)。
    """
    df_test = pd.read_csv(test_csv)
    props, _ = load_standardized_aaindex_props(json_path, corr_threshold=corr_threshold)
    prop_matrix = build_prop_matrix(props)
    X_test, y_test = build_X_Y_from_time_series(df_test, prop_matrix)
    return X_test, y_test


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

    train_ds = PairDataset(X_train, y_train, y_scaler)
    test_ds  = PairDataset(X_test, y_test, y_scaler)

    if X_val is not None and y_val is not None:
        val_ds = PairDataset(X_val, y_val, y_scaler)
    else:
        # 保持向后兼容：若未提供验证集，则从训练集中随机划分
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

    print("\n=== Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k:10}: {v:.4f}")

    scores = extract_average_attention_weights(model.attn_weights, model.head_weights, prop_names)
    print(f"\nTop-{config['top_n']} property importances:")
    for name, score in scores[:config['top_n']]:
        print(f"{name:15}: {score:.4f}")
    
    return model, scores, prop_names, test_metrics

def slice_features(X, Y, indices):
    return X[:, :, indices], Y


def auto_select_and_retrain(config):
    print("=== Step 0: Load time_series CSVs (train/val/test) ===")
    X_train, y_train, X_val, y_val, X_test, y_test, prop_names, props = get_time_series_datasets(
        config["train_csv"],
        config["val_csv"],
        config["test_csv"],
        config["json_path"],
        config["corr_threshold"]
    )

    print("=== Step 1: Full property training ===")
    model, scores, _, test_metrics = main(
        config, X_train, y_train, X_test, y_test, prop_names, X_val=X_val, y_val=y_val
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

    test_year = 2026
    config = {
        # 文件路径
        "json_path": "data/prd/aaindex1_dicts.json",
        "train_csv": f"data/all_time/H3N2/train.csv",
        "val_csv":   f"data/all_time/H3N2/val.csv",
        "test_csv":  f"data/all_time/H3N2/test.csv",
        "out_path":  f"fluAttn/prop/{test_year}/",

        # 模型与训练参数
        "batch_size": 256,
        "n_heads": 4,
        "n_retrain_heads": 2,
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
    scores_df.to_csv(config["out_path"] + 'prop1.csv', index=False)

    result_path = os.path.join('time_result', 'find1.csv')
    result_row = {
        'year': test_year,
        **test_metrics_topn
    }
    result_df = pd.DataFrame([result_row], columns=['year', 'RMSE', 'MAE', 'R2', 'Pearson_r'])
    write_header = not os.path.exists(result_path)
    result_df.to_csv(result_path, mode='a', header=write_header, index=False)
