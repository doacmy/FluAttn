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
import copy

from itertools import product
import pandas as pd
import os

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

def load_standardized_aaindex_props(json_path, corr_threshold=0.95, selected_names=None):
    with open(json_path, "r") as f:
        raw_props = json.load(f)

    if selected_names is not None:
        raw_props = {k: raw_props[k] for k in selected_names if k in raw_props}

    filtered_props = remove_highly_correlated_props(raw_props, threshold=corr_threshold)
    props = [standardize_property_dict(d) for d in filtered_props.values()]
    prop_names = list(filtered_props.keys())
    return props, prop_names


class SequencePairDataset(Dataset):
    """
    基于成对序列CSV的按需特征计算数据集（序列已对齐，均为标准氨基酸）。
    - df: 包含列 S1, S2, distance
    - prop_matrix: 形状 (P, 20) 的AAIndex属性矩阵，已标准化
    - seq_len: 序列长度（各样本一致）
    - prop_indices: 可选，只选择部分属性维度（Top-N 重训阶段）
    - y_scaler: 可选，用于y标准化的字典 {mean, std}
    """
    def __init__(self, df: pd.DataFrame, prop_matrix: np.ndarray, seq_len: int, y_scaler=None, prop_indices=None):
        self.df = df.reset_index(drop=True)
        self.prop_matrix = prop_matrix  # (P, 20)
        self.seq_len = seq_len
        self.y_scaler = y_scaler
        self.prop_indices = prop_indices

    def __len__(self):
        return len(self.df)

    def _encode_seq(self, s: str) -> np.ndarray:
        # 序列均为标准氨基酸且长度一致，直接映射
        return np.fromiter((aa_to_idx[c] for c in s), dtype=np.int32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s1, s2 = str(row['S1']), str(row['S2'])
        y = float(row['distance'])

        si = self._encode_seq(s1)  # (L,)
        sj = self._encode_seq(s2)  # (L,)

        # prop_matrix: (P, 20); 索引得到 (P, L)
        pi = self.prop_matrix[:, si]
        pj = self.prop_matrix[:, sj]
        diff = np.abs(pi - pj).T  # (L, P)
        x = diff.astype(np.float32, copy=False)

        if self.prop_indices is not None:
            x = x[:, self.prop_indices]

        if self.y_scaler:
            y = (y - self.y_scaler['mean']) / (self.y_scaler['std'] + 1e-8)

        return torch.from_numpy(x).float(), torch.tensor(y).float()


class SequencePairBinaryDataset(Dataset):
    """
    仅使用病毒对序列的0-1比对特征：
    - 对齐序列同位点相同 -> 0；不同 -> 1
    - 特征形状为 (L, 1)，不依赖AAIndex属性
    """
    def __init__(self, df: pd.DataFrame, seq_len: int, y_scaler=None):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.y_scaler = y_scaler

    def __len__(self):
        return len(self.df)

    def _encode_seq(self, s: str) -> np.ndarray:
        return np.fromiter((aa_to_idx[c] for c in s), dtype=np.int32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s1, s2 = str(row['S1']), str(row['S2'])
        y = float(row['distance'])

        si = self._encode_seq(s1)  # (L,)
        sj = self._encode_seq(s2)  # (L,)

        mismatch = (si != sj).astype(np.float32)  # (L,)
        x = mismatch[:, None]  # (L, 1)

        if self.y_scaler:
            y = (y - self.y_scaler['mean']) / (self.y_scaler['std'] + 1e-8)

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
            nn.Linear(seq_len, 256),
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


class PureMLPRegressor(nn.Module):
    """
    纯MLP回归器：不使用多头静态注意力，直接对 (L, P) 特征展平后回归。
    输入: x -> (B, L, P)
    展平: (B, L*P)
    """
    def __init__(self, seq_len, n_props):
        super().__init__()
        in_dim = seq_len * n_props
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, L, P)
        x = x.view(x.size(0), -1)  # (B, L*P)
        return self.net(x).squeeze(-1)


def standardize_property_dict(prop_dict):
    vals = np.array(list(prop_dict.values()))
    mean, std = vals.mean(), vals.std()
    return {aa: (v - mean) / std for aa, v in prop_dict.items()}


def build_prop_matrix(props):
    """将属性列表转换为 (P, 20) 的矩阵。"""
    prop_matrix = np.zeros((len(props), 20), dtype=np.float32)
    for p, prop in enumerate(props):
        for aa, idx in aa_to_idx.items():
            prop_matrix[p, idx] = prop.get(aa, 0.0)
    return prop_matrix


def load_props_as_matrix(json_path, corr_threshold=0.95, selected_names=None):
    props, prop_names = load_standardized_aaindex_props(json_path, corr_threshold=corr_threshold, selected_names=selected_names)
    prop_matrix = build_prop_matrix(props)
    return prop_matrix, prop_names


def make_dataloaders_from_time_series(config, selected_names=None, prop_indices=None):
    """
    读取 data/time_series 下的 train/val/test.csv，构建按需计算的 DataLoader。
    返回：train_dl, val_dl, test_dl, prop_names, y_scaler, seq_len, n_props
    """
    train_csv = config.get('train_csv')
    val_csv = config.get('val_csv')
    test_csv = config.get('test_csv')

    use_aaindex = bool(config.get('use_aaindex', True))

    prop_matrix = None
    prop_names = None
    if use_aaindex:
        prop_matrix_full, prop_names_full = load_props_as_matrix(
            config['json_path'],
            corr_threshold=config.get('corr_threshold', 0.95),
            selected_names=selected_names
        )

        if prop_indices is None:
            prop_matrix = prop_matrix_full
            prop_names = prop_names_full
        else:
            prop_matrix = prop_matrix_full[prop_indices, :]
            prop_names = [prop_names_full[i] for i in prop_indices]

    # 读取CSV
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # 序列已对齐且长度一致，直接从训练集首条推断长度
    seq_len = int(len(str(df_train.iloc[0]['S1'])))

    # y标准化仅用训练集统计量
    y_scaler = None
    if config.get('standardize_y', False):
        y_vals = df_train['distance'].astype(float).values
        y_scaler = {'mean': float(np.mean(y_vals)), 'std': float(np.std(y_vals) + 1e-8)}

    # 数据集与加载器
    if use_aaindex:
        train_ds = SequencePairDataset(df_train, prop_matrix, seq_len, y_scaler=y_scaler)
        val_ds = SequencePairDataset(df_val, prop_matrix, seq_len, y_scaler=y_scaler)
        test_ds = SequencePairDataset(df_test, prop_matrix, seq_len, y_scaler=y_scaler)
    else:
        # 使用0-1比对，不依赖AAIndex属性
        train_ds = SequencePairBinaryDataset(df_train, seq_len, y_scaler=y_scaler)
        val_ds = SequencePairBinaryDataset(df_val, seq_len, y_scaler=y_scaler)
        test_ds = SequencePairBinaryDataset(df_test, seq_len, y_scaler=y_scaler)

    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], num_workers=8, pin_memory=True)

    if use_aaindex:
        n_props = prop_matrix.shape[0]
        names = prop_names
    else:
        n_props = 1
        names = ["mismatch"]
    return train_dl, val_dl, test_dl, names, y_scaler, seq_len, n_props



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
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
        # 评估训练集与验证集指标（以非训练模式计算）
        train_metrics = evaluate(model, train_dl, device, y_scaler)
        val_metrics = evaluate(model, val_dl, device, y_scaler)
        val_rmse = val_metrics['RMSE']
        scheduler.step(val_rmse)
        print(
            f"[Epoch {epoch:03d}] "
            f"Train RMSE={train_metrics['RMSE']:.4f} | "
            f"Val RMSE={val_rmse:.4f}, Pearson={val_metrics['Pearson_r']:.3f}"
        )
        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
    # 训练结束后回滚到验证集最优模型，并打印最优信息
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best Val RMSE={best_val_rmse:.4f} at epoch {best_epoch}")
    else:
        print("Warning: best_state is None; model will remain at last epoch state.")
    return model


def main_from_dataloaders(config, train_dl, val_dl, test_dl, prop_names, seq_len, n_props, y_scaler=None):
    """
    新数据路径（显式train/val/test CSV）训练入口。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_pure_mlp = bool(config.get("use_pure_mlp", False))
    use_aaindex = bool(config.get("use_aaindex", True))
    if use_pure_mlp:
        model = PureMLPRegressor(
            seq_len=seq_len,
            n_props=n_props,
        ).to(device)
    else:
        model = WeightedMultiHeadAttentionMLP(
            seq_len=seq_len,
            n_props=n_props,
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

    scores = []
    can_rank_props = (not use_pure_mlp) and use_aaindex and (n_props > 1)
    if can_rank_props:
        scores = extract_average_attention_weights(model.attn_weights, model.head_weights, prop_names)
        print(f"\nTop-{config['top_n']} property importances:")
        for name, score in scores[:config['top_n']]:
            print(f"{name:15}: {score:.4f}")

    return model, scores, prop_names, test_metrics

def slice_features(X, Y, indices):
    return X[:, :, indices], Y


def auto_select_and_retrain_time_series(config):
    print("=== Step 0: Load CSV-based data (train/val/test) ===")
    use_pure_mlp = bool(config.get("use_pure_mlp", False))
    use_aaindex = bool(config.get("use_aaindex", True))

    # 首轮：使用全部属性
    train_dl, val_dl, test_dl, prop_names, y_scaler, seq_len, n_props = make_dataloaders_from_time_series(
        config, selected_names=None, prop_indices=None
    )

    print("=== Step 1: Full property training ===")
    model, scores, _, test_metrics = main_from_dataloaders(
        config, train_dl, val_dl, test_dl, prop_names, seq_len, n_props, y_scaler
    )

    if use_pure_mlp or (not use_aaindex) or (n_props <= 1) or (len(scores) == 0):
        print("\nSkip Top-N selection and retraining (pure MLP or no AAIndex or insufficient props).")
        return model, scores, prop_names, test_metrics

    scores_wo_diff = [(name, s) for name, s in scores]
    top_n_props = [name for name, _ in scores_wo_diff[:config["top_n"]]]
    print(f"\nTop-{config['top_n']} selected AAIndex properties:")
    for name in top_n_props:
        print(f" - {name}")

    print("\n=== Step 2: Filter features and retrain (Top-N) ===")
    prop_indices = [prop_names.index(name) for name in top_n_props]

    # 使用Top-N属性的重训数据加载器
    train_dl2, val_dl2, test_dl2, prop_names2, y_scaler2, seq_len2, n_props2 = make_dataloaders_from_time_series(
        config, selected_names=None, prop_indices=prop_indices
    )

    config_topn = config.copy()
    config_topn["n_heads"] = config_topn.get("n_retrain_heads", config["n_heads"])  # 回退为原值

    model_topn, scores_topn, _, test_metrics_topn = main_from_dataloaders(
        config_topn, train_dl2, val_dl2, test_dl2, prop_names2, seq_len2, n_props2, y_scaler2
    )

    return model_topn, scores_topn, prop_names2, test_metrics_topn


if __name__ == "__main__":

    # 切换至使用 time_series 拆分的CSV文件
    config = {
        # 文件路径
        "json_path": "data/prd/aaindex1_dicts.json",
        "train_csv": "data/time_series/train.csv",
        "val_csv":   "data/time_series/test.csv",
        "test_csv":  "data/time_series/test.csv",
        "out_path":  "data/time_series/prop1.csv",

        # 模型与训练参数
        "batch_size": 256,
        "n_heads": 4,
        "use_pure_mlp": False,  # True 启用纯MLP回归（禁用多头静态注意力）
        "use_aaindex": True,     # False 时使用0-1比对特征，跳过AAIndex
        "n_retrain_heads": 2,
        "epochs": 1000,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "patience": 60,

        # 数据处理参数
        "standardize_y": False,
        "corr_threshold": 0.6,
        "random_state": 42,

        # Top-N 属性选择
        "top_n": 5
    }

    torch.manual_seed(config["random_state"])
    np.random.seed(config["random_state"])
    random.seed(config["random_state"]) 

    model_final, scores_final, selected_props_final, test_metrics_final = auto_select_and_retrain_time_series(config)

    # 仅在存在注意力权重时输出属性重要性
    if scores_final:
        scores_df = pd.DataFrame(scores_final, columns=['prop_name', 'weight'])
        scores_df.to_csv(config["out_path"], index=False)
    else:
        print("No attention weights in Pure MLP mode; skip saving property importances.")
