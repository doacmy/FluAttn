#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import copy
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class Config:
    # 数据路径
    vtype = 'H1N1'

    # train_csv: str = f"data/prd/all_time/{vtype}/train.csv"
    # val_csv: str = f"data/prd/all_time/{vtype}/val.csv"
    # test_csv: str = f"data/prd/all_time/{vtype}/test.csv"


    test_year = 2023

    train_csv = f"data/time_series/{test_year}/train.csv"
    val_csv = f"data/time_series/{test_year}/val.csv"
    test_csv = f"data/time_series/{test_year}/test.csv"

    aaindex_2_path: str = "data/prd/AAIndex/aaindex2_dicts.json"
    aaindex_3_path: str = "data/prd/AAIndex/aaindex3_dicts.json"

    # 选点参数（只在 train.csv 上执行）
    mi_threshold: float = 1e-4
    conservation_max_freq_cutoff: float = 0.99
    gap_chars: str = "-."
    # 注意：distance_threshold 现在仅用于 MI 选点（二值化 variant），
    # 回归标签本身使用原始 distance
    distance_threshold: float = 2.0
    # 当真实距离大于该阈值时，认为发生“重大变异”（用于变异分类指标）
    mutation_distance_threshold: float = 4.0

    # 模型与训练
    batch_size: int = 512
    num_epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # 特征维度（num_positions 会在选点后被修改）
    num_positions: int = 96  # 初始值，选点后会被覆盖为 len(selected_positions)
    num_aaindex: int = 10
    in_channels: int = 1  # 不再使用（1D CNN 用 num_aaindex 作为通道）


cfg = Config()


def set_random_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """
    Set seeds for Python, NumPy, and PyTorch; return generator for DataLoader shuffle.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a reproducible, different seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# 2. AA 编码 & AAindex 工具
# ============================================================

VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(VALID_AA)}

# 论文最终使用的 10 个 AAindex（来自 AAindex2 + AAindex3）
AAINDEX_IDS: List[str] = [
    "BENS940104",
    "LUTR910108",
    "MUET010101",
    "KOLA920101",
    "AZAE970101",
    "BONM030104",
    "TANS760101",
    "ZHAC000106",
    "BETM990101",
    "BONM030103",
]


def matrix_to_scalar_property(mat: np.ndarray) -> Dict[str, float]:
    """
    将 20x20 替换矩阵 -> 每个 AA 的行均值属性 {AA: scalar}
    """
    if mat.shape != (20, 20):
        raise ValueError(f"AAindex 矩阵应为 20x20，实际 {mat.shape}")
    row_means = mat.mean(axis=1)  # (20,)
    return {aa: float(row_means[AA_TO_IDX[aa]]) for aa in VALID_AA}


def load_aaindex_2_props(config: Config) -> Dict[str, Dict[str, float]]:
    with open(config.aaindex_2_path, "r") as f:
        json_dict = json.load(f)

    props: Dict[str, Dict[str, float]] = {}
    for name, mat in json_dict.items():
        m = np.array(mat, dtype=np.float32)
        props[name] = matrix_to_scalar_property(m)
    return props


def load_aaindex_3_props(config: Config) -> Dict[str, Dict[str, float]]:
    with open(config.aaindex_3_path, "r") as f:
        json_dict = json.load(f)

    props: Dict[str, Dict[str, float]] = {}
    for name, mat in json_dict.items():
        m = np.array(mat, dtype=np.float32)
        props[name] = matrix_to_scalar_property(m)
    return props


def load_selected_aaindex_props(config: Config) -> Dict[str, Dict[str, float]]:
    """
    只保留 AAINDEX_IDS 中那 10 个 index。
    返回: {AAindexID: {AA: scalar}}
    """
    aa2 = load_aaindex_2_props(config)
    aa3 = load_aaindex_3_props(config)

    merged: Dict[str, Dict[str, float]] = {}
    merged.update(aa2)
    merged.update(aa3)

    selected: Dict[str, Dict[str, float]] = {}
    for name in AAINDEX_IDS:
        if name not in merged:
            raise KeyError(f"AAindex '{name}' 未在 aaindex2/3 JSON 中找到")
        aa_dict = merged[name]
        # 确保 20AA 都有值
        for aa in VALID_AA:
            if aa not in aa_dict:
                raise KeyError(f"AAindex '{name}' 缺少氨基酸 '{aa}' 的数值")
        selected[name] = aa_dict

    if len(selected) != cfg.num_aaindex:
        raise ValueError(
            f"选出的 AAindex 数量应为 {cfg.num_aaindex}，当前为 {len(selected)}"
        )
    return selected


# ============================================================
# 3. 从 CSV 读入样本 + 选取关键位点（只用 train）
# ============================================================

def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"S1", "S2", "distance"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"{path} 必须包含列 {expected_cols}")
    return df[["S1", "S2", "distance"]]


def check_sequence_lengths_single(df: pd.DataFrame, name: str) -> int:
    """
    检查单个数据集内部 S1/S2 是否等长，并返回统一长度。
    """
    lengths = []
    for idx, row in df.iterrows():
        s1 = str(row["S1"])
        s2 = str(row["S2"])
        if len(s1) != len(s2):
            raise ValueError(
                f"[{name}] 第 {idx} 行 S1 和 S2 长度不一致：len(S1)={len(s1)}, len(S2)={len(s2)}"
            )
        lengths.append(len(s1))

    lengths = np.array(lengths, dtype=int)
    if not np.all(lengths == lengths[0]):
        unique = np.unique(lengths)
        raise ValueError(
            f"[{name}] 不同样本的序列长度不一致：{unique}。"
            "说明尚未对多序列做统一对齐，请先用 MUSCLE/MAFFT 对齐。"
        )

    L = int(lengths[0])
    print(f"[info] [{name}] 所有样本 S1/S2 已对齐，统一长度 L = {L}")
    return L


def ensure_same_length(train_L: int, df: pd.DataFrame, name: str) -> None:
    """
    检查 val/test 的长度是否与 train 一致。
    """
    lengths = []
    for _, row in df.iterrows():
        s1 = str(row["S1"])
        s2 = str(row["S2"])
        if len(s1) != len(s2):
            raise ValueError(
                f"[{name}] 存在 S1/S2 长度不一致的样本，说明该集合内部未对齐。"
            )
        lengths.append(len(s1))

    lengths = np.array(lengths, dtype=int)
    if not np.all(lengths == train_L):
        unique = np.unique(lengths)
        raise ValueError(
            f"[{name}] 序列长度 {unique} 与 train 的长度 {train_L} 不一致，"
            "说明对齐参考不同，需先在全集上统一对齐。"
        )
    print(f"[info] [{name}] 序列长度与 train 一致，L = {train_L}")


def find_candidate_positions(
    df: pd.DataFrame,
    L: int,
    gap_chars: str,
    conservation_max_freq_cutoff: float,
) -> List[int]:
    """
    近似论文的 ConSurf + conservation<cutoff：

    - 丢弃任一 S1/S2 中出现 gap 的位点
    - 统计该列主氨基酸频率 max_freq，若 <cutoff，则认为“非高度保守”，保留
    """
    n = len(df)
    print(f"[info] [train] 总样本数: {n}")

    all_s1 = df["S1"].astype(str).tolist()
    all_s2 = df["S2"].astype(str).tolist()

    candidate_positions = []

    for pos in range(L):
        chars = []
        for s in all_s1:
            chars.append(s[pos])
        for s in all_s2:
            chars.append(s[pos])

        chars_arr = np.array(chars, dtype="<U1")

        # gap 检查
        if np.isin(chars_arr, list(gap_chars)).any():
            continue

        # 主氨基酸频率
        unique, counts = np.unique(chars_arr, return_counts=True)
        max_freq = counts.max() / counts.sum()

        if max_freq < conservation_max_freq_cutoff:
            candidate_positions.append(pos)

    print(
        f"[info] [train] 候选位点（无 gap 且 max_freq<{conservation_max_freq_cutoff}）数量: "
        f"{len(candidate_positions)}"
    )
    return candidate_positions


def compute_mi_for_position(
    df: pd.DataFrame,
    pos: int,
    distance_threshold: float,
    random_state: int | None = None,
) -> float:
    """
    单个位点 pos:
      x_i = 1{S1_i[pos] != S2_i[pos]}
      y_i = 1{distance_i > distance_threshold}
    逻辑回归拟合后，用 predict_proba 得到 y_hat，再计算 MI(y_hat>=0.5, y)

    注意：这里仍然是二值标签，只用于位点筛选，不影响回归标签本身。
    """
    s1_list = df["S1"].astype(str).tolist()
    s2_list = df["S2"].astype(str).tolist()
    dist_list = df["distance"].astype(float).to_numpy()

    n = len(df)
    x = np.zeros((n, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)

    for i in range(n):
        s1 = s1_list[i]
        s2 = s2_list[i]
        x[i, 0] = 1.0 if s1[pos] != s2[pos] else 0.0
        y[i] = 1 if dist_list[i] > distance_threshold else 0

    # 若全 0 或全 1，MI 一定为 0
    if np.all(x == x[0, 0]):
        return 0.0

    clf = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )

    clf.fit(x, y)
    proba = clf.predict_proba(x)[:, 1]
    y_hat = (proba >= 0.5).astype(np.int32)

    mi = mutual_info_score(y, y_hat)
    return float(mi)


def select_positions_by_mi(
    df: pd.DataFrame,
    candidate_positions: List[int],
    mi_threshold: float,
    distance_threshold: float,
    random_state: int | None = None,
) -> List[Tuple[int, float]]:
    """
    对候选位点计算 MI，并筛选 MI>mi_threshold 的位点。
    返回 [(pos, mi), ...]，按 MI 从大到小排序。
    """
    results = []
    for pos in candidate_positions:
        mi = compute_mi_for_position(
            df, pos, distance_threshold=distance_threshold, random_state=random_state
        )
        results.append((pos, mi))

    filtered = [(p, mi) for (p, mi) in results if mi > mi_threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    print(f"[info] [train] 通过 MI 阈值 {mi_threshold} 的位点数: {len(filtered)}")
    if len(filtered) > 0:
        print("[info] [train] 前 10 个位点及其 MI：")
        for p, mi in filtered[:10]:
            print(f"  pos={p}, MI={mi:.6e}")

    return filtered


# ============================================================
# 4. AAIndexEncoder 与 Dataset
# ============================================================

class AAIndexEncoder:
    """
    对每个选定位点 p ∈ selected_positions、每个 AAindex k：

        feature[i, j] = f_k(AA_i^p) - f_k(AA_j^p)

    输出矩阵 shape = (num_positions, num_aaindex)
    """

    def __init__(
        self,
        selected_positions: List[int],
        aaindex_ids: List[str],
        aaindex_props: Dict[str, Dict[str, float]],
    ):
        self.positions = selected_positions
        self.aaindex_ids = aaindex_ids
        self.aaindex_props = aaindex_props

        if len(self.aaindex_ids) != cfg.num_aaindex:
            raise ValueError("AAindex 数量必须为 cfg.num_aaindex")

    def encode_pair(self, s1: str, s2: str) -> np.ndarray:
        if len(s1) != len(s2):
            raise ValueError("S1 和 S2 长度不一致")

        L = len(s1)
        feat = np.zeros((len(self.positions), len(self.aaindex_ids)), dtype=np.float32)

        for i, pos in enumerate(self.positions):
            if not (0 <= pos < L):
                raise ValueError(f"位点 {pos} 越界，序列长度为 {L}")
            aa1 = s1[pos]
            aa2 = s2[pos]

            if aa1 not in VALID_AA or aa2 not in VALID_AA:
                continue

            for j, aa_id in enumerate(self.aaindex_ids):
                props = self.aaindex_props[aa_id]
                v1 = props[aa1]
                v2 = props[aa2]
                # 直接写入 float32，避免依赖全局 float 名可能被覆盖
                feat[i, j] = np.float32(v1 - v2)

        return feat  # (num_positions, num_aaindex)


class AntigenicityDataset(Dataset):
    """
    从 CSV 读入样本：
      - 输入：Conv1d 形式 (C=num_aaindex, L=num_positions)
      - 标签：直接为 distance（回归）
    """

    def __init__(
        self,
        csv_path: str,
        encoder: AAIndexEncoder,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        expected_cols = {"S1", "S2", "distance"}
        if not expected_cols.issubset(self.df.columns):
            raise ValueError(f"{csv_path} 必须包含列 {expected_cols}")

        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        s1 = str(row["S1"])
        s2 = str(row["S2"])
        dist = float(row["distance"])

        # 回归标签：直接用原始 distance
        label = dist

        feat = self.encoder.encode_pair(s1, s2)  # (num_positions, num_aaindex)
        # 转成 (C=num_aaindex, L=num_positions)，用于 Conv1d
        feat_cl = feat.T  # shape: (num_aaindex, num_positions)
        x = torch.from_numpy(feat_cl)  # (C, L)

        y = torch.tensor([label], dtype=torch.float32)  # (1,)
        return x, y


# ============================================================
# 5. CNN 模型（1D 卷积版本，回归输出）
# ============================================================

class AntigenicityCNN(nn.Module):
    """
    1D 卷积版本基线：
    - 输入: (B, C=10, L=num_positions)
    - Conv1: out=193, kernel_size=3, stride=1, padding=1
    - Conv2: out=212, kernel_size=5, stride=1, padding=2
    - Conv3: out=109, kernel_size=5, stride=1, padding=2
    - 全局展平 + FC(256) + 输出(1) 线性（回归）

    输出为实数抗原距离，不再使用 sigmoid。
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=cfg.num_aaindex,   # 10 个 AAindex 作为通道
            out_channels=193,
            kernel_size=3,
            stride=1,
            padding=1,                     # 保持长度
        )
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv1d(
            in_channels=193,
            out_channels=212,
            kernel_size=5,
            stride=1,
            padding=2,                     # 保持长度
        )
        self.dropout2 = nn.Dropout(p=0.165)

        self.conv3 = nn.Conv1d(
            in_channels=212,
            out_channels=109,
            kernel_size=5,
            stride=1,
            padding=2,                     # 保持长度
        )
        self.dropout3 = nn.Dropout(p=0.1)

        # 用 dummy 推一遍，自动算 flatten 维度
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.num_aaindex, cfg.num_positions)  # (B, C, L)
            h = self._forward_features(dummy)
            flat_dim = h.shape[1]

        self.fc1 = nn.Linear(flat_dim, 256)
        self.dropout_fc1 = nn.Dropout(p=0.241)
        self.fc_out = nn.Linear(256, 1)  # 回归输出：1 维实数

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)  # 展平
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=10, L=num_positions)
        """
        x = self._forward_features(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc_out(x)  # 不做激活，直接回归
        return x  # (B, 1)


# ============================================================
# 6. 训练 & 评估函数（回归：RMSE / MAE / R2）
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)  # (B, C, L)
        y = y.to(device)  # (B, 1)

        optimizer.zero_grad()
        y_pred = model(x)  # (B, 1)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    回归评估：RMSE, MAE, R2
    """
    model.eval()
    ys = []
    ps = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            ys.append(y.cpu().numpy())
            ps.append(y_pred.cpu().numpy())

    ys = np.concatenate(ys, axis=0).reshape(-1)
    ps = np.concatenate(ps, axis=0).reshape(-1)

    # RMSE
    mse = np.mean((ps - ys) ** 2)
    rmse = float(np.sqrt(mse))

    # MAE
    mae = float(np.mean(np.abs(ps - ys)))

    # R2
    ss_res = np.sum((ys - ps) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    if ss_tot > 0:
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = float("nan")

    # 变异分类指标：以距离 > mutation_distance_threshold 判为“重大变异”
    mut_acc = None
    mut_prec = None
    mut_rec = None
    mut_f1 = None
    thr = float(cfg.mutation_distance_threshold)
    if ys.size > 0:
        true_mut = ys > thr
        pred_mut = ps > thr

        mut_acc = float((true_mut == pred_mut).mean())

        tp = float(np.logical_and(true_mut, pred_mut).sum())
        fp = float(np.logical_and(~true_mut, pred_mut).sum())
        fn = float(np.logical_and(true_mut, ~pred_mut).sum())

        mut_prec = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        mut_rec = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        denom = mut_prec + mut_rec
        mut_f1 = 2.0 * mut_prec * mut_rec / denom if denom > 0.0 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mutation_accuracy": mut_acc,
        "mutation_precision": mut_prec,
        "mutation_recall": mut_rec,
        "mutation_f1": mut_f1,
    }


# ============================================================
# 7. 主流程（位点筛选仅用 train）
# ============================================================

def main():
    data_rng = set_random_seed(cfg.seed)
    print(f"[info] 使用设备: {cfg.device} | Seed: {cfg.seed}")

    # ---------- Step 1: 仅从 train.csv 选位点 ----------
    train_df = load_pairs(cfg.train_csv)
    train_L = check_sequence_lengths_single(train_df, name="train")

    # 确认 val/test 序列长度与 train 一致（同一对齐参考）
    val_df = load_pairs(cfg.val_csv)
    test_df = load_pairs(cfg.test_csv)
    ensure_same_length(train_L, val_df, name="val")
    ensure_same_length(train_L, test_df, name="test")

    candidate_positions = find_candidate_positions(
        train_df,
        L=train_L,
        gap_chars=cfg.gap_chars,
        conservation_max_freq_cutoff=cfg.conservation_max_freq_cutoff,
    )

    pos_mi_list = select_positions_by_mi(
        train_df,
        candidate_positions=candidate_positions,
        mi_threshold=cfg.mi_threshold,
        distance_threshold=cfg.distance_threshold,
        random_state=cfg.seed,
    )

    selected_positions = [p for (p, mi) in pos_mi_list]
    cfg.num_positions = len(selected_positions)

    print(f"[info] 最终选出的位点数量: {cfg.num_positions}")
    print(f"[info] 选中位点示例（前 20 个）: {selected_positions[:20]}")

    # ---------- Step 2: 加载 AAindex 属性 ----------
    aaindex_props = load_selected_aaindex_props(cfg)
    encoder = AAIndexEncoder(selected_positions, AAINDEX_IDS, aaindex_props)

    # ---------- Step 3: 构造 Dataset & DataLoader ----------
    train_ds = AntigenicityDataset(cfg.train_csv, encoder)
    val_ds = AntigenicityDataset(cfg.val_csv, encoder)
    test_ds = AntigenicityDataset(cfg.test_csv, encoder)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=data_rng,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=data_rng,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=data_rng,
    )

    # ---------- Step 4: 初始化模型 ----------
    model = AntigenicityCNN(cfg).to(cfg.device)
    criterion = nn.MSELoss()  # 回归损失
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ---------- Step 5: 训练（加入 Early Stopping） ----------
    best_rmse = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.num_epochs + 1):
        train_mse = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device
        )
        train_rmse = np.sqrt(train_mse)

        val_metrics = evaluate_regression(model, val_loader, cfg.device)
        val_rmse = val_metrics["rmse"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f}"
        )

        # -------- Early Stopping 逻辑 --------
        if val_rmse < best_rmse - 1e-8:  # 有显著提升（防止浮点数抖动）
            best_rmse = val_rmse
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())  # 保存最好模型
        else:
            bad_epochs += 1
            print(f"[info] RMSE 无提升: {bad_epochs}/{cfg.patience}")

        if bad_epochs >= cfg.patience:
            print(f"[Early Stopping] 验证集 RMSE 连续 {cfg.patience} 次未提升，提前停止训练。")
            break

    # 训练结束后恢复最佳权重
    model.load_state_dict(best_state)



    # ---------- Step 6: 测试评估 ----------
    test_metrics = evaluate_regression(model, test_loader, cfg.device)
    print("==== Test Metrics (Regression + Mutation Classification) ====")
    print(f"rmse: {test_metrics['rmse']:.4f}")
    print(f"mae: {test_metrics['mae']:.4f}")
    print(f"r2: {test_metrics['r2']:.4f}")
    print(
        f"mutation_accuracy: {test_metrics['mutation_accuracy']:.4f}, "
        f"mutation_precision: {test_metrics['mutation_precision']:.4f}, "
        f"mutation_recall: {test_metrics['mutation_recall']:.4f}, "
        f"mutation_f1: {test_metrics['mutation_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
