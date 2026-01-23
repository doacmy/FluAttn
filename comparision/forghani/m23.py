#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy
import random
from pathlib import Path
from dataclasses import dataclass
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class Config:
    # 数据路径
    vtype = 'H3N2'

    # train_csv: str = f"data/prd/all_time/{vtype}/train.csv"
    # val_csv: str = f"data/prd/all_time/{vtype}/val.csv"
    # test_csv: str = f"data/prd/all_time/{vtype}/test.csv"

    test_year = 2024

    train_csv = f"data/time_series/{test_year}/train.csv"
    val_csv = f"data/time_series/{test_year}/val.csv"
    test_csv = f"data/time_series/{test_year}/test.csv"

    # AAIndex PCA 文件
    aaindex_csv: str = "comparision/forghani/AAindec_PCA_Factors.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输入片段裁剪
    ha_start: int = 17      # 对齐后的 HA 全长序列，从第 18 个氨基酸开始
    ha_length: int = 304    # 输入 M23 的固定长度

    # 训练参数
    batch_size_train: int = 512
    batch_size_eval: int = 2
    base_lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0002
    # 以 epoch 为单位控制训练总轮数
    max_epochs: int = 100
    lr_power: float = 1.0

    # Early Stopping：验证集 RMSE 连续 patience 次未提升则停止
    early_stop_patience: int = 10

    # 当真实距离大于该阈值时，认为发生“重大变异”（用于变异分类指标）
    mutation_distance_threshold: float = 4.0

    # 训练设备
    num_workers: int = 0
    seed: int = 42


def set_random_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    """
    Set seeds for Python, NumPy and PyTorch to keep runs reproducible.
    Returns a CPU generator that can be used by DataLoader for deterministic shuffles.
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
    """Ensure each DataLoader worker has a distinct, but deterministic, seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)



# ============================================================
# 2. AAIndex PCA 嵌入加载
# ============================================================

class AAIndexPCAMapper:
    """
    从 AAindec_PCA_Factors.csv 加载 11×20 矩阵，构建 aa -> 11 维向量映射
    """
    def __init__(self, csv_path: str | Path, normalize_to_01: bool = False):
        df = pd.read_csv(csv_path)
        self.aa_order: List[str] = list(df.columns)
        mat = df.to_numpy(dtype=np.float32)

        if normalize_to_01:
            mat = mat / 255.0

        self.mat = mat
        self.aa2vec = {aa: mat[:, i] for i, aa in enumerate(self.aa_order)}
        self.zero_vec = np.zeros((mat.shape[0],), dtype=np.float32)
        # 支持对齐缺口和未知位点
        for k in ["-", "X"]:
            self.aa2vec.setdefault(k, self.zero_vec)

    def encode_residue(self, aa: str) -> np.ndarray:
        aa = aa.upper()
        return self.aa2vec.get(aa, self.zero_vec)

    def encode_subseq(self, seq: str, start: int, length: int) -> np.ndarray:
        seq = (seq or "").strip().upper()
        subseq = seq[start:start + length]
        if len(subseq) < length:
            subseq += "-" * (length - len(subseq))
        # 预分配并逐位填充
        feats = np.zeros((self.mat.shape[0], length), dtype=np.float32)
        for i, a in enumerate(subseq):
            v = self.aa2vec.get(a, self.zero_vec)
            feats[:, i] = v
        return feats

    def encode_pair_to_tensor(self, s1: str, s2: str, start: int, length: int) -> torch.Tensor:
        f1 = self.encode_subseq(s1, start, length)
        f2 = self.encode_subseq(s2, start, length)
        arr = np.stack([f1, f2], axis=1)  # (11, 2, length)
        return torch.from_numpy(arr)


# ============================================================
# 3. Dataset
# ============================================================

class CsvPairDataset(Dataset):
    """
    读取 CSV（S1, S2, distance），裁剪子序列并编码为 (11,2,304)
    """
    def __init__(self, csv_path: str | Path, aa_mapper: AAIndexPCAMapper,
                 ha_start: int, ha_length: int):
        # 读取数据（调用方已保证 distance 为数值；序列可能包含 '-' 与 'X'）
        self.df = pd.read_csv(csv_path)
        required_cols = {"S1", "S2", "distance"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing} in {csv_path}")

        self.aa_mapper = aa_mapper
        self.ha_start = ha_start
        self.ha_length = ha_length

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self.aa_mapper.encode_pair_to_tensor(row["S1"], row["S2"], self.ha_start, self.ha_length)
        y = torch.tensor([float(row["distance"])], dtype=torch.float32)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    return xs, ys


def build_dataloaders(cfg: Config, rng: torch.Generator | None = None):
    aa_mapper = AAIndexPCAMapper(cfg.aaindex_csv)

    train_ds = CsvPairDataset(cfg.train_csv, aa_mapper, cfg.ha_start, cfg.ha_length)
    val_ds   = CsvPairDataset(cfg.val_csv, aa_mapper, cfg.ha_start, cfg.ha_length)
    test_ds  = CsvPairDataset(cfg.test_csv, aa_mapper, cfg.ha_start, cfg.ha_length)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size_train, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        worker_init_fn=seed_worker, generator=rng,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size_eval, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        worker_init_fn=seed_worker, generator=rng,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size_eval, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        worker_init_fn=seed_worker, generator=rng,
    )
    # 简要汇总
    print(
        f"[Data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"(batch_train={cfg.batch_size_train}, batch_eval={cfg.batch_size_eval})"
    )
    return train_loader, val_loader, test_loader


# ============================================================
# 4. M23 模型
# ============================================================

class M23Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(11, 256, kernel_size=(2, 7), stride=(1, 3), padding=(1, 3))
        self.pool0 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.lrn0 = nn.LocalResponseNorm(3, alpha=0.0001, beta=0.75)

        self.conv1 = nn.Conv2d(256, 256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))
        self.pool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.lrn1 = nn.LocalResponseNorm(3, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.pool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.lrn2 = nn.LocalResponseNorm(3, alpha=0.0001, beta=0.75)

        self.fc6 = nn.Linear(256 * 3 * 1, 256)
        self.dropout6 = nn.Dropout(0.5)
        self.fc_last = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.pool0(self.lrn0(F.relu(self.conv0(x))))
        x = self.pool1(self.lrn1(F.relu(self.conv1(x))))
        x = self.pool2(self.lrn2(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout6(F.relu(self.fc6(x)))
        return self.fc_last(x).squeeze(-1)


# ============================================================
# 5. 训练与评估
# ============================================================

def build_poly_scheduler(optimizer, max_epochs: int, power: float):
    """
    多项式衰减学习率（按 epoch 调度）：
    lr = base_lr * (1 - epoch / max_epochs) ** power
    """
    def lr_lambda(current_epoch: int):
        t = min(current_epoch / max_epochs, 1.0)
        return (1 - t) ** power
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, device):
    criterion = nn.L1Loss()
    model.train()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    return None


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    preds, trues = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    trues = np.concatenate(trues).reshape(-1)

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return mae, rmse, r2


@torch.no_grad()
def evaluate_mutation_metrics(model, loader, device, mutation_threshold: float):
    """
    变异分类指标（以距离 > mutation_threshold 判为“重大变异”）：
      - mutation_accuracy
      - mutation_precision
      - mutation_recall
      - mutation_f1
    """
    model.eval()
    preds, trues = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)
    trues = np.concatenate(trues).reshape(-1)

    mut_acc = None
    mut_prec = None
    mut_rec = None
    mut_f1 = None

    if trues.size > 0:
        thr = float(mutation_threshold)
        true_mut = trues > thr
        pred_mut = preds > thr

        mut_acc = float((true_mut == pred_mut).mean())

        tp = float(np.logical_and(true_mut, pred_mut).sum())
        fp = float(np.logical_and(~true_mut, pred_mut).sum())
        fn = float(np.logical_and(true_mut, ~pred_mut).sum())

        mut_prec = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        mut_rec = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        denom = mut_prec + mut_rec
        mut_f1 = 2.0 * mut_prec * mut_rec / denom if denom > 0.0 else 0.0

    return mut_acc, mut_prec, mut_rec, mut_f1


# ============================================================
# 6. 主程序（含 RMSE 早停）
# ============================================================

def main():
    cfg = Config()
    data_rng = set_random_seed(cfg.seed)
    
    print(f"Using device: {cfg.device} | Seed: {cfg.seed}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg, data_rng)
    model = M23Net().to(cfg.device)

    optimizer = SGD(
        model.parameters(),
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_poly_scheduler(optimizer, cfg.max_epochs, cfg.lr_power)
    epoch = 0

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(1, cfg.max_epochs + 1):
        train_one_epoch(model, train_loader, optimizer, cfg.device)

        # 计算 Train / Val 指标
        train_mae, train_rmse, train_r2 = evaluate_metrics(model, train_loader, cfg.device)
        val_mae, val_rmse, val_r2 = evaluate_metrics(model, val_loader, cfg.device)
        val_metrics = {"rmse": val_rmse, "mae": val_mae, "r2": val_r2}

        # 每轮统一格式日志输出
        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f}"
        )
        
        

        # ---- Early Stopping 逻辑（基于验证集 RMSE）----
        if val_rmse < best_val_rmse - 1e-6:  # 允许一个极小容差
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            print(f"[EarlyStop] RMSE 未提升计数: {epochs_no_improve} / {cfg.early_stop_patience}")

        if epochs_no_improve >= cfg.early_stop_patience:
            print(f"[EarlyStop] 验证集 RMSE 连续 {cfg.early_stop_patience} 次未提升，提前结束训练。")
            break

        # 每个 epoch 结束后再进行一次 LR 调度步进
        scheduler.step()

    # 使用验证集最佳模型进行测试
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print("\n========== Final Test ==========")
    test_mae, test_rmse, test_r2 = evaluate_metrics(model, test_loader, cfg.device)
    test_mut_acc, test_mut_prec, test_mut_rec, test_mut_f1 = evaluate_mutation_metrics(
        model, test_loader, cfg.device, cfg.mutation_distance_threshold
    )
    print(
        f"[Test] RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | R2={test_r2:.4f} | "
        f"MutAcc={test_mut_acc:.4f} | MutPrec={test_mut_prec:.4f} | "
        f"MutRec={test_mut_rec:.4f} | MutF1={test_mut_f1:.4f}"
    )


if __name__ == "__main__":
    main()
