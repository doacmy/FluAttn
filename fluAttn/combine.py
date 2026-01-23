import json
import torch
import numpy as np
import pandas as pd
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from itertools import combinations
import os
import copy


AA_TO_IDX = {aa: idx for idx, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def standardize_prop_1_dict(prop_dict):
    vals = np.array(list(prop_dict.values()))
    mean, std = vals.mean(), vals.std()
    return {aa: (v - mean) / std for aa, v in prop_dict.items()}

def load_standardized_aaindex_1_props():
    with open(config["aaindex_1_path"], "r") as f:
        raw_props = json.load(f)
    
    prop_1 = pd.read_csv(config["prop1_path"])
    selected_names = prop_1['prop_name'].tolist()

    if selected_names is not None:
        raw_props = {k: raw_props[k] for k in selected_names if k in raw_props}

    props = [standardize_prop_1_dict(d) for d in raw_props.values()]
    weights = prop_1["weight"].tolist()
    return props, np.array(weights)

def load_standardized_aaindex_2_props():
    with open(config["aaindex_2_path"], "r") as f:
        json_dict = json.load(f)
    
    prop_2 = pd.read_csv(config["prop2_path"])
    matrix_names = prop_2["prop_name"]

    matrix_dict = {k: np.array(v) for k, v in json_dict.items()}

    matrix_list = []
    for mname in matrix_names:
        matrix = matrix_dict[mname]

        min_val = matrix.min()
        max_val = matrix.max()
        if max_val > min_val:
            matrix = matrix / (max_val - min_val)
        else:
            matrix = np.zeros_like(matrix)


        matrix_list.append(matrix)

    weights = prop_2["weight"].tolist()
    return matrix_list, np.array(weights)

def get_binary_diff_X(pairs, seq_indices):
    L = len(seq_indices[0])
    n_pairs = len(pairs)

    X = np.zeros((n_pairs, L), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        si = seq_indices[i]
        sj = seq_indices[j]

        binary_diff = np.fromiter((a != b for a, b in zip(si, sj)), dtype=np.float32)

        X[k] = binary_diff
    return X

def get_aaindex_1_X(pairs, seq_indices, props, weights):
    props, weights = load_standardized_aaindex_1_props()

    L = len(seq_indices[0])
    n_pairs = len(pairs)
    n_props = len(props)
    
    props_matrix = np.zeros((n_props, 20), dtype=np.float32)
    for p, prop in enumerate(props):
        for aa, idx in AA_TO_IDX.items():
            props_matrix[p, idx] = prop.get(aa, 0.0)

    X = np.zeros((n_pairs, L), dtype=np.float32)

    for k, (i, j) in enumerate(pairs):
        si = seq_indices[i]
        sj = seq_indices[j]

        pi = props_matrix[:, si]
        pj = props_matrix[:, sj]
        prop_1_diff = np.abs(pi - pj)

        X[k] = np.dot(weights, prop_1_diff)

    return X

def get_aaindex_2_X(pairs, seq_indices, matrix_list, weights):
    L = len(seq_indices[0])
    n_pairs = len(pairs)

    matrix_num = len(weights)

    X = np.zeros((n_pairs, L), dtype=np.float32)

    M = np.zeros((matrix_num, L), dtype=np.float32)

    for k, (i, j) in enumerate(pairs):
        si = seq_indices[i]
        sj = seq_indices[j]

        for m in range(matrix_num):
            mat = matrix_list[m]
            M[m] = mat[si, sj]
            
        X[k] = np.dot(weights, M)
    return X

def build_X_Y(pairs, seq_indices, dist_mat, props_1, weights_1, props_2, weights_2):
    n_pairs = len(pairs)

    X0 = get_binary_diff_X(pairs, seq_indices)
    X1 = get_aaindex_1_X(pairs, seq_indices, props_1, weights_1)
    X2 = get_aaindex_2_X(pairs, seq_indices, props_2, weights_2)

    X = np.concatenate([X0, X1, X2], axis=1)

    Y = np.empty(n_pairs, dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        Y[k] = dist_mat[i, j]

    return X, Y


def _props_to_matrix(props):
    n_props = len(props)
    props_matrix = np.zeros((n_props, 20), dtype=np.float32)
    for p, prop in enumerate(props):
        for aa, idx in AA_TO_IDX.items():
            props_matrix[p, idx] = prop.get(aa, 0.0)
    return props_matrix

def _seq_to_idx_array(seq: str) -> np.ndarray:
    return np.array([AA_TO_IDX.get(ch, -1) for ch in seq], dtype=np.int32)

def _build_features_from_pairs(df: pd.DataFrame, props_1, weights_1, matrix_list, weights_2):
    s1_list = df["S1"].astype(str).tolist()
    s2_list = df["S2"].astype(str).tolist()

    assert len(s1_list) == len(s2_list), "S1 and S2 must have same number of rows"
    n = len(s1_list)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    L = len(s1_list[0])
    for i, (a, b) in enumerate(zip(s1_list, s2_list)):
        if len(a) != len(b):
            raise ValueError(f"Row {i} has different sequence lengths: {len(a)} vs {len(b)}")
        if len(a) != L:
            raise ValueError("All pairs must have consistent sequence length")

    idx1 = np.vstack([_seq_to_idx_array(s) for s in s1_list])
    idx2 = np.vstack([_seq_to_idx_array(s) for s in s2_list])

    # X0: binary difference, ignore invalid positions
    valid_mask = (idx1 >= 0) & (idx2 >= 0)
    X0 = ((idx1 != idx2) & valid_mask).astype(np.float32)

    # Prepare AAindex-1 matrix
    props_matrix = _props_to_matrix(props_1)  # shape: (P,20)
    w1 = weights_1.astype(np.float32)

    # X1: weighted AAindex-1 difference per position
    X1 = np.zeros((n, L), dtype=np.float32)
    for i in range(n):
        vi = valid_mask[i]
        if not np.any(vi):
            continue
        ii1 = idx1[i, vi]
        ii2 = idx2[i, vi]
        pi = props_matrix[:, ii1]  # (P, M)
        pj = props_matrix[:, ii2]  # (P, M)
        diff = np.abs(pi - pj)     # (P, M)
        X1[i, vi] = np.dot(w1, diff)

    # X2: weighted AAindex-2 substitution per position
    K = len(weights_2)
    w2 = weights_2.astype(np.float32)
    X2 = np.zeros((n, L), dtype=np.float32)
    for i in range(n):
        vi = valid_mask[i]
        if not np.any(vi):
            continue
        ii1 = idx1[i, vi]
        ii2 = idx2[i, vi]
        M = np.zeros((K, np.sum(vi)), dtype=np.float32)
        for m, mat in enumerate(matrix_list):
            M[m] = mat[ii1, ii2]
        X2[i, vi] = np.dot(w2, M)

    X = np.concatenate([X0, X1, X2], axis=1)
    return X

def get_data_from_pairs_df(df):
    props_1, weights_1 = load_standardized_aaindex_1_props()
    props_2, weights_2 = load_standardized_aaindex_2_props()

    X = _build_features_from_pairs(df, props_1, weights_1, props_2, weights_2)
    Y = df["distance"].to_numpy(dtype=np.float32)
    return X, Y

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

class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(dims[-1], 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train(model, optimizer, loss_fn, dataloader, device):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(Y_batch)
    return total_loss / len(dataloader.dataset)

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

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    
    return rmse, mae, r2



if __name__ == "__main__":

    for test_year in range(2022,2025):
        config = {
            "random_state": 42,
            "batch_size": 256,
            "aaindex_1_path": "data/prd/aaindex1_dicts.json",
            "aaindex_2_path": "data/prd/aaindex2_dicts.json",
            "prop1_path": f"fluAttn/prop/{test_year}/prop1.csv",
            "prop2_path": f"fluAttn/prop/{test_year}/prop2.csv",
            "train_csv": f"data/time_series/{test_year}/train.csv",
            "val_csv": f"data/time_series/{test_year}/val.csv",
            "test_csv": f"data/time_series/{test_year}/test.csv",
            "standardize_y": True,
            "patience": 20,
            "stopping_delta": 1e-4,
            "epochs": 200
        }

        set_global_seed(config["random_state"])

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

        # Load train/val/test directly from time_series CSVs
        X_train, y_train = get_data_from_pairs_df(df_train)
        X_val,   y_val   = get_data_from_pairs_df(df_val)
        X_test,  y_test  = get_data_from_pairs_df(df_test)

        y_scaler = None
        if config["standardize_y"]:
            y_scaler = {'mean': y_train.mean(), 'std': max(1e-8, y_train.std())}

        train_ds = PairDataset(X_train, y_train, y_scaler)
        val_ds   = PairDataset(X_val, y_val, y_scaler)
        test_ds  = PairDataset(X_test, y_test, y_scaler)

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=8, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=config["batch_size"], num_workers=8, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MLPRegressor(input_dim=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        best_val_rmse = float("inf")
        best_model = None
        patience = config["patience"]
        delta = config["stopping_delta"]
        epochs_no_improve = 0

        for epoch in range(config["epochs"]):
            train_loss = train(model, optimizer, loss_fn, train_loader, device)
            val_rmse, val_mae, val_r2 = evaluate(model, val_loader, device, y_scaler)

            print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f} | Val RMSE = {val_rmse:.4f}, MAE = {val_mae:.4f}, R2 = {val_r2:.4f}")

            if val_rmse < best_val_rmse - delta:
                best_val_rmse = val_rmse
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch:02d}. Best Val RMSE: {best_val_rmse:.4f}")
                break

        if best_model is not None:
            model.load_state_dict(best_model)
        model.eval()

        test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device, y_scaler)
        print(f"Test RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}, R2 = {test_r2:.4f}")
        result_path = 'time_result/combine.csv'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        row = {"year": test_year, "RMSE": test_rmse, "MAE": test_mae, "R2": test_r2}
        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(result_path) or os.stat(result_path).st_size == 0
        df_row.to_csv(result_path, mode='a', index=False, header=write_header)

