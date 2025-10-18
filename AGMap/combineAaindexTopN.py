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


def get_train_data(config):
    seq_df = pd.read_csv(config["train_seq_path"])
    dist_df = pd.read_csv(config["dist_path"], index_col=0)

    seq_df = seq_df[['short_name', 'HA1_sequence']].set_index('short_name')
    common = seq_df.index.intersection(dist_df.index)
    seq_df = seq_df.loc[common]
    dist_df = dist_df.loc[common, common]

    seqs_array = np.array(seq_df['HA1_sequence'].apply(list).to_list())
    variable_sites = [i for i in range(seqs_array.shape[1]) if len(set(seqs_array[:, i]) - {'-'}) > 1]
    seqs_var = seqs_array[:, variable_sites]
    seqs = ["".join(row) for row in seqs_var]
    dist_mat = dist_df.values

    n = len(seqs)
    pairs = list(combinations(range(n), 2))

    seq_indices = np.full((len(seqs), len(seqs[0])), -1, dtype=np.int32)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            seq_indices[i, j] = AA_TO_IDX.get(aa, -1)


    props_1, weights_1 = load_standardized_aaindex_1_props()
    props_2, weights_2 = load_standardized_aaindex_2_props()


    X_train, Y_train = build_X_Y(pairs, seq_indices, dist_mat, props_1, weights_1, props_2, weights_2)

    return X_train, Y_train

def get_test_data(config):

    X_test, Y_test = 0, 0



    return X_test, Y_test

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
    def __init__(self, input_dim, hidden_dims=[1024, 256, 64], dropout=0):
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

    data_set = "2003-2025" # "1963-2002" or "2003-2025"

    config = {
        "random_state": 42,
        "batch_size": 256,
        "aaindex_1_path": "data/prd/aaindex1_dicts.json",
        "aaindex_2_path": "data/prd/aaindex2_dicts.json",
        "standardize_y": True,
        "patience": 20,
        "stopping_delta": 1e-4,
        "test_size": 0.2,
        "epochs": 200
    }

    if data_set == "1963-2002":
        config["train_seq_path"] = "data/prd/1963-2002/sequences.csv"
        config["dist_path"] = "data/prd/1963-2002/distance_matrix.csv"
        config["prop1_path"] = "data/prd/1963-2002/prop1.csv"
        config["prop2_path" ]= "data/prd/1963-2002/prop2.csv"

    else:
        config["train_seq_path"] = "data/prd/2003-2025/final_sequences.csv"
        config["dist_path"] = "data/prd/2003-2025/2003_2025_distance_matrix.csv"
        config["prop1_path"] = "data/prd/2003-2025/prop1.csv"
        config["prop2_path" ]= "data/prd/2003-2025/prop2.csv"



    set_global_seed(config["random_state"])

    X_train, y_train = get_train_data(config)
    X_test, y_test = get_test_data(config)

    y_scaler = None
    if config["standardize_y"]:
        y_scaler = {'mean': y_train.mean(), 'std': y_train.std()}

    train_ds = PairDataset(X_train, y_train, y_scaler)
    test_ds  = PairDataset(X_test, y_test, y_scaler)

    val_size = max(1, int(0.1 * len(train_ds)))
    train_ds, val_ds = random_split(
        train_ds,
        [len(train_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(config["random_state"])
    )

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
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch:02d}. Best Val RMSE: {best_val_rmse:.4f}")
            break



    model.load_state_dict(best_model)
    model.eval()

    test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device, y_scaler)
    print(f"Test RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}, R2 = {test_r2:.4f}")


# 1963-2002
# X0          RMSE = 0.8200, MAE = 0.5997, R2 = 0.9787
# X0,X1       RMSE = 0.7737, MAE = 0.5596, R2 = 0.9810
# X0,X2       RMSE = 0.7888, MAE = 0.5729, R2 = 0.9803
# X0,X1,X2    RMSE = 0.7666, MAE = 0.5585, R2 = 0.9814

# 2003-2025
# X0          RMSE = 0.9227, MAE = 0.6168, R2 = 0.7825
# X0,X1       RMSE = 0.7266, MAE = 0.4685, R2 = 0.8651
# X0,X2       RMSE = 0.7435, MAE = 0.4813, R2 = 0.8587
# X0,X1,X2    RMSE = 0.6520, MAE = 0.4211, R2 = 0.8914
