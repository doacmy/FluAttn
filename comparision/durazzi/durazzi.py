import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import os


# ProtBERT 序列嵌入函数
def get_protbert_embedding(seq):
    seq = ' '.join(list(seq))
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
    tokens = tokenizer(seq, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)
    # 使用 [CLS] 向量作为序列表示
    return output.last_hidden_state[0][0].numpy()


def build_X_Y_from_df(df: pd.DataFrame, cache: dict):
    """
    从包含列 S1, S2, distance 的 DataFrame 构造训练/测试数据。
    - 对每条序列用 ProtBERT 取嵌入向量 emb
    - 以 |emb(S1) - emb(S2)| 作为特征向量 X
    - 以 distance 作为标注 y
    通过 cache 复用已计算的序列嵌入以加速。
    """
    required = {"S1", "S2", "distance"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building X/Y"):
        s1 = str(row["S1"]).strip()
        s2 = str(row["S2"]).strip()

        if s1 in cache:
            e1 = cache[s1]
        else:
            e1 = get_protbert_embedding(s1)
            cache[s1] = e1

        if s2 in cache:
            e2 = cache[s2]
        else:
            e2 = get_protbert_embedding(s2)
            cache[s2] = e2

        feat = np.abs(e1 - e2)
        X.append(feat)
        y.append(float(row["distance"]))

    return np.array(X), np.array(y)


if __name__ == "__main__":
    for test_year in range(2022,2025):
        train_csv = f"data/time_series/{test_year}/train.csv"
        val_csv = f"data/time_series/{test_year}/val.csv"
        test_csv = f"data/time_series/{test_year}/test.csv"

        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        df_train = pd.concat([df_train, df_val], axis=0, ignore_index=True)
        df_test = pd.read_csv(test_csv)

        # 加载 ProtBERT
        tokenizer = BertTokenizer.from_pretrained("comparision/durazzi/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("comparision/durazzi/prot_bert")
        model.eval()

        # 构建特征与标签（带缓存避免重复计算）
        embed_cache = {}
        X_train, y_train = build_X_Y_from_df(df_train, embed_cache)
        X_test, y_test = build_X_Y_from_df(df_test, embed_cache)

        # 归一化 + 岭回归
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        reg = Ridge(alpha=1.0)
        reg.fit(X_train_scaled, y_train)

        y_pred = reg.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Distance RMSE: {rmse:.4f}, R^2: {r2:.4f}, MAE: {mae:.4f}")

        result_path = 'time_result/durazzi.csv'
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        row = {"year": test_year, "RMSE": rmse, "MAE": mae, "R2": r2}
        df_row = pd.DataFrame([row])
        write_header = not os.path.exists(result_path) or os.stat(result_path).st_size == 0
        df_row.to_csv(result_path, mode='a', index=False, header=write_header)
