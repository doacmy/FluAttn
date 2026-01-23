import pandas as pd
import networkx as nx


df = pd.read_csv('data/prd/2003-2025/2003-2025_final_HI.csv')
hi_matrix = df.pivot(index='Test Virus', columns='Reference Virus', values='Titre')
hi_matrix = hi_matrix.fillna("*")
min_valid = 4

def is_numeric_df(df):
    return df.apply(lambda col: col.map(lambda x: str(x).isnumeric()))

while True:
    numeric_mask = is_numeric_df(hi_matrix)

    row_valid_counts = numeric_mask.sum(axis=1)
    col_valid_counts = numeric_mask.sum(axis=0)

    rows_to_keep = row_valid_counts[row_valid_counts >= min_valid].index
    cols_to_keep = col_valid_counts[col_valid_counts >= min_valid].index

    new_matrix = hi_matrix.loc[rows_to_keep, cols_to_keep]

    if new_matrix.shape == hi_matrix.shape:
        break
    else:
        hi_matrix = new_matrix

edges = []
for test_virus in hi_matrix.index:
    for ref_virus in hi_matrix.columns:
        value = hi_matrix.loc[test_virus, ref_virus]
        if value.isnumeric():
            edges.append((test_virus, ref_virus))

G = nx.Graph()
G.add_edges_from(edges)

largest_component = max(nx.connected_components(G), key=len)
print(f"最大连通分量中病毒数：{len(largest_component)}")

filtered_matrix = hi_matrix.loc[
    hi_matrix.index.intersection(largest_component),
    hi_matrix.columns.intersection(largest_component)
]

filtered_matrix.to_csv("data/prd/2003-2025/2003-2025_HI_matrix.csv")

print(f"清洗后的 HI 矩阵大小：{filtered_matrix.shape}")
