import pandas as pd
import networkx as nx

seq_df = pd.read_csv('data/prd/2003-2025/sequences.csv')
seq_df = seq_df[~seq_df['HA1_Sequence'].astype(str).str.contains('X', na=False)].copy()


raw_df = pd.read_csv('data/prd/2003-2025/2003-2025_final_HI.csv')
raw_df['Titre'] = pd.to_numeric(raw_df['Titre'], errors='coerce')
raw_df = raw_df[raw_df['Titre'].notna()].copy()

valid_seq_nodes = set(seq_df['Virus'].astype(str).str.strip())
raw_df = raw_df[
    raw_df['Test Virus'].astype(str).str.strip().isin(valid_seq_nodes)
    & raw_df['Reference Virus'].astype(str).str.strip().isin(valid_seq_nodes)
].copy()

def extract_last_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().str.extract(r"(\d{4})(?!.*\d)")[0],
        errors='coerce'
    )


def get_train_nodes(raw_df, year_min, year_max, top_n):
    test_year = extract_last_year(raw_df['Test Virus'])
    ref_year = extract_last_year(raw_df['Reference Virus'])
    mask = test_year.between(year_min, year_max) & ref_year.between(year_min, year_max)
    df = raw_df.loc[mask].copy()

    edges = (
        df[["Test Virus", "Reference Virus"]]
        .dropna()
        .astype(str)
        .apply(lambda s: (s["Test Virus"].strip(), s["Reference Virus"].strip()), axis=1)
        .tolist()
    )

    G = nx.Graph()
    G.add_edges_from(edges)

    degree_items = list(G.degree())
    degree_items.sort(key=lambda x: (-x[1], x[0]))

    top_degree = degree_items[:top_n]

    top_df = pd.DataFrame(top_degree, columns=["Virus", "Degree"])

    train_nodes = set(top_df["Virus"].astype(str).str.strip())

    return train_nodes


def get_extra_nodes(raw_df, train_nodes, val_year, conn_top_n):
    raw_test_year = extract_last_year(raw_df['Test Virus'])
    raw_ref_year = extract_last_year(raw_df['Reference Virus'])
    mask = (raw_test_year == val_year) | (raw_ref_year == val_year)
    df = raw_df.loc[mask].copy()

    edges = (
        df[["Test Virus", "Reference Virus"]]
        .dropna()
        .astype(str)
        .apply(lambda s: (s["Test Virus"].strip(), s["Reference Virus"].strip()), axis=1)
        .tolist()
    )
    G = nx.Graph()
    G.add_edges_from(edges)

    test = raw_df.loc[raw_test_year == val_year, 'Test Virus'].astype(str).str.strip()
    ref = raw_df.loc[raw_ref_year == val_year, 'Reference Virus'].astype(str).str.strip()
    viruses = set(pd.concat([test, ref], ignore_index=True).unique())

    conn_counts = []
    for v in viruses:
        if v in G:
            neighbors = set(G.neighbors(v))
            conn_to_top = neighbors & train_nodes
            conn_counts.append((v, len(conn_to_top)))

    conn_df = pd.DataFrame(conn_counts, columns=["Virus", "ConnectionsToTopNodes"])\
        .sort_values(["ConnectionsToTopNodes", "Virus"], ascending=[False, True])

    conn_top_df = conn_df.head(conn_top_n)
    val_nodes = set(conn_top_df["Virus"].astype(str).str.strip())

    return val_nodes

train_nodes = get_train_nodes(raw_df, 2015, 2022, 400)
val_nodes = get_extra_nodes(raw_df, train_nodes, 2023, 20)
test_nodes = get_extra_nodes(raw_df, train_nodes, 2024, 20)


all_nodes = set().union(train_nodes, val_nodes, test_nodes)
df_in_all = raw_df[
    raw_df['Test Virus'].astype(str).str.strip().isin(all_nodes)
    & raw_df['Reference Virus'].astype(str).str.strip().isin(all_nodes)
].copy()

hi_matrix = df_in_all.pivot(index='Test Virus', columns='Reference Virus', values='Titre')

# 删除行中数字少于3个的行（以非NaN计数作为有效数字计数）
row_valid_counts = hi_matrix.notna().sum(axis=1)
hi_matrix = hi_matrix.loc[row_valid_counts >= 3]
hi_matrix = hi_matrix.fillna("*")

hi_matrix.to_csv("data/time_series/HI_matrix_2024.csv")
