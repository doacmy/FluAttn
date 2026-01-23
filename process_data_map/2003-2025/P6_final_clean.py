import pandas as pd

df = pd.read_csv("data/prd/2003-2025/2003_2025_distance_matrix.csv", index_col=0)
virus_names = df.index.tolist()


df = pd.read_csv("data/prd/2003-2025/sequences.csv")

filtered_df = df[df['Virus'].isin(virus_names)]

filtered_df.columns = ['short_name', 'HA1_sequence']



df_unique = filtered_df.drop_duplicates(subset=["HA1_sequence"], keep="first")

print(f"原始序列数: {len(df)}, 去重后: {len(df_unique)}")

df_unique.to_csv("data/prd/2003-2025/final_sequences.csv", index=False)
