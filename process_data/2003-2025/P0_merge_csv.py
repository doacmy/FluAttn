# 清洗从pdf文件中提取的原始滴度数据，并合并为一个文件
import os
import pandas as pd

def safe_avg_titre(group_df):
    titres = group_df['Titre'].astype(str).str.strip()

    numeric_mask = titres.str.isnumeric()
    star_mask = titres == "*"
    less_than_mask = titres.str.match(r"<\d+")

    num_numeric = numeric_mask.sum()
    num_star = star_mask.sum()
    num_less = less_than_mask.sum()
    total = len(titres)

    if num_numeric == total:
        return float(titres.astype(float).mean())
    if num_star == total:
        return "*"
    if num_less == total:
        values = [int(t[1:]) for t in titres]
        return f"<{min(values)}"
    if num_numeric + num_star == total:
        numeric_values = titres[numeric_mask].astype(float)
        return float(numeric_values.mean())
    if num_less + num_star == total and num_numeric == 0:
        values = [int(t[1:]) for t in titres[less_than_mask]]
        return f"<{min(values)}"
    if num_numeric > 0 and num_less + num_star + num_numeric == total:
        numeric_values = titres[numeric_mask].astype(float)
        return float(numeric_values.mean())

    raise ValueError(f"非法的 Titre 组合: {titres.tolist()}")


def merge_and_process_csvs(folder_path, output_csv=None):
    all_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            path = os.path.join(folder_path, filename)
            df = pd.read_csv(path, dtype=str)

            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_df['Pair Key'] = combined_df.apply(
        lambda row: tuple(sorted([row['Test Virus'].strip(), row['Reference Virus'].strip()])),
        axis=1
    )

    grouped = combined_df.groupby('Pair Key')
    result_rows = []

    for pair_key, group_df in grouped:
        try:
            titre_value = safe_avg_titre(group_df)
            virus1, virus2 = pair_key
            result_rows.append({
                'Test Virus': virus1,
                'Reference Virus': virus2,
                'Titre': titre_value
            })
        except ValueError as e:
            print(f"[错误] {pair_key}: {e}")

    result_df = pd.DataFrame(result_rows)

    def convert_titre(val):
        if isinstance(val, float):
            return int(val)
        return val

    result_df['Titre'] = result_df['Titre'].apply(convert_titre)

    if output_csv:
        result_df.to_csv(output_csv, index=False)


folder = "data/prd/WHOCC/"
output_file = "data/prd/2003-2025.csv"
merge_and_process_csvs(folder, output_file)