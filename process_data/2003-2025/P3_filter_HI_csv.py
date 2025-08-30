import pandas as pd
import re

def extract_year(virus_name):
    match = re.search(r'/(\d{2,4})$', virus_name)
    year = match.group(1)
    return int(year) if len(year) == 4 else int("20" + year) 

def canonical_pair(a, b):
    return tuple(sorted([a, b]))

def convert_titre(val):
    if isinstance(val, float):
        return int(val)
    return val

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

seq_df = pd.read_csv('data/prd/2003-2025/sequences.csv')
seq_df['Year'] = seq_df['Virus'].apply(extract_year)

seq_df_sorted = seq_df.sort_values(by='Year', ascending=True)
seq_rep_df = seq_df_sorted.drop_duplicates(subset='HA1_Sequence', keep='first')
sequence_to_rep = dict(zip(seq_rep_df['HA1_Sequence'], seq_rep_df['Virus']))

virus_to_seq = dict(zip(seq_df['Virus'], seq_df['HA1_Sequence']))
virus_to_rep = {virus: sequence_to_rep[seq] for virus, seq in virus_to_seq.items()}

df = pd.read_csv('data/prd/2003-2025/2003-2025_filter.csv', dtype=str)
valid_viruses = set(seq_df['Virus'])
mask = df['Test Virus'].isin(valid_viruses) & df['Reference Virus'].isin(valid_viruses)
filtered_df = df[mask].copy()

filtered_df['Test Virus'] = filtered_df['Test Virus'].map(virus_to_rep)
filtered_df['Reference Virus'] = filtered_df['Reference Virus'].map(virus_to_rep)

filtered_df['Pair Key'] = filtered_df.apply(
    lambda row: tuple(sorted([row['Test Virus'].strip(), row['Reference Virus'].strip()])),
    axis=1
)

grouped = filtered_df.groupby('Pair Key')
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

filtered_df = pd.DataFrame(result_rows)



filtered_df['Titre'] = filtered_df['Titre'].apply(convert_titre)
filtered_df = filtered_df[filtered_df['Test Virus'] != filtered_df['Reference Virus']]

filtered_df.to_csv('data/prd/2003-2025/2003-2025_final_HI.csv', index=False)

mapping_df = pd.DataFrame(list(virus_to_rep.items()), columns=['Original Virus', 'Representative Virus'])
mapping_df.to_csv('data/prd/2003-2025/virus_name_mapping_by_year.csv', index=False)
