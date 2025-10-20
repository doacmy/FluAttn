import os
import re
from typing import Dict, Iterable

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def to_numeric_titre(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == '*':
        return np.nan
    if s.startswith('<'):
        return np.nan
    try:
        v = float(s)
        return np.nan if v <= 0 else v
    except Exception:
        return np.nan


def compute_antigenic_distance(hi_long: pd.DataFrame) -> pd.DataFrame:
    required_cols = {'Test Virus', 'Reference Virus', 'Titre'}
    missing = required_cols - set(hi_long.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df = hi_long.copy()

    # 数值化 Titre，并取 log2
    df['Titre_num'] = df['Titre'].apply(to_numeric_titre)
    df = df[df['Titre_num'].notna()].copy()
    df['Log2_Titre'] = np.log2(df['Titre_num'])

    # 列基线：同一 Reference Virus 下 log2 滴度的最大值
    col_basis = df.groupby('Reference Virus')['Log2_Titre'].max().rename('Column_Basis_Log2')

    # 合并并计算距离（以二倍稀释步为单位）
    df = df.merge(col_basis, on='Reference Virus', how='left')
    df['distance'] = df['Column_Basis_Log2'] - df['Log2_Titre']

    # 整理输出列
    out_cols = [
        'Test Virus', 'Reference Virus', 'distance'
    ]
    return df[out_cols].sort_values(['Reference Virus', 'Test Virus']).reset_index(drop=True)

def attach_sequences(
    dist_df: pd.DataFrame,
    seq_csv_path: str = 'data/prd/2003-2025/sequences.csv',
    virus_col: str = 'Virus',
    seq_col: str = 'HA1_Sequence',
    test_col: str = 'Test Virus',
    ref_col: str = 'Reference Virus',
    test_seq_out: str = 'S1',
    ref_seq_out: str = 'S2',
):
    seq_df = pd.read_csv(seq_csv_path, usecols=[virus_col, seq_col])
    seq_df = seq_df.drop_duplicates(subset=[virus_col], keep='first')
    seq_df[virus_col] = seq_df[virus_col].astype(str).str.strip()

    out = dist_df.copy()
    out[test_col] = out[test_col].astype(str).str.strip()
    out[ref_col] = out[ref_col].astype(str).str.strip()

    seq_for_test = seq_df.rename(columns={virus_col: test_col, seq_col: test_seq_out})
    seq_for_ref = seq_df.rename(columns={virus_col: ref_col, seq_col: ref_seq_out})

    out = out.merge(seq_for_test, on=test_col, how='left')
    out = out.merge(seq_for_ref, on=ref_col, how='left')

    keep_cols = [
        'Test Virus', 'Reference Virus', 'distance',
        'S1', 'S2'
    ]
    out = out[keep_cols].dropna(subset=['S1', 'S2'])

    return out

def filter_by_year_range(
    df: pd.DataFrame,
    year_min: int = 2007,
    year_max: int = 2017,
    test_col: str = 'Test Virus',
    ref_col: str = 'Reference Virus',
):
    # 从病毒名末尾提取四位年份，并筛选区间
    tmp = df.copy()
    test_year = tmp[test_col].astype(str).str.strip().str.extract(r'(\d{4})\s*$')[0]
    ref_year = tmp[ref_col].astype(str).str.strip().str.extract(r'(\d{4})\s*$')[0]
    test_year = pd.to_numeric(test_year, errors='coerce')
    ref_year = pd.to_numeric(ref_year, errors='coerce')
    mask = (
        test_year.between(year_min, year_max, inclusive='both') &
        ref_year.between(year_min, year_max, inclusive='both')
    )
    return tmp[mask].copy()[['S1', 'S2', 'distance']]

def filter_year_vs_range(
    df: pd.DataFrame,
    range_min: int = 2007,
    range_max: int = 2017,
    year: int = 2018,
    test_col: str = 'Test Virus',
    ref_col: str = 'Reference Virus',
):
    # 一列年份为 2018，另一列在 [2007, 2017]
    tmp = df.copy()
    test_year = tmp[test_col].astype(str).str.strip().str.extract(r'(\d{4})\s*$')[0]
    ref_year = tmp[ref_col].astype(str).str.strip().str.extract(r'(\d{4})\s*$')[0]
    test_year = pd.to_numeric(test_year, errors='coerce')
    ref_year = pd.to_numeric(ref_year, errors='coerce')
    mask = (
        ((test_year == year) & ref_year.between(range_min, range_max, inclusive='both')) |
        ((ref_year == year) & test_year.between(range_min, range_max, inclusive='both'))
    )
    return tmp[mask].copy()[['S1', 'S2', 'distance']]

def get_seq_info(df):
    fasta_dir = 'data/raw/GISAID'
    raw_out_fasta = 'new_data_1/raw.fasta'
    ali_out_fasta= 'new_data_1/ali.fasta'

    test_names = df['Test Virus'].astype(str).str.strip()
    ref_names = df['Reference Virus'].astype(str).str.strip()
    all_viruses: Iterable[str] = pd.unique(pd.concat([test_names, ref_names], ignore_index=True))

    records = []
    if os.path.isdir(fasta_dir):
        for fname in os.listdir(fasta_dir):
            fpath = os.path.join(fasta_dir, fname)
            for record in SeqIO.parse(fpath, 'fasta'):
                name = str(record.description).split('|')[0].strip()
                if name not in all_viruses:
                    continue

                seq = str(record.seq).strip()
                if ('-' in seq) or ('X' in seq):
                    continue
                records.append(record)

    
    with open(raw_out_fasta, 'w') as handle:
        SeqIO.write(records, handle, 'fasta')

    os.system(f"muscle -in {raw_out_fasta} -out {ali_out_fasta} -maxiters 16 -diags")





def main():

    hi_df = pd.read_csv('new_data/2003-2025.csv')
    dist_df = compute_antigenic_distance(hi_df)

    get_seq_info(dist_df)


    # df = attach_sequences(dist_df)

    # train_df = filter_by_year_range(df, 2007, 2017)
    # train_df.to_csv('data/time_series/train.csv', index=False)

    # val_df = filter_year_vs_range(df, range_min=2007, range_max=2017, year=2018)
    # val_df.to_csv('data/time_series/val.csv', index=False)

    # test_df = filter_year_vs_range(df, range_min=2007, range_max=2017, year=2019)
    # test_df.to_csv('data/time_series/test.csv', index=False)

    






if __name__ == '__main__':
    main()
