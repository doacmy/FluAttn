import re
import os
import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
import hashlib
import pandas as pd
from collections import defaultdict, Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split
import numpy

RANDOM_SEED = 42
START_PEPTIDE_LEN = 16

def read_virus_sequences(fasta_dir):
    virus_seq_list = []
    count = 0
    # 固定读取顺序，避免不同文件系统导致的不稳定顺序
    for filename in sorted(os.listdir(fasta_dir)):
        if filename.endswith(".fasta"):
            fasta_path = os.path.join(fasta_dir, filename)
            for record in SeqIO.parse(fasta_path, "fasta"):
                if 'H3N2' in fasta_dir:
                    virus_name = record.description.split("|")[0].strip()
                    seq = str(record.seq).strip()

                    if len(seq) >= 566 and 'X' not in seq:
                        virus_seq_list.append({
                            'virus_name': virus_name,
                            'sequence': seq
                        })
                else:
                    virus_name = record.description.split("|")[2].strip()
                    seq = str(record.seq).strip()

                    if len(seq) >= 566 and 'X' not in seq:
                        virus_seq_list.append({
                            'virus_name': virus_name,
                            'sequence': seq
                        })

    df = pd.DataFrame(virus_seq_list).drop_duplicates()
    return df

def normalize_region(region: str) -> str:
    if region is None:
        return ""
    region = region.strip().lower()
    region = re.sub(r"[^a-z0-9]+", "", region)
    return region

def split_virus_name(name: str):
    """返回 (subtype, region, vid, year) 或 None"""
    if not isinstance(name, str):
        return None
    parts = name.split('/')
    if len(parts) < 4:
        return None
    subtype = parts[0].strip()
    year = parts[-1].strip()
    vid = parts[-2].strip()
    region = '/'.join(p.strip() for p in parts[1:-2])
    return subtype, region, vid, year

# ---------- 新增：地区变体映射 & 验证式改名 ----------

def build_region_variant_map(names) -> dict:
    """
    收集 seqs_df['virus_name'] 中所有出现过的地区写法，
    返回 {region_norm: [candidate_region1, candidate_region2, ...]}。
    候选按出现频次由高到低排序。
    """
    buckets = defaultdict(Counter)
    for raw in pd.Series(names).dropna().unique():
        parts = split_virus_name(raw)
        if not parts:
            continue
        _, region, _, _ = parts
        buckets[normalize_region(region)][region] += 1

    variant_map = {}
    for rnorm, ctr in buckets.items():
        variant_map[rnorm] = [r for (r, _) in ctr.most_common()]
    return variant_map

def rewrite_name_to_existing(raw: str, variant_map: dict, valid_names_set: set) -> str:
    """
    仅替换地区部分；对同一归一化键下的所有候选地区逐一尝试，
    返回第一个能在 seqs_df['virus_name'] 中命中的完整写法；否则原样返回。
    """
    parts = split_virus_name(raw)
    if not parts:
        return raw
    subtype, region, vid, year = parts
    rnorm = normalize_region(region)

    for cand_region in variant_map.get(rnorm, []):
        candidate = f"{subtype}/{cand_region}/{vid}/{year}"
        if candidate in valid_names_set:
            return candidate
    return raw

# ----------------------------------------------------

def read_hi_data(source_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(source_dir) if f.lower().endswith('.csv')]
    files.sort()
    frames = []
    for f in files:
        path = os.path.join(source_dir, f)
        try:
            df = pd.read_csv(path, dtype=str)
            frames.append(df)
        except Exception as e:
            print(f"跳过文件 {path}: {e}")
    if not frames:
        return pd.DataFrame()
    hi_df = pd.concat(frames, ignore_index=True)
    # 统一处理 '<' 或 '<40' 为 '20'
    hi_df = hi_df.map(lambda x: '20' if x == '<' else x)
    hi_df = hi_df.map(lambda x: '20' if x == '<40' else x)
    hi_df = hi_df.map(lambda x: '5' if x == '<10' else x)
    hi_df = hi_df[hi_df['Titre'] != '*'].copy()
    hi_df = hi_df[hi_df['Titre'] != '0'].copy()
    return hi_df

def clean_hi_df(hi_df, seqs_df):
    """
    1) 先保留滴度表中两端病毒名都能直接在序列表命中的记录；
    2) 对剩余记录，按地区归一化键收集 seqs_df 中所有地区写法候选；
       对 Test/Reference 两列逐一替换候选并验证完整名称是否存在，命中即接受；
    3) 合并两部分，去重。
    """
    # 直接命中的部分
    valid_virus_names = seqs_df['virus_name'].tolist()
    valid_names_set = set(valid_virus_names)

    mask = hi_df['Test Virus'].isin(valid_names_set) & hi_df['Reference Virus'].isin(valid_names_set)
    filtered_df = hi_df[mask].copy()

    # 尝试修复未命中的部分
    rest_df = hi_df[~mask].copy()
    if not rest_df.empty:
        variant_map = build_region_variant_map(seqs_df['virus_name'])

        for col in ["Test Virus", "Reference Virus"]:
            cache = {}
            def _fix(name):
                if name in cache:
                    return cache[name]
                fixed = rewrite_name_to_existing(name, variant_map, valid_names_set)
                cache[name] = fixed
                return fixed
            rest_df[col] = rest_df[col].apply(_fix)

        mask2 = rest_df['Test Virus'].isin(valid_names_set) & rest_df['Reference Virus'].isin(valid_names_set)
        rest_df = rest_df[mask2]

        if not rest_df.empty:
            filtered_df = pd.concat([filtered_df, rest_df], ignore_index=True)

    # 将滴度数据中的病毒名称替换为序列信息，保留同名多序列的所有配对
    name_to_seqs = {}
    for _, seq_row in seqs_df.iterrows():
        virus_name = seq_row['virus_name']
        seq_value = seq_row['sequence']
        if virus_name not in name_to_seqs:
            name_to_seqs[virus_name] = []
        if seq_value not in name_to_seqs[virus_name]:
            name_to_seqs[virus_name].append(seq_value)

    records = []
    for _, row in filtered_df.iterrows():

        test_year = int(row['Test Virus'].split('/')[-1])
        ref_year = int(row['Reference Virus'].split('/')[-1])

        s1_list = name_to_seqs.get(row['Test Virus'], [])
        s2_list = name_to_seqs.get(row['Reference Virus'], [])
        for s1 in s1_list:
            for s2 in s2_list:
                records.append({
                    'Test Virus': s1,
                    'Reference Virus': s2,
                    'Test Year': test_year,
                    'Reference Year': ref_year,
                    'Titre': row['Titre']
                })

    final_df = pd.DataFrame.from_records(records, columns=['Test Virus', 'Reference Virus', 'Test Year', 'Reference Year', 'Titre']).drop_duplicates()
    return final_df

def merge_hi_df(seq_hi_df):
    def safe_gmt_titre(group_df):
        titres = group_df['Titre'].astype(str).str.strip().astype(float)
        log2_vals = numpy.log2(titres)
        mean_log2 = log2_vals.mean()
        return float(numpy.power(2.0, mean_log2))

    # 先在全局上为每条序列算“最早年份”
    virus_year_df = pd.concat([
        seq_hi_df[['Test Virus', 'Test Year']].rename(columns={'Test Virus': 'Virus', 'Test Year': 'Year'}),
        seq_hi_df[['Reference Virus', 'Reference Year']].rename(columns={'Reference Virus': 'Virus', 'Reference Year': 'Year'}),
    ], ignore_index=True)
    virus_min_year = virus_year_df.groupby('Virus')['Year'].min()

    # 再构造无向 pair key
    seq_hi_df['Pair Key'] = seq_hi_df.apply(
        lambda row: tuple(sorted((row['Test Virus'].strip(), row['Reference Virus'].strip()))),
        axis=1
    )

    grouped = seq_hi_df.groupby('Pair Key')
    result_rows = []
    for pair_key, group_df in grouped:
        try:
            titre_value = safe_gmt_titre(group_df)
            virus1, virus2 = pair_key
            result_rows.append({
                'Test Virus': virus1,
                'Reference Virus': virus2,
                'Test Year': virus_min_year[virus1],
                'Reference Year': virus_min_year[virus2],
                'Titre': titre_value
            })
        except ValueError as e:
            print(f"[错误] {pair_key}: {e}")

    result_df = pd.DataFrame(result_rows)
    if not result_df.empty:
        result_df = result_df.sort_values(
            by=["Test Virus", "Reference Virus", "Test Year", "Reference Year", "Titre"],
            kind="mergesort"
        ).reset_index(drop=True)
    return result_df


def calculate_distance(df):
    df['Log2_Titre'] = numpy.log2(df['Titre'])
    col_basis = df.groupby('Reference Virus')['Log2_Titre'].max().rename('Column_Basis_Log2')
    df = df.merge(col_basis, on='Reference Virus', how='left')
    df['distance'] = df['Column_Basis_Log2'] - df['Log2_Titre']

    out_cols = [
        'Test Virus', 'Reference Virus', 'Test Year', 'Reference Year', 'distance'
    ]

    return df[out_cols].sort_values(['Reference Virus', 'Test Virus']).reset_index(drop=True)


def aligne_sequences_in_hi_df(
    hi_df,
    seqs_df,
    muscle_path = "muscle",
    maxiters = "16",
    diags = True,
    max_chunk_size = 1000,   # 每块最多多少条序列；可按内存调整
    prefer_chunks = None     # 若想强制块数，如 10，则设置它；否则按max_chunk_size切
):

    # 1) 收集涉及到的唯一序列
    seq_to_name = dict(zip(seqs_df['sequence'], seqs_df['virus_name']))
    involved_sequences = set(hi_df['Test Virus']).union(set(hi_df['Reference Virus']))

    # 2) 为每条原序列分配稳定 uid（基于原始字符串）
    fasta_records, rawseq_to_uid = [], {}
    # 固定集合迭代顺序，确保对齐输入稳定
    for sequence in sorted(involved_sequences):
        uid = hashlib.md5(sequence.encode('utf-8')).hexdigest()
        rawseq_to_uid[sequence] = uid
        fasta_records.append(
            SeqRecord(Seq(sequence), id=uid, description=seq_to_name.get(sequence, "")))

    # 无需对齐的边界情形
    if not fasta_records:
        return hi_df

    out_dir = Path('process_data')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) 准备临时目录
    tmp_root = Path(tempfile.mkdtemp(prefix="muscle_profile_merge_", dir=str(out_dir)))

    def _run(cmd, **kwargs):
        # 小工具：运行 shell 命令并检查
        subprocess.run(cmd, check=True, **kwargs)

    try:
        # 4) 切块策略
        N = len(fasta_records)
        if prefer_chunks and prefer_chunks > 0:
            chunk_count = min(prefer_chunks, N)
            chunk_size = (N + chunk_count - 1) // chunk_count
        else:
            chunk_size = max_chunk_size
            chunk_count = (N + chunk_size - 1) // chunk_size

        # 5) 每块各自对齐
        aligned_chunk_paths = []
        for idx in range(chunk_count):
            start = idx * chunk_size
            end = min(N, start + chunk_size)
            chunk = fasta_records[start:end]
            if not chunk:
                continue
            chunk_in = tmp_root / f"chunk_{idx+1}.fasta"
            chunk_aln = tmp_root / f"chunk_{idx+1}_aligned.fasta"
            SeqIO.write(chunk, chunk_in, "fasta")

            cmd = [muscle_path, "-in", str(chunk_in), "-out", str(chunk_aln), "-maxiters", str(maxiters)]
            if diags: cmd += ["-diags"]
            _run(cmd)
            aligned_chunk_paths.append(chunk_aln)

        # 6) profile-profile 迭代合并
        if not aligned_chunk_paths:
            return hi_df

        # 若只有一个块，直接就是全局对齐
        current_profile = aligned_chunk_paths[0]
        if len(aligned_chunk_paths) > 1:
            # 把后续每个块与 current_profile 做 profile-profile 合并
            for i in range(1, len(aligned_chunk_paths)):
                next_profile = aligned_chunk_paths[i]
                merged_out = tmp_root / f"merged_{i}.fasta"
                cmd = [
                    muscle_path, "-profile",
                    "-in1", str(current_profile),
                    "-in2", str(next_profile),
                    "-out", str(merged_out)
                ]
                _run(cmd)
                current_profile = merged_out  # 更新累计结果

        # 7) 读取最终全局对齐（统一坐标系）
        aligned_records = list(SeqIO.parse(str(current_profile), "fasta"))
        uid_to_aligned = {rec.id: str(rec.seq) for rec in aligned_records}

        # 8) 映射回 hi_df（原序列 -> 对齐序列）
        seq_to_aligned = {}
        miss = 0
        for raw_seq, uid in rawseq_to_uid.items():
            aln = uid_to_aligned.get(uid)
            if aln is None:
                miss += 1
            else:
                seq_to_aligned[raw_seq] = aln

        # 一致性检查：所有对齐序列长度应一致
        if seq_to_aligned:
            lens = {len(s) for s in seq_to_aligned.values()}
            if len(lens) != 1:
                raise RuntimeError(f"Profile 合并后列数不一致：{sorted(lens)}，请检查 MUSCLE 版本/输入质量。")

        # 9) 更新 hi_df
        hi_df = hi_df.copy()
        hi_df['Test Virus'] = hi_df['Test Virus'].map(seq_to_aligned).fillna(hi_df['Test Virus'])
        hi_df['Reference Virus'] = hi_df['Reference Virus'].map(seq_to_aligned).fillna(hi_df['Reference Virus'])
        hi_df.drop_duplicates(inplace=True)



        if miss > 0:
            print(f"[警告] 有 {miss} 条序列未在最终对齐中找到 uid（可能被过滤或命名不一致）。")

        return hi_df

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_root, ignore_errors=True)

def random_split_data(out_dir, df_aligned, random_state: int = RANDOM_SEED):
    """
    将 df_aligned 按 80% / 10% / 10% 随机划分为
    训练集、验证集和测试集，并返回 (train_df, val_df, test_df)。
    """
    # 先划分出 80% 训练集、20% 验证+测试集
    train_df, valtest_df = train_test_split(
        df_aligned,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )

    # 再将剩余 20% 按 1:1 划分为验证集和测试集
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )

    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    val_df.to_csv(f"{out_dir}/val.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)

    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")

def time_split_data(out_dir, df_aligned):
    os.makedirs(out_dir, exist_ok=True)

    train_range_min = 2009
    train_range_max = 2019
    mask = (
        df_aligned['year1'].between(train_range_min, train_range_max, inclusive='both') &
        df_aligned['year2'].between(train_range_min, train_range_max, inclusive='both')
    )
    train_df = df_aligned[mask].copy()[['S1', 'S2', 'year1', 'year2', 'distance']]
    val_mask = (train_df['year1'] == train_range_max) | (train_df['year2'] == train_range_max)
    val_df = train_df[val_mask]
    train_df = train_df[~val_mask]

    keep_n = int(len(val_df) * 0.1)
    if keep_n > 0:
        val_keep = val_df.sample(n=keep_n, random_state=RANDOM_SEED)
        train_df = pd.concat([train_df, val_df.drop(val_keep.index)], ignore_index=True)
        val_df = val_keep

    test_range_min = 2020
    test_range_max = 2024
    mask = (
        (df_aligned['year1'].between(test_range_min, test_range_max, inclusive='both') & df_aligned['year2'].between(test_range_min, test_range_max, inclusive='both')) 
    )
    test_df = df_aligned[mask].copy()[['S1', 'S2', 'year1', 'year2','distance']]


    train_df.to_csv(f"{out_dir}/Period/train.csv", index=False)
    val_df.to_csv(f"{out_dir}/Period/val.csv", index=False)
    test_df.to_csv(f"{out_dir}/Period/test.csv", index=False)

    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")

def main(fasta_dir: str = 'data/raw/GISAID/HA',
         source_dir: str = "data/raw/csv",
         time_out_dir: str = 'data/prd/new',
         all_out_dir: str = 'data/prd/all'):
    seqs_df = read_virus_sequences(fasta_dir)
    hi_df = read_hi_data(source_dir)
    seq_hi_df = clean_hi_df(hi_df, seqs_df)
    seq_hi_df = merge_hi_df(seq_hi_df)

    dis_df = calculate_distance(seq_hi_df)

    df_aligned = aligne_sequences_in_hi_df(dis_df, seqs_df)

    df_aligned = df_aligned.rename(columns={
        "Test Virus": "S1",
        "Reference Virus": "S2",
        'Test Year': "year1", 
        'Reference Year': "year2",
        "distance": "distance"
    })
    
    time_split_data(time_out_dir, df_aligned)
    random_split_data(all_out_dir, df_aligned)



if __name__ == '__main__':

    vtype = 'H3N2' # or 'H1N1'

    fasta_dir = f'data/raw/{vtype}/GISAID/HA'
    source_dir = f"data/raw/{vtype}/csv"
    time_out_dir = f"data/prd/time_series/{vtype}"
    all_out_dir = f"data/prd/all_time/{vtype}"

    main(fasta_dir, source_dir, time_out_dir, all_out_dir)
