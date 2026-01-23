import os
import pandas as pd
import numpy as np
from Bio import SeqIO, AlignIO

# 从全部数据中抽取出目标序列
df = pd.read_csv("data/raw/HI/virus_name.csv")
virus_name = df.iloc[:, 0].tolist()

result = []
for record in SeqIO.parse("data/raw/Seq/FASTA.fa", "fasta"):
    desc_parts = record.description.split()
    full_name = desc_parts[1]

    if full_name.count('/') < 2:
        full_name = full_name + " " + desc_parts[2]

    if full_name in virus_name:
        result.append(record)

output_path = "data/prd/1963-2002/seq_raw.fa"
SeqIO.write(result, output_path, "fasta")

# 使用 MUSCLE 进行序列对齐
os.system("muscle -in data/prd/1963-2002/seq_raw.fa -out data/prd/1963-2002/seq_aligned.fa -maxiters 16 -diags")


# 从对齐后的序列中提取HA1片段
alignment = AlignIO.read("data/prd/1963-2002/seq_aligned.fa", "fasta")
ha1_only = alignment[:, 16:345]
AlignIO.write(ha1_only, "data/prd/1963-2002/ha1_trimmed.fa", "fasta")

# 从HA1片段中去除重复的序列，只保留gap最少的序列
record_dict = {}
for record in SeqIO.parse("data/prd/1963-2002/ha1_trimmed.fa", "fasta"):
    desc_parts = record.description.split()
    full_name = desc_parts[1]
    if full_name.count('/') < 2:
        full_name = full_name + " " + desc_parts[2]

    gap_count = record.seq.count('-')

    if (full_name not in record_dict) or (gap_count < record_dict[full_name][1]):
        record_dict[full_name] = (record, gap_count)

filtered_records = [v[0] for v in record_dict.values()]

SeqIO.write(filtered_records, "data/prd/1963-2002/ha1.fa", "fasta")

# 保存到 CSV 文件
name_df = pd.read_csv("data/raw/HI/virus_name.csv")
name_dict = dict(zip(name_df['full_name'], name_df['short_name']))

rows = []
for record in SeqIO.parse("data/prd/1963-2002/ha1.fa", "fasta"):
    desc_parts = record.description.split()
    full_name = desc_parts[1]

    if full_name.count('/') < 2:
        full_name = full_name + " " + desc_parts[2]

    short_name = name_dict.get(full_name, None)

    rows.append({
        "short_name": short_name,
        "HA1_sequence": record.seq,
    })

df = pd.DataFrame(rows)
df.to_csv("data/prd/1963-2002/sequences.csv", index=False)

    

