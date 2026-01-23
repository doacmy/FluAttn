import os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def extract_unique_virus_sequences(fasta_dir):
    import os
    virus_seq_dict = {}
    for filename in os.listdir(fasta_dir):
        if filename.endswith(".fasta"):
            fasta_path = os.path.join(fasta_dir, filename)
            for record in SeqIO.parse(fasta_path, "fasta"):
                virus_name = record.description.split("|")[0].strip()
                seq = str(record.seq).strip()

                clean_seq = seq.replace("-", "").replace("X", "")
                has_gap_or_x = '-' in seq or 'X' in seq
                seq_score = (not has_gap_or_x, len(clean_seq))  # True > False; longer is better

                if virus_name not in virus_seq_dict:
                    virus_seq_dict[virus_name] = (record.seq, seq_score)
                else:
                    _, existing_score = virus_seq_dict[virus_name]
                    if seq_score > existing_score:
                        virus_seq_dict[virus_name] = (record.seq, seq_score)

    return {k: v[0] for k, v in virus_seq_dict.items()}



def filter_csv_and_save_sequences(csv_path, fasta_dir, output_csv, output_raw_seq_fasta, output_ali_seq_fasta):
    virus_seq_dict = extract_unique_virus_sequences(fasta_dir)
    valid_virus_names = set(virus_seq_dict.keys())

    # 过滤 CSV 文件
    df = pd.read_csv(csv_path)
    mask = df['Test Virus'].isin(valid_virus_names) & df['Reference Virus'].isin(valid_virus_names)
    filtered_df = df[mask]
    filtered_df.to_csv(output_csv, index=False)
    print(f"保留了 {len(filtered_df)} 条记录，结果已保存到：{output_csv}")

    # 提取过滤后所涉及的病毒名，并保存对应序列
    involved_viruses = set(filtered_df['Test Virus']) | set(filtered_df['Reference Virus'])
    seq_records = []
    for virus in involved_viruses:
        if virus in virus_seq_dict:
            seq = Seq(virus_seq_dict[virus])
            record = SeqRecord(seq, id=virus, description="")
            seq_records.append(record)

    print(f"提取了 {len(seq_records)} 个病毒序列")
    SeqIO.write(seq_records, output_raw_seq_fasta, "fasta")

    os.system(f"muscle -in {output_raw_seq_fasta} -out {output_ali_seq_fasta} -maxiters 16 -diags")

    

fasta_dir = 'data/raw/GISAID/'
csv_path = 'data/prd/2003-2025/2003-2025.csv'
output_csv = 'data/prd/2003-2025/2003-2025_filter.csv'
output_raw_seq_fasta = 'data/prd/2003-2025/sequences.fasta'
output_ali_seq_fasta = 'data/prd/2003-2025/align_seq.fasta'


filter_csv_and_save_sequences(csv_path, fasta_dir, output_csv, output_raw_seq_fasta, output_ali_seq_fasta)

