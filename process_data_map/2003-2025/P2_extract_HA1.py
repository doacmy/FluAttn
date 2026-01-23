from Bio import SeqIO
import pandas as pd

start = 16
end = 345

ha1_data = []
discarded_count = 0

for record in SeqIO.parse('data/prd/2003-2025/align_seq.fasta', "fasta"):
    virus_name = record.description.split("|")[0].strip()
    ha1_seq = str(record.seq[start:end])
    
    if '-' in ha1_seq or 'X' in ha1_seq:
        discarded_count += 1
        continue

    ha1_data.append((virus_name, ha1_seq))

df = pd.DataFrame(ha1_data, columns=["Virus", "HA1_Sequence"])
df.to_csv('data/prd/2003-2025/sequences.csv', index=False)

print(f"成功保存 {len(df)} 条 HA1 序列到 CSV")
print(f"丢弃了 {discarded_count} 条包含 gap ('-') 的序列")
