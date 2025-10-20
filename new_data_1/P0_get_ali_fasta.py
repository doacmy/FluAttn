import os
import re
from typing import Dict, Iterable

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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
    get_seq_info(hi_df)
    

if __name__ == '__main__':
    main()
