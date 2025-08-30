import re
import json
import pandas as pd
import numpy as np

file_path = "data/raw/AAIndex/aaindex2.txt"

def parse_aaindex2():
    with open(file_path, "r") as f:
        content = f.read()
    records = content.strip().split("//\n")

    keywords = [
        r"hydrophobic", r"polarity", r"volume", r"charge", r"composition",
        r"physicochemical", r"chemical", r"distance", r"structural", r"property",
        r"hydration", r"mutation values", r"replace ability"
    ]
    pattern = re.compile("|".join(keywords), re.IGNORECASE)

    matrix_dict = {}
    for record in records:
        lines = record.splitlines()
        m = re.match(r'^H\s+(\S+)', lines[0])
        if not m:
            continue
        code = m.group(1)
        for i, L in enumerate(lines):
            if L.startswith('M rows'):
                m_idx = i
                break
        else:
            continue

        mline = lines[m_idx]
        rm = re.search(r'rows\s*=\s*([A-Z\-]+)', mline)
        cm = re.search(r'cols\s*=\s*([A-Z\-]+)', mline)
        if not rm or not cm:
            continue
        rows = list(rm.group(1))
        cols = list(cm.group(1))

        if rm.group(1) != 'ARNDCQEGHILKMFPSTWYV' or cm.group(1) != 'ARNDCQEGHILKMFPSTWYV':
            continue

        n_rows, n_cols = len(rows), len(cols)

        vals = []
        for L in lines[m_idx+1:]:
            parts = L.strip().split()
            if all(re.fullmatch(r'-?\d+(\.\d*)?', p) for p in parts):
                vals.extend(parts)
            else:
                break
        vals = [float(v) for v in vals]

        total = len(vals)
        if total == n_rows * n_cols:
            mat = np.array(vals).reshape(n_rows, n_cols)
        elif n_rows == n_cols and total == n_rows * (n_rows + 1) // 2:
            mat = np.zeros((n_rows, n_rows), dtype=float)
            idx = 0
            for i in range(n_rows):
                for j in range(i + 1):
                    mat[i, j] = mat[j, i] = vals[idx]
                    idx += 1
        else:
            continue
        
        tag = False
        for i in range(n_rows):
            if mat[i, i] != mat[0, 0]:
                tag = True
                break
        
        if tag == False:
            mat = mat - mat[0,0]
            matrix_dict[code] = mat

        






    return matrix_dict

if __name__ == '__main__':
    matrix_dict = parse_aaindex2()
    json_dict = {k: v.tolist() for k, v in matrix_dict.items()}

    with open("data/prd/aaindex2_dicts.json", "w") as f:
        json.dump(json_dict, f, indent=2)




