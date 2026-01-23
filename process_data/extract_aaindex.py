import re
import os
import json
import numpy as np


def extract_aaindex_1(file_path):

    def parse_i_record_to_dict(lines):
        aa_pairs = ["A/L", "R/K", "N/M", "D/F", "C/P", "Q/S", "E/T", "G/W", "H/Y", "I/V"]
        aa_list = [aa for pair in aa_pairs for aa in pair.split('/')]

        values = []
        for line in lines:
            if 'NA' in line:
                return None
            values += [float(val) for val in line.strip().split()]

        if len(aa_list) != len(values):
            return None

        return dict(zip(aa_list, values))

    all_dicts = {}
    with open(file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("H "):
            record_id = line.strip().split()[1]
        if line.startswith("I"):
            i_values = []
            i += 1
            while i < len(lines) and not lines[i].startswith("//"):
                i_values.append(lines[i])
                i += 1
            parsed = parse_i_record_to_dict(i_values)
            if parsed:
                all_dicts[record_id] = parsed
        else:
            i += 1
    
    with open(f"{output_dir}/aaindex1_dicts.json", "w") as json_file:
        json.dump(all_dicts, json_file, indent=4)


def extract_aaindex_2(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    records = content.strip().split("//\n")

    canonical = 'ARNDCQEGHILKMFPSTWYV'

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
        rows_str = rm.group(1)
        cols_str = cm.group(1)

        has_gap = False
        if rows_str == canonical and cols_str == canonical:
            pass
        elif rows_str == '-' + canonical and cols_str == '-' + canonical:
            has_gap = True
        else:
            continue

        rows = list(rows_str)
        cols = list(cols_str)

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
            mat_full = np.array(vals).reshape(n_rows, n_cols)
        elif n_rows == n_cols and total == n_rows * (n_rows + 1) // 2:
            mat_full = np.zeros((n_rows, n_rows), dtype=float)
            idx = 0
            for i in range(n_rows):
                for j in range(i + 1):
                    mat_full[i, j] = mat_full[j, i] = vals[idx]
                    idx += 1
        else:
            continue

        if has_gap:
            mat = mat_full[1:, 1:]
        else:
            mat = mat_full

        matrix_dict[code] = mat.tolist()

    with open(f"{output_dir}/aaindex2_dicts.json", "w") as f:
        json.dump(matrix_dict, f, indent=2)


def extract_aaindex_3(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    records = content.strip().split("//\n")

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

        matrix_dict[code] = mat.tolist()

    with open(f"{output_dir}/aaindex3_dicts.json", "w") as f:
        json.dump(matrix_dict, f, indent=2)

if __name__ == "__main__":

    aaindex_dir = 'data/raw/AAIndex'
    output_dir = 'data/prd/AAIndex'

    os.makedirs(output_dir, exist_ok=True)

    extract_aaindex_1(f'{aaindex_dir}/aaindex1.txt')
    extract_aaindex_2(f'{aaindex_dir}/aaindex2.txt')
    extract_aaindex_3(f'{aaindex_dir}/aaindex3.txt')
