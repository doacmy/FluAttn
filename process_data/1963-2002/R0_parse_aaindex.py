import re
import pandas as pd
import numpy as np

def parse_aaindex2(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按条目末尾的 '//' 分割
    entries = [e.strip() for e in content.split('//') if e.strip()]
    for entry in entries:
        lines = entry.splitlines()
        # 第一行 H 行，提取代码（如 TANS760101）
        m = re.match(r'^H\s+(\S+)', lines[0])
        if not m:
            continue
        code = m.group(1)

        # 找到以 'M rows' 开头的那一行
        for i, L in enumerate(lines):
            if L.startswith('M rows'):
                m_idx = i
                break
        else:
            # 如果没找到，跳过
            continue

        # 解析 rows 和 cols 标签
        mline = lines[m_idx]
        rm = re.search(r'rows\s*=\s*([A-Z\-]+)', mline)
        cm = re.search(r'cols\s*=\s*([A-Z\-]+)', mline)
        if not rm or not cm:
            continue
        rows = list(rm.group(1))
        cols = list(cm.group(1))
        n_rows, n_cols = len(rows), len(cols)

        # 收集紧接其后的所有数字行
        vals = []
        for L in lines[m_idx+1:]:
            parts = L.strip().split()
            # 匹配整数、带小数点且小数部分可为空（如 "3.", "-2.75"）
            if all(re.fullmatch(r'-?\d+(\.\d*)?', p) for p in parts):
                vals.extend(parts)
            else:
                break
        vals = [float(v) for v in vals]

        total = len(vals)
        # 判断是完整矩阵还是对称下三角
        if total == n_rows * n_cols:
            # 完整矩阵，按行优先 reshape
            mat = np.array(vals).reshape(n_rows, n_cols)
        elif n_rows == n_cols and total == n_rows * (n_rows + 1) // 2:
            # 对称下三角矩阵，补全为对称矩阵
            mat = np.zeros((n_rows, n_rows), dtype=float)
            idx = 0
            for i in range(n_rows):
                for j in range(i + 1):
                    mat[i, j] = mat[j, i] = vals[idx]
                    idx += 1
        else:
            print(f"[{code}] 格式不匹配：共读到 {total} 个数，"
                  f"期望 {n_rows*n_cols}（完整）或 {n_rows*(n_rows+1)//2}（下三角）")
            continue

        # 构造 DataFrame 并写 CSV
        df = pd.DataFrame(mat, index=rows, columns=cols)
        out_name = f'./data/prd/AAIndex/{code}.csv'
        df.to_csv(out_name, float_format='%.4f')
        print(f'Wrote matrix {code} → {out_name}')

if __name__ == '__main__':
    parse_aaindex2('./data/raw/AAIndex/aaindex2.txt')
