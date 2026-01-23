import pandas as pd
import re

"""
脚本用途：
1) 读取 2003-2025 年的 HA1 序列与 HI 数据；
2) 将具有相同 HA1 序列的毒株合并为“代表毒株”（按年份最早出现者作为代表）；
3) 在 HI 数据中把所有毒株名称映射到对应的代表毒株；
4) 按无序成对键（Test/Reference 顺序无关）聚合重复条目，对 Titre 进行安全求平均；
5) 过滤掉自反（self）比较，并输出最终的 HI 数据与名称映射文件。

说明：仅添加注释，不改变任何原有逻辑与输出。
"""

def extract_year(virus_name):
    """从毒株名称中提取年份。

    约定：毒株名以 '/YY' 或 '/YYYY' 结尾，例如 'A/xxx/07' 或 'A/xxx/2007'。
    - 若匹配 4 位年份，直接返回该年份的整数；
    - 若匹配 2 位年份，以 2000 年代补全（'07' -> 2007）。
    """
    match = re.search(r'/(\d{2,4})$', virus_name)
    year = match.group(1)
    return int(year) if len(year) == 4 else int("20" + year)

def canonical_pair(a, b):
    """返回无序成对键（按字典序排序后组成的二元组）。

    用于把 (Test, Reference) 和 (Reference, Test) 视为同一对。
    """
    return tuple(sorted([a, b]))

def convert_titre(val):
    """将浮点型整值（如 160.0）转为整型（160），保持其它类型不变。"""
    if isinstance(val, float):
        return int(val)
    return val

def safe_avg_titre(group_df):
    """对同一对病毒的多条 HI 记录进行“安全平均”。

    支持三类值混合：
    - 纯数字（如 '160'）：做算术平均；
    - 全部为 '*'：返回 '*'；
    - 全部为小于号形式（如 '<40'）：取最小阈值，返回 '<min'；
    - 数字 + '*'：忽略 '*'，对数字求平均；
    - '<N' + '*' 且无数字：取所有阈值的最小值，返回 '<min'；
    - 数字 + '<N' + '*'：忽略非数字项，仅对数字求平均；
    其余未覆盖的组合视为非法并抛出异常，便于追踪数据问题。
    """
    titres = group_df['Titre'].astype(str).str.strip()

    # 分类掩码：纯数字、星号、以及形如 "<数字" 的阈值
    numeric_mask = titres.str.isnumeric()
    star_mask = titres == "*"
    less_than_mask = titres.str.match(r"<\d+")

    num_numeric = numeric_mask.sum()
    num_star = star_mask.sum()
    num_less = less_than_mask.sum()
    total = len(titres)

    # 情形 1：全为数字 -> 直接均值
    if num_numeric == total:
        return float(titres.astype(float).mean())
    # 情形 2：全为 '*' -> 返回 '*'
    if num_star == total:
        return "*"
    # 情形 3：全为 '<N' -> 取最小阈值
    if num_less == total:
        values = [int(t[1:]) for t in titres]
        return f"<{min(values)}"
    # 情形 4：数字 + '*' -> 仅对数字求均值
    if num_numeric + num_star == total:
        numeric_values = titres[numeric_mask].astype(float)
        return float(numeric_values.mean())
    # 情形 5：'<N' + '*' 且无数字 -> 取最小阈值
    if num_less + num_star == total and num_numeric == 0:
        values = [int(t[1:]) for t in titres[less_than_mask]]
        return f"<{min(values)}"
    # 情形 6：混合（含数字）-> 忽略非数字，仅对数字求均值
    if num_numeric > 0 and num_less + num_star + num_numeric == total:
        numeric_values = titres[numeric_mask].astype(float)
        return float(numeric_values.mean())

    # 其它情况：抛出异常，便于排查
    raise ValueError(f"非法的 Titre 组合: {titres.tolist()}")

# 读取序列表，并从毒株名中提取年份
seq_df = pd.read_csv('data/prd/2003-2025/sequences.csv')
seq_df['Year'] = seq_df['Virus'].apply(extract_year)

##
## 基于年份选择“代表毒株”：
## - 先按 Year 升序排序；
## - 对相同的 HA1_Sequence 仅保留第一条（最早年份的毒株）；
## - 构建序列到代表毒株名的映射。
seq_df_sorted = seq_df.sort_values(by='Year', ascending=True)
seq_rep_df = seq_df_sorted.drop_duplicates(subset='HA1_Sequence', keep='first')
sequence_to_rep = dict(zip(seq_rep_df['HA1_Sequence'], seq_rep_df['Virus']))

## 原始毒株名 -> 序列 -> 代表毒株名 的两级映射
virus_to_seq = dict(zip(seq_df['Virus'], seq_df['HA1_Sequence']))
virus_to_rep = {virus: sequence_to_rep[seq] for virus, seq in virus_to_seq.items()}

## 读取 HI 数据，仅保留两端毒株都在序列表中的记录
df = pd.read_csv('data/prd/2003-2025/2003-2025_filter.csv', dtype=str)
valid_viruses = set(seq_df['Virus'])
mask = df['Test Virus'].isin(valid_viruses) & df['Reference Virus'].isin(valid_viruses)
filtered_df = df[mask].copy()

## 将 Test/Reference 毒株名映射为对应的代表毒株名
filtered_df['Test Virus'] = filtered_df['Test Virus'].map(virus_to_rep)
filtered_df['Reference Virus'] = filtered_df['Reference Virus'].map(virus_to_rep)

## 生成无序成对键，保证 (A,B) 与 (B,A) 被视为同一对
filtered_df['Pair Key'] = filtered_df.apply(
    lambda row: tuple(sorted([row['Test Virus'].strip(), row['Reference Virus'].strip()])),
    axis=1
)

grouped = filtered_df.groupby('Pair Key')
result_rows = []

## 对每个无序成对键聚合 Titre；对异常组合打印提示
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



## 转换 Titre 为整型（若为如 160.0 的浮点整值）；过滤掉自反比较
filtered_df['Titre'] = filtered_df['Titre'].apply(convert_titre)
filtered_df = filtered_df[filtered_df['Test Virus'] != filtered_df['Reference Virus']]

## 输出最终聚合后的 HI 数据
filtered_df.to_csv('data/prd/2003-2025/2003-2025_final_HI.csv', index=False)

## 输出原始毒株名 -> 代表毒株名 的映射表，便于溯源
mapping_df = pd.DataFrame(list(virus_to_rep.items()), columns=['Original Virus', 'Representative Virus'])
mapping_df.to_csv('data/prd/2003-2025/virus_name_mapping_by_year.csv', index=False)
