# 该代码未采用
# 使用低秩矩阵分解进行HI矩阵补全，再进行MDS降维
# 论文原文中使用的是Racmacs包进行抗原图谱构建和抗原距离计算
# 这里使用Python实现相同的功能，使用fancyimpute库进行矩阵低秩补全
# 使用scikit-learn的MDS进行降维
# 最终生成抗原图谱和抗原距离矩阵

import pandas as pd
import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from fancyimpute import SoftImpute
import matplotlib.pyplot as plt

hi_df = pd.read_csv("data/1968-2002.csv", index_col=0)

def convert_value(x):
    if isinstance(x, str):
        if '<' in x:
            return float(x.replace('<','')) / 2
        elif '*' in x:
            return np.nan
        else:
            return float(x)
    else:
        return x

hi_df_numeric = hi_df.applymap(convert_value)

hi_log2 = np.log2(hi_df_numeric)
c = np.nanmax(hi_log2.values)
hi_distance = c - hi_log2

hi_distance_filled = pd.DataFrame(
    SoftImpute().fit_transform(hi_distance.values),
    index=hi_distance.index,
    columns=hi_distance.columns
)

embedding = MDS(n_components=2, dissimilarity="euclidean", random_state=42)
coords = embedding.fit_transform(hi_distance_filled.values)


plt.figure(figsize=(10,8))
plt.scatter(coords[:,0], coords[:,1], c='blue')
for i, name in enumerate(hi_distance.index):
    plt.text(coords[i,0]+0.1, coords[i,1]+0.1, name, fontsize=8)

plt.title("Influenza Antigenic Map (Metric MDS)")
plt.grid(True)
plt.savefig('AntigenicMap.png')

pairwise_distance = squareform(pdist(coords, metric='euclidean'))
pairwise_df = pd.DataFrame(pairwise_distance, index=hi_distance.index, columns=hi_distance.index)
pairwise_df.to_csv("virus_pairwise_antigenic_distance.csv")
