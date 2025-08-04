import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import json
import numpy as np

corpus = []

with open('/root/pku/yusen/social_agent/data/records.jsonl', 'r', encoding='utf-8') as file:
    for _ in range(7000):
        line = file.readline()
        data = json.loads(line)
        corpus.append(json.dumps(data))
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# 2. 聚类分析
kmeans = KMeans(n_clusters=7)
X_clustered = kmeans.fit_predict(X)

# 3. 降维到二维空间
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 4. 颜色加权计算
# 归一化聚类中心
pca_centers = pca.transform(kmeans.cluster_centers_)
norm = Normalizer()
pca_centers = norm.fit_transform(pca_centers)

# 计算每个点到其聚类中心的距离
distances = []
for i in range(len(X_pca)):
    cluster_center = pca_centers[kmeans.labels_[i]]
    distance = np.linalg.norm(X_pca[i] - cluster_center)
    distances.append(distance)

# 计算每个聚类的平均距离
# cluster_distances = [np.mean(distances[X_clustered == i]) for i in range(kmeans.n_clusters)]

# 创建一个颜色映射
colors = plt.cm.Spectral(np.linspace(0, 1, kmeans.n_clusters))

# 绘制散点图
plt.figure(figsize=(12, 8))

for i in range(kmeans.n_clusters):
    points = X_pca[X_clustered == i]
    color = colors[i]  # 使用颜色映射中的颜色
    plt.scatter(points[:, 0], points[:, 1], color=color, label=f'# Task cluster {i}')

plt.title('2D visualization of clustered\n HSII social task scenes',fontsize=24)
# plt.spines['top'].set_linewidth(2)
plt.xlabel('# Feature Dim 1',fontsize=24)
plt.ylabel('# Feature Dim 2',fontsize=24)
plt.tick_params(axis='x', labelsize=15)  # 设置x轴刻度标签字体大小为20
plt.tick_params(axis='y', labelsize=15)  # 设置y轴刻度标签字体大小为20
plt.legend(fontsize=16)
plt.savefig("/root/pku/yusen/social_agent/cluster_hsii1.pdf")