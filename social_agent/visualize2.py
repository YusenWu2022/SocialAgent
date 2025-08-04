import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 数据
labels=np.array(['# Clt 1', '# Clt 2', '# Clt 3', '# Clt 4', '# Clt 5', '# Clt 6', '# Clt 7'])
stats=np.array([0.18, 0.19, 0.16, 0.11, 0.12, 0.09, 0.15])

# 计算每个维度的角度
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()

# 闭合雷达图
stats=np.concatenate((stats,[stats[0]]))
angles+=angles[:1]

# 绘图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, stats, color='red', alpha=0.25)
ax.plot(angles, stats, color='red', linewidth=2)  # 绘制线

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)

count = 0
for label, angle in zip(ax.get_xticklabels(), angles):
    if count == 3 or count==4:
        label.set_horizontalalignment('left')
        label.set_position((angle, -0.2))
    else:
        label.set_horizontalalignment('left')
        label.set_position((angle, 0.0))
    count += 1


plt.title('Distribution of Sample Nums in 7 Clusters',fontsize=20)
# plt.spines['top'].set_linewidth(2)
# plt.legend(fontsize=16)
plt.savefig("/root/pku/yusen/social_agent/cluster_hsii2.pdf")