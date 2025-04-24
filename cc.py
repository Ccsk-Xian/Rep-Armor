import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 定义两个正态分布
x = np.linspace(-10, 10, 500)
mu1, sigma1 = -3, 1
mu2, sigma2 = 3, 1

f1 = norm.pdf(x, mu1, sigma1)  # 第一个正态分布
f2 = norm.pdf(x, mu2, sigma2)  # 第二个正态分布

# 添加权重
w1 = 1  # 第一个正态分布的权重
w2 =  1 # 第二个正态分布的权重

# 未归一化的相加结果
f_weighted_sum = w1 * f1 + w2 * f2

# 归一化
normalization_factor = np.trapz(f_weighted_sum, x)  # 计算积分（总面积）
f_normalized = f_weighted_sum / normalization_factor
print(normalization_factor)
print(np.trapz(f_normalized,x))
# # 绘图
plt.plot(x, f1, label='Normal Distribution 1')
plt.plot(x, f2, label='Normal Distribution 2')
plt.plot(x, f_weighted_sum, label='Weighted Sum (Unnormalized)', linestyle='dashed')
plt.plot(x, f_normalized, label='Weighted Sum (Normalized)', linestyle='dashdot')
plt.legend()
plt.title("Weighted Sum of Two Normal PDFs with Normalization")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.savefig("x.png")
plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle

# # 创建高斯掩码
# def gaussian_mask(size):
#     x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
#     d = np.sqrt(x*x + y*y)
#     sigma, mu = 0.5, 0.0
#     mask = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))
#     return mask

# # 创建四周值高、中间值低的掩码
# def inverse_gaussian_mask(size):
#     gaussian = gaussian_mask(size)
#     return 1 - gaussian

# # 设置掩码大小
# mask_size = 9

# # 生成掩码
# mask_gaussian = gaussian_mask(mask_size)
# mask_inverse = inverse_gaussian_mask(mask_size)

# # 绘图
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# # 左图：重参数化模块结构
# axes[0].add_patch(Rectangle((0.2, 0.4), 0.6, 0.2, edgecolor='black', facecolor='lightgreen'))
# axes[0].add_patch(Rectangle((0.2, 0.6), 0.6, 0.2, edgecolor='black', facecolor='orange'))
# axes[0].add_patch(Rectangle((0.2, 0.8), 0.6, 0.2, edgecolor='black', facecolor='gold'))
# axes[0].arrow(0.5, 0.2, 0, 0.2, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
# axes[0].text(0.5, 0.05, 'Input', ha='center', va='center')
# axes[0].text(0.5, 1.05, 'Output', ha='center', va='center')
# axes[0].axis('off')
# axes[0].set_title('(a) Structural Reparameterization')

# # 中图：高斯掩码
# im1 = axes[1].imshow(mask_gaussian, cmap='viridis', interpolation='nearest')
# axes[1].set_title('(b) Gaussian Mask')
# fig.colorbar(im1, ax=axes[1])

# # 右图：四周值高、中间值低的掩码
# im2 = axes[2].imshow(mask_inverse, cmap='viridis', interpolation='nearest')
# axes[2].set_title('(c) Inverse Gaussian Mask')
# fig.colorbar(im2, ax=axes[2])

# plt.tight_layout()
# plt.savefig("r.png")
# plt.show()
