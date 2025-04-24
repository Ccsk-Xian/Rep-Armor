# import matplotlib.pyplot as plt
# import numpy as np

# # 示例数据
# x = np.linspace(1, 100, 100)
# y = np.sin(x)

# # 设置圆圈的大小，x 轴值越大，圆圈越大
# sizes = x * 10  # 这里可以调整倍数以达到所需的效果

# # 创建图表
# plt.scatter(x, y, s=sizes, facecolors='none', edgecolors='b', label='Data Points')

# # 添加标签和标题
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.title('Scatter Plot with Circle Size Based on X axis Value')
# plt.legend()

# # 显示图表
# plt.show()[]

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm

# # 设置均值和标准差
# mu1, sigma1 = -5, 1  # 第一个正态分布的均值和标准差
# mu2, sigma2 = 5, 1   # 第二个正态分布的均值和标准差

# # 生成 x 轴的数据范围
# x = np.linspace(-10, 10, 1000)

# # 计算每个正态分布的概率密度函数值
# f1 = norm.pdf(x, mu1, sigma1)
# f2 = norm.pdf(x, mu2, sigma2)
# f3 = norm.pdf(x, 0, sigma2)
# f4 = norm.pdf(x, 2, sigma2)

# # 计算两个正态分布的概率密度函数相加后的结果
# f_sum = f1 + f2+f3+f4

# # 绘制图像
# plt.figure(figsize=(10, 6))

# # 绘制第一个正态分布的概率密度函数
# # plt.plot(x, f1, label=r'$f_1(x)$, $\mu_1=-5, \sigma_1=1$', color='blue')

# # 绘制第二个正态分布的概率密度函数
# # plt.plot(x, f2, label=r'$f_2(x)$, $\mu_2=5, \sigma_2=1$', color='red')

# # 绘制两个正态分布相加后的概率密度函数
# plt.plot(x, f_sum, label=r'$f_1(x) + f_2(x)$', color='green', linestyle='dashed')

# # 添加图例
# plt.legend()

# # 设置标题和标签
# plt.title('Sum of Two Normal Distributions', fontsize=14)
# plt.xlabel('x', fontsize=12)
# plt.ylabel('Probability Density', fontsize=12)

# # 显示图像
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('z.png')
# plt.show()

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
def generate_coordinates(values):
    """
    生成二维坐标对，i 和 o 分别对应 x 和 y 坐标。
    
    参数：
    values (list or tensor): 包含多个数的列表，例如 [-1, 0, 1]
    
    返回：
    tuple: 返回生成的 i 和 o 张量
    """
    # 创建 i 和 o 坐标
    i = torch.tensor([x for x in values for _ in values]).to(torch.device('cpu'))  # 生成 x 坐标
    o = torch.tensor([y for _ in values for y in values]).to(torch.device('cpu'))  # 生成 y 坐标


    return i, o

# 参数设定
step=3
KH, KW = 3, 3  # 网格大小 (3x3)
# sigma_value = torch.ones((step**2), dtype=torch.float32) * 1.0  # sigma设置为5
# sigma_value = torch.rand(step**2) * 4 + 1
# sigma_value = torch.tensor([0.2561, 0.2984, 0.2521, 0.1921, 0.2748, 0.1874, 0.2525, 0.2981, 0.2492])
sigma_value = torch.ones((step**2), dtype=torch.float32) * 5.0
print(sigma_value)
circle = True  # 是否启用圆形限制

# 创建网格坐标
grid_x = torch.linspace(-(KW - 1) / 2., (KW - 1) / 2., step).to(torch.device('cpu'))
grid_y = torch.linspace(-(KH - 1) / 2., (KH - 1) / 2., step).to(torch.device('cpu'))
# half_KW = (KW - 1) / 2.0
# half_KH = (KH - 1) / 2.0

# # 生成随机数并映射到 [-half, half] 范围
# grid_x = (torch.rand(step) * (half_KW * 2)) - half_KW
# grid_y = (torch.rand(step) * (half_KH * 2)) - half_KH

# 创建网格（meshgrid）
grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy')

# 创建偏移量 i 和 o
value = torch.linspace(-1,1,steps=step)
i,o = generate_coordinates(values=value)

# i = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).to(torch.device('cpu'))
# o = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).to(torch.device('cpu'))

# 将 i 和 o 重新排列为 (KH*KW, 1, 1)
i = i.view(len(i), 1, 1)
o = o.view(len(o), 1, 1)

# 将网格坐标扩展为 (KH*KW, -1, -1) 的大小
grid_x = grid_x.expand(len(i), -1, -1)
grid_y = grid_y.expand(len(o), -1, -1)

# 初始化 b
# weight = torch.ones(len(i))
# weight = torch.rand(len(i))
# weight = torch.tensor([0.05,0.05,0.05,0.05,0.55,0.05,0.05,0.05,0.05])
weight = torch.ones([9])
b = torch.softmax(weight, dim=0)  # 对权重进行 softmax 归一化
b = b.view(len(i), 1, 1)
print(b)

# 设置 sigma
sigma = sigma_value.view(len(i), 1, 1)

# 如果circle为True，则对sigma进行调整
if circle:
    sigma = sigma.view(len(i), 1, 1)
print(b)
# 计算高斯分布

gaussian = torch.exp(-0.5 * ((grid_x - i)**2 + (grid_y - o)**2) / sigma**2) * b

# 求和
gaussian = gaussian.sum(dim=0)
print(gaussian.shape)
# 可视化高斯分布 (三维图)
gaussian_image = gaussian.squeeze().cpu().numpy()

# 创建 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 网格坐标转换为 X, Y, Z
X, Y = np.meshgrid(np.linspace(-1, 1, step), np.linspace(-1, 1, step))
Z = gaussian_image
dense_points = 1000  # 插值后的密集点数
xi = np.linspace(-1, 1, dense_points)
yi = np.linspace(-1, 1, dense_points)
xi, yi = np.meshgrid(xi, yi)

# 使用插值函数生成平滑数据
zi = griddata((X.flatten(), Y.flatten()), Z.flatten(), (xi, yi), method='cubic')
# print(X.shape)
# print(Y.shape)
# print(Z.shape)
# 绘制三维表面图
print(len(xi))
print(len(yi))
print(len(zi))
ax.plot_surface(xi, yi, zi, cmap='hot', edgecolor='none')

# 设置标题和轴标签
ax.set_title(f"Gaussian Distribution with sigma=5")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Density")

# 显示颜色条
fig.colorbar(ax.plot_surface(xi, yi, zi, cmap='hot', edgecolor='none'))


plt.savefig('inial.png')
plt.show()