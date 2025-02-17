import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
from matplotlib import font_manager

plt.style.use('seaborn-v0_8-white')
font_path = 'times.ttf'
font_manager.fontManager.addfont(font_path)

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# epoch, acc, loss, val_acc, val_loss
x_axis_data = [1, 1.2, 1.4, 1.6, 1.8, 2]

y_axis_data1 = [0.776, 0.76, 0.758, 0.754, 0.751, 0.747]
y_axis_data1_2 = [0.766, 0.765, 0.764, 0.759, 0.757, 0.753]
y_axis_data1_3 = [0.756, 0.755, 0.752, 0.749, 0.745, 0.741]
y_axis_data1_std = [np.std([y_axis_data1[i], y_axis_data1_2[i], y_axis_data1_3[i]]) for i in range(6)]

y_axis_data2 = [0.7665, 0.7518, 0.7461, 0.7431, 0.7418, 0.7398]
y_axis_data2_2 = [0.754, 0.7525, 0.750, 0.748, 0.747, 0.734]
y_axis_data2_3 = [0.7615, 0.7432, 0.7439, 0.7409, 0.7422, 0.7282]
y_axis_data2_std = [np.std([y_axis_data2[i], y_axis_data2_2[i], y_axis_data2_3[i]]) for i in range(6)]

y_axis_data3 = [3.435, 2.858, 2.531, 2.451, 2.178, 1.799]
y_axis_data3_2 = [3.806, 3.182, 2.738, 2.660, 2.163, 1.873]
y_axis_data3_3 = [3.364, 2.834, 2.624, 2.342, 1.893, 1.725]
y_axis_data3_std = [np.std([y_axis_data3[i], y_axis_data3_2[i], y_axis_data3_3[i]]) for i in range(6)]

y_axis_data4 = [3.673, 3.032, 2.864, 2.612, 2.256, 1.868]
y_axis_data4_2 = [3.981, 3.327, 3.159, 2.905, 2.552, 2.175]
y_axis_data4_3 = [3.365, 2.737, 2.569, 2.319, 1.960, 1.561]
y_axis_data4_std = [np.std([y_axis_data3[i], y_axis_data3_2[i], y_axis_data3_3[i]]) for i in range(6)]

# 创建主图 (左侧y轴，使用蓝色系)
fig, ax1 = plt.subplots()

ax1.plot(x_axis_data, y_axis_data1, 'b-', alpha=0.9, linewidth=2.5, label='$S\'\'$:Accuracy')
ax1.fill_between(x_axis_data, [y_axis_data1[i] - y_axis_data1_std[i] for i in range(len(x_axis_data))], 
                 [y_axis_data1[i] + y_axis_data1_std[i] for i in range(len(x_axis_data))], color='b', alpha=0.2, edgecolor='none')

ax1.plot(x_axis_data, y_axis_data2, 'c-', alpha=0.9, linewidth=2.5, label='$\hat{Y}$:  Accuracy')  # 使用较浅的蓝色
ax1.fill_between(x_axis_data, [y_axis_data2[i] - y_axis_data2_std[i] for i in range(len(x_axis_data))], 
                 [y_axis_data2[i] + y_axis_data2_std[i] for i in range(len(x_axis_data))], color='c', alpha=0.2, edgecolor='none')

# 设置左侧轴的属性
ax1.set_xlabel(r'$\gamma$', fontsize=18)
ax1.set_ylabel('Accuracy', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.grid(alpha=0.35)
ax1.legend(loc="lower left", fontsize=14)

ax1.set_ylim(0.705, 0.785)

# 创建第二个Y轴 (右侧y轴，使用红色系)
ax2 = ax1.twinx()

ax2.plot(x_axis_data, y_axis_data3, 'r-', alpha=0.9, linewidth=2.5, label='$S\'\'$:MMD')  # 使用深红色
ax2.fill_between(x_axis_data, [y_axis_data3[i] - y_axis_data3_std[i] for i in range(len(x_axis_data))], 
                 [y_axis_data3[i] + y_axis_data3_std[i] for i in range(len(x_axis_data))], color='r', alpha=0.2, edgecolor='none')

ax2.plot(x_axis_data, y_axis_data4, 'orange', alpha=0.9, linewidth=2.5, label='$\hat{Y}$:  MMD')  # 使用浅红色（橙色）
ax2.fill_between(x_axis_data, [y_axis_data4[i] - y_axis_data4_std[i] for i in range(len(x_axis_data))], 
                 [y_axis_data4[i] + y_axis_data4_std[i] for i in range(len(x_axis_data))], color='orange', alpha=0.2, edgecolor='none')

ax2.set_ylabel('MMD', fontsize=18)  # 设置右侧Y轴的标签
ax2.tick_params(axis='y', labelsize=16)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax2.legend(loc="lower left", fontsize=14, bbox_to_anchor=(0.35, 0))

ax2.set_ylim(1.5, 6)
# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig(f'scatter_alpha_acc_with_right_axis_adult.jpg', dpi=1000)
