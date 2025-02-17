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

y_axis_data1 = [0.8640000000000001, 0.8740000000000001, 0.893, 0.904, 0.91, 0.915]
y_axis_data1_2 = [0.867, 0.877, 0.8959999999999999, 0.9059999999999999, 0.9129999999999999, 0.9179999999999999]
y_axis_data1_3 = [0.861, 0.871, 0.89, 0.902, 0.907, 0.911]
y_axis_data1_std = [np.std([y_axis_data1[i], y_axis_data1_2[i], y_axis_data1_3[i]]) for i in range(6)]

y_axis_data2 = [0.8809999999999999, 0.8865000000000001, 0.9039, 0.915, 0.9238, 0.927]
y_axis_data2_2 = [0.878, 0.883, 0.9009999999999999, 0.912, 0.9209999999999999, 0.924]
y_axis_data2_3 = [0.8740000000000001, 0.8795000000000001, 0.898, 0.909, 0.9179999999999999, 0.921]
y_axis_data2_std = [np.std([y_axis_data2[i], y_axis_data2_2[i], y_axis_data2_3[i]]) for i in range(6)]

y_axis_data3 = [5.000, 4.24, 3.8, 3.5, 2.37, 1.55]
y_axis_data3_2 = [4.497, 3.824, 3.368, 3.099, 2.024, 1.335]
y_axis_data3_3 = [4, 3.3, 2.9, 2.6, 1.78, 1.13]
y_axis_data3_std = [np.std([y_axis_data3[i], y_axis_data3_2[i], y_axis_data3_3[i]]) for i in range(6)]

y_axis_data4 = [5.141, 4.54, 4.021, 3.821, 2.656, 2.2149999999999997]
y_axis_data4_2 = [4.641, 4.040, 3.521, 3.321, 2.156, 1.515]
y_axis_data4_3 = [4.141, 3.54, 3.021, 2.821, 1.6560000000000001, 0.915]
y_axis_data4_std = [np.std([y_axis_data3[i], y_axis_data3_2[i], y_axis_data3_3[i]]) for i in range(6)]

# 创建主图 (左侧y轴，使用蓝色系)
fig, ax1 = plt.subplots()

ax1.plot(x_axis_data, y_axis_data1, 'b-', alpha=0.9, linewidth=2.5, label='$S\'\'$:RMSE')
ax1.fill_between(x_axis_data, [y_axis_data1[i] - y_axis_data1_std[i] for i in range(len(x_axis_data))], 
                 [y_axis_data1[i] + y_axis_data1_std[i] for i in range(len(x_axis_data))], color='b', alpha=0.2, edgecolor='none')

ax1.plot(x_axis_data, y_axis_data2, 'c-', alpha=0.9, linewidth=2.5, label='$\hat{Y}$:  RMSE')  # 使用较浅的蓝色
ax1.fill_between(x_axis_data, [y_axis_data2[i] - y_axis_data2_std[i] for i in range(len(x_axis_data))], 
                 [y_axis_data2[i] + y_axis_data2_std[i] for i in range(len(x_axis_data))], color='c', alpha=0.2, edgecolor='none')

# 设置左侧轴的属性
ax1.set_xlabel(r'$\gamma$', fontsize=18)
ax1.set_ylabel('RMSE', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.grid(alpha=0.35)
ax1.legend(loc="lower left", fontsize=14)

ax1.set_ylim(0.8, 0.932)

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
ax2.legend(loc="lower left", fontsize=14, bbox_to_anchor=(0.32, 0))

ax2.set_ylim(1.2, 10)
# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig(f'scatter_alpha_acc_with_right_axis.jpg', dpi=1000)
