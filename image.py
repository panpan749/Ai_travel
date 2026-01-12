import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
# 查找系统中包含 "Sim" 或 "Song" 关键字的可用字体名
print([f.name for f in fm.fontManager.ttflist if 'Sim' in f.name or 'Song' in f.name])
# ===========================
# 1. 全局绘图设置
# ===========================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'FangSong', 'STSong']
zh_font = fm.FontProperties(family='SimSun', size=12)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.unicode_minus'] = False
# ===========================
# 2. 数据准备
# ===========================
groups = ['Deepseek-V3.2', 'Deepseek-V3.2-Thinking']
methods = ['Baseline', 'Our Method']
metrics = ['P50', 'P95', 'Average']

data = {
    'Deepseek-V3.2': {
        'Baseline': [3.39, 11.79, 4.73],
        'Our Method': [1.40, 6.05, 3.46]
    },
    'Deepseek-V3.2-Thinking': {
        'Baseline': [3.00, 10.02, 4.06],
        'Our Method': [1.46, 23.42, 4.24]
    }
}

# ===========================
# 3. 样式定义
# ===========================
colors = {
    'Baseline': '#E0E0E0', 
    'Our Method': '#85C1E9'
}
edge_color = '#333333'

hatches = {
    'P50': '', 
    'P95': '////', 
    'Average': '...'
}

# ===========================
# 4. 绘图参数计算
# ===========================
x = np.arange(len(groups))
n_bars = len(methods) * len(metrics)
bar_width = 0.10  
gap_between_methods = 0.04 

offsets = []
half_gap = gap_between_methods / 2
for i in range(n_bars):
    if i < 3:
        pos = (i - 2.5) * bar_width - half_gap
    else:
        pos = (i - 2.5) * bar_width + half_gap
    offsets.append(pos)

# ===========================
# 5. 开始绘图
# ===========================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

bar_containers = []
bar_idx = 0
for method_idx, method in enumerate(methods):
    for metric_idx, metric in enumerate(metrics):
        values = [data[group][method][metric_idx] for group in groups]
        current_x = x + offsets[bar_idx]
        
        rects = ax.bar(current_x, values, bar_width,
                       color=colors[method],
                       hatch=hatches[metric],
                       edgecolor=edge_color,
                       linewidth=0.8,
                       zorder=3)
        bar_containers.append(rects)
        bar_idx += 1

# ===========================
# 6. 细节美化
# ===========================
for rects in bar_containers:
    ax.bar_label(rects, fmt='%.2f', padding=2, fontsize=8.5, color='#333333', fontname='Times New Roman')

ax.set_ylabel('Generate Time (s)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=13, fontweight='bold')
ax.set_ylim(0, 30)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)
ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)

# ===========================
# 7. 图例设置
# ===========================
# 左上角颜色图例
color_patches = [
    mpatches.Patch(facecolor=colors['Baseline'], edgecolor=edge_color, label='Baseline (Few-shot)'),
    mpatches.Patch(facecolor=colors['Our Method'], edgecolor=edge_color, label='Our Method')
]
leg1 = ax.legend(handles=color_patches, loc='upper left', 
                 bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=11)
ax.add_artist(leg1)

# 底部纹理图例
hatch_patches = []
for metric in metrics:
    p = mpatches.Patch(facecolor='white', edgecolor=edge_color, hatch=hatches[metric], label=metric)
    hatch_patches.append(p)

leg2 = ax.legend(handles=hatch_patches, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.12), # 稍微往下一点，给X轴标签留空
                 ncol=3, frameon=False, fontsize=11)

# ===========================
# 8. 添加底部标题 (Caption)
# ===========================
# 使用 fig.text 在整个画布坐标系上书写
# x=0.5 (水平居中), y=0.02 (底部边缘), ha='center' (对齐方式)
caption_text = "Figure 2: 基于Baseline(few shot)与我们的方法生成的代码在Mock数据上的生成时间\n对比，其中每个question的输入数据实体总数在100到200之间"

fig.text(0.5, 0.02, caption_text, ha='center', fontsize=12, fontproperties=zh_font)

# ===========================
# 9. 布局调整与保存
# ===========================
# 增加 bottom 的值，给图例和 Caption 留出足够的白色空间
plt.subplots_adjust(bottom=0.22) 

plt.savefig('academic_chart_with_caption.png', dpi=300, bbox_inches='tight') # bbox_inches='tight' 会自动包裹所有内容
# plt.show()