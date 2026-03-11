import matplotlib
matplotlib.use('Agg') # Linux服务器必备

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==========================================
# 0. 基础配置
# ==========================================
plt.rcParams['font.family'] = 'DejaVu Sans'

# ==========================================
# 1. 数据准备 (已加入吞吐量/带宽数据)
# ==========================================
data = [
    # --- QKV Linear 部分 ---
    {"name": "Sample Noise",    "time": 2.074, "type": "cpu"},
    {"name": "Linear Enc",      "time": 0.032, "type": "cpu"},
    
    # 增加 bw 字段
    {"name": "H2D (QKV)",       "time": 1.590, "type": "h2d", "bw": "4.6 GiB/s"},
    
    {"name": "Gen QKV",         "time": 0.072, "type": "gpu"},
    
    # 增加 bw 字段
    {"name": "D2H (QKV)",       "time": 2.199, "type": "d2h", "bw": "5.8 GiB/s"},
    
    # --- 连续微小步骤 (4阶梯) ---
    {"name": "Split & Norm",    "time": 0.178, "type": "cpu"}, 
    {"name": "Decryption",      "time": 0.032, "type": "cpu"}, 
    {"name": "RoPE",            "time": 0.073, "type": "cpu"}, 
    {"name": "Ortho Enc",       "time": 0.134, "type": "cpu"}, 
    
    # --- 瓶颈与核心 ---
    # 增加 bw 字段 (详细列出 Q/K/V)
    {"name": "H2D (Attn)",      "time": 6.358, "type": "h2d", "bw": "Q:7.9 K:4.6 V:5.0 GiB/s"},
    
    {"name": "Flash Attn",      "time": 0.223, "type": "gpu"}, 
    
    # 增加 bw 字段
    {"name": "D2H (Output)",    "time": 2.055, "type": "d2h", "bw": "3.6 GiB/s"},
    
    {"name": "Out Proj",        "time": 0.160, "type": "cpu"},
    
    # 补齐误差
    {"name": "",                "time": 0.095, "type": "cpu"}, 
]

# 计算总时间
total_time = sum(item["time"] for item in data)

# ==========================================
# 2. 配色方案
# ==========================================
colors = {
    "cpu": "#4E79A7",  
    "gpu": "#59A14F",  
    "h2d": "#F28E2B",  
    "d2h": "#E15759"   
}

fig, ax = plt.subplots(figsize=(18, 7)) 
bar_height = 0.5
y_pos = 1
current_x = 0
label_threshold = 3.5 

# ==========================================
# 3. 绘图循环
# ==========================================
label_offsets = [0.4, 0.8, 1.2, 1.6] 
offset_index = 0

for i, item in enumerate(data):
    # 跳过补齐块
    if item["name"] == "":
        current_x += (item["time"] / total_time) * 100
        continue

    duration = item["time"]
    percent = (duration / total_time) * 100
    
    color = colors.get(item["type"], "#999999")
    
    # 画方块
    rect = patches.Rectangle(
        (current_x, y_pos - bar_height/2), 
        percent, 
        bar_height,
        linewidth=0.5, 
        edgecolor='white', 
        facecolor=color
    )
    ax.add_patch(rect)
    
    center_x = current_x + percent / 2
    
    # 画标签
    if percent >= label_threshold:
        # 大块：写在内部
        # 【关键修改】：如果存在 bw 字段，则加在第三行
        label_text = f"{item['name']}\n{percent:.1f}%"
        if "bw" in item:
            label_text += f"\n({item['bw']})"
            
        ax.text(
            center_x, y_pos, 
            label_text, 
            ha='center', va='center', 
            fontsize=8.5, color='white', fontweight='bold' # 稍微调小一点字体以容纳三行
        )
        offset_index = 0
    else:
        # 小块：写在外部 (这里通常没有带宽数据，保持原样)
        y_offset = label_offsets[offset_index % 4]
        text_y = y_pos + bar_height/2 + y_offset
        
        ax.annotate(
            f"{item['name']}\n({percent:.1f}%)",
            xy=(center_x, y_pos + bar_height/2), 
            xytext=(center_x, text_y),           
            arrowprops=dict(arrowstyle="->", color='black', lw=0.6),
            ha='center', fontsize=8, color='black'
        )
        offset_index += 1

    current_x += percent

# ==========================================
# 4. 坐标轴与图例
# ==========================================
ax.set_xlim(0, 100)
ax.set_ylim(0, 3.5) 
ax.set_yticks([])

ax.set_xlabel("Percentage of Total Latency (%)", fontsize=12)
ax.set_xticks(np.arange(0, 101, 10))
ax.grid(axis='x', linestyle='--', alpha=0.5)

for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

plt.title(f"Normalized Timeline with Throughput (Total: {total_time:.2f} ms)", fontsize=14, pad=20)

legend_elements = [
    patches.Patch(color=colors['cpu'], label='CPU Operation'),
    patches.Patch(color=colors['gpu'], label='GPU Operation'),
    patches.Patch(color=colors['h2d'], label='Host to Device (H2D)'),
    patches.Patch(color=colors['d2h'], label='Device to Host (D2H)'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4, frameon=False)

# ==========================================
# 5. 保存
# ==========================================
output_filename = 'timeline_base.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Chart with bandwidth saved to: {output_filename}")