import matplotlib.pyplot as plt
import numpy as np

def plot_bandwidth():
    # ==========================================
    # 1. 数据填入区
    # 请将 thoughout.py 最后输出的 X 和 Y 列表复制到这里
    # ==========================================
    
    # 示例数据 (请替换为你实际跑出来的数据)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    bandwidths  = [np.float64(13.16), np.float64(16.67), np.float64(17.54), np.float64(22.73), np.float64(23.39), np.float64(24.77), np.float64(24.54), np.float64(25.06), np.float64(25.23), np.float64(25.15)]
    # ==========================================
    # 2. 绘图设置
    # ==========================================
    plt.figure(figsize=(10, 6), dpi=150)  # 设置画布大小和清晰度
    
    # 绘制折线图
    # marker='o': 圆点标记
    # linewidth=2: 线宽
    # color='#1f77b4': 经典的蓝色
    plt.plot(batch_sizes, bandwidths, marker='o', linestyle='-', linewidth=2.5, color='#1f77b4', label='Decode Bandwidth')

    # ==========================================
    # 3. 坐标轴与标签美化
    # ==========================================
    plt.title('Decode Phase Bandwidth with Batch Size', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Batch Size', fontsize=12, fontweight='bold')
    plt.ylabel('Bandwidth (GiB/s)', fontsize=12, fontweight='bold')
    
    # 设置 X 轴为 Log 刻度，这样 1, 2, 4, 8... 会均匀分布
    plt.xscale('log', base=2)
    
    # 自定义 X 轴的刻度标签 (确保显示所有 Batch Size)
    plt.xticks(batch_sizes, labels=[str(bs) for bs in batch_sizes], fontsize=10)
    plt.yticks(fontsize=10)
    
    # 开启网格 (主网格和次网格)
    plt.grid(True, which="major", ls="-", alpha=0.6)
    plt.grid(True, which="minor", ls=":", alpha=0.3)

    # ==========================================
    # 4. 标注数据点数值
    # ==========================================
    for x, y in zip(batch_sizes, bandwidths):
        # 在每个点上方显示具体的带宽数值
        plt.annotate(f"{y:.1f}", 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 8), 
                     ha='center', 
                     fontsize=9,
                     fontweight='bold',
                     color='#333333')

    # 添加图例
    plt.legend(loc='lower right', fontsize=11)
    
    # 调整布局防止标签被截断
    plt.tight_layout()

    # ==========================================
    # 5. 保存图片
    # ==========================================
    output_filename = 'bandwidth_with_bs.png'
    plt.savefig(output_filename)
    print(f"✅ 图表已保存为: {output_filename}")
    
    # 如果是在本地运行且有图形界面，可以取消注释下面这行来显示图片
    # plt.show()

if __name__ == "__main__":
    plot_bandwidth()