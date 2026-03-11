import pandas as pd
import glob
import os

def probe_bs1_raw_stats():
    # 1. 设定文件路径
    file_pattern = os.path.join("nsys_log", "stats", "bs1_trace_cuda_gpu_trace.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"❌ 未找到文件: {file_pattern}")
        return

    csv_file = files[0]
    print(f"🕵️‍♂️ 正在读取原始 CSV: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # 2. 自动寻找名为 "Bytes (MB)" 的列
        # 我们寻找列名中同时包含 'Bytes' 和 'MB' 的那一列
        bytes_col = next((c for c in df.columns if 'Bytes' in c and 'MB' in c), None)
        
        # 寻找 Name 或 Operation 列
        name_col = next((c for c in df.columns if c in ['Name', 'Operation']), None)
        
        if not bytes_col:
            print(f"❌ 未找到类似 'Bytes (MB)' 的列。现有列名: {list(df.columns)}")
            return
            
        print(f"📋 锁定数据列: [{bytes_col}] (保持原始单位)")

        # 3. 筛选 PCIe 传输 (Memcpy)
        # 排除 Device-to-Device，保留 H2D 和 D2H
        if name_col:
            mask_memcpy = df[name_col].astype(str).str.contains('memcpy', case=False, na=False)
            mask_not_p2p = ~df[name_col].astype(str).str.contains('Device-to-Device', case=False, na=False)
            target = df[mask_memcpy & mask_not_p2p].copy()
        else:
            target = df.copy()

        # 4. 过滤掉 Prefill 的大包 (比如 > 1.0 MB)
        # 我们只关心 Decode 阶段的小数点后几位的小包
        target = target[target[bytes_col] < 1.0]

        # 5. 【核心】直接按原始数值分组统计
        # 统计每个原始数值出现了多少次
        stats = target[bytes_col].value_counts().reset_index()
        stats.columns = ['Raw_Value (MB)', 'Count']
        
        # 按出现频次降序排列，取前 10 名
        stats = stats.sort_values('Count', ascending=False).head(10)
        
        print("\n=== 📊 BS=1 原始数据包大小分布 (Top 10) ===")
        print(f"{'Raw Value (MB)':<20} | {'Count':<10} | {'(参考: 约等于 KiB)'}")
        print("-" * 60)
        
        for _, row in stats.iterrows():
            raw_val = row['Raw_Value (MB)']
            count = row['Count']
            
            # 仅仅为了人类阅读方便，后面加个备注，不影响原始数据
            ref_kib = raw_val * 1024.0
            
            # 标记高频包
            mark = ""
            if count > 100: mark = "👈 主力包"
            
            print(f"{raw_val:<20} | {count:<10} | {ref_kib:.2f} KiB {mark}")
            
        print("-" * 60)
        print("💡 结论分析：")
        print("   1. 请查看 'Raw Value' 列的数值。")
        print("   2. 如果显示 0.004，说明 Nsys 对 4KB 进行了取整。")
        print("   3. 如果显示 0.003906...，说明 Nsys 保留了高精度。")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    probe_bs1_raw_stats()