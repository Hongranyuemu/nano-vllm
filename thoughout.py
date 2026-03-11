import pandas as pd
import glob
import os

def get_peak_bandwidth_mb(csv_file, batch_size):
    try:
        df = pd.read_csv(csv_file)
        
        # === 1. 列名识别 (保持鲁棒性) ===
        col_map = {c: c for c in df.columns}
        name_col = next((c for c in df.columns if c in ['Name', 'Operation']), None)
        bytes_mb_col = next((c for c in df.columns if 'Bytes' in c and 'MB' in c), None)
        thru_col = next((c for c in df.columns if 'Throughput' in c), None) # MiB/s
        dur_col = next((c for c in df.columns if 'Duration' in c), None)

        if not name_col or not bytes_mb_col:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # === 2. 定义基准单位 (Unit Size per Batch) ===
        # 根据你的观察：
        # - Linear Input (H2D) 约为每 Batch 0.004 MB (4KB)
        # - QKV Output (D2H)   约为每 Batch 0.008 MB (8KB)
        UNIT_A_MB = 0.004
        UNIT_B_MB = 0.008
        
        # 计算当前 Batch Size 下的目标大小 (MB)
        target_A_mb = batch_size * UNIT_A_MB
        target_B_mb = batch_size * UNIT_B_MB
        
        # === 3. 筛选逻辑 (H2D + D2H) ===
        mask_memcpy = df[name_col].astype(str).str.contains('memcpy', case=False, na=False)
        mask_not_p2p = ~df[name_col].astype(str).str.contains('Device-to-Device', case=False, na=False)
        valid_ops = df[mask_memcpy & mask_not_p2p].copy()

        # === 4. 内部函数：寻找峰值带宽 (GiB/s) ===
        def find_peak_gib_s(target_mb):
            # 设定宽容度 (Tolerance)
            # 在 MB 维度下，小数值的相对误差可能较大，所以对于小包我们放宽一点
            if target_mb < 0.1:
                tolerance = 0.30 # 小包 (如 BS=1, 0.004) 给 30% 空间，防截断
            else:
                tolerance = 0.15 # 大包 给 15% 即可
            
            min_mb = target_mb * (1 - tolerance)
            max_mb = target_mb * (1 + tolerance)
            
            # 直接在原始 MB 列上搜索
            mask_size = (valid_ops[bytes_mb_col] >= min_mb) & (valid_ops[bytes_mb_col] <= max_mb)
            ops = valid_ops[mask_size]
            
            if ops.empty:
                return 0.0
            
            # 计算带宽
            if thru_col:
                # Nsys Throughput (MB/s) -> GiB/s
                # 取 Max 捕获峰值，或 nlargest(5).mean() 捕获平均
                return ops[thru_col].max() / 1024.0
            else:
                # 手动计算: MB / 1024 / Seconds
                calc = (ops[bytes_mb_col] / 1024.0) / (ops[dur_col] / 1e9)
                return calc.max()

        # 分别寻找
        bw_A_gib = find_peak_gib_s(target_A_mb)
        bw_B_gib = find_peak_gib_s(target_B_mb)
        
        # 取最大值
        best_bw_gib = max(bw_A_gib, bw_B_gib)
        
        return best_bw_gib, bw_A_gib, bw_B_gib, target_A_mb, target_B_mb

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0

def main():
    stats_dir = os.path.join("nsys_log", "stats")
    search_pattern = os.path.join(stats_dir, "bs*_trace_cuda_gpu_trace.csv")
    
    print(f"🚀 开始分析 (基于 MB 线性缩放)...")
    files = glob.glob(search_pattern)
    
    if not files:
        print("❌ 未找到文件")
        return

    results = []
    for f in files:
        try:
            filename = os.path.basename(f)
            bs = int(filename.split('_')[0].replace('bs', ''))
            best, bw_A, bw_B, t_A, t_B = get_peak_bandwidth_mb(f, bs)
            results.append((bs, best, bw_A, bw_B, t_A, t_B))
        except: continue

    results.sort(key=lambda x: x[0])
    
    # === 输出表格 ===
    print("\n" + "="*100)
    print(f"{'BS':<5} | {'Best BW':<12} | {'Target A':<12} | {'Peak BW A':<10} | {'Target B':<12} | {'Peak BW B':<10}")
    print(f"{'':<5} | {'(GiB/s)':<12} | {'(MB)':<12} | {'(GiB/s)':<10} | {'(MB)':<12} | {'(GiB/s)':<10}")
    print("-" * 100)
    
    for bs, best, val_a, val_b, ta, tb in results:
        # 格式化
        s_best = f"{best:.2f}"
        s_val_a = f"{val_a:.2f}" if val_a > 0 else "-"
        s_val_b = f"{val_b:.2f}" if val_b > 0 else "-"
        
        print(f"{bs:<5} | {s_best:<12} | {ta:<12.4f} | {s_val_a:<10} | {tb:<12.4f} | {s_val_b:<10}")
    print("="*100 + "\n")

    print("=== 📊 绘图数据 ===")
    print(f"batch_sizes = {[r[0] for r in results]}")
    print(f"bandwidths  = {[round(r[1], 2) for r in results]}")

if __name__ == "__main__":
    main()