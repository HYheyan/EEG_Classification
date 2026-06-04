"""
将 data/train/ 中第一个 .npy 文件转换为可读的 txt 格式。

EEG 数据形状: (1, 59, 282) → 1 通道, 59 电极, 282 时间点
输出格式: 每行一个电极, 282 个时间点的值以空格分隔。
"""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
OUTPUT_FILE = PROJECT_ROOT / "eeg_sample.txt"


def main() -> None:
    # ---- 找到第一个 .npy 文件 ----
    npy_files = sorted(TRAIN_DIR.glob("*.npy"))
    if not npy_files:
        print(f"错误: {TRAIN_DIR} 中没有找到 .npy 文件")
        return

    npy_path = npy_files[0]
    print(f"读取: {npy_path.name}")

    # ---- 加载数据 ----
    eeg = np.load(npy_path).astype(np.float32)  # shape: (1, 59, 282)
    print(f"形状: {eeg.shape}  (通道, 电极, 时间点)")

    # 去掉第一维 (通道), 得到 (59, 282)
    data = eeg[0]  # (59, 282)

    # ---- 写入 txt ----
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 文件头
        f.write(f"# 文件: {npy_path.name}\n")
        f.write(f"# 形状: {eeg.shape}  (1=通道, 59=电极, 282=时间点)\n")
        f.write(f"# 电极数: {data.shape[0]}\n")
        f.write(f"# 时间点数: {data.shape[1]}\n")
        f.write(f"# 每行 = 一个电极, 每列 = 一个时间点\n")
        f.write(f"#\n")
        # 列头: 时间点编号
        f.write("# Electrode\\Time")
        for t in range(data.shape[1]):
            f.write(f"\tT{t+1:04d}")
        f.write("\n")

        # 数据行
        for elec_idx in range(data.shape[0]):
            f.write(f"E{elec_idx+1:02d}")
            for t in range(data.shape[1]):
                f.write(f"\t{data[elec_idx, t]:.6f}")
            f.write("\n")

    print(f"已保存: {OUTPUT_FILE}")
    print(f"文件大小: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
