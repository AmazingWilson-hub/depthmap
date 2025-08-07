import numpy as np
import time
from scipy.ndimage import generic_filter
import cv2

# 模擬一張有洞的深度圖
H, W = 256, 512
np.random.seed(42)
depth_map = np.random.rand(H, W).astype(np.float32)
depth_map[depth_map < 0.6] = 0

# 傳統：Python sliding window
def fill_depth(depth):
    def replace_zeros(values):
        center = values[len(values)//2]
        if center != 0:
            return center
        nonzero = values[values > 0]
        return nonzero.mean() if len(nonzero) > 0 else 0
    return generic_filter(depth, replace_zeros, size=3)

def iterative(depth, times=3):
    filled = depth.copy()
    for _ in range(times):
        filled = fill_depth(filled)
    return filled

def iterative_section(depth_section, times=10):
    filled = depth_section.copy()
    for _ in range(times):
        filled = fill_depth(filled)
    return filled

# 捲積法：OpenCV 卷積
def fast_fill_depth(depth, iterations=3):
    filled = depth.copy()
    for _ in range(iterations):
        zero_mask = (filled == 0)
        nonzero = (filled != 0).astype(np.float32)
        kernel = np.ones((3, 3), np.float32)
        sum_ = cv2.filter2D(filled, -1, kernel)
        count = cv2.filter2D(nonzero, -1, kernel)
        avg = np.divide(sum_, count, out=np.zeros_like(sum_), where=(count > 0))
        filled[zero_mask] = avg[zero_mask]
    return filled

# === 🔵 傳統方法：上半3次 + 下半100次 ===
start = time.time()
depth_trad = iterative(depth_map, times=3)
depth_trad[H//2:, :] = iterative_section(depth_trad[H//2:, :], times=100)
time_trad = time.time() - start
print(f"[傳統] fill_depth: Top(3x) + Bottom(100x) → {time_trad:.4f} sec")

# === 🟢 捲積方法：上半3次 + 下半100次 ===
start = time.time()
depth_conv = fast_fill_depth(depth_map, iterations=3)
depth_conv[H//2:, :] = fast_fill_depth(depth_conv[H//2:, :], iterations=100)
time_conv = time.time() - start
print(f"[捲積] fast_fill_depth: Top(3x) + Bottom(100x) → {time_conv:.4f} sec")

# ⚖️ 差異
print(f"Speedup: {time_trad / time_conv:.1f}x faster")
