import open3d as o3d
import numpy as np
import cv2
import os
from pathlib import Path
from cv2.ximgproc import guidedFilter  # 要先 pip install opencv-contrib-python
from scipy.ndimage import generic_filter


# === [1] 路徑設定 ===
base_dir = Path(__file__).parent
scene_dir = base_dir / "heighway_sunny_day_2024-06-12-08-43-21"
image_path = scene_dir / "image" / "000000.png"
pcd_path = scene_dir / "VLS128_pcdnpy" / "000000.pcd"
output_path = base_dir / "output" / "iterate_lidar_v1.png"
os.makedirs(output_path.parent, exist_ok=True)





max_depth = 80.0


# === [2] 相機內參矩陣 K ===
K = np.array([
    [1418.667, 0.0, 640.0],
    [0.0, 1418.667, 360.0],
    [0.0, 0.0, 1.0]
])

# === [3] LiDAR to Camera 外參矩陣 ===
T_lidar_to_cam = np.array([
    [ 0.037, -0.999,  0.009,  0.0],
    [-0.094, -0.012, -0.996, -0.3],
    [ 0.995,  0.036, -0.094, -0.43],
    [ 0.0,    0.0,    0.0,    1.0]
])

# === [4] 載入圖片與點雲 ===
pcd = o3d.io.read_point_cloud(str(pcd_path))
points = np.asarray(pcd.points)
points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])

img_bgr = cv2.imread(str(image_path))
H, W = img_bgr.shape[:2]

# === [5] 投影 ===
cam_points = (T_lidar_to_cam @ points_hom.T).T
cam_points = cam_points[cam_points[:, 2] > 0]
z = cam_points[:, 2]
print(f"🔹 投影後最大深度 = {z.max():.2f} m")

x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
u = (K[0, 0] * x / z + K[0, 2]).astype(np.int32)
v = (K[1, 1] * y / z + K[1, 2]).astype(np.int32)

# === [6] 建立 depth map + 最近點保留
depth_map = np.zeros((H, W), dtype=np.float32)
valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
for i in range(len(u)):
    if valid[i]:
        px, py = u[i], v[i]
        if depth_map[py, px] == 0 or z[i] < depth_map[py, px]:
            depth_map[py, px] = z[i]


depth_raw_vis = np.clip(depth_map, 0, max_depth)
depth_raw_vis_norm = (depth_raw_vis / max_depth * 255).astype(np.uint8)
cv2.imwrite(str(base_dir / "output" / "depth_raw_000000_gray.png"), depth_raw_vis_norm)
depth_raw_colored = cv2.applyColorMap(depth_raw_vis_norm, cv2.COLORMAP_INFERNO)
cv2.imwrite(str(base_dir / "output" / "depth_raw_000000_colored.png"), depth_raw_colored)


def fill_depth(depth):
    def replace_zeros(values):
        center = values[len(values)//2]
        if center != 0:
            return center
        nonzero = values[values > 0]
        return nonzero.mean() if len(nonzero) > 0 else 0
    return generic_filter(depth, replace_zeros, size=3)

depth_filled = fill_depth(depth_map)

def iterative(depth, times=3):
    filled = depth.copy()
    for _ in range(times):
        filled = fill_depth(filled)
    return filled

depth_filled = iterative(depth_map, times=3)

# === [補強下半部空洞] ===
H, W = depth_filled.shape

# 我們只對下半部 y > H // 2 做更強的 iterative
bottom_half = depth_filled[H//2:, :]

# 強化補洞：再跑 10 次 iterative
def iterative_section(depth_section, times=10):
    filled = depth_section.copy()
    for _ in range(times):
        filled = fill_depth(filled)
    return filled

depth_filled[H//2:, :] = iterative_section(bottom_half, times=100)

np.save(str(base_dir / "output" / "lidar_depth_000000.npy"), depth_filled)
print(f"Depth shape = {depth_filled.shape}, min = {depth_filled.min():.3f}, max = {depth_filled.max():.3f}, mean = {depth_filled.mean():.3f}")




# === [9] 彩色視覺化
# depth_vis = np.clip(refined_depth, 0, max_depth)

depth_vis = np.clip(depth_filled, 0, max_depth)
depth_vis_norm = (depth_vis / max_depth * 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_vis_norm, cv2.COLORMAP_INFERNO)

# === [10] 拼接上下圖並輸出
# combined = np.vstack([img_bgr, depth_colored])
# cv2.imwrite(str(output_path), combined)

cv2.imwrite(str(output_path),  depth_colored)

print(f"✅ 已輸出：{output_path}")




