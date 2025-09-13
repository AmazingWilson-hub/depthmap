import open3d as o3d
import numpy as np
import cv2
import os
from pathlib import Path
from cv2.ximgproc import guidedFilter  
from scipy.ndimage import generic_filter
from concurrent.futures import ThreadPoolExecutor, as_completed

# === [1] 路徑設定（改為遞迴處理 data/ 下所有場景） ===
base_dir = Path(__file__).parent
data_dir = base_dir / "elan_dataset"  # <- 你要的根目錄名稱
output_root = base_dir / "output"
os.makedirs(output_root, exist_ok=True)

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


def load_points(pcd_path: Path) -> np.ndarray:
    """讀取點雲，支援 .pcd 與 .npy（形狀 Nx3）。"""
    if pcd_path.suffix.lower() == ".pcd":
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        pts = np.asarray(pcd.points)
        return pts
    elif pcd_path.suffix.lower() == ".npy":
        arr = np.load(str(pcd_path))
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
        raise ValueError(f"npy 形狀不正確: {pcd_path}，期望 Nx3")
    else:
        raise ValueError(f"不支援的點雲格式: {pcd_path}")


def fast_fill_depth(depth: np.ndarray, iterations: int = 3) -> np.ndarray:
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


def process_pair(image_path: Path, pcd_path: Path, out_dir: Path) -> None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"讀不到圖片: {image_path}")
        return
    H, W = img_bgr.shape[:2]

    # 讀取點雲
    try:
        points = load_points(pcd_path)
    except Exception as e:
        print(f"讀點雲失敗 {pcd_path}: {e}")
        return

    # 投影
    points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    cam_points = (T_lidar_to_cam @ points_hom.T).T
    cam_points = cam_points[cam_points[:, 2] > 0]
    if cam_points.size == 0:
        print(f"無有效前方點: {pcd_path}")
        return
    z = cam_points[:, 2]
    x, y = cam_points[:, 0], cam_points[:, 1]
    u = (K[0, 0] * x / z + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * y / z + K[1, 2]).astype(np.int32)

    # 建立 depth map + 最近點保留
    depth_map = np.zeros((H, W), dtype=np.float32)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    for i in range(len(u)):
        if valid[i]:
            px, py = u[i], v[i]
            if depth_map[py, px] == 0 or z[i] < depth_map[py, px]:
                depth_map[py, px] = z[i]

    # 補洞（加速版）並加強下半部
    depth_filled = fast_fill_depth(depth_map, iterations=3)
    depth_filled[H // 2 :, :] = fast_fill_depth(depth_filled[H // 2 :, :], iterations=100)

    # 視覺化與輸出
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # 儲存 dense depth npy
    depth_npy_path = out_dir / f"{stem}.npy"
    np.save(str(depth_npy_path), depth_filled)

    # 儲存彩色圖
    depth_vis = np.clip(depth_filled, 0, max_depth)
    depth_vis_norm = (depth_vis / max_depth * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis_norm, cv2.COLORMAP_INFERNO)
    out_png_path = out_dir / f"{stem}.png"
    cv2.imwrite(str(out_png_path), depth_colored)

    print(f"Done: {image_path} -> {out_png_path}")


# === [Main] 遞迴尋找 data/ 下的所有影像，配對同名點雲 ===
# 預期目錄結構：
#   data/<scene>/image/XXXXX.jpg
#   data/<scene>/VLS128_pcdnpy/XXXXX.pcd  或  XXXXX.npy
# 若結構不同，可自行調整下面的配對邏輯。

image_exts = {".jpg", ".jpeg", ".png"}

if not data_dir.exists():
    raise FileNotFoundError(f"找不到資料夾: {data_dir}")

def _iter_jobs():
    for img_path in data_dir.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue
        try:
            # 期待 image 在 scene_root/image
            if img_path.parent.name == "image":
                scene_root = img_path.parent.parent
            else:
                # 退一步：假設影像所在目錄就是 scene_root
                scene_root = img_path.parent
            stem = img_path.stem
            pcd_candidate_pcd = scene_root / "VLS128_pcdnpy" / f"{stem}.pcd"
            pcd_candidate_npy = scene_root / "VLS128_pcdnpy" / f"{stem}.npy"

            if pcd_candidate_pcd.exists():
                pcd_path = pcd_candidate_pcd
            elif pcd_candidate_npy.exists():
                pcd_path = pcd_candidate_npy
            else:
                print(f"跳過，找不到同名點雲: {img_path}")
                continue

            # 輸出放在 output/ 下，保留相對於 data/ 的子路徑
            rel_parent = img_path.parent.relative_to(data_dir)
            # 將目錄名稱 'image' 改為 'depth_output'
            if rel_parent.parts[-1] == "image":
                rel_parent = Path(*rel_parent.parts[:-1], "depth_output")
            out_dir = output_root / rel_parent

            yield (img_path, pcd_path, out_dir)
        except Exception as e:
            print(f"準備任務失敗 {img_path}: {e}")


try:
    cv2.setNumThreads(1)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

jobs = list(_iter_jobs())
if not jobs:
    print("⚠️ 沒有可處理的影像任務。")
else:
    max_workers = min(8, (os.cpu_count() or 4))
    print(f"🚀 並行處理 {len(jobs)} 張影像，workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_pair, img, pcd, outdir) for (img, pcd, outdir) in jobs]
        for fut in as_completed(futures):
            
            fut.result()