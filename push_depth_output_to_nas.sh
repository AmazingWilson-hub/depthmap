#!/opt/homebrew/bin/bash
set -euo pipefail

# ==== 依你的情況修改（照 FileZilla 的設定填）====
SRC_ROOT="/Users/wilson/code/depthmap/output/2024-05-09"   # 本機根目錄
HOST="evilab.synology.me"                                  # NAS 位址（跟 FileZilla 一樣）
PORT="22"                                                  # SFTP 連接埠（FileZilla 用幾就填幾，常見 22/2222）
USER="113C52027"                                           # 帳號（跟 FileZilla 一樣）
PASS=""                                                    # 每次執行會互動輸入密碼（不再使用環境變數）
NAS_ROOT="/1323LAB_FTP/Labeled_RealCar_Dataset"            # 遠端根路徑（請用你實際可列到的前綴）
DRY_RUN="${DRY_RUN:-1}"                                    # 1=只顯示命令(不執行)，0=真的執行

# 調試選項：DEBUG=1 會開詳細輸出與較短逾時，方便看卡在哪
DEBUG="${DEBUG:-0}"
if [[ "$DEBUG" == "1" ]]; then
  set -x
fi

# SFTP 連線與逾時參數（避免長時間卡住），並改用 OpenSSH 作為後端以提升相容性
LFTP_SETTINGS=(
  "set net:max-retries 1"
  "set net:persist-retries 0"
  "set net:timeout 15"
  "set net:reconnect-interval-base 5"
  "set net:reconnect-interval-max 20"
  "set net:idle 10"
  "set sftp:auto-confirm yes"
  "set sftp:connect-program ssh -a -x -p ${PORT}"
)

# 小工具：包一層 lftp，DEBUG=1 時帶 -d 顯示底層交握
lftp_cmd() {
  local cmd="$1"  # 例如：cls -1 "/path"; bye
  if [[ "$DEBUG" == "1" ]]; then
    lftp -d -u "$USER","$PASS" sftp://"$HOST":"$PORT" -e "${LFTP_SETTINGS[*]}; $cmd"
  else
    lftp    -u "$USER","$PASS" sftp://"$HOST":"$PORT" -e "${LFTP_SETTINGS[*]}; $cmd"
  fi
}

# 每次執行都要求輸入密碼（螢幕不顯示）
need_pass() {
  read -rsp "SFTP 密碼: " PASS; echo
}

# lftp 列遠端目錄（純 SFTP）
remote_ls() {
  local path="$1"
  lftp_cmd "cls -1 \"$path\"; bye" 2>/dev/null || true
}

# 遞迴建目錄
remote_mkdir_p() {
  local path="$1"
  lftp_cmd "mkdir -p \"$path\"; bye"
}

# 鏡像上傳（只較新的；parallel=1 更穩）
remote_mirror_upload() {
  local local_dir="$1"; shift
  local remote_dir="$1"; shift
  lftp_cmd "mirror -R --only-newer --verbose=1 --parallel=1 \"$local_dir\" \"$remote_dir\"; bye"
}

# ====== 執行前提示 ======
if [[ "$DRY_RUN" == "1" ]]; then
  echo "⚠️  現在是 DRY-RUN 測試模式（不會上傳檔案）。要真的上傳請在執行時加：DRY_RUN=0"
fi
echo "[INFO] HOST=$HOST PORT=$PORT USER=$USER"
echo "[INFO] SRC_ROOT=$SRC_ROOT"
echo "[INFO] NAS_ROOT=$NAS_ROOT"
echo

# 互動輸入密碼
need_pass

# ====== 主流程 ======
for scene_dir in "$SRC_ROOT"/*; do
  [[ -d "$scene_dir" ]] || continue
  scene="$(basename "$scene_dir")"  # e.g. citystreet_sunny_day

  for evil_dir in "$scene_dir"/EVIL_*; do
    [[ -d "$evil_dir/depth_output" ]] || { echo "略過: $evil_dir (沒有 depth_output)"; continue; }

    base="$(basename "$evil_dir")"            # EVIL_2024-05-07-06-26-46
    ts="${base#EVIL_}"                        # 2024-05-07-06-26-46
    ymd="$(echo "$ts" | cut -d'-' -f1-3)"     # 2024-05-07

    day_scene_dir="$NAS_ROOT/$ymd/$scene"

    echo "[STEP] 掃描遠端目錄：$day_scene_dir"
    listing="$(remote_ls "$day_scene_dir")"
    if [[ -z "$listing" ]]; then
      echo "[WARN] 遠端沒有目錄：$day_scene_dir ；請先在 NAS 建好（日期/scene）"
      continue
    fi

    # 用時間戳過濾符合的子資料夾（名稱不同沒關係，只要包含同一段 ts）
    mapfile -t matches < <(echo "$listing" | grep "$ts" || true)
    if (( ${#matches[@]} == 0 )); then
      echo "[WARN] $day_scene_dir 下找不到含 '$ts' 的資料夾"
      continue
    elif (( ${#matches[@]} > 1 )); then
      echo "[WARN] 匹配到多個，請人工確認："
      printf '  - %s/%s\n' "$day_scene_dir" "${matches[@]}"
      continue
    fi

    target_dir="$day_scene_dir/${matches[0]}/depth_output"
    echo "[SYNC] $evil_dir/depth_output -> sftp://$HOST:$PORT$target_dir"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[DRY-RUN] 準備上傳到：$target_dir"
      echo "[DRY-RUN] remote_mkdir_p \"$target_dir\""
      echo "[DRY-RUN] remote_mirror_upload \"$evil_dir/depth_output/\" \"$target_dir/\""
    else
      echo "[STEP] 建立遠端目錄：$target_dir"
      remote_mkdir_p "$target_dir"
      echo "[STEP] 開始鏡像上傳（only-newer, parallel=1）：$evil_dir/depth_output -> $target_dir"
      remote_mirror_upload "$evil_dir/depth_output/" "$target_dir/"
    fi
  done
done

echo "完成。"