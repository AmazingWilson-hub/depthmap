#!/usr/bin/env bash
set -euo pipefail

# === 修改這裡 ===
SCENE="highway_sunny_day"   # 場景名稱 (要加在前綴)
ROOT="/Users/wilson/code/depthmap/output/2024-05-07/${SCENE}"  # A 場景的路徑
DRY_RUN="${DRY_RUN:-1}"   # 1=只顯示不改，0=真的改名

shopt -s nullglob
for d in "$ROOT"/EVIL_*; do
  [[ -d "$d" ]] || continue
  base="$(basename "$d")"          # EVIL_2024-05-07-06-26-46
  suffix="${base#EVIL_}"           # 2024-05-07-06-26-46
  new="${SCENE}_${suffix}"         # A_2024-05-07-06-26-46

  from="$d"
  to="$(dirname "$d")/$new"

  if [[ -e "$to" ]]; then
    echo "⚠️  衝突：$to 已存在，略過 $from"
    continue
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY] mv \"$from\" \"$to\""
  else
    mv "$from" "$to"
    echo "[OK ] $from -> $to"
  fi
done

echo "完成。DRY_RUN=$DRY_RUN"