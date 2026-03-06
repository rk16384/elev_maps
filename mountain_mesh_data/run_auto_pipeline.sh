#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MOUNTAINS=(
  "cayambe|Cayambe|18,996"
  "chimborazo|Chimborazo|20,549"
  "cotopaxi|Cotopaxi|19,347"
)
# MOUNTAINS=(
#   "glacier-peak|Glacier Peak|10,541"
#   "mt-baker|Mount Baker|10,786"
#   "mt-hood|Mount Hood|11,249"
#   "mt-jefferson|Mount Jefferson|10,495"
#   "mt-rainier|Mount Rainier|14,410"
#   "mt-sthelens|Mount St. Helens|8,363"
#   "mt-shasta|Mount Shasta|14,179"
#   "pikes-peak|Pikes Peak|14,115"
#   "mt-washington|Mount Washington|6,288"
# )

for entry in "${MOUNTAINS[@]}"; do
  IFS='|' read -r folder title elevation <<< "$entry"

  echo "============================================================"
  echo "Processing: ${folder}"
  echo "============================================================"

  #python interactive_mesh.py "$folder" radius
  #python drop_shadow_editor.py "$folder"
  python add_mountain_titles.py \
    "$folder" \
    --title "$title" \
    --elevation "Elevation - ${elevation} ft" \
    --title_weight_boost_px 2 \
    --subtitle_weight_boost_px 2 
done

echo "All mountains processed."
