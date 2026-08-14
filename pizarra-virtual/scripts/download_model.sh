#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../models/hand_landmarker.task"

mkdir -p "$(dirname "$DEST")"
echo "Descargando modelo Hand Landmarker en $DEST..."
curl -L -o "$DEST" "$MODEL_URL"
echo "Listo: $(du -h "$DEST" | cut -f1)"
