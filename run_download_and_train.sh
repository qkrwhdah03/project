#!/usr/bin/env bash
set -euo pipefail

# Minimal helper: download Tartanair2 and train depth estimator.
# Run from anywhere; it will locate the repo root automatically.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "==> Step 1/2: Download Tartanair2"
python download_tartanair2.py

echo "==> Step 2/2: Train depth estimator"
python forward_warp/train_depth_estimator.py

echo "✅ Done."

