#!/usr/bin/env bash
# Build pobim_splats.zip for installing into Blender
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f pobim_splats.zip
zip -r pobim_splats.zip pobim_splats -x "*/__pycache__/*"
echo "created $(pwd)/pobim_splats.zip"
