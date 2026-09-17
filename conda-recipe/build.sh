#!/usr/bin/env bash
set -euo pipefail

cd "$SRC_DIR"
source scripts/linux-target.sh
conda list

./scripts/build-native-unix.sh -j "${CMAKE_BUILD_PARALLEL_LEVEL:-${CPU_COUNT:-2}}"
./scripts/publish-unix.sh

PUBLISH_DIR="$SRC_DIR/Release/$WARP_RUNTIME_ID/publish"
mkdir -p "$PREFIX/bin" "$PREFIX/lib"

# Executables, debug symbols and configs belong in bin; shared libraries in lib.
cp "$PUBLISH_DIR"/{EstimateWeights,Frankenmap,MCore,MTools,MrcConverter,Noise2Half,Noise2Map,Noise2Mic,Noise2Tomo,WarpTools,WarpWorker,WarpWorker2} "$PREFIX/bin/"
cp "$PUBLISH_DIR"/*.pdb "$PREFIX/bin/"
cp "$PUBLISH_DIR"/*.config "$PREFIX/bin/"
cp "$PUBLISH_DIR"/{libLibTorchSharp.so,libNativeAcceleration.so,libSkiaSharp.so} "$PREFIX/lib/"
