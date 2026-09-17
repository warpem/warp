#!/usr/bin/env bash

# This script should be run from the repository root.
set -euo pipefail
source scripts/linux-target.sh

NUM_JOBS_MAKE=${CMAKE_BUILD_PARALLEL_LEVEL:-${CPU_COUNT:-8}}
while getopts ":j:" opt; do
  case $opt in
    j) NUM_JOBS_MAKE=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done
if ! [[ "$NUM_JOBS_MAKE" =~ ^[1-9][0-9]*$ ]]; then
  echo "The number of build jobs must be a positive integer." >&2
  exit 1
fi

rm -rf NativeAcceleration/build LibTorchSharp/build
# Conda provides CMAKE_ARGS as a space-separated list of CMake options.
cmake ${CMAKE_ARGS:-} -S NativeAcceleration -B NativeAcceleration/build
cmake --build NativeAcceleration/build --parallel "$NUM_JOBS_MAKE"
TORCH_CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake ${CMAKE_ARGS:-} -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX_PATH" -S LibTorchSharp -B LibTorchSharp/build
cmake --build LibTorchSharp/build --parallel "$NUM_JOBS_MAKE"

PUBLISH_DIR="Release/$WARP_RUNTIME_ID/publish"
mkdir -p "$PUBLISH_DIR"
cp NativeAcceleration/build/lib/libNativeAcceleration.so "$PUBLISH_DIR/"
cp LibTorchSharp/build/LibTorchSharp/libLibTorchSharp.so "$PUBLISH_DIR/"
