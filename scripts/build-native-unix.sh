#!/usr/bin/env bash

# This script should be run from the repository root

# Default number of jobs
NUM_JOBS_MAKE=8

# Parse command line options
while getopts "j:" opt; do
  case $opt in
    j)
      NUM_JOBS_MAKE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done


set -e
cd NativeAcceleration
rm -rf build
mkdir build
cd build
cmake ..
make -j ${NUM_JOBS_MAKE}
cd ../..
cd LibTorchSharp
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j ${NUM_JOBS_MAKE}
cd ../..
mkdir -p Release/linux-x64/publish
cp NativeAcceleration/build/lib/libNativeAcceleration.so Release/linux-x64/publish/
cp LibTorchSharp/build/LibTorchSharp/libLibTorchSharp.so Release/linux-x64/publish/
