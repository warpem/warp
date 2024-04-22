#!/usr/bin/env bash
set -e
cd NativeAcceleration
rm -rf build
mkdir build
cd build
cmake ..
make -j 8
cd ../..
cd LibTorchSharp
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j 8
cd ../..
mkdir -p Release/linux-x64/publish
cp NativeAcceleration/build/lib/libNativeAcceleration.so Release/linux-x64/publish/
cp LibTorchSharp/build/LibTorchSharp/libLibTorchSharp.so Release/linux-x64/publish/
