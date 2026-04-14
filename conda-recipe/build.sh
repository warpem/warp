#!/bin/bash
#LD_LIBRARY_PATH=${PREFIX}/lib
#ls ${PREFIX}/lib

PROJECT_ROOT=$(pwd)

echo conda list
conda list

# build NativeAcceleration — skipped to speed up cmake debugging
# echo building NativeAcceleration
# cd NativeAcceleration
# rm -rf build
# mkdir build
# cd build
# cmake ${CMAKE_ARGS} ..
# make -j 2
# cd ${PROJECT_ROOT}

# build LibTorchSharp
echo building LibTorchSharp
TORCH_CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
CUSTOM_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH}"
cd LibTorchSharp
rm -rf build
mkdir build
cd build
cmake ${CMAKE_ARGS} ${CUSTOM_CMAKE_ARGS} --log-level=VERBOSE \
  --trace-expand --trace-redirect=cmake_trace.log .. 2>&1 || {
  echo "=== CMakeError.log ===" && cat CMakeFiles/CMakeError.log 2>/dev/null
  echo "=== cmake_trace.log (last 200 lines) ===" && tail -200 cmake_trace.log 2>/dev/null
  exit 1
}
make -j 2
cd ${PROJECT_ROOT}

mkdir -p Release/linux-x64/publish
# cp NativeAcceleration/build/lib/libNativeAcceleration.so Release/linux-x64/publish/
cp LibTorchSharp/build/LibTorchSharp/libLibTorchSharp.so Release/linux-x64/publish/

./scripts/publish-unix.sh

# Create bin and lib directories
mkdir -p $PREFIX/bin
mkdir -p $PREFIX/lib

# Copy binaries to the bin directory
cp $SRC_DIR/Release/linux-x64/publish/{EstimateWeights,Frankenmap,MCore,MTools,MrcConverter,Noise2Half,Noise2Map,Noise2Mic,Noise2Tomo,WarpTools,WarpWorker} $PREFIX/bin/

# PDB and configs also go into bin
cp $SRC_DIR/Release/linux-x64/publish/*.pdb $PREFIX/bin/
cp $SRC_DIR/Release/linux-x64/publish/*.config $PREFIX/bin/

# Copy libraries to the lib directory
cp $SRC_DIR/Release/linux-x64/publish/{libLibTorchSharp.so,libNativeAcceleration.so,libSkiaSharp.so} $PREFIX/lib/
