#!/bin/bash
#LD_LIBRARY_PATH=${PREFIX}/lib
#ls ${PREFIX}/lib

PROJECT_ROOT=$(pwd)

echo conda list
conda list

echo "=== Build environment diagnostics ==="
echo "CONDA_BUILD_SYSROOT=${CONDA_BUILD_SYSROOT:-<unset>}"
echo "CC=${CC:-<unset>}"
echo "CXX=${CXX:-<unset>}"
echo "CFLAGS=${CFLAGS:-<unset>}"
echo "CXXFLAGS=${CXXFLAGS:-<unset>}"
echo "CMAKE_ARGS=${CMAKE_ARGS:-<unset>}"
echo "System glibc: $(ldd --version 2>&1 | head -1)"
if [ -n "${CONDA_BUILD_SYSROOT}" ] && [ -d "${CONDA_BUILD_SYSROOT}" ]; then
  echo "Sysroot libc: $(ls ${CONDA_BUILD_SYSROOT}/lib/libc-* 2>/dev/null || echo 'not found')"
  echo "Sysroot contents: $(ls ${CONDA_BUILD_SYSROOT}/usr/include/features.h 2>/dev/null && grep __GLIBC_MINOR__ ${CONDA_BUILD_SYSROOT}/usr/include/features.h | head -3)"
else
  echo "WARNING: CONDA_BUILD_SYSROOT is unset or does not exist!"
fi
echo "=== End diagnostics ==="

# build NativeAcceleration
echo building NativeAcceleration
cd NativeAcceleration
rm -rf build
mkdir build
cd build
cmake ${CMAKE_ARGS} ..
make -j 2
cd ${PROJECT_ROOT}

# build LibTorchSharp
echo building LibTorchSharp
TORCH_CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
CUSTOM_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${TORCH_CMAKE_PREFIX_PATH}"
cd LibTorchSharp
rm -rf build
mkdir build
cd build
cmake ${CMAKE_ARGS} ${CUSTOM_CMAKE_ARGS} ..
make -j 2
cd ${PROJECT_ROOT}

mkdir -p Release/linux-x64/publish
cp NativeAcceleration/build/lib/libNativeAcceleration.so Release/linux-x64/publish/
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
