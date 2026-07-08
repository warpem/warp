CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(NativeAcceleration LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)


FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(FFTW COMPONENTS FLOAT_THREADS_LIB REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#NativeAcceleration

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

INCLUDE(FindTIFF)

INCLUDE_DIRECTORIES(include ${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}") 

FILE(GLOB_RECURSE SOURCES src/*.cu src/*.cpp gtom/src/*.cu gtom/src/*.cpp liblion/*.cpp liblion/*.cc)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -w -fPIC -mavx2 -mf16c")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

add_compile_definitions("FLOAT_PRECISION")

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-w")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-restrict")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-use_fast_math")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lineinfo")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
# -cudart=static removed: combining a statically-embedded CUDA runtime with dynamically-linked
# libcufft.so creates two independent runtime instances with separate per-thread device TLS.
# When cudaSetDevice(N) is called through the static runtime, libcufft.so still sees the default
# device (0) through its own dynamic runtime, causing cufftExec* to fail on any GPU != 0.
# The failure is intermittent because it depends on CUDA driver context initialisation order,
# which varies with system load — hence the Heisenbug on multi-GPU nodes.
# Using the shared (dynamic) CUDA runtime ensures a single runtime instance is shared by
# NativeAcceleration.so, libcufft.so, libcublas.so, and libcurand.so.
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets")
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--ptxas-options=-v")

ADD_LIBRARY(NativeAcceleration SHARED ${SOURCES})

set_target_properties(NativeAcceleration PROPERTIES CUDA_ARCHITECTURES "61;70;75;80;86;89;90;100;120")

# libcudart.so is intentionally listed explicitly so the linker records a direct SONAME
# dependency rather than relying on libcufft.so's transitive pull, which guarantees the
# dynamic linker loads exactly one libcudart.so instance regardless of link order.
TARGET_LINK_LIBRARIES(NativeAcceleration ${FFTW_LIBRARIES} libtiff.so libcudart.so libcufft.so libcublas.so libcurand.so ${CMAKE_THREAD_LIBS_INIT})
