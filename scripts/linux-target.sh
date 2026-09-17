#!/usr/bin/env bash

# Source from Linux build scripts. Conda's target name differs from the .NET RID.
case "${target_platform:-$(uname -m)}" in
  linux-64|x86_64)
    WARP_RUNTIME_ID=linux-x64
    WARP_PLATFORM_TARGET=x64
    ;;
  linux-aarch64|aarch64|arm64)
    WARP_RUNTIME_ID=linux-arm64
    WARP_PLATFORM_TARGET=ARM64
    ;;
  *)
    echo "Unsupported Linux target: ${target_platform:-$(uname -m)}" >&2
    return 1
    ;;
esac
