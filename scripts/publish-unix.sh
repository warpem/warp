#!/usr/bin/env bash

# This script should be run from the repository root.
set -euo pipefail
source scripts/linux-target.sh

PROJECTS=(Noise2Map Noise2Mic Noise2Tomo Noise2Half EstimateWeights Frankenmap MrcConverter WarpWorker WarpWorker2 WarpTools MTools MCore)
for PROJECT in "${PROJECTS[@]}"; do
  dotnet publish \
    -nowarn:CS0219,CS0162,CS0168,CS0649,CS0067,CS0414,CS0661,CS0659,CS0169,CS0618,CS1998,MSB3270,SYSLIB0011 \
    --configuration Release --framework net10.0 \
    --runtime "$WARP_RUNTIME_ID" --self-contained true \
    -p:PlatformTarget="$WARP_PLATFORM_TARGET" -p:PublishSingleFile=true \
    "$PROJECT/$PROJECT.csproj"
done
