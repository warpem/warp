# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Linux: `conda env create -f warp_build.yml && conda activate warp_build && ./scripts/build-native-unix.sh && ./scripts/publish-unix.sh`
- Windows: `publish-windows.bat`
- Output location: `Release/linux-x64/publish` (Linux) or `Release\win-x64\publish` (Windows)

## Environment Setup
- C# projects use .NET 8.0
- Native components require CUDA 11.8, PyTorch 2.0.1, FFTW, MKL 2024.0.0
- Manage dependencies with Conda: `conda env create -f warp_build.yml`

## Code Style Guidelines
- Follow existing code style in the project
- C# naming: PascalCase for public members, camelCase for private/local variables
- Minimize runtime exceptions - prefer validation checks
- Warnings suppressed: CS0219, CS0162, CS0168, CS0649, CS0067, CS0414, CS0661, CS0659, CS0169, CS0618, CS1998
- Match indent style of existing files (4 spaces)

## Documentation
- Documentation is in Markdown format in the `docs/` directory
- Preview with `mkdocs serve`