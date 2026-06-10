# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the code accompanying the paper _Embedding Optimization of Layouts via Distortion Minimization_. It optimizes how a coarse quad-layout (a low-poly "cage" mesh) is embedded onto a target triangle mesh by minimizing parametric distortion via gradient descent using LibTorch for autodiff.

## Setup

Clone with submodules:
```bash
git clone --recurse-submodules <repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

Download LibTorch 2.4.0 (CPU) into `extern/libtorch/`:
```bash
bash download_libtorch.sh   # Linux/macOS; fetches the Linux build
```

On Windows download the **Windows** LibTorch build instead (the script above fetches the Linux build):
```
https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%2Bcpu.zip
```
extract it to `extern/libtorch/`. The Windows build is compiled with MSVC, which is why the Windows build below uses MSVC (matching ABI).

## Build (Linux/macOS — Clang)

Prerequisites: CMake >= 3.16, Clang/Clang++, Ninja, OpenMP, Eigen3.

Build output lands **outside** the source directory (`../Clang-Debug` or `../Clang-Release`).

```bash
# Configure + build (release)
cmake --preset clang-release && cmake --build --preset clang-release

# Configure + build (debug — includes AddressSanitizer)
cmake --preset clang-debug && cmake --build --preset clang-debug
```

There are no tests. The apps in `apps/` serve as integration entry points.

## Build (Windows — MSVC)

Prerequisites: Visual Studio 2022/2026 with the C++ toolset, CMake >= 3.23, Ninja. Eigen is downloaded automatically (see below). Build with **MSVC** (not the bundled clang) to match the MSVC-built Windows LibTorch ABI. Output lands in `../MSVC-Release` / `../MSVC-Debug`.

The MSVC toolchain must be on `PATH` for both configure and build — use one of the provided helpers (they run `vcvars64.bat` via `vswhere`):

```powershell
# One-shot configure + build (release)
.\build_msvc.ps1                      # or: .\build_msvc.ps1 -Preset msvc-debug

# Or import the env into the current shell, then drive cmake yourself:
. .\setup_msvc_env.ps1
cmake --preset msvc-release
cmake --build --preset msvc-release
```

Build **release**: the prebuilt LibTorch uses the dynamic release CRT (`/MD`); a debug build (`/MDd`) would mismatch.

Windows-specific build notes (already handled in the tree):
- The `msvc-*` presets set `CMAKE_POLICY_VERSION_MINIMUM=3.5` because CMake 4.x rejects the old `cmake_minimum_required` in several submodules.
- `Types.hh` needs Eigen **3.4** (`Eigen::Vector3<T>` aliases); geometry-central bundles 3.3.9, so `extern/eigen-3.4.0/` is downloaded and given include priority in `CMakeLists.txt`. Get it from `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip`.
- A post-build step copies LibTorch's DLLs next to each `.exe`, so the apps run without LibTorch on `PATH`.
- A few `extern/` submodule edits were needed for MSVC (e.g. `glow-extras/.../viewer/configure.hh`: `#undef near`/`far` and explicit `tg::pos3` construction). These live outside the main repo and must be re-applied if those submodules are re-checked-out.

## Running

Each `.cc` file in `apps/` becomes its own executable. The main entry point is `optimize`, which loads a target mesh + layout mesh from `data/`, runs the optimization loop, and writes screenshots to `data/<Model>/images/`:

```bash
# Linux/macOS
../Clang-Release/optimize
../Clang-Release/extract_layout_from_qm   # extracts a layout from a quad mesh
../Clang-Release/load_state               # loads and views a saved state
```
```powershell
# Windows
..\MSVC-Release\optimize.exe
..\MSVC-Release\extract_layout_from_qm.exe
..\MSVC-Release\load_state.exe
```

`optimize` runs headless (writes screenshots); `load_state` and `extract_layout_from_qm` open an interactive viewer window. Input meshes are `.obj` files. The `DATA_PATH` and `OUTPUT_PATH` CMake variables (set at configure time) are baked into the binary as preprocessor macros.

## Architecture

### Core data structures (`library/LayoutOpt/DataStructures/`)

The optimization operates on four parallel data structures that are kept in sync throughout the loop:

- **`TargetMeshData`** — the fixed input triangle mesh; holds polymesh topology, positions, and optional curvature / scalar-field / direction-field data as both polymesh attributes and pre-converted `torch::Tensor` copies.
- **`LayoutData`** — the coarse layout (quad cage) mesh; holds topology and positions. Mutable layout node positions are the optimization variables.
- **`PathNetworkData`** — the embedded boundary network: the layout edges traced onto the target surface as sequences of surface points. The intrinsic representation (`SurfacePoint` per vertex, stored as barycentric coordinates on a target face) is what actually gets differentiated.
- **`OverlayMeshData`** — the combined mesh formed by inserting the path network into the target mesh; used to evaluate per-patch parameterization and compute objective terms.

### Optimization loop (`apps/optimize.cc`, `library/LayoutOpt/Optimization.*`)

1. **Init** (`Init.cc`): normalize both meshes, call `compute_layout_embedding_init` to project layout vertices onto the target surface and trace the initial path network.
2. **Resample** (`Resample.cc`): periodically re-distribute path-network vertices along each arc for numerical stability.
3. **Eval** (`Optimization.cc` → `ObjectiveFunctions.cc`): build the overlay mesh, compute objective losses via LibTorch (gradients flow through the intrinsic surface-point positions), then back-propagate to get update directions.
4. **Apply** (`Optimization.cc`): move surface points along their update directions, rebuild the overlay mesh.

### Objective functions (`ObjectiveFunctions.hh/cc`)

All losses are differentiable via LibTorch autograd:
- `harmonic_distortion_loss` — main isometric/area-preserving distortion term (AIAP / area-preserving SVD variants); controlled by `HarmonicOptions` weights.
- `principal_curvature_alignment_loss` — aligns patch boundaries to principal curvature directions.
- `distorion_loss` / `distorion_loss_extrinsic` — Yamabe-equation-based distortion (experimental).
- `variance_loss`, `total_length_loss`, `inner_angle_loss`, `repel_loss_*` — additional regularizers (most disabled by default).

Weights are set in `OptimizationOptions`; the optimizer is Adam by default (`Optimizers.cc`).

### Type conventions (`DataStructures/Types.hh`)

- Positions/vectors: `pos3 = tg::dpos3`, `vec3 = tg::dvec3` (double-precision typed-geometry).
- Mesh handles: `VH`, `EH`, `HEH`, `FH` (polymesh).
- Math: `vec3d = Eigen::Vector3d`, `mat2d = Eigen::Matrix2d` for linear algebra.
- Container: `cc::vector<T>` aliased as `LayoutOpt::vector<T>`.

### External dependencies

All live under `extern/` as git submodules:
- **polymesh** — halfedge mesh data structure (attributes, iterators).
- **typed-geometry** — type-safe geometry math (positions, vectors, matrices).
- **clean-core** — lightweight STL alternatives (`cc::vector`, asserts, etc.).
- **glow / glow-extras** — OpenGL renderer + viewer used for visualization and headless screenshots.
- **geometry-central** — geodesic shortest-path solver used during path network computation.
- **libigl** (header-only, `extern/libigl/include`) — used for cotangent weights and mesh utilities.
- **libtorch** — CPU-only PyTorch C++ for autodiff. See [`documentation/libTorchUsage.md`](documentation/libTorchUsage.md) for a detailed breakdown of how autograd, tensors, and the custom optimizer are used.
