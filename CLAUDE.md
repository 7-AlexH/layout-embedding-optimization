# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the code accompanying the paper _Embedding Optimization of Layouts via Distortion Minimization_. It optimizes how a coarse quad-layout (a low-poly "cage" mesh) is embedded onto a target triangle mesh by minimizing parametric distortion via gradient descent. Gradients come from a hand-written reverse-mode adjoint chain over plain doubles/Eigen (`library/LayoutOpt/Adjoint/`) — see [`documentation/adjointDifferentiation.md`](documentation/adjointDifferentiation.md). (The chain replaced an earlier LibTorch/autograd implementation; [`torchRemoval.md`](torchRemoval.md) documents that port and the conventions the hand code preserves.)

## Setup

Clone with submodules:
```bash
git clone --recurse-submodules <repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

## Build (Linux/macOS — Clang)

Prerequisites: CMake >= 3.16, Clang/Clang++, Ninja, OpenMP, Eigen3.

Build output lands **outside** the source directory (`../Clang-Debug` or `../Clang-Release`).

```bash
# Configure + build (release)
cmake --preset clang-release && cmake --build --preset clang-release

# Configure + build (debug — includes AddressSanitizer)
cmake --preset clang-debug && cmake --build --preset clang-debug
```

There is no unit-test framework. Two apps are permanent validation harnesses (exit nonzero on failure, write `gradcheck_fd.txt` / `e2e_determinism.txt` to the working directory):

- `gradcheck_fd` — hand gradients vs central finite differences on the canonical banana state.
- `e2e_determinism` — two full production runs must be bitwise identical, plus an iter-0 reference-loss gate.

Run both after touching anything in `library/LayoutOpt/Adjoint/` or the OpenMP parallelism.

## Build (Windows — MSVC)

Prerequisites: Visual Studio 2022/2026 with the C++ toolset, CMake >= 3.23, Ninja. Eigen is downloaded automatically (see below). Output lands in `../MSVC-Release` / `../MSVC-Debug`.

The MSVC toolchain must be on `PATH` for both configure and build — use one of the provided helpers (they run `vcvars64.bat` via `vswhere`):

```powershell
# One-shot configure + build (release)
.\build_msvc.ps1                      # or: .\build_msvc.ps1 -Preset msvc-debug

# Or import the env into the current shell, then drive cmake yourself:
. .\setup_msvc_env.ps1
cmake --preset msvc-release
cmake --build --preset msvc-release
```

Windows-specific build notes (already handled in the tree):
- The `msvc-*` presets set `CMAKE_POLICY_VERSION_MINIMUM=3.5` because CMake 4.x rejects the old `cmake_minimum_required` in several submodules.
- `Types.hh` needs Eigen **3.4** (`Eigen::Vector3<T>` aliases); geometry-central bundles 3.3.9, so `extern/eigen-3.4.0/` is downloaded and given include priority in `CMakeLists.txt`. Get it from `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip`.
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

`optimize` runs headless (writes screenshots); `load_state` and `extract_layout_from_qm` open an interactive viewer window. The validation harnesses `gradcheck_fd` and `e2e_determinism` (see Build) also run headless. Input meshes are `.obj` files. The `DATA_PATH` and `OUTPUT_PATH` CMake variables (set at configure time) are baked into the binary as preprocessor macros.

## Architecture

### Core data structures (`library/LayoutOpt/DataStructures/`)

The optimization operates on four parallel data structures that are kept in sync throughout the loop:

- **`TargetMeshData`** — the fixed input triangle mesh; holds polymesh topology, positions, and optional curvature / scalar-field / direction-field data (as polymesh attributes plus plain Eigen matrix copies).
- **`LayoutData`** — the coarse layout (quad cage) mesh; holds topology and positions. Mutable layout node positions are the optimization variables.
- **`PathNetworkData`** — the embedded boundary network: the layout edges traced onto the target surface as sequences of surface points. The intrinsic representation (`SurfacePoint` per vertex, stored as barycentric coordinates on a target face) is what actually gets differentiated.
- **`OverlayMeshData`** — the combined mesh formed by inserting the path network into the target mesh; used to evaluate per-patch parameterization and compute objective terms.

### Optimization loop (`apps/optimize.cc`, `library/LayoutOpt/Optimization.*`)

1. **Init** (`Init.cc`): normalize both meshes, call `compute_layout_embedding_init` to project layout vertices onto the target surface and trace the initial path network.
2. **Resample** (`Resample.cc`): periodically re-distribute path-network vertices along each arc for numerical stability.
3. **Eval** (`Optimization.cc` → `Adjoint/HandGradients.*`): build the overlay mesh, run the hand forward loss, then the hand reverse pass — gradients flow back to the intrinsic surface-point barycentric coordinates and become update directions.
4. **Apply** (`Optimization.cc`): move surface points along their update directions, rebuild the overlay mesh.

### Objective functions (`library/LayoutOpt/Adjoint/`)

The loss and its gradient are computed by a hand-written forward + reverse-mode adjoint chain (`HandGradients.hh/cc` orchestrates the per-stage files, see `documentation/adjointDifferentiation.md`):
- **Harmonic distortion** — main isometric/area-preserving distortion term (AIAP / area-preserving SVD variants on the per-face Jacobian of each patch's harmonic parameterization); controlled by `HarmonicOptions` weights.
- **Curvature alignment** — aligns patch boundaries to principal curvature directions (`CurvatureAlignmentStage`).

Weights are set in `OptimizationOptions`; the optimizer is Adam by default (`Optimizers.cc`). `apps/gradcheck_fd.cc` (finite-difference gradient check) and `apps/e2e_determinism.cc` (bitwise run-to-run determinism + reference-loss gate) are the permanent validation harnesses.

**Determinism contract:** the pipeline is bitwise run-to-run deterministic and `e2e_determinism` gates it. `hand_loss_forward` parallelizes the per-patch S6/S7 work under OpenMP, but each patch writes only its own slot and the loss reduction stays sequential in patch order, so results are independent of thread count/schedule. Two rules when touching this code: keep reductions/iteration orders fixed (e.g. the sparse-triplet insertion order in `HarmonicStage.cc` is load-bearing), and never create polymesh attributes inside a parallel region (attribute registration is not thread-safe — per-thread scratch attributes are created and destroyed outside the loop).

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
