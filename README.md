# Embedding Optimization of Layouts via Distortion Minimization

Code accompanying the paper **_Embedding Optimization of Layouts via Distortion Minimization_**.

## Cloning with Submodules

This repository uses git submodules. Clone with:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

# Building the Project

This project uses [CMake Presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) to simplify configuration and building. Two presets are available: `clang-debug` and `clang-release`.

## Prerequisites

Make sure the following tools are installed and available on your `PATH`:

- CMake >= 3.16
- Clang / Clang++
- Ninja
- OpenMP
- Eigen3

You can verify the key tools with:

```bash
cmake --version
clang --version
ninja --version
```
## Configure

```bash
# Debug
cmake --preset clang-debug

# Release
cmake --preset clang-release
```
Build output is placed outside the source directory:

| Preset          | Binary Directory        |
|-----------------|-------------------------|
| `clang-debug`   | `../Clang-Debug`        |
| `clang-release` | `../Clang-Release`      |

> The binary directory for each preset can be changed by updating the `binaryDir` field in `CMakePresets.json`.

## Build

```bash
# Debug
cmake --build --preset clang-debug

# Release
cmake --build --preset clang-release
```

Both presets build with 4 parallel jobs by default.

## Configure and Build in One Step

```bash
cmake --preset clang-release && cmake --build --preset clang-release

cmake --preset clang-debug && cmake --build --preset clang-debug
```