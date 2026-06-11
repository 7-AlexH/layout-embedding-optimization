# LibTorch Usage in Layout Embedding Optimization

This document describes how LibTorch (the C++ PyTorch API) is used in this codebase. LibTorch is used exclusively for **automatic differentiation**, not for neural networks or GPU acceleration. All computation runs on CPU with double-precision tensors (`torch::kFloat64`).

---

## Overview

The optimization problem is: given a coarse quad layout embedded onto a triangle mesh as a path network (a set of surface points with barycentric coordinates), minimize parametric distortion by gradient descent. LibTorch provides the autodiff machinery that makes this tractable — the loss functions are written as differentiable tensor expressions, and `loss.backward()` propagates gradients back to the leaf variables.

---

## Optimization Variables

**File:** `library/LayoutOpt/DataStructures/SurfacePoint.hh`

The fundamental differentiable quantity is `SurfacePoint::bary_coords`, a `torch::Tensor` locating a path-network vertex on the target surface (shape contract enforced by `is_tensor_valid()`):

- **Vertex point:** no coordinates (the position is the vertex itself)
- **Edge point:** shape `[1]` — α, the weight on the halfedge's from-vertex (1 − α on the to-vertex)
- **Face point:** shape `[2]` — α, β; the third barycentric coordinate is implicit (γ = 1 − α − β)

Only a subset of these are **leaf tensors** of the autograd graph: `requires_grad` is set exclusively on the face-point `bary_coords` of path-network vertices that map to **layout nodes** (`compute_differentiable_surface_points`, `DifferentiableIntersection.cc`) — 2 degrees of freedom per layout node. All other surface points, including every strip-interior path vertex, are constants; the interior overlay positions are *derived* from the layout-node leaves through differentiable edge–edge intersections (`compute_differentiable_intersection_for_overlay`). `SurfacePoint::get_pos()` converts barycentric coordinates back to a position via differentiable linear interpolation, connecting downstream loss computations to the graph.

`SurfacePoint::copy()` creates a detached copy (not part of the computation graph), used when computing world-space step sizes in the optimizer without creating spurious gradient paths.

---

## Tensor Mirrors of Mesh Data

Each of the four core data structures stores a pre-converted `torch::Tensor` copy of its geometric data alongside the polymesh representation. These are fixed inputs (no `requires_grad`) that are combined with the differentiable `bary_coords` to build the computation graph each iteration.

| Data structure | Tensor field | Shape | Description |
|---|---|---|---|
| `TargetMeshData` | `torch_pos_` | `[V × 3]` | Target mesh vertex positions |
| `PathNetworkData` | `torch_t_` | per-halfedge | Parametric arc-length positions along layout edges |
| `PathNetworkData` | `torch_uvs_` | per-halfedge `[2]` | UV coordinates of path-network vertices |
| `OverlayMeshData` | `torch_pos_` | `[V × 3]` | Overlay mesh vertex positions |
| `LayoutData` | `torch_embedded_edge_length_` | per-edge | Arc lengths of embedded layout edges |

**Files:** `library/LayoutOpt/DataStructures/LayoutEmbedding.hh`

---

## TorchUtils — Differentiable Geometric Primitives

**Files:** `library/LayoutOpt/TorchUtils.hh/.cc`

A layer of mesh-aware helper functions that serve as building blocks for the objective functions. All are fully differentiable:

| Function | Description |
|---|---|
| `torch_face_areas` | Per-face areas, shape `[F]` |
| `torch_cotans` | Per-edge cotangent weights, shape `[E]` |
| `torch_compute_2D_face_embedding` | Local 2D coordinate frame for a face |
| `torch_compute_bary_cords_2D` | Barycentric coordinates of a 2D point in a triangle |
| `torch_face_normal` | Face normal vector |
| `torch_compute_intersection_parameter` | Line-line intersection parameter |
| `torch_singular_values` | SVD singular values of a 2×2 matrix |

---

## Objective Functions

**Files:** `library/LayoutOpt/ObjectiveFunctions.hh/.cc`

All loss functions return a `torch::Tensor` scalar built from differentiable operations. The pipeline per iteration is:

1. `torch_compute_embedded_layout_edge_lengths` — compute differentiable arc lengths
2. `torch_compute_t_for_pn_halfedge` — compute parametric positions along arcs
3. `torch_compute_pn_uvs` — compute UV coordinates for path-network vertices
4. Per-patch harmonic parameterization and distortion evaluation

### Active Losses

**`harmonic_distortion_loss`** (primary loss)

Computes per-patch harmonic parameterizations using cotangent weights, then measures the distortion of the resulting map. Steps:

1. Assemble UV boundary conditions from `torch_uvs_`
2. Solve for interior UVs via `torch_harmonic_param` — a differentiable cotangent-weight linear solve. By default this uses `sparse_harmonic_solve` (`Adjoint/SparseHarmonicSolve.cc`), a custom `torch::autograd::Function` with an Eigen `SimplicialLDLT` forward and hand-derived adjoint; the dense `torch::linalg` path is kept as a validation oracle behind `HarmonicOptions::use_sparse_harmonic_solve`
3. For each overlay triangle, compute the Jacobian of the 3D→2D map
4. Call `compute_distortion()`, which applies SVD via `torch_singular_values` to the Jacobian and returns a distortion value based on the singular values

`HarmonicOptions` selects between variants (AIAP isometric, area-preserving, etc.).

**`principal_curvature_alignment_loss`**

Aligns patch boundary directions to the principal curvature field of the target mesh using `torch::dot` and `torch::norm`.

### Inactive Losses (implemented, currently disabled)

- `variance_loss` — penalizes variance of per-patch parameterization areas
- `distorion_loss` — Yamabe-equation-based distortion (experimental)
- `distorion_loss_extrinsic` — modified Yamabe with mean curvature
- `total_length_loss` — penalizes total path network length
- `inner_angle_loss` — encourages 90-degree patch corners
- `repel_loss_v1/v2` — repels surface points from each other

---

## Backpropagation

**File:** `library/LayoutOpt/ObjectiveFunctions.cc` (`compute_gradients`)

After the weighted total loss scalar is assembled in `eval()` (`Optimization.cc`):

```cpp
loss.backward();
```

`compute_gradients` iterates over all path-network vertices and, for face points, collects `bary_coords.grad()` — a `[2]`-shaped tensor giving the gradient of the loss with respect to the point's barycentric coordinates. The free variables are exactly the **layout-node** face points (the only tensors with `requires_grad`); other path-network vertices carry no gradient of their own — their overlay positions are derived from the layout-node leaves via the differentiable intersections, so the chain rule routes their sensitivity back to those leaves automatically.

---

## Optimizer — Custom Vector Adam

**Files:** `library/LayoutOpt/Optimizers.hh/.cc`

Rather than LibTorch's built-in `torch::optim::Adam`, the codebase implements a **custom per-vertex Adam variant** that accounts for the intrinsic geometry of the surface. Standard Adam normalizes by the parameter-space gradient magnitude, but the surface metric distorts parameter space — so the second moment is instead computed from the *world-space* displacement magnitude:

```cpp
// world-space displacement for the proposed step
auto pos_from = sp.get_pos(tmd.torch_pos_, mesh);
sp.bary_coords = sp.bary_coords + gradients[pn_vh];
auto pos_to   = sp.get_pos(tmd.torch_pos_, mesh);

// second moment tracks world-space step size, not parameter-space
od.v[pn_vh] = beta2 * od.v[pn_vh] + (1 - beta2) * torch::norm(pos_to - pos_from).square();
```

This follows Algorithm 1 from [arXiv:2205.13599](https://arxiv.org/pdf/2205.13599) (Riemannian Adam). The moment accumulators `m` (first moment, `[2]`) and `v` (second moment scalar) are stored as `pm::vertex_attribute<at::Tensor>` — one tensor per path-network vertex.

`OptimizerData` default hyperparameters: β₁ = 0.9, β₂ = 0.9, ε = 1e-8, step size = 0.01.

Gradient descent (plain `-lr * grad`) is also available as a fallback.

**`preprocess_gradients`** applies MAD-based outlier filtering before the optimizer step: gradients whose world-space magnitude deviates more than 85 modified Z-scores from the median are clamped, preventing a single badly-conditioned patch from dominating the update.

---

## Optimization Loop

**Files:** `library/LayoutOpt/Optimization.hh/.cc`, `apps/optimize.cc`

Each iteration:

1. **`eval()`** — build the differentiable loss, call `backward()`, collect gradients, compute Adam updates → returns `EvalInfo` with gradient tensors and update directions
2. **`apply()`** — move each `SurfacePoint::bary_coords` along its update direction (tracing along the surface if the step crosses a triangle boundary), then rebuild the overlay mesh

The path network is periodically resampled (`Resample.cc`) to redistribute vertices along each arc for numerical stability; resampling detaches and re-initializes the `bary_coords` tensors.

---

## Key Design Decisions

- **No GPU, no `requires_grad` on mesh data.** Only `bary_coords` participate in the computation graph. Target mesh positions are fixed constants fed as tensors.
- **Per-vertex optimizer state as `at::Tensor`.** Moment vectors are tensors so arithmetic stays in the LibTorch ecosystem, but they are always detached — they accumulate statistics, they do not propagate gradients.
- **Double precision throughout.** All tensors use `torch::kFloat64` to match the `double`-precision polymesh/Eigen pipeline.
- **Custom Adam instead of `torch::optim`.** The built-in optimizers assume Euclidean parameter spaces; the custom implementation corrects the step size normalization to use world-space displacement.
