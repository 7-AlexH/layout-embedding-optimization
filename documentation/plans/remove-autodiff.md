# Plan: Remove LibTorch Autodiff (Hand-Rolled Adjoint Differentiation)

**Goal:** Eliminate LibTorch entirely from the codebase by replacing its autodiff with
hand-written reverse-mode (adjoint) differentiation over plain `double` / Eigen / typed-geometry
code, to test the performance limits of the method.

**Non-goals:** Changing the optimization algorithm, the loss definitions, or the
path-network/overlay data model. The optimizer trajectory should be bitwise-explainable
relative to the current implementation (identical up to floating-point reassociation),
except where we *deliberately* change a gradient boundary (see Phase 2).

---

## 1. Current state (what actually needs differentiating)

A full audit of the 442 `torch::`/`at::Tensor` references shows two distinct categories:

### A. The real autodiff graph (needs hand adjoints)

One scalar loss, reverse mode, one backward pass. Leaves are the `bary_coords` of
path-network vertices that map to **layout nodes** (face points, 2 effective DOF each) —
`compute_differentiable_surface_points` (`DifferentiableIntersection.cc:38`) sets
`requires_grad` only on those; strip-interior path vertices are *derived* via
differentiable intersections.

> ⚠ `documentation/libTorchUsage.md` claims the opposite (layout-mapped nodes skipped).
> The code is authoritative; confirm during the Phase 2 audit and fix the doc.

Forward stage graph:

| Stage | Code | Math | Adjoint difficulty |
|---|---|---|---|
| S1: bary → 3D node pos / 2D strip endpoints | `SurfacePoint::get_pos` | linear interp | trivial |
| S2: strip 2D flattening (endpoint faces only) | `torch_compute_2D_face_embedding`, parts of `FlattenTriangleStrip.cc` | norms, atan2, cos/sin | easy calculus, fuzzy boundary (see Phase 2) |
| S3: intersections → overlay positions | `DifferentiableIntersection.cc`, `torch_compute_intersection_parameter` | 2D line–line closed form | easy |
| S4: arc lengths → `t` → boundary UVs | `torch_compute_embedded_layout_edge_lengths`, `torch_compute_t_for_pn_halfedge`, `torch_compute_pn_uvs` | norms, running sums along halfedge walks | easy math, **bookkeeping-heavy** (reverse-order walks) |
| S5: cotans, face areas | `torch_cotans`, `torch_face_areas` | closed form per edge/face | easy |
| S6: per-patch harmonic solve → interior UVs | `torch_harmonic_param` | `u = L⁻¹ b` | **textbook adjoint + sparse refactor** (see §3) |
| S7: Jacobian → 2×2 SVD → distortion → loss | `compute_distortion`, `torch_singular_values` | closed form (SVD already hand-rolled!) | easy |
| (parallel) curvature alignment | `principal_curvature_alignment_loss` | dot/norm/atan2 | easy |

### B. Tensor-flavored plain geometry (no adjoints — mechanical port)

These use tensors but are **already detached** from the graph (constructed from
`.item<double>()` values, or run on optimizer outputs):

- `Update.cc` (`do_step` step tracing) — entirely detached.
- `Resample.cc` — detaches and re-initializes `bary_coords` by design.
- Most of `FlattenTriangleStrip.cc` (`puzzle_triangle_fan`, `transform_*`,
  `adapt_metric`) — `.item()` calls cut the graph mid-chain.
- `Optimizers.cc` — Adam state `m`/`v` are tensors used as plain 2-vectors/scalars.
- `preprocess_gradients`, `EmbeddingUtils.cc` helpers, debug/visualization conversions.

Port these to `tg::pos2/pos3` / `Eigen::Vector2d` directly; the code gets *simpler*.

### Inactive losses — decision needed

`variance_loss`, `distorion_loss`, `distorion_loss_extrinsic`, `total_length_loss`,
`inner_angle_loss`, `repel_loss_v1/v2` are implemented but disabled in `eval()`.
**Recommendation: do not port them.** Move them to an `attic/` file that only compiles
under the torch flag (Phase 0), and delete them with Phase 6. Porting them later is
mechanical once the adjoint framework exists. If any are wanted for the paper, flag it
now and add them to Phase 4 scope (each is a day or less; none touch the solve except
the two Yamabe ones, which reuse the S6 adjoint).

---

## 2. Strategy

**Oracle-driven, back-to-front, one stage at a time.**

1. Keep the entire torch path compiling behind a CMake option
   (`LAYOUTOPT_WITH_TORCH`, default ON until Phase 6).
2. Port stages in reverse order (S7 first). At each step the hand-rolled stage's
   input adjoints are compared against the torch graph's intermediate gradients
   (`tensor.retain_grad()` on stage boundaries) on a small model, to ~1e-10.
3. Only after a stage matches do we move one stage earlier. We are never debugging
   more than one stage of hand AD at a time.
4. Finite differences are the backstop oracle for anything torch itself gets wrong
   or where we deliberately change the gradient boundary.

### Gradient representation (design decision)

No general-purpose tape / operator-overloading type. Each stage gets an explicit
`forward` + `backward` function pair:

- Forward functions store whatever intermediates backward needs in a per-stage
  context struct (a manual, stage-granular tape). For the halfedge-walk stages (S4),
  forward records visit order + running values into flat arrays; backward iterates
  them in reverse.
- Stage boundaries exchange plain adjoint buffers, e.g.
  `d_overlay_pos : Eigen::MatrixX3d` (∂loss/∂ overlay positions),
  `d_uvs`, `d_cotans`, `d_arc_lengths`, finally `d_bary : Eigen::Vector2d` per leaf.
- `SurfacePoint::bary_coords` becomes a fixed-size `tg::dpos3`-style value
  (3 doubles + the existing type/heh fields); gradients live in a separate
  per-eval buffer, not on the point. `SurfacePoint::copy()` and all
  detach-dance code disappears.
- New files: `library/LayoutOpt/Adjoint/` (one .cc per stage pair is fine), plus
  `GradCheck.hh/cc` for the comparison harness. `TorchUtils` shrinks until it dies
  in Phase 6.

### The solve adjoint (S6) — the only non-elementwise op

Forward, per patch: assemble the interior cotan Laplacian `L` (symmetric) as
`Eigen::SparseMatrix<double>`, factor **once** with `SimplicialLDLT`, backsolve for
`u` and `v`.

Backward: given `ḡ_u, ḡ_v` (∂loss/∂ interior UVs):

```
λ_u = L⁻ᵀ ḡ_u   (same factorization, L symmetric)
λ_v = L⁻ᵀ ḡ_v
b̄_u = λ_u ;  b̄_v = λ_v                    → scatter into boundary-UV / cotan adjoints via rhs assembly
L̄ᵢⱼ = −(λ_u)ᵢ uⱼ − (λ_v)ᵢ vⱼ              → only for stored sparse entries (i,j)
                                            → scatter into cotan adjoints via the same
                                              index maps used in assembly (map_to_vec)
```

One factorization serves all four solves. This replaces the current **four effective
dense O(n³) solves per patch per iteration** and is the largest single perf win in
the codebase — which is why it lands first (Phase 1) while the torch oracle still
covers everything.

---

## 3. Execution protocol: per-phase checkpoint

**Before starting each phase, Claude pauses and presents:**

1. A recap of what the phase will touch and any findings from the previous phase
   that change its scope.
2. A **recommended model + effort level** for the phase (with one-line rationale),
   re-evaluated at that moment — the table below is the provisional starting point,
   but actual recommendations account for what earlier phases revealed.
3. Waits for explicit go-ahead (and the user switching model/effort via
   `/model` / `/effort` if they accept the recommendation) before writing any code.

Provisional recommendations:

| Phase | Model | Effort | Rationale |
|---|---|---|---|
| 0 Scaffolding | Sonnet 4.6 | medium | CMake/harness plumbing; design is already specified here |
| 1 Sparse solve | Fable 5 | high | Adjoint correctness + `torch::autograd::Function` subtleties; bugs here poison everything downstream |
| 2 Boundary audit | Fable 5 | high | Pure careful reading/analysis; the riskiest unknown in the project |
| 3 Core types | Sonnet 4.6 | medium | Mechanical type refactor against a spec |
| 4 Adjoint port | Fable 5 | high | The hand-derived math; S4's reverse-walk bookkeeping especially |
| 5 De-torchification | Sonnet 4.6 | medium | Mechanical; large-volume edits verifiable by compile + identical loss values; good subagent fan-out candidate |
| 6 Removal + cleanup | Sonnet 4.6 | medium | Deletion, docs, final timing runs (bump to Fable 5 if the perf writeup needs analysis) |

Within Phase 4, the checkpoint applies **per stage item** (S7, S6, S5, S4, alignment,
S3, S2, S1), since difficulty varies: S7/S5/S3 are easy calculus (Sonnet-capable if
budget matters), while S6, S4, and S2 warrant Fable 5 at high effort.

---

## 4. Phases

### Phase 0 — Scaffolding (1–2 days)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Sonnet 4.6, medium).

- [ ] Add `LAYOUTOPT_WITH_TORCH` CMake option; gate `find_package(Torch)`
      (`CMakeLists.txt:95-97`) and all torch-including TUs on it. (Until Phase 4 the
      flag is required-ON; the point is to make the dependency boundary explicit
      and let the new path compile both ways.)
- [ ] Deterministic comparison mode: fixed model + layout, fixed iteration count,
      resampling at fixed iterations, no viewer. Verify two consecutive runs produce
      identical loss sequences (they should — there is no RNG in the loop; confirm).
- [ ] Gradient-diff harness: run one `eval()` under torch, dump per-leaf
      `bary_coords.grad()` and selected intermediate grads
      (`retain_grad()` on `_omd.torch_pos_`, `torch_uvs_`, cotans, per-patch `L`/rhs)
      to a file; a check mode reruns with the hand-rolled stage and asserts
      max-abs-diff < 1e-9 (relative where magnitudes are large).
- [ ] Capture baseline timings (existing `TimerCollection`: EvalObjective,
      Backpropagation, Update, Embedding) on 2–3 models; record in
      `documentation/plans/remove-autodiff-baseline.md`. These are the numbers the
      whole exercise is measured against.
- [ ] Move inactive losses to `ObjectiveFunctionsAttic.cc` (torch-flag-only).

### Phase 1 — Sparse solve inside torch-land (1–2 days)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Fable 5, high).

- [ ] Replace the dense `torch::linalg::solve` in `torch_harmonic_param` with a
      custom autograd function: forward assembles `Eigen::SparseMatrix` from the
      (torch) cotan values' doubles, factors with `SimplicialLDLT`, solves u and v;
      backward implements the adjoint formulas above and returns grads w.r.t. the
      cotan tensor and boundary-UV tensors. (`torch::autograd::Function` subclass.)
- [ ] Validate: loss + leaf gradients match the dense path to tolerance; timings
      re-captured (expect a large drop in EvalObjective + Backpropagation already).
- This validates the *exact* assembly/scatter plumbing the final version will use,
  while the full torch oracle still surrounds it.

### Phase 2 — Gradient-boundary audit (1 day)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Fable 5, high).

The current graph is cut implicitly wherever `.item<double>()` intervenes. Map it
precisely; the port must replicate it (or deliberately improve it — record any
deviation and validate those cases by finite differences instead of the oracle):

- [ ] Catalog every `.item()` on the path from leaves to loss. Known cut points to
      classify: `puzzle_triangle_fan` (`FlattenTriangleStrip.cc:308,335,352` — angles
      and rotations detached), `transform_*` (new edge `SurfacePoint`s built from
      `params[1].item()` — detached), `compute_distortion` early-outs (control flow
      only, fine), `adapt_metric` rotations (mixed).
- [ ] Determine exactly which strip-flattening outputs carry gradients (expectation:
      only the two endpoint-face embeddings via `set_triangle_pos` →
      `torch_compute_2D_face_embedding`; everything downstream of the first fan
      unfold is detached). Write the result down as the **normative gradient
      boundary** in this document before porting S2/S3.
- [ ] Resolve the doc/code contradiction about which vertices are free
      (code: layout-node face points only); update `libTorchUsage.md`.

### Phase 3 — Core types (1 day, overlaps Phase 4)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Sonnet 4.6, medium).

- [ ] `SurfacePoint` → plain doubles (`bary[3]` + type + heh). Keep API
      (`get_pos`, `fh`, `is_inside_element`) over `tg`/Eigen types.
- [ ] Replace tensor mirrors (`torch_pos_`, `torch_uvs_`, `torch_t_`,
      `torch_embedded_edge_length_`) with `Eigen::MatrixX3d` / flat
      `std::vector<double>` equivalents (`pos_mat_`, `uvs_`, ...). During Phases 3–5
      both representations coexist behind the flag.
- [ ] `OptimizerData::m/v` → `Eigen::Vector2d` / `double` attributes.

### Phase 4 — Stage-by-stage adjoint port (6–9 days, back to front)

> ⏸ **Checkpoint before starting, and before each stage item below** — confirm
> model/effort per item (provisional: Fable 5 high for S6/S4/S2; Sonnet 4.6 high is
> acceptable for S7/S5/S3/alignment/S1 if budget matters).

Each item = forward port + hand adjoint + oracle match on the small model before
moving on. Order:

- [ ] **S7** `compute_distortion` + closed-form 2×2 SVD derivative + per-patch
      summation. Inputs treated as leaves for validation. Includes the AIAP
      min/max branch (subgradient choice must match torch's).
- [ ] **S6** harmonic solve — port the Phase 1 custom function out of
      `torch::autograd::Function` into the plain stage-pair form; add the
      `torch_prepare_param` indexing (port as-is, it's already torch-free logic).
- [ ] **S5** cotans + areas (vectorize over Eigen; adjoint scatters into
      `d_overlay_pos`).
- [ ] **S4** arc lengths → `t` → boundary UVs. Forward records halfedge walk order;
      backward replays in reverse. The fiddliest bookkeeping in the port — budget
      accordingly and gradcheck against torch *per arc* if the full-stage diff
      doesn't converge quickly.
- [ ] **Curvature alignment loss** (independent branch into `d_overlay_pos`;
      port `PrincipalCurvature.cc` tensor fields to Eigen).
- [ ] **S3** intersection parameters → overlay positions (adjoint of the 2D
      line–line closed form; scatter `d_overlay_pos` into endpoint 2D positions
      and fixed target positions — the latter are constants, drop them).
- [ ] **S2** endpoint-face 2D embedding per the Phase 2 normative boundary.
- [ ] **S1** `get_pos` interpolation adjoint → accumulate `d_bary`; wire into
      `compute_gradients` replacement. Full-pipeline leaf-gradient match +
      finite-difference check.
- [ ] End-to-end: N iterations (across at least one resample) with both paths;
      loss trajectories and final embeddings match to tolerance.

### Phase 5 — De-torchify the detached code (2–4 days, parallelizable with Phase 4)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Sonnet 4.6,
> medium; candidate for parallel subagent fan-out, one agent per file).

Mechanical; verifiable by compile + identical loss values (these paths don't carry
gradients, but they *do* affect forward values — keep the comparison mode running):

- [ ] `Update.cc` (`do_step`) — pure 2D geometry on doubles.
- [ ] `Resample.cc`, `FlattenTriangleStrip.cc` non-differentiable remainder,
      `EmbeddingUtils.cc`, `Embedding.cc`, `IO.cc`, `Optimizers.cc`,
      `preprocess_gradients`.
- [ ] Visualization/debug helpers (`torch_to_pos2/3` call sites).

### Phase 6 — Removal & cleanup (1–2 days)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Sonnet 4.6, medium).

- [ ] Delete the torch path, `TorchUtils.*`, attic losses, the gradient-diff
      harness's torch side (keep the finite-difference checker permanently —
      it's the regression test this repo doesn't have).
- [ ] Remove `find_package(Torch)`, `extern/libtorch*`, `download_libtorch.sh`
      reference from README/CLAUDE.md; update `libTorchUsage.md` → replace with
      `documentation/adjointDifferentiation.md` describing the hand-rolled system.
- [ ] Re-enable the per-patch OpenMP loop in `harmonic_distortion_loss`
      (`ObjectiveFunctions.cc:58` — commented out today, almost certainly because
      torch autograd graph construction isn't thread-safe across shared leaves;
      plain doubles + per-patch buffers make it embarrassingly parallel).
- [ ] Final timings vs. Phase 0 baseline; write results into this document.

---

## 5. Validation summary

| Check | When | Tolerance |
|---|---|---|
| Per-stage adjoint vs torch `retain_grad()` oracle | each Phase 4 item | 1e-9 abs / 1e-7 rel |
| Leaf gradients vs torch | end of Phase 4 | 1e-9 abs |
| Leaf gradients vs central finite differences (h≈1e-6, double precision) | S1 done + any deliberate boundary change | 1e-5 rel |
| Loss trajectory over N iters incl. resample, both paths | Phases 4–5 continuously | 1e-8 rel drift |
| Determinism (two identical runs) | Phase 0, then continuously | exact |

## 6. Risks

- **Implicit gradient boundary** (the big one): if the port accidentally
  differentiates *more or less* than torch did, gradients diverge for non-bug
  reasons. Mitigated by the Phase 2 audit producing a written normative boundary
  before S2/S3 are ported, and by finite differences arbitrating disagreements.
- **Subgradient/branch choices**: `compute_distortion` early-outs, AIAP min/max,
  `torch::clamp` ends, `max(0.02, ·)` in `torch_compute_pn_uvs` — each branch's
  derivative convention must match torch (clamp: zero gradient outside; max: gradient
  to the larger arg).
- **Reverse-walk bugs in S4** manifest as *almost*-right gradients. Per-arc
  gradcheck localizes them.
- **Doc/code mismatch on free variables** — resolve in Phase 2 before it confuses
  the port.
- **Numerical divergence accumulating over iterations** (reassociation): trajectories
  will drift slowly even when both are correct; that's why per-eval gradient checks,
  not only trajectory checks, are the acceptance criterion.

## 7. Effort estimate

| Phase | Effort |
|---|---|
| 0 Scaffolding | 1–2 days |
| 1 Sparse solve (in torch) | 1–2 days |
| 2 Boundary audit | 1 day |
| 3 Core types | 1 day |
| 4 Adjoint port | 6–9 days |
| 5 De-torchification | 2–4 days (parallelizable) |
| 6 Removal + cleanup | 1–2 days |
| **Total** | **~2–3 weeks focused** |

Expected outcome: per-iteration eval+backward drops from torch-graph-dominated cost
to roughly the cost of the sparse factorizations plus linear scans — order(s) of
magnitude, with Phase 1 alone delivering a large fraction of it as an early checkpoint.
