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

> ⚠ `documentation/libTorchUsage.md` claimed the opposite (layout-mapped nodes skipped).
> **Resolved in Phase 2:** the code is authoritative; the doc was corrected 2026-06-11.

Forward stage graph:

| Stage | Code | Math | Adjoint difficulty |
|---|---|---|---|
| S1: bary → 3D node pos / 2D strip endpoints | `SurfacePoint::get_pos` | linear interp | trivial |
| S2: strip 2D flattening (endpoint faces only) | `torch_compute_2D_face_embedding`, parts of `FlattenTriangleStrip.cc` | norms, atan2, cos/sin | **eliminated by the Phase 2 audit** — entirely constant, no adjoint needed (see Phase 2 result) |
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
- **All** of `FlattenTriangleStrip.cc` (`puzzle_triangle_fan`, `transform_*`,
  `adapt_metric`) — it runs before `requires_grad` is ever set, so its inputs are
  constants and its `.item()` calls never cut live graph (Phase 2 result).
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

- [x] Add `LAYOUTOPT_WITH_TORCH` CMake option; gate `find_package(Torch)`
      (`CMakeLists.txt:95-97`) and all torch-including TUs on it. (Until Phase 4 the
      flag is required-ON; the point is to make the dependency boundary explicit
      and let the new path compile both ways.)
- [x] Deterministic comparison mode: fixed model + layout, fixed iteration count,
      resampling at fixed iterations, no viewer. Verify two consecutive runs produce
      identical loss sequences (they should — there is no RNG in the loop; confirm).
      *(Confirmed bitwise — see baseline doc; required the Windows xmm14/15 and
      stack-size fixes first.)*
- [x] Gradient-diff harness: run one `eval()` under torch, dump per-leaf
      `bary_coords.grad()` and selected intermediate grads
      (`retain_grad()` on `_omd.torch_pos_`, `torch_uvs_`, cotans, per-patch `L`/rhs)
      to a file; a check mode reruns with the hand-rolled stage and asserts
      max-abs-diff < 1e-9 (relative where magnitudes are large).
      *(`GradCheck.hh/cc` + file-dump; the comparison itself must run torch-free —
      `compare_gradcheck.py` — see baseline doc caveat.)*
- [x] Capture baseline timings (existing `TimerCollection`: EvalObjective,
      Backpropagation, Update, Embedding) on 2–3 models; record in
      `documentation/plans/remove-autodiff-baseline.md`. These are the numbers the
      whole exercise is measured against. *(Spot + Banana; Bunny blocked by a
      pre-existing init bug, documented there.)*
- [x] Move inactive losses to `ObjectiveFunctionsAttic.cc` (torch-flag-only).

### Phase 1 — Sparse solve inside torch-land (1–2 days)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Fable 5, high).

- [x] Replace the dense `torch::linalg::solve` in `torch_harmonic_param` with a
      custom autograd function: forward assembles `Eigen::SparseMatrix` from the
      (torch) cotan values' doubles, factors with `SimplicialLDLT`, solves u and v;
      backward implements the adjoint formulas above and returns grads w.r.t. the
      cotan tensor and boundary-UV tensors. (`torch::autograd::Function` subclass.)
      *(`library/LayoutOpt/Adjoint/SparseHarmonicSolve.{hh,cc}`, toggled by
      `HarmonicOptions::use_sparse_harmonic_solve`, default ON.)*
- [x] Validate: loss + leaf gradients match the dense path to tolerance; timings
      re-captured (expect a large drop in EvalObjective + Backpropagation already).
      *(PASSED: max abs leaf-grad diff 1.7e-16 over 578 leaves, loss bitwise
      identical; timings in the baseline doc — gains modest (Spot −14%/−9%) because
      the element-wise graph dominates; the qualitative win is removing the
      ABI-violating MKL kernel from the production path.)*
- This validates the *exact* assembly/scatter plumbing the final version will use,
  while the full torch oracle still surrounds it.

### Phase 2 — Gradient-boundary audit (1 day)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Fable 5, high).

The current graph is cut implicitly wherever `.item<double>()` intervenes. Map it
precisely; the port must replicate it (or deliberately improve it — record any
deviation and validate those cases by finite differences instead of the oracle):

- [x] Catalog every `.item()` on the path from leaves to loss. Known cut points to
      classify: `puzzle_triangle_fan` (`FlattenTriangleStrip.cc:308,335,352` — angles
      and rotations detached), `transform_*` (new edge `SurfacePoint`s built from
      `params[1].item()` — detached), `compute_distortion` early-outs (control flow
      only, fine), `adapt_metric` rotations (mixed). *(Done — full catalog below;
      headline: the FlattenTriangleStrip sites are not cuts at all, they operate on
      constants.)*
- [x] Determine exactly which strip-flattening outputs carry gradients (expectation:
      only the two endpoint-face embeddings via `set_triangle_pos` →
      `torch_compute_2D_face_embedding`; everything downstream of the first fan
      unfold is detached). Write the result down as the **normative gradient
      boundary** in this document before porting S2/S3. *(Done — the expectation was
      too pessimistic: **none** of the strip flattening carries gradients; see below.)*
- [x] Resolve the doc/code contradiction about which vertices are free
      (code: layout-node face points only); update `libTorchUsage.md`. *(Done —
      code confirmed authoritative; `libTorchUsage.md` corrected 2026-06-11.)*

#### Phase 2 result — the normative gradient boundary (2026-06-11)

**The boundary is much simpler than §1 feared: the entire strip flattening
(`TriangleStrip::heh_pos_2d`) is constant w.r.t. the autograd graph.** S2 disappears
as a differentiation stage.

**Why `heh_pos_2d` is constant (proof, two facts):**

1. *Call order.* In `compute_layout_embedding_shared_part` (`Embedding.cc:29-53`),
   `compute_2D_embeddig_per_strip` (strip flattening) runs first;
   `compute_differentiable_intersections_for_overlay_stable` — the only place
   `requires_grad` is ever set (`compute_differentiable_surface_points`,
   `DifferentiableIntersection.cc:38-41`) — runs last (`Embedding.cc:50`). Setting
   `requires_grad` on a tensor does not retroactively create graph for ops that
   already executed.
2. *Detach invariant.* At strip-build time every `bary_coords` in the path network is
   a fresh detached tensor: `do_step` (`Update.cc`) replaces **every** pn vertex's
   surface point each iteration — via `copy()` (= `clone().detach()`,
   `SurfacePoint.cc:6`) at `Update.cc:26/:31`, or fresh `torch::tensor(...)` /
   `torch::stack(...)` at `:68/:136/:195`. `init_path_network_surface_points`
   (`Embedding.cc`) and `Resample.cc` likewise create `requires_grad(false)` tensors.
   No leaf survives into the next embedding build.

Consequence: the `.item<double>()` calls inside `FlattenTriangleStrip.cc` are **not
graph cuts** — they are plain forward computation on constants. Even the
endpoint-face embeddings (`set_triangle_pos` → `torch_compute_2D_face_embedding`)
carry no gradients.

**Leaves (the free variables):** `bary_coords` (shape `[2]`: α, β; `FacePoint`) of
exactly those pn vertices with a valid `map_to_layout_vertices_` entry — i.e.
**layout nodes only**, 2 DOF each (`DifferentiableIntersection.cc:38-41`).
Everything else in the system is constant.

**Differentiable overlay rows.** `omd.torch_pos_` is rebuilt from constants
(`attr_to_torch(_omd.pos_)`); only two classes of rows are then overwritten with
graph-connected values (`DifferentiableIntersection.cc`):

1. *Layout-node rows* — `sp.get_pos(_tmd.torch_pos_)`: leaf-bary interpolation of
   constant 3D target-face corners (= S1).
2. *Strip-interior path-vertex rows* — per layout edge, per interior vertex:
   `intersect_pos = from_3D * params[1] + to_3D * params[0]`
   (`DifferentiableIntersection.cc:109`) with
   `params = torch_compute_intersection_parameter(intersect_seg, layout_seg)`, where
   `intersect_seg` (crossed target edge's 2D endpoints from `heh_pos_2d`) and
   `from_3D`/`to_3D` (its 3D endpoints) are **constant**, and
   `layout_seg = stack(A, B)` with A/B the *endpoint* leaves'
   `get_pos(heh_pos_2d)` — leaf-bary interpolation of **constant** 2D corners.
   So each strip-interior row depends on **only the two endpoint leaves of its
   strip**: leaf bary → 2D endpoint pos (linear) → intersection params (2D
   line–line closed form) → 3D position (linear in params).

All other rows of `omd.torch_pos_` are constants. Downstream of `omd.torch_pos_`
(S4–S7 + curvature alignment) everything is differentiable with no cuts: lengths →
t → uvs (`torch_compute_embedded_layout_edge_lengths`,
`torch_compute_t_for_pn_halfedge`, `torch_compute_pn_uvs`), `torch_cotans` /
`torch_face_areas`, `torch_harmonic_param` (sparse or dense), per-face triangle
rebuild + `compute_distortion`, `principal_curvature_alignment_loss`
(`direction_data.basis/.dir` are constants).

**`.item()` catalog (every site on or near the leaf→loss path):**

| Category | Sites | Verdict |
|---|---|---|
| Pre-boundary (operates on constants) | `FlattenTriangleStrip.cc`: `puzzle_triangle_fan` :308/:335/:352, `transform_single_v_sp` :508, `transform_multiple_v_sp` :650, `transform_v_sp_to_e_sp` :793 (+`_front_back` :944), `transform_special_case` :1016/:1028/:1072, `compute_strip_segment_length` :126 | not cuts — plain forward math; port as plain doubles (Phase 5) |
| Control flow / guards | parallel-line guard `TorchUtils.cc:341-346`; `compute_distortion` early-outs `ObjectiveFunctions.cc:426/:429`; `SurfacePoint::is_inside_element`; `pn_contains_degenerate_edges` `EmbeddingUtils.cc:895`; TORCH_CHECKs `ObjectiveFunctions.cc:464-465` | replicate the branch; no derivative through the condition |
| Forward-only outputs (off/after graph) | loss extraction `Optimization.cc:82`; eval-info logging `ObjectiveFunctions.cc:156/:170`; `compute_embedded_length_per_layout_edge` `EmbeddingUtils.cc:801`; arc-index bookkeeping `EmbeddingUtils.cc:772` (int64); geodesic-trace input `Embedding.cc:228-240` | no adjoint needed |
| Detached by design (§1 category B) | `Update.cc` `do_step` :68/:136/:195; `Resample.cc`; `Optimizers.cc:86`; `set_layout_pos_based_on_pn_sp` | mechanical port (Phase 5) |

**Subgradient / branch conventions the port must replicate:**

- `compute_distortion` early-outs (`:426` ref/param area < EPS, `:429` NaN) → the
  whole triangle contributes zero (zero gradient).
- AIAP min/max ordering (`ObjectiveFunctions.cc:510`) → gradient flows only through
  the selected tensors.
- `torch::max(0.02, ·)` in `torch_compute_pn_uvs` (`:707-708`) → zero gradient into
  the clamped argument when the constant wins.
- `+1e-10` in cotan denominators (`TorchUtils.cc:162-163`) → smooth; replicate verbatim.
- `torch::clamp(cos, −1, 1)` in the fan unfold → pre-boundary (constant); irrelevant.
- Parallel-line guard (`TorchUtils.cc:341-346`) returns an undefined tensor →
  control flow; replicate the branch, no derivative.

**Impact on Phase 4 scope:**

- **S2 is no longer a differentiation stage.** `heh_pos_2d` enters the port as a
  constant input array. The only "S2" math left is A/B = leaf-bary interpolation of
  constant 2D corners — S1's interpolation adjoint applied in 2D. No fan-unfold,
  `transform_*`, or `adapt_metric` differentiation, ever.
- **S3 shrinks:** only ∂`intersect_pos`/∂`params` (linear) and
  ∂`params`/∂`layout_seg` (line–line closed form) are needed; the `intersect_seg`,
  `from_3D`, `to_3D` adjoints drop out (constants).
- Perf note: strip flattening currently builds large dead autograd graphs from
  constants on every rebuild — pure waste; disappears with the Phase 5 port.

### Phase 3 — Core types (1 day, overlaps Phase 4) — ✅ COMPLETE (2026-06-11, Opus 4.8 medium)

> ⏸ **Checkpoint before starting** — confirm model/effort (provisional: Sonnet 4.6, medium).
> *Done with Opus 4.8 / medium. Approach: **additive coexistence** — the torch
> tensors stay authoritative (so the gradient oracle is bitwise-unchanged); the
> plain-double / Eigen representations are added alongside, populated at the
> existing single chokepoints. Phase 4 flips each consuming stage to read them.*

- [x] `SurfacePoint` → plain doubles. Added `vec3d bary_full()` (derives the full
      `(α, β, γ)` from the authoritative tensor: vertex→`(1,0,0)`, edge→`(α,1-α,0)`,
      face→`(α,β,1-α-β)`) and a plain-double `get_pos(Eigen::MatrixX3d, mesh)` that
      mirrors the torch interpolation. Tensor `bary_coords` remains the leaf; the
      flip of `bary_full()` to a stored authoritative field is deferred to Phase 5/6
      (decided at the S1 closeout). `fh`/`is_inside_element`
      unchanged. *Validated: Eigen `get_pos` reproduces torch `get_pos` to **0** abs
      diff over 3675 surface points (the `+0·C` edge term is a bitwise no-op).*
- [x] Tensor mirrors added (NOT replaced — both coexist behind the flag through
      Phases 3–5): `TargetMeshData::pos_mat_` (`Eigen::MatrixX3d`, ctor),
      `OverlayMeshData::pos_mat_` (refreshed in `sync_tg_and_torch`),
      `LayoutData::embedded_edge_length_` (`Eigen::VectorXd`), and
      `PathNetworkData::t_flat_` / `uvs_flat_` (`Eigen::VectorXd` / `Eigen::MatrixX2d`,
      flat by pn-halfedge idx) — the differentiable mirrors are derived from each
      torch tensor right after its `torch::stack`, and reset alongside the torch
      resets in `reset_embedding_data`. *Validated: all mirrors = torch to **0**
      abs diff; loss bitwise-identical to the banana baseline (1.2734862918788761).*
- [x] `OptimizerData::m_eigen`/`v_eigen` (`vec2d` / `double` attributes) added and
      zero-initialized in `init_vetor_adam_param`. Per-step writes deferred to Phase 5
      (the optimizer is detached/torch today and is de-torchified there); no Phase 4
      stage consumes them, so wiring the writes now would be untestable dead code.

> **Validation harness:** `apps/gradcheck_phase3.cc` (banana, post init+resample+init
> + one forward loss eval) compares every mirror to its torch counterpart and dumps
> raw max-diffs to `gradcheck_phase3.txt` (file-dump per the xmm-safe protocol).
> Phase 1 gradcheck re-run after these changes: still PASSED (loss bitwise, 578
> leaves, external max abs diff 5.7e-17) — the gradient path is unperturbed.

### Phase 4 — Stage-by-stage adjoint port (6–9 days, back to front)

> ⏸ **Checkpoint before starting, and before each stage item below** — confirm
> model/effort per item (provisional: Fable 5 high for S6/S4/S2; Sonnet 4.6 high is
> acceptable for S7/S5/S3/alignment/S1 if budget matters).

Each item = forward port + hand adjoint + oracle match on the small model before
moving on. Order:

- [x] **S7** `compute_distortion` + closed-form 2×2 SVD derivative + per-patch
      summation. Inputs treated as leaves for validation. Includes the AIAP
      min/max branch (subgradient choice must match torch's).
      **DONE 2026-06-11 (Fable 5, xhigh).** Stage pair in
      `library/LayoutOpt/Adjoint/DistortionStage.{hh,cc}` (face forward/backward,
      patch gather/scatter, loss assembly; early-out faces contribute zero
      value+gradient, AIAP selection replicated via the same `s0 < s1` value
      comparison). The per-face rebuild was extracted verbatim from the
      `harmonic_distortion_loss` patch loop into `torch_face_distortion`
      (`ObjectiveFunctions.cc`) so the gradcheck oracle IS the production code.
      `apps/gradcheck_phase4_s7.cc` (banana, leaves = per-patch inner/boundary
      UVs + overlay positions): oracle loss == production loss **bitwise** in
      both configs (production weights 1.1873013392084315, all-branches
      2.3348588562371528); hand adjoints worst-metric ≤ 1 at 1e-9/1e-7
      (max abs diffs: d_inner 1.1e-14, d_boundary 4.7e-13, d_overlay_pos
      3.6e-10 — the latter from the SDE_DirectComputation branch's literal
      `factor1` chain). Raw dumps in `gradcheck_phase4_s7.txt`. Phase 1
      gradcheck re-run: PASSED (loss bitwise 1.2734862918788761, external max
      abs diff 1.7e-16); Phase 3 mirror check re-run: PASSED (all diffs 0) —
      the refactor is bitwise-safe.
- [x] **S6** harmonic solve — port the Phase 1 custom function out of
      `torch::autograd::Function` into the plain stage-pair form; add the
      `torch_prepare_param` indexing (port as-is, it's already torch-free logic).
      **DONE 2026-06-11 (Fable 5, xhigh).** Stage pair in
      `library/LayoutOpt/Adjoint/HarmonicStage.{hh,cc}`: `prepare_param`
      (line-for-line port of `torch_prepare_param`, boundary UVs gathered from
      a plain per-halfedge `Eigen::MatrixX2d`) + `harmonic_param_forward/
      backward` (Phase 1 solve restructured into ctx form; assembly/solve
      helpers duplicated from `SparseHarmonicSolve.cc` rather than refactoring
      validated Phase 1 code — the duplication dies in Phase 6). **No
      production code changed**, so Phase 1/3/S7 re-runs were unnecessary.
      `apps/gradcheck_phase4_s6.cc` (banana, composed S6+S7 chain; leaves =
      cotans + per-patch boundary UVs + overlay positions) PASSED both weight
      configs: oracle loss == production loss **bitwise** (1.1873013392084315 /
      2.3348588562371528); `prepare_param` port handle- and UV-identical
      (0 mismatches over all patches); hand inner UVs **bitwise-equal** to the
      production sparse solve (max abs 0 — identical traversal, triplet order,
      LDLT); adjoint max abs diffs: d_cotans 5.5e-18 / 2.7e-17, d_boundary_uvs
      4.7e-13 / 4.5e-13, d_overlay_pos 4.6e-13 / 3.6e-10 (worst ratio 0.357 in
      the all-branches config — the same SDE_DirectComputation `factor1` chain
      seen in S7). Raw dumps in `gradcheck_phase4_s6.txt`.
- [x] **S5** cotans + areas (vectorize over Eigen; adjoint scatters into
      `d_overlay_pos`). **DONE 2026-06-11 (Opus 4.8, high).** Stage pair in
      `library/LayoutOpt/Adjoint/CotanStage.{hh,cc}`: `cotans_forward`
      (per-edge `e_ia·e_ja/(|e_ia×e_ja|+1e-10) + e_ib·e_jb/(|e_ib×e_jb|+1e-10)`,
      indexed by edge idx; ctx stows only the 4 incident vertex indices) +
      `cotans_backward` (recomputes edge vectors/crosses, scatters into
      `d_overlay_pos`; cross-product adjoint `d_a = b×d_c`, `d_b = d_c×a`).
      **Face areas: deliberately NOT ported** — `torch_face_areas`' result is
      dead in the live loss (computed at `ObjectiveFunctions.cc:25`, never read;
      only the experimental Yamabe losses in `ObjectiveFunctionsAttic.cc`
      consume face areas, and those stay torch-gated). No production code
      changed. `apps/gradcheck_phase4_s5.cc` (banana, composed S5→S6→S7; leaves
      = overlay positions + per-patch boundary UVs; cotans derived from the
      position leaf with `retain_grad()` so the intermediate `d_cotans` is still
      comparable) PASSED both configs: oracle == production loss **bitwise**
      (1.1873013392084315 / 2.3348588562371528); `cotans_forward` **bitwise ==
      `torch_cotans`** (max abs 0 — Eigen cross/norm/dot matched torch on this
      data), so the downstream solve also stayed **bitwise** (inner solve max
      abs 0); adjoint max abs diffs: d_cotans 5.5e-18 / 2.7e-17, d_boundary_uvs
      4.7e-13 / 4.5e-13, d_overlay_pos 4.6e-13 / 3.6e-10 (worst ratio 0.357 in
      the all-branches config — unchanged from S6; the cotan path adds a
      negligible 16th-digit contribution to the dominant direct-3D path, the
      same SDE_DirectComputation `factor1` chain). Raw dumps in
      `gradcheck_phase4_s5.txt`.
- [x] **S4** arc lengths → `t` → boundary UVs. Forward records halfedge walk order;
      backward replays in reverse. **DONE 2026-06-11 (Fable 5, xhigh).** Stage
      triple in `library/LayoutOpt/Adjoint/BoundaryUvStage.{hh,cc}`:
      `edge_lengths_forward/backward` (per-layout-edge scatter-add of pn-segment
      norms; adjoint `±(d_len/|seg|)·(pa−pb)` into `d_overlay_pos`),
      `pn_t_forward/backward` (per layout-halfedge arc chain: walk recorded as
      steps with saved seg norms + prefix sums `acc_before` + chain `total`;
      backward replays in reverse carrying `d_acc`, with the quotient adjoint
      `d_total += d_t·acc_before/total²` and `d_lengths[e] += d_total` over the
      chain edges), `pn_uvs_forward/backward` (per-face 4-side bilinear corner
      blend; `max(0.02, res)` gate follows ATen `max.other` — grad to `res` iff
      `res ≥ 0.02`; torch's exit-B-then-overwrite-with-next-A aliasing proven
      value- and gradient-identical, so side starts are treated uniformly as A).
      `prepare_param` gained an optional `_boundary_src_hehs` out-param (records
      the overlay-halfedge row each boundary UV was gathered from — gradcheck
      scatter only, existing call sites unaffected); otherwise **no production
      code changed**. Backward ordering constraint documented in the header:
      pn_uvs and pn_t both accumulate into `d_lengths`, so edge_lengths_backward
      must run LAST. `apps/gradcheck_phase4_s4.cc` (banana, composed
      S4→S5→S6→S7; single leaf = overlay positions, oracle built by temporarily
      swapping the leaf into `_omd.torch_pos_` and running the production torch
      S4 functions with `retain_grad()` on lengths/t/uvs/cotans) PASSED both
      weight configs: oracle == production loss **bitwise** (1.1873013392084315
      / 2.3348588562371528); prepare_param structural mismatches 0. **Finding:
      S4's forward is NOT bitwise vs torch** (unlike S5's cotans) — hand Eigen
      segment norms differ from torch's norm kernel by ≤1 ulp on some segments
      (lengths 1.39e-17, t 3.33e-16, uvs 4.16e-17; downstream solve stays at
      2.78e-17), so the S5/S6-style bitwise boundary-UV memcmp was demoted to a
      reported `bnd_gather_max_abs` (~ulp), keeping the plan's tolerance-based
      adjoint criterion (1e-9/1e-7) as the gate. Adjoint max abs diffs
      (production / all-branches): d_uvs 1.2e-12 / 9.0e-13, d_t 1.3e-13 /
      9.4e-14, d_lengths 1.3e-14 / 6.7e-14, d_cotans 6.7e-18 / 2.7e-17,
      d_overlay_pos 2.5e-13 / 3.6e-10 (worst ratio 0.357 in all-branches —
      unchanged from S5/S6/S7, the known direct-3D-path ulp effect; note
      d_overlay_pos is now the TOTAL loss gradient w.r.t. overlay positions:
      S4 uv path + S5 cotan path + S7 direct path composed). Raw dumps in
      `gradcheck_phase4_s4.txt`.
- [x] **Curvature alignment loss** (independent branch into `d_overlay_pos`;
      port `PrincipalCurvature.cc` tensor fields to Eigen). **DONE 2026-06-11
      (Fable 5, xhigh; S5-grade difficulty in practice — no walk bookkeeping).**
      Stage pair in `library/LayoutOpt/Adjoint/CurvatureAlignmentStage.{hh,cc}`:
      per pn edge `vec → |vec| → basis·vec → atan2 → 4-rosy (cos 4θ, sin 4θ) →
      |dir − rosy|²`, loss = Σ(n·align)/Σ(n); backward chains the quotient rule
      (d_n gets numerator AND denominator paths), the rosy/trig adjoint, the
      atan2 adjoint `(−v2.y, v2.x)/|v2|²`, and `d_vec = basisᵀ·d_v2 +
      (d_n/n)·vec` scattered ± into `d_overlay_pos`. The basis/dir field data
      are constants (fixed target mesh) — no gradient flows into them; the dead
      commented-out `confidence` factor is not mirrored. Production change:
      `DirectionFieldData` gained Phase-3-style Eigen mirrors `basis_eigen` /
      `dir_eigen`, filled in `smooth_direction_field` from the same doubles the
      tensors are built from (verified **bitwise-equal**, mirror max abs 0).
      `apps/gradcheck_phase4_curvalign.cc` (banana, branch validated
      standalone; single leaf = overlay positions via the S4 leaf-swap; two
      backward seeds w=1.0 / w=0.7) PASSED: oracle == production loss
      **bitwise** (0.86184952670444603); hand loss within 8.9e-16 (summation
      order over ~3.9k edges; like S4, std::atan2/cos/sin vs torch kernels make
      bitwise unattainable); d_overlay_pos max abs 1.8e-15 / 1.3e-15 (worst
      ratio 2.4e-7 — 7 orders inside the 1e-9/1e-7 gate). S4 gradcheck re-run
      clean after the `PrincipalCurvature` edit (additive struct fields only).
      Raw dumps in `gradcheck_phase4_curvalign.txt`.
- [x] **S3** intersection parameters → overlay positions (adjoint of the 2D
      line–line closed form; scatter `d_overlay_pos` into endpoint 2D positions
      and fixed target positions — the latter are constants, drop them).
      **DONE 2026-06-11 (Fable 5, xhigh; easy as predicted).** Stage pair in
      `library/LayoutOpt/Adjoint/IntersectionStage.{hh,cc}`: per strip, with
      endpoint 2D positions A/B as the only differentiable inputs (Phase 2
      audit), forward replays `torch_compute_intersection_parameter` with
      line_a = (to_2D, from_2D), line_b = (A, B) and overwrites each interior
      pn vertex's overlay row with `from_3D·t + to_3D·(1−t)`; backward routes
      `d_t = d_pos[row]·(from_3D−to_3D)` through the quotient
      (`d_t_numer = d_t/cross`, `d_cross = −d_t·t_numer/cross²`) and the two
      2D-cross bilinear forms into `d_A = d_b0_a0 − d_b`, `d_B = d_b`
      (crossed-edge segment + 3D target rows are constants and drop). Setup
      lore: strips are LOCAL to `compute_layout_embedding_shared_part` and
      cannot be rebuilt post-init (`compute_2D_embeddig_per_strip` mutates the
      path network), so `apps/gradcheck_phase4_s3.cc` replicates the final
      `compute_layout_embedding_init(false)` inline from its public
      constituents keeping the strips — the bitwise-known reference losses
      double as the replication self-check. Oracle: per-layout-edge A/B leaves
      (recomputed via `sp.get_pos(strip.heh_pos_2d)`, detached) + verbatim
      replay of the production write loop against a detached clone of the
      overlay positions, `retain_grad()` after the last in-place write →
      `.grad` is the TOTAL post-write `d_overlay_pos`; then the S4 torch chain.
      PASSED both weight configs: ref losses **bitwise** the known values
      (1.1873013392084315 / 2.3348588562371528 — replication exact), oracle ==
      production **bitwise**, oracle rewrite max_abs **0** AND hand rewrite
      max_abs **0** over 3097 rewritten rows (the hand Eigen intersection
      arithmetic is bitwise == torch, unlike S4's norms). Adjoint max abs
      diffs (production / all-branches): d_overlay_pos 2.5e-13 / 3.6e-10
      (worst ratio 0.357, unchanged from S4), d_seg_A 4.4e-14 / 3.5e-11,
      d_seg_B 1.4e-14 / 7.7e-11 (worst ratio 0.077) — all inside the
      1e-9/1e-7 gate. Raw dumps in `gradcheck_phase4_s3.txt`.
- [x] **S2** endpoint-face 2D embedding per the Phase 2 normative boundary.
      **CLOSED AS NO-OP 2026-06-11 (Fable 5, medium).** Eliminated as a
      differentiation stage by the Phase 2 audit (see "Phase 2 result" above):
      the entire strip flattening `heh_pos_2d` — including the endpoint-face
      embeddings (`set_triangle_pos` → `torch_compute_2D_face_embedding`) — is
      constant w.r.t. the autograd graph, so no adjoint exists to write.
      Re-verified at closeout: (1) the only `requires_grad(true)` site in the
      library is `DifferentiableIntersection.cc:41`
      (`compute_differentiable_surface_points`), which runs AFTER
      `compute_2D_embeddig_per_strip` (`Embedding.cc:29-53`); all other sites
      are explicit `requires_grad(false)` (`Embedding.cc:142/177/183`);
      `torch_compute_2D_face_embedding`'s only callers are
      `FlattenTriangleStrip.cc:145` (pre-boundary) and `Update.cc:38`
      (detached-by-design, §1 category B). (2) Empirically: the S3 gradcheck
      treats `heh_pos_2d` as constant throughout and its oracle reproduces the
      production loss bitwise with all 3097 rewritten rows matching to 0 — if
      any gradient flowed through the flattening, those checks could not hold.
      The residual "S2 math" (A/B = leaf-bary interpolation of constant 2D
      corners) is S1's interpolation adjoint applied in 2D and is handled
      there; the forward flattening itself is ported as plain doubles in
      Phase 5. No code written for this item.
- [x] **S1** `get_pos` interpolation adjoint → accumulate `d_bary`; wire into
      `compute_gradients` replacement. Full-pipeline leaf-gradient match +
      finite-difference check. **DONE 2026-06-11 (Fable 5, xhigh) —
      `Adjoint/LeafStage.{hh,cc}` + `Adjoint/HandGradients.{hh,cc}` +
      `apps/gradcheck_phase4_s1.cc` PASSED both configs.** The leaf stage
      handles BOTH `get_pos` corner conventions: (a) 3D layout-node overlay
      rows (corners `hh.from / hh.to / hh.next.to` into
      `TargetMeshData::pos_mat_`) and (b) per-layout-edge 2D strip endpoints
      A/B (corners `pos[hh] / pos[hh.next()] / pos[hh.next().next()]` over the
      constant strip flattening, copied out of the torch tensors at collect
      time). Each mapped pn vertex owns one node row but valence-many endpoint
      records; the adjoint (`d_α += d_out·(A−C)`, `d_β += d_out·(B−C)`)
      accumulates across all of them. `compute_gradients_hand`
      (`HandGradients.cc`) is the production `compute_gradients` replacement:
      full forward (S1→S3→S4/S5/S6/S7 + curvature, `eval()`'s assembly order)
      + the reverse chain, returning the per-pn-vertex bary gradient with
      production's FacePoint-only collection convention. Validation (banana
      pair, post init+resample+init; 578 leaves, all FacePoint): config A =
      the FULL production configuration (w_h=1, AIAP+AP 0.5/0.5, w_c=0.1) —
      the first gradcheck combining harmonic + curvature — ref_loss
      **bitwise** the Phase 3 baseline 1.2734862918788761
      (= 1.1873013392084315 + 0.1·0.86184952670444603); config B =
      all-branches + w_c=0.7, ref 2.9381535249302653. Per config: graph
      rebuild idempotent (max_abs **0**), hand rewritten rows vs production
      `pos_mat_` **0**, hand seg A/B vs torch `get_pos` **0**, packaged
      wrapper vs direct calls **0**. Hand loss vs torch: 2.2e-16 / 8.9e-16
      abs. **d_bary vs torch oracle max_abs 1.18e-15 (production) / 5.46e-12
      (all-branches)** — comfortably inside the end-of-Phase-4 1e-9 abs leaf
      gate (worst gated ratios 1.2e-6 / 5.5e-3 against 1e-9/1e-7). Central
      finite differences (h=1e-6, gate max(1e-8, 1e-5·|fd|), 61/60 components
      = top-10 |grad| + even stride): worst gated 0.016 / 0.077 — no
      subgradient-kink failures. Raw dumps in `gradcheck_phase4_s1.txt`.
      Lore: production `compute_gradients`' attribute default
      `torch::zeros(2)` is **float32** — reading non-FacePoint entries with a
      double accessor throws c10::Error (found via cdb; the gradcheck reads
      FacePoint rows only, the rest are semantically zero). `loss.backward()`
      frees the graph, so each config rebuilds the differentiable overlay
      exactly as production iterations do
      (`compute_differentiable_intersections_for_overlay_stable` +
      `sync_tg_and_torch` — verified bitwise idempotent) and resets the leaf
      grads. DEFERRED: the bary authority flip (`bary_full()` reading a
      stored field, announced for S1 in the Phase 3 notes) moves to
      Phase 5/6 — the tensor must stay authoritative while the torch oracle
      exists, and flipping now would churn the `do_step`/`Resample`
      sp-creation sites with zero validation benefit.
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
  reasons. **Resolved by the Phase 2 audit** — the normative boundary is written
  in the Phase 2 result section above; finite differences remain the arbiter for
  any deliberate deviation.
- **Subgradient/branch choices**: `compute_distortion` early-outs, AIAP min/max,
  `torch::clamp` ends, `max(0.02, ·)` in `torch_compute_pn_uvs` — each branch's
  derivative convention must match torch (clamp: zero gradient outside; max: gradient
  to the larger arg).
- **Reverse-walk bugs in S4** manifest as *almost*-right gradients. Per-arc
  gradcheck localizes them.
- **Doc/code mismatch on free variables** — resolved in Phase 2
  (`libTorchUsage.md` corrected; layout-node face points are the only free variables).
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
