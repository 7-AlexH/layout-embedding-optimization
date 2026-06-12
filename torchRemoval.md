# Report: Removing LibTorch from layout-embedding-optimization

**Status: complete (2026-06-11).** LibTorch is gone from the codebase — source,
build system, and documentation. Gradients now come from a hand-written
reverse-mode adjoint chain over plain doubles and Eigen
(`library/LayoutOpt/Adjoint/`). The eval+backprop cost that motivated the
project dropped by three-plus orders of magnitude, with correctness pinned by
two permanent regression harnesses.

This report summarizes what was done and why. The working documents it
condenses:

- `documentation/plans/remove-autodiff.md` — the phased plan, with every
  checkbox annotated with results as it was executed.
- `documentation/plans/remove-autodiff-baseline.md` — before/after timings.
- `documentation/adjointDifferentiation.md` — how the replacement system works
  (the doc to read for maintenance).

---

## 1. Motivation

The optimizer's inner loop built a LibTorch autograd graph every iteration to
differentiate one scalar loss with respect to the barycentric coordinates of
layout-node surface points. Torch's per-element tensor overhead (graph
construction, dispatch, graph destruction) dominated the cost: on the Spot
model, a single iteration spent ~27 s in objective evaluation and ~73 s in
backpropagation. The math being differentiated is modest — interpolation,
closed-form 2D intersections, halfedge-walk arc lengths, cotan Laplacians, one
sparse solve per patch, a hand-rolled 2×2 SVD — so a hand-written adjoint
chain promised orders of magnitude.

A precise audit (plan §1) found only ~8 genuine differentiation stages; the
remaining 400+ `torch::` references were tensor-flavored plain geometry already
detached from the graph, portable mechanically.

## 2. Strategy

**Oracle-driven, back-to-front, one stage at a time.** The torch path was kept
compiling behind a CMake flag until the very end; each hand stage was validated
against torch's own intermediate gradients (`retain_grad()` on stage
boundaries) before the next stage was started, so at no point was more than one
stage of hand AD being debugged at once. Finite differences served as the
backstop oracle.

No general-purpose tape or operator-overloading type was introduced. Each stage
is an explicit `forward`/`backward` function pair; forward stores whatever
backward needs in a per-stage context struct (a manual, stage-granular tape),
and stage boundaries exchange plain adjoint buffers (`d_pos`, `d_uvs`,
`d_cotans`, …, finally a 2-vector `d_bary` per leaf).

## 3. What happened, phase by phase

### Phase 0 — Scaffolding and baseline

Added the `LAYOUTOPT_WITH_TORCH` gate, a deterministic comparison mode, a
gradient-dump harness, and captured baseline timings on Spot and Banana
(Bunny is blocked by a pre-existing init bug, documented in the baseline doc).

Two Windows-specific torch bugs had to be fixed before a valid baseline
existed:

1. **Stack overflow on Spot** — torch destroys its autograd graph recursively,
   one frame per node; mesh-sized graphs blow Windows' 1 MB default stack
   (Linux's 8 MB is why upstream never saw it). Fixed with `/STACK:67108864`.
2. **xmm14/xmm15 ABI violation** — an MKL dense-solve kernel inside
   `torch_cpu.dll` (attributed by cdb bisect to `torch::linalg::solve`)
   clobbered callee-saved xmm14/15. MSVC parks by-value struct fields and
   constexpr FP loop constants in exactly those registers, producing wild
   symptoms: the optimizer enum flipping Adam→GradientDescent, the step size
   collapsing bitwise to 0, and self-contradictory in-process FP comparisons.
   Mitigated with `const&` option passing and `volatile` loop constants; the
   lasting consequence was a protocol: **all gradient comparisons were dumped
   to files and compared in a separate torch-free process.**

### Phase 1 — Sparse solve (still inside torch)

Replaced the four effective dense O(n³) `torch::linalg::solve` calls per patch
with a custom `torch::autograd::Function`: sparse cotan-Laplacian assembly,
one `Eigen::SimplicialLDLT` factorization serving all solves, and the
hand-derived solve adjoint (λ = L⁻ᵀḡ; L̄ᵢⱼ = −λᵢuⱼ−λᵢvⱼ on stored entries
only). Validated to 1.7e-16 max leaf-gradient difference with bitwise-identical
loss. Timing gains were modest (the element-wise graph dominated), but this
removed the ABI-violating MKL kernel from the production path and validated
the exact assembly/scatter plumbing the final version reuses.

### Phase 2 — Gradient-boundary audit

The critical analysis phase. Every `.item<double>()` on the leaf→loss path was
cataloged and classified. Headline finding: **the entire triangle-strip
flattening (`heh_pos_2d`) is constant with respect to the autograd graph** —
`requires_grad` is set only *after* the flattening runs, and no leaf survives
between iterations (every surface point is recreated detached each step). The
feared "S2" flattening stage therefore vanished as a differentiation problem.
The free variables are exactly the barycentric coordinates of layout-node face
points (2 DOF each); each strip-interior overlay vertex depends on only the two
endpoint leaves of its strip. The audit result was written into the plan as the
**normative gradient boundary**, including every subgradient convention the
port had to replicate (AIAP min/max selection, `max(0.02, ·)` clamp, distortion
early-outs, cotan `+1e-10` denominators).

### Phase 3 — Core types (additive coexistence)

Plain-double/Eigen mirrors were added *alongside* the torch tensors —
`pos_mat_` on the target and overlay meshes, flat `t`/`uv` arrays on the path
network, `bary_full()` on `SurfacePoint` — populated at the existing single
chokepoints, with the tensors staying authoritative so the oracle stayed
bitwise-unchanged. Every mirror validated to **0** absolute difference.

### Phase 4 — The adjoint port (back to front)

One stage pair per file under `library/LayoutOpt/Adjoint/`, each gradchecked
against the torch oracle at 1e-9 abs / 1e-7 rel before moving on:

| Stage | File | Notes |
|---|---|---|
| S7 distortion + 2×2 SVD | `DistortionStage` | per-face rebuild extracted so the oracle IS the production code; AIAP branch replicated |
| S6 harmonic solve | `HarmonicStage` | Phase 1 solve restructured into ctx form; inner UVs **bitwise** equal to production |
| S5 cotans | `CotanStage` | forward **bitwise** == `torch_cotans`; face areas not ported (dead in the live loss) |
| S4 lengths → t → boundary UVs | `BoundaryUvStage` | walk order recorded forward, replayed in reverse; first stage where bitwise vs torch is unattainable (ulp-level norm-kernel differences) |
| Curvature alignment | `CurvatureAlignmentStage` | independent branch into `d_pos`; atan2/4-rosy chain |
| S3 intersections | `IntersectionStage` | line–line closed form; hand arithmetic bitwise == torch |
| S2 | — | **closed as a no-op** per the Phase 2 audit |
| S1 leaf interpolation | `LeafStage` + `HandGradients` | both `get_pos` corner conventions; `compute_gradients_hand` is the full-chain entry point |

End-of-phase validation: full-pipeline leaf gradients matched torch to
1.18e-15 (production config), central finite differences to a worst gated
ratio of 0.016 against the 1e-5 gate, and a 12-iteration end-to-end run
(crossing a resample) showed the hand-driven optimizer tracking the
torch-driven one to 4.3e-9 relative loss drift with identical discrete
behavior.

### Phase 5 — De-torchify the detached code

Mechanical port of everything in category B — `Update.cc`, `Resample.cc`,
`FlattenTriangleStrip.cc`, `Optimizers.cc`, `EmbeddingUtils.cc`, IO and
visualization — to typed-geometry/Eigen doubles. Validated continuously by the
gradcheck suite plus the e2e trajectory gate. One recalibration was needed and
diagnosed honestly: a sliver triangle in the banana overlay makes two gradient
components ill-conditioned (κ ~ 1e9), so ulp-level forward changes legitimately
move the late trajectory; the e2e gate became a strict *structural* equality
(vertex/type/anchor fingerprints per iteration) plus loosened numeric drift
bounds, while the per-state gradient gates stayed at 1e-9/1e-7.

### Phase 6 — Removal and cleanup

- **6a** — bary authority flip: `SurfacePoint`'s plain-double barycentric
  coordinates became authoritative; the tensor died.
- **6b** — deleted the torch path: 20 files including `TorchUtils.*`, the attic
  losses, and the torch sides of the gradcheck harnesses. Two harnesses were
  kept **permanently**: `apps/gradcheck_fd.cc` (hand gradients vs central
  finite differences) and `apps/e2e_determinism.cc` (two 12-iteration
  production runs bitwise-equal, iter-0 loss gated against the torch-era
  reference at 1e-12 rel).
- **6c** — unlinked torch from CMake (find_package, DLL copies, prefix paths,
  `TORCH_CXX_FLAGS`), deleted `download_libtorch.sh`, removed the volatile
  xmm workarounds (the clobbering DLL is gone), replaced
  `libTorchUsage.md` with `adjointDifferentiation.md`, scrubbed README and
  CLAUDE.md. Both harnesses re-passed bitwise after the unlink. The 64 MB
  stack reservation was kept as generic deep-recursion headroom.
- **6d** — parallelized the per-patch S6+S7 forward loop in
  `hand_loss_forward` with OpenMP (the thing torch's non-thread-safe graph
  construction had forced off). Determinism by construction: each patch writes
  only its own pre-sized slot, the loss reduction stays sequential in
  patch-index order, and per-thread scratch polymesh attributes are created
  and destroyed outside the parallel region (polymesh attribute registration
  is not thread-safe). Both harnesses passed **bitwise unchanged** on the
  parallel build.

## 4. Results

Same protocol as the Phase 0 baseline (`optimize.exe`, MSVC-Release, 20
iterations, headless, no screenshots):

| Model | EvalObjective (ms/iter) | Backpropagation (ms/iter) | Update | Embedding | Total run |
|---|---|---|---|---|---|
| Spot | 26907 → **13.6** (×1980) | 72525 → **17.1** (×4240) | 110 → 0.79 | 319383 → 28457 (×11) | ≈140 min → ≈9.5 min |
| Banana | 6012 → **7.7** (×780) | 15568 → **4.8** (×3240) | 21 → 0.35 | 17453 → 1305 (×13) | ≈13 min → ≈27 s |

The plan projected "order(s) of magnitude" for eval+backward; the result is
three-plus. Embedding (the re-embedding inside `apply`: strip flattening, path
tracing, overlay rebuild) got its ~12× from the Phase 5 de-torchification of
per-element tensor traffic and now dominates wall-clock (>99%) — it is the
only remaining optimization target, and it is outside this project's scope.

Correctness at these numbers, on the final build:

- `gradcheck_fd` PASS — worst hand-vs-FD ratio 0.017 of the 1e-5 gate over the
  smooth components (sliver-triangle nonsmooth components excluded by an
  FD-smoothness guard).
- `e2e_determinism` PASS — two production runs bitwise-equal at every
  iteration; iter-0 loss 1.2734862918788759 vs the torch-era reference
  1.2734862918788761 (1 ulp, inside the 1e-12 gate).

## 5. Lessons worth keeping

1. **The gradient boundary audit paid for the whole project.** Reading the
   code precisely (Phase 2) deleted an entire feared stage (S2) and shrank S3,
   before any of that code was written.
2. **Back-to-front with a live oracle** meant no debugging session ever
   involved more than one unverified stage. Every stage landed on its first
   gradcheck or within one fix.
3. **Never trust FP comparisons inside a process linked against a DLL you
   don't control.** The xmm14/15 clobber made torch-linked harnesses report
   self-contradictory verdicts; file-dump + out-of-process comparison was the
   only reliable protocol.
4. **Bitwise determinism is achievable under OpenMP** if reductions are kept
   sequential and parallel work writes to disjoint pre-sized slots — and a
   bitwise e2e gate makes parallelism changes safe to make.
5. **polymesh attribute registration is not thread-safe**; scratch attributes
   must be created/destroyed outside parallel regions (one per thread, reset
   per iteration).

## 6. Maintenance notes

- After any change touching the adjoint chain, evaluation order, or
  parallelism: run `gradcheck_fd` and `e2e_determinism` (both exit nonzero on
  failure).
- `extern/libtorch/` and `extern/libtorch-2.4.0-win.zip` may still exist on
  disk; they are untracked, unused, and safe to delete.
- `optimize.cc` prints `LOSS[i]` every iteration permanently, so future timing
  captures need no temporary edits.
