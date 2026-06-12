# Remove-Autodiff Baseline Timings

Captured at Phase 0 before any adjoint work. Reference for measuring improvement.
All timings use the `TimerCollection` instrumentation already in the code
(EvalObjective, Backpropagation, Update, Embedding labels).

Run: `optimize.exe` (MSVC-Release), fixed 20 iterations, no screenshots,
single model per row.

## Results

| Model | Iterations | EvalObjective (ms/iter) | Backpropagation (ms/iter) | Update (ms/iter) | Embedding (ms/iter) | Notes |
|---|---|---|---|---|---|---|
| Spot | 20 | 26907 | 72525 | 110 | 319383 | loss 1.7695→1.4700; total run ≈140 min |
| Banana | 20 | 6012 | 15568 | 21 | 17453 | loss 1.2735→1.2396; total run ≈13 min |
| Bunny | — | — | — | — | — | **blocked** — init fails, see below |

## Capture notes (2026-06-10, Windows/MSVC-Release)

Two Windows-specific bugs had to be fixed before a valid baseline could be captured
(both caused by LibTorch 2.4.0 CPU on Windows; both documented in the remove-autodiff plan):

1. **Stack overflow (0xc00000fd) on spot.obj** — torch destroys its autograd graph
   recursively (one stack frame per node); graph depth scales with mesh size and
   blows Windows' 1 MB default stack (Linux defaults to 8 MB, which is why the
   original authors never hit it). Fixed by `/STACK:67108864` (64 MB) on all MSVC
   app targets in CMakeLists.txt.
2. **xmm14/xmm15 ABI violation** — a torch_cpu.dll kernel clobbers callee-saved
   xmm14/15 during eval(). MSVC parks by-value option-struct fields AND main's
   constexpr FP loop constants in those registers. Consequences observed: optimizer
   enum flipping Adam→GradientDescent (fixed by passing `OptimizationOptions`/
   `HarmonicOptions` by `const&`), and `od.step_size` collapsing to exactly 0 after
   iteration 0 → bitwise-frozen loss for 19 iterations (fixed by making main's loop
   FP constants `static volatile double` in apps/optimize.cc).
   *Attributed by cdb breakpoint bisect (cdb_out12.txt): the clobbering call is
   `torch::linalg::solve` (ObjectiveFunctions.cc:344, MKL LAPACK dense solve) — xmm14
   flipped 0→garbage across the solve on the 110th per-patch solve of iteration 0,
   i.e. a matrix-size-dependent MKL kernel path. The Phase 1 Eigen sparse-solve
   replacement removes this kernel entirely.*

Determinism check (Phase 0 gate): repeated Spot runs reproduce the loss sequence
bitwise (LOSS[0]=1.7695223340424289, LOSS[1]=1.7450757401038559,
LOSS[2]=1.725120104898781) with Adam active throughout.

Timer caveat: `timers.print_stats` was added at end of main; Init/Resample/
PerIteration/Total rows are not instrumented (print as 0) — only EvalObjective,
Backpropagation, Update, Embedding are meaningful, and Embedding (re-embedding
inside apply) dominates wall-clock.

**Bunny is blocked**: `compute_layout_embedding_init` → `insert_path_network` →
`insert_surface_point_edge_connections` hits `cannot add face` (overlay face that
`polymesh can_add()` rejects, observed at target face 904). The failure path in
EmbeddingOverlay.cc:1039-1080 opens a *blocking interactive debug viewer* (hangs
any headless run) and then — in release, where `assert` is a no-op — calls
`faces().add()` on the rejected face anyway (UB). Fixing bunny init is a library
correctness issue out of scope for Phase 0; Banana (data/Spot/bananna.obj +
bananna_layout.obj) is used as the second baseline model instead.

## Instructions for capturing

Build release, then run with the timing output captured:

```powershell
..\MSVC-Release\optimize.exe 2>&1 | Tee-Object -FilePath baseline_timings.txt
```

Fill in the table above from the `TimerCollection` summary printed at program exit.

## Expected outcome after Phase 1

The `torch::linalg::solve` replacement with `Eigen::SimplicialLDLT` should
produce a significant drop in EvalObjective + Backpropagation (the dense O(n³)
solve per patch per iteration is the dominant cost today).

## Phase 1 results (2026-06-11, Windows/MSVC-Release)

Implementation: `library/LayoutOpt/Adjoint/SparseHarmonicSolve.{hh,cc}` — a
custom `torch::autograd::Function` assembling A := -L sparsely (Eigen
`SimplicialLDLT`, `SparseLU` fallback) with a hand-derived solve adjoint;
toggled by `HarmonicOptions::use_sparse_harmonic_solve` (default ON, dense
path kept as validation oracle).

### Validation gate — PASSED

`gradcheck_phase1.exe` (banana, same embedding state as optimize.cc's first
eval, dense then sparse eval on the same graph with leaf-grad zeroing between):

- loss dense  = 1.2734862918788761, loss sparse = 1.2734862918788761
  (bitwise identical; matches the Phase 0 banana baseline LOSS[0])
- 578 leaf gradients: **max abs diff 1.7e-16, max rel diff 8.3e-13**, zero
  failures at atol 1e-9 / rtol 1e-7 (plan §5 tolerances)
- Spot production run cross-check: sparse LOSS[0]=1.7695223340424291 vs dense
  baseline 1.7695223340424289 (1 ulp); LOSS[1..2] agree to ~13 digits before
  optimizer chaos amplifies the 1e-16 solver differences.

**Caveat for future gradchecks:** the *in-process* tolerance comparison inside
a torch-linked binary is untrustworthy on Windows — the xmm14/15 clobber
poisons MSVC register-parked FP constants, producing self-contradictory
verdicts (observed: "max abs diff 0.969 / 0.155" with zero failures, while the
true max was 1.7e-16). gradcheck_phase1 therefore dumps raw grads to
`gradcheck_phase1_{dense,sparse}.txt` and the authoritative comparison is the
torch-free `compare_gradcheck.py`.

### Timings (same protocol as Phase 0: 20 iters, no screenshots, headless)

| Model | EvalObjective (ms/iter) | Backpropagation (ms/iter) | Update (ms/iter) | Embedding (ms/iter) | Notes |
|---|---|---|---|---|---|
| Spot | 23224 | 65662 | 107 | 327220 | loss 1.7695→1.4700 (LOSS[19]=1.4700084043778932, matches dense endpoint) |
| Banana | 5666 | 14880 | 21 | 17396 | loss 1.2735→1.2396 (LOSS[19]=1.2395567156654854, matches dense endpoint) |

Deltas vs the Phase 0 dense baseline: Spot EvalObjective −14%, Backpropagation
−9%; Banana EvalObjective −6%, Backpropagation −4%; Update and Embedding
unchanged (Embedding differences are run-to-run noise). The gains are modest
because the element-wise autograd graph (compute_distortion, SVD terms, etc.)
dominates eval/backprop cost, not the solve — that graph is what the later
phases remove. The qualitative win of Phase 1 is correctness: the
ABI-violating MKL dense-solve kernel (xmm14/15 clobber) is no longer in the
production path.

## Phase 6 final timings (2026-06-11, Windows/MSVC-Release)

Captured after Phase 6 completion: torch fully deleted and unlinked, gradients
from the hand adjoint chain (`Adjoint/HandGradients.*`), the per-patch S6+S7
forward loop OpenMP-parallel (deterministic: per-patch slots + sequential
patch-order reduction; gated bitwise by `e2e_determinism`). Same protocol as
Phase 0: `optimize.exe`, 20 iterations, no screenshots, headless.

| Model | EvalObjective (ms/iter) | Backpropagation (ms/iter) | Update (ms/iter) | Embedding (ms/iter) | Notes |
|---|---|---|---|---|---|
| Spot | 13.6 | 17.1 | 0.79 | 28457 | total run ≈9.5 min (Phase 0: ≈140 min) |
| Banana | 7.7 | 4.8 | 0.35 | 1305 | total run ≈27 s (Phase 0: ≈13 min) |

Deltas vs the Phase 0 torch baseline:

| Model | EvalObjective | Backpropagation | Update | Embedding |
|---|---|---|---|---|
| Spot | **×1980 faster** (26907 → 13.6) | **×4240 faster** (72525 → 17.1) | ×139 (110 → 0.79) | ×11 (319383 → 28457) |
| Banana | **×780 faster** (6012 → 7.7) | **×3240 faster** (15568 → 4.8) | ×60 (21 → 0.35) | ×13 (17453 → 1305) |

The eval+backward cost that motivated the project (torch graph construction +
destruction + element-wise kernels) is gone — three-plus orders of magnitude,
beyond the "order(s) of magnitude" the plan projected. Embedding (the
re-embedding inside apply: strip flattening, path tracing, overlay rebuild)
got its ~12× from the Phase 5 de-torchification of its per-element tensor
traffic and now utterly dominates wall-clock (>99%); further speedups would
have to come from that stage, which is outside this project's scope.

Correctness at these numbers is pinned by the two permanent harnesses on the
banana pair (both PASS on the final build): `gradcheck_fd` (hand vs central
FD, worst ratio 0.017 of the 1e-5 gate) and `e2e_determinism` (two production
runs bitwise EQ per iteration; iter-0 loss 1.2734862918788759 vs the Phase 3
torch-era reference 1.2734862918788761, |Δ| = 1 ulp = 2.2e-16).
