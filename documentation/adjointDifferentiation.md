# Hand-Written Adjoint Differentiation

The optimization loop needs the gradient of the objective (harmonic distortion
+ curvature alignment) with respect to the barycentric coordinates of the
path-network surface points. This was originally computed with LibTorch
autograd; it is now a hand-written reverse-mode adjoint chain over plain
doubles and Eigen, living in `library/LayoutOpt/Adjoint/`. No autodiff
library is involved.

## Entry point

`Adjoint/HandGradients.hh/cc` orchestrates everything:

- `hand_loss_forward(...)` — full forward pass from explicit bary values
  (`[n_pn_vertices x 2]`, see `collect_bary`) to the weighted total loss.
  Every intermediate the backward pass needs is captured in a `HandLossCtx`.
- `hand_loss_backward(...)` — runs the stage adjoints in reverse, accumulating
  the total-loss gradient into a caller-zeroed `[n_pn_vertices x 2]` matrix.
- `compute_gradients_hand(...)` — the production call: leaf collection +
  forward + backward in one shot. Returns a per-path-network-vertex
  `vec2d` attribute; only `FacePoint` surface points receive gradient
  (the collection convention inherited from the torch-era production code —
  gradient reaching other surface-point types is discarded).

`eval()` in `Optimization.cc` calls `compute_gradients_hand`, then converts
the bary gradients into update directions for the Adam optimizer
(`Optimizers.cc`).

## Stage decomposition

The forward chain mirrors the original objective evaluation; each stage is a
forward/backward pair in its own file:

| Stage | File | Forward |
|-------|------|---------|
| S1 leaf | `LeafStage.*` | bary → 3D layout-node overlay rows + 2D triangle-strip endpoints A/B per layout edge |
| S3 intersections | `IntersectionStage.*` | A/B endpoints → overwritten interior overlay vertex rows |
| S4 boundary UVs | `BoundaryUvStage.*` | edge lengths → arc parameters `t` → boundary UVs per path-network halfedge |
| S5 cotans | `CotanStage.*` | overlay positions → cotangent weights |
| S6 harmonic | `HarmonicStage.*` | per patch: boundary UVs + cotans → interior UVs via harmonic parameterization (sparse solve; the backward pass solves the transposed system) |
| S7 distortion | `DistortionStage.*` | per patch: UVs + 3D positions → per-face Jacobian SVD → AIAP / area-preserving distortion energy |
| curvature | `CurvatureAlignmentStage.*` | boundary alignment to principal curvature directions |

`loss = w_harmonic * harmonic + w_curvature * curvature`.

### Backward ordering constraints

The reverse pass is not a free permutation; `HandGradients.cc` encodes:

- `pn_uvs_backward` and `pn_t_backward` both feed the edge-length adjoint, so
  `edge_lengths_backward` runs **last** of the S4 trio.
- Every `d_pos` accumulator (S7 direct path, S4, S5, curvature) must run
  **before** `intersections_backward` and `leaf_backward`, which only *read*
  `d_pos`.

## Validation harnesses (permanent)

Both build as apps and exit nonzero on failure; run them from the repo root
after any change touching the adjoint chain or evaluation order:

- **`apps/gradcheck_fd.cc`** — compares `compute_gradients_hand` against
  central finite differences of the hand forward loss (h = 1e-6) on sampled
  `FacePoint` bary coordinates. Gate: 1e-5 relative, 1e-8 absolute floor.
  Each sample is cross-checked with a second step size (h2 = 1e-5); samples
  where the two FD estimates disagree (relative to the FD magnitude, abs
  floor 1e-9) sit in locally nonsmooth sliver-triangle regions where no
  finite difference is meaningful — they are reported and excluded.
  Raw values go to `gradcheck_fd.txt`.
- **`apps/e2e_determinism.cc`** — two identical 12-iteration production runs
  on the banana pair must match **bitwise** (per-iteration loss, discrete
  state fingerprints, post-apply bary arrays, final layout-node positions),
  and the iteration-0 loss must match the recorded reference
  (`1.2734862918788761`, rel 1e-12). Raw values go to `e2e_determinism.txt`.
  A failure means nondeterminism crept into the pipeline (unordered
  iteration, uninitialized reads, parallel reduction).

The full porting/validation history is in
`documentation/plans/remove-autodiff.md`.
