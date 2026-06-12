# Driving the Layout Optimizer from a MIQ-Style Cross Field

Notes on the question: *could the embedding optimizer work from a MIQ-style
singularity + cross field instead of a hand-made layout mesh, and would that
require motorcycling parameter space to connect the singularities?*

Short answer: yes, it could — and the repo already half-supports one pipeline
for it — but the cross field alone is not a sufficient input, and while a
motorcycle graph is the robust way to build the missing structure, it is not
the only way. The main codebase adaptation cost is T-node handling in the
per-patch parameter domains.

## 1. What the optimizer actually requires

The method (see `CLAUDE.md`, `documentation/adjointDifferentiation.md`) is an
*embedding* optimizer: the layout's connectivity is frozen and only where its
nodes and arcs sit on the target surface is optimized. Its two real inputs
are:

1. **A pure-quad cage topology.** The loss is "sum of parametric distortion
   over per-patch harmonic parameterizations", and the patches *are* the
   layout faces. Each quad face gets a fixed unit-square parameter domain:
   the S4 boundary-UV stage (`Adjoint/BoundaryUvStage.*`) walks each patch
   boundary and pins the four layout-node corners to the unit square's
   corner factors, interpolating ordinary path-network vertices by arc
   length in between. Each layout edge becomes a path-network arc, a
   triangle strip, and a run of boundary UVs; each layout vertex becomes a
   patch corner.

2. **An initial embedding of that cage.** `compute_layout_embedding_init`
   needs only layout node positions: it projects them to surface points and
   connects them with geodesic shortest paths (geometry-central). Everything
   afterwards is intrinsic (barycentric `SurfacePoint`s); the ambient layout
   geometry is written *back* from the embedding, not read.

A cross field with singularities supplies neither directly. What it does
supply:

- **The 0-cells, for free.** Cross-field singularities are exactly the
  irregular layout vertices (index ±1/4 → valence 3/5), and Poincaré–Hopf
  guarantees the singularity set is consistent with the surface topology —
  a constraint a hand-made cage has to satisfy by careful authoring.
- **A consistent direction field for the objective.** The curvature
  alignment loss (`Adjoint/CurvatureAlignmentStage.*`) already aligns patch
  boundaries to a 4-RoSy field (currently `smooth_direction_field`).
  Swapping in the MIQ field means the optimizer pulls boundaries toward the
  same field that generated the layout: the separatrix network is close to
  a stationary configuration of the alignment term, so the alignment and
  distortion terms should cooperate rather than fight.

The missing piece is the 1- and 2-cells: which singularity connects to
which, along what arc, bounding which quad patch.

## 2. Three ways to build the missing cage

### (a) Direct separatrix tracing on the surface — fragile

Numerically integrating separatrices from singularity to singularity is the
classically broken approach: separatrices spiral, near-miss their target
singularities, and never close into a finite cage without snapping
heuristics. Not recommended as the primary mechanism.

### (b) Motorcycle graph — the robust tracing formulation

The motorcycle graph fixes (a) by changing the termination rule: a trace
stops when it hits a *previously traced curve*, not when it reaches another
singularity. This guarantees termination and produces a partition into
combinatorially rectangular patches. Doing it in the parameter space of a
seamless map (rather than by streamline integration on the surface) makes it
exact — separatrices are iso-parameter lines and collisions are grid-line
intersections.

Cost: motorcycle graphs produce **T-nodes** (see §3), which this codebase
does not currently support.

### (c) Integer-grid map → quad mesh → base complex — supported today

The pipeline the repo already accommodates:

    MIQ field → integer-grid map → extract dense quad mesh (QEx-style)
            → extract_layout_from_qm (base-complex coarsening) → optimize

The integer rounding in MIQ implicitly resolves every connectivity question
the motorcycle graph answers explicitly; the base complex of the extracted
quad mesh *is* the separatrix partition. It is heavier — a dense quad mesh
is built and immediately thrown away — but every stage is standard and
robust, and it lands directly on the input format `optimize` expects, with
zero changes to this codebase.

Crucially, **the map quality barely matters**. The parameterization is only
scaffolding for topology + initialization; the optimizer re-embeds every
node and arc from scratch by distortion minimization. A cheap, low-quality,
even locally injective-only MIQ map suffices.

## 3. The real adaptation cost of (b): T-nodes

A motorcycle that crashes into the side of another track creates a
valence-3 node that is a *corner* for the two patches on its side but a
*flat* (angle-π) boundary point for the through-patch. Combinatorially the
through-patch becomes a pentagon, and the codebase hard-assumes pure quads:
the S4 UV walk assigns exactly four unit-square corner factors per layout
face, and the per-patch boundary classification in `prepare_param`
(`Adjoint/HarmonicStage.*`) keys off layout-face halfedge walks.

Two ways out:

1. **Heal the T-junctions** — extend each crashed motorcycle until it
   terminates at a node. Standard, keeps the codebase unchanged, but can
   multiply patch count substantially (each extension can crash into new
   tracks and cascade).
2. **Generalize the parameter-domain assignment** so a layout node may sit
   *on* a patch edge for one incident patch (UV interpolated by arc length,
   exactly as ordinary path-network vertices already are) while remaining a
   corner for its other incident patches. This is a modest, local change:
   the arc-length interpolation machinery already exists in the S4 walk;
   the work is in the corner-factor assignment (per (face, node) rather
   than per node) and in re-checking the harmonic boundary classification
   and the adjoint bookkeeping (`d_boundary_uvs` scatter through
   `_boundary_src_hehs`) for the flat-node case. The distortion stage (S7)
   is untouched — it only sees per-face UVs.

Option 2 is the more interesting one: it would let the optimizer consume
motorcycle-graph partitions natively and would also relax the pure-quad
constraint for other layout sources.

## 4. Initialization bonus: traced arcs beat geodesic init

Field-traced separatrices come pre-embedded on the surface in the correct
homotopy class. The current init connects projected node positions with
geodesic *shortest* paths, which can pick the wrong homotopy class on
high-genus inputs (a known failure mode of shortest-path initialization).
Seeding the path network with the traced arcs directly — rather than only
their endpoints — would remove that failure mode. The optimizer's resampling
(`Resample.cc`) would immediately regularize the traced polylines, so trace
quality is uncritical.

## 5. Other compatibility notes

- **Singularity placement is not pinned.** Layout nodes are free FacePoint
  barycentrics; nothing constrains an irregular node to stay at the field
  singularity that spawned it. In practice the curvature-alignment term
  pulls it there, but if exact placement matters, a (trivially
  differentiable) soft penalty `w * |node_pos - sing_pos|²` per irregular
  node would be a one-stage addition to the adjoint chain — it reads only
  S1 outputs, like the curvature branch, so it accumulates straight into
  `d_overlay_pos` with no ordering constraints.
- **Holonomy/validity comes free.** A valid MIQ singularity set
  automatically satisfies the index sum the cage needs; hand-made layouts
  have to get this right by construction.
- **Determinism.** Whatever construction is used runs once, up front,
  outside the optimization loop — it does not interact with the
  bitwise-determinism contract (`e2e_determinism` gates the loop, not the
  input construction).

## 6. Recommendation

For a first experiment, use pipeline (c): generate a MIQ quad mesh with any
standard tool, run `extract_layout_from_qm`, and feed the result to
`optimize` unchanged. If and when the throwaway dense quadrangulation
becomes the bottleneck — or T-rich partitions are desirable for coarseness —
implement (b) with the §3 option-2 domain generalization, and seed the path
network with the traced separatrices (§4) while doing so.
