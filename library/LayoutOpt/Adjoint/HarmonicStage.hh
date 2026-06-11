#pragma once
// Phase 4 / S6 of the remove-autodiff plan: hand-rolled forward + reverse-mode
// adjoint of the per-patch harmonic parameterization stage, plus the plain
// (torch-free) port of the patch-collection indexing.
//
// This restructures the Phase 1 sparse solve (Adjoint/SparseHarmonicSolve.cc,
// a torch::autograd::Function) into the explicit stage-pair form used by
// Phase 4: forward returns a context with everything backward needs, backward
// scatters the upstream adjoint into d_cotans / d_boundary_uvs directly
// instead of handing off to autograd. The math, assembly order, traversal
// order, and solver (SimplicialLDLT with SparseLU fallback) are identical to
// the Phase 1 path, so the forward solution is bitwise-equal to the
// production sparse solve.
//
// Stage boundary (validation per plan section 4/S6 — inputs treated as
// leaves):
//   inputs  : per-overlay-edge cotans [n_edges] (S5 output), per-patch
//             boundary UVs (S4 output)
//   output  : per-patch inner UVs [n_inner x 2]
//   adjoints: d_cotans (accumulated across patches), d_boundary_uvs

#include <cstdint>
#include <vector>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/ObjectiveFunctions.hh> // MappingIndex (include chain de-torched in Phase 6)

namespace LayoutOpt
{

// Plain port of torch_prepare_param: collect one patch's faces and classify
// its overlay vertices into boundary (UV pinned from the path network) and
// inner (solved for). Identical traversal and index assignment; the boundary
// UVs are gathered from a plain per-overlay-halfedge UV array (from-vertex
// convention, same layout as the `uvs` tensor in harmonic_distortion_loss).
// If _boundary_src_hehs is non-null it receives, per boundary entry, the
// overlay halfedge idx the UV was gathered from (the row of _o_uvs) — the S4
// gradcheck uses this to scatter d_boundary_uvs back to per-halfedge UV rows.
void prepare_param(int const _patch_value,
                   PathNetworkData const& _pnd,
                   OverlayMeshData const& _omd,
                   Eigen::MatrixX2d const& _o_uvs,
                   pm::vertex_attribute<MappingIndex>& _map_to_vec,
                   std::vector<VH>& _inner_patch_vhs,
                   std::vector<VH>& _boundary_patch_vhs,
                   std::vector<vec2d>& _boundary_uvs,
                   std::vector<FH>& _patch_fhs,
                   std::vector<int>* _boundary_src_hehs = nullptr);

// Forward context: the system structure (same flattened encoding as
// torch_harmonic_param's sparse path) plus the solution. Backward re-assembles
// and re-factors A from these (mirrors the Phase 1 custom function, which
// cannot stow the factorization either).
struct PatchHarmonicCtx
{
    int n_inner = 0;
    std::vector<int64_t> off;  // flattened (i, j, e): A(i,j) = -cotans[e], inner neighbor j
    std::vector<int64_t> diag; // flattened (i, e):    A(i,i) += cotans[e], per outgoing halfedge
    std::vector<int64_t> rhs;  // flattened (i, b, e): B(i,:) += cotans[e] * boundary_uvs[b]
    Eigen::MatrixX2d inner_uvs; // X, the solve output [n_inner x 2]
};

// Build the system structure (same traversal as torch_harmonic_param), then
// assemble A := -L / B and solve A X = B. n_inner == 0 yields an empty ctx.
PatchHarmonicCtx harmonic_param_forward(pm::vertex_attribute<MappingIndex> const& _map_to_vec,
                                        std::vector<VH> const& _inner_patch_vhs,
                                        std::vector<vec2d> const& _boundary_uvs,
                                        Eigen::VectorXd const& _cotans);

// Adjoint of harmonic_param_forward. _cotans / _boundary_uvs must hold the
// same values as in forward (the caller owns them; they are not copied into
// the ctx). With mu := A^-1 d_inner (A symmetric):
//   d_cotans[e] += mu(i,:)·X(j,:)            (off-diagonal entries)
//   d_cotans[e] -= mu(i,:)·X(i,:)            (diagonal entries)
//   d_cotans[e] += mu(i,:)·uvs(b,:)          (rhs entries)
//   d_boundary_uvs[b] += cotans[e] * mu(i,:) (rhs entries)
// Accumulates into _d_cotans (global, across patches) and _d_boundary_uvs
// (per patch, sized like the forward input).
void harmonic_param_backward(PatchHarmonicCtx const& _ctx,
                             std::vector<vec2d> const& _boundary_uvs,
                             Eigen::VectorXd const& _cotans,
                             Eigen::MatrixX2d const& _d_inner_uvs,
                             Eigen::VectorXd& _d_cotans,
                             std::vector<vec2d>& _d_boundary_uvs);

} // namespace LayoutOpt
