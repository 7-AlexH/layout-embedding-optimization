#pragma once
// S6 of the adjoint chain (see documentation/adjointDifferentiation.md): the
// per-patch harmonic parameterization — assemble the interior cotan-Laplacian
// system A := -L, factor once (SimplicialLDLT, SparseLU fallback), solve for
// the interior UVs, and hand-derive the solve adjoint. prepare_param is the
// patch-collection indexing that feeds it.
//
// Stage boundary (inputs treated as leaves for validation):
//   inputs  : per-overlay-edge cotans [n_edges] (S5 output), per-patch
//             boundary UVs (S4 output)
//   output  : per-patch inner UVs [n_inner x 2]
//   adjoints: d_cotans (accumulated across patches), d_boundary_uvs

#include <cstdint>
#include <vector>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/ObjectiveFunctions.hh> // MappingIndex

namespace LayoutOpt
{

// Collect one patch's faces and classify its overlay vertices into boundary
// (UV pinned from the path network) and inner (solved for). The boundary UVs
// are gathered from a per-overlay-halfedge UV array in the from-vertex
// convention (the o_uvs gather in hand_loss_forward). If _boundary_src_hehs
// is non-null it receives, per boundary entry, the overlay halfedge idx the
// UV was gathered from (the row of _o_uvs) — used to scatter d_boundary_uvs
// back to per-halfedge UV rows.
//
// NOTE: _map_to_vec must arrive in the fresh {-1, true} state and is the only
// argument shared across patches; hand_loss_forward's parallel loop hands each
// thread its own scratch attribute and resets exactly the entries written here
// (the inner/boundary lists) after every patch.
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

// Forward context: the flattened system structure plus the solution. Backward
// re-assembles and re-factors A from these rather than stowing the
// factorization (one factorization object per patch would dominate the ctx
// memory; the re-factor is bitwise-identical input, so the adjoint solve is
// exact w.r.t. the forward solve).
struct PatchHarmonicCtx
{
    int n_inner = 0;
    std::vector<int64_t> off;  // flattened (i, j, e): A(i,j) = -cotans[e], inner neighbor j
    std::vector<int64_t> diag; // flattened (i, e):    A(i,i) += cotans[e], per outgoing halfedge
    std::vector<int64_t> rhs;  // flattened (i, b, e): B(i,:) += cotans[e] * boundary_uvs[b]
    Eigen::MatrixX2d inner_uvs; // X, the solve output [n_inner x 2]
};

// Build the system structure, then assemble A := -L / B and solve A X = B.
// n_inner == 0 yields an empty ctx.
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
