#pragma once
// Phase 4 / S1 of the remove-autodiff plan: hand-rolled forward + reverse-mode
// adjoint of the LEAF stage — the SurfacePoint::get_pos interpolations that
// connect the free variables (the layout-node face-point barycentrics, plan
// "Phase 2 result") to the rest of the chain:
//
//  (a) 3D layout-node overlay rows (compute_differentiable_surface_points):
//      per pn vertex with a valid map_to_layout_vertices_ entry,
//        pos[o_row] = alpha*A + beta*B + (1 - alpha - beta)*C
//      with A/B/C the target positions of (hh.from, hh.to, hh.next.to) for the
//      sp's halfedge hh — the get_pos(|V|x3 tensor, mesh) corner convention.
//
//  (b) 2D strip-endpoint positions A/B per layout edge
//      (compute_differentiable_intersection_for_overlay): the same bary
//      interpolation over the CONSTANT strip flattening heh_pos_2d, with the
//      halfedge-attribute corner convention (pos[hh], pos[hh.next()],
//      pos[hh.next().next()]) — the "residual S2 math" per the S2 closeout.
//      These feed S3 (IntersectionStage).
//
// Adjoint (FacePoint, the only leaf type in practice):
//   d_alpha += d_out . (A - C) ;  d_beta += d_out . (B - C)
// accumulated from BOTH (a) — d_overlay_pos at the node row — and (b) — the
// d_seg_A/d_seg_B rows produced by intersections_backward. EdgePoint sps use
// d_alpha += d_out . (A - B); VertexPoint sps carry no gradient. (Production
// sets requires_grad on every mapped sp regardless of type, but
// compute_gradients only collects FacePoint grads — mirrored in
// compute_gradients_hand, not here.)
//
// Faithfulness notes:
//  * the 2D corners are copied out of the strips' torch tensors at collect
//    time (constants per the Phase 2 audit; heh_pos_2d turns plain in
//    Phase 5/6) — forward/backward are then torch-free, so a finite-
//    difference loop over leaf_forward costs no torch traffic here.
//  * each mapped pn vertex owns exactly one overlay node row, but appears as
//    a strip endpoint once per incident layout edge (valence-many records) —
//    the adjoint accumulates across all of them, exactly like autograd.
//  * arithmetic matches get_pos_intern's op order (validated bitwise in the
//    Phase 3 mirror check: alpha*A + beta*B + (1.0 - alpha - beta)*C).

#include <vector>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/TriangleStrip.hh>
#include <LayoutOpt/DataStructures/Types.hh>

namespace LayoutOpt
{

// One record per mapped pn vertex (= layout node), in pn-vertex order (the
// production write order of compute_differentiable_surface_points).
struct LeafNodeRec
{
    int pn_v = -1;  // pn vertex idx (row of bary / d_bary)
    int o_row = -1; // overlay position row this leaf overwrites
    SurfacePointType type = SurfacePointType::Invalid;
    int rA = -1, rB = -1, rC = -1; // target vertex rows of the sp's corners (constants)
};

// One record per (layout edge, endpoint side): the strip-endpoint 2D
// interpolation. side 0 = pn_vhs.front() -> row of seg_a; side 1 =
// pn_vhs.back() -> row of seg_b.
struct LeafEndpointRec
{
    int l_edge = -1;
    int side = 0;
    int pn_v = -1;
    SurfacePointType type = SurfacePointType::Invalid;
    vec2d A2 = vec2d::Zero(), B2 = vec2d::Zero(), C2 = vec2d::Zero(); // constant 2D corners
};

struct LeafCtx
{
    std::vector<LeafNodeRec> nodes;
    std::vector<LeafEndpointRec> ends;
};

// One-time structural collection (topology + the constant 2D corners read out
// of the strips). Valid as long as the path network / strips stay alive.
void leaf_collect(std::vector<TriangleStrip> const& _strips,
                  TargetMeshData const& _tmd,
                  LayoutData const& _ld,
                  PathNetworkData const& _pnd,
                  LeafCtx& _ctx);

// Current bary values per pn vertex [n_pn_v x 2] (tensor-authoritative during
// Phases 4-5): FacePoint -> (alpha, beta); EdgePoint -> (alpha, 0);
// VertexPoint / invalid -> (0, 0).
Eigen::MatrixX2d collect_bary(PathNetworkData const& _pnd);

// Overwrite the layout-node rows of _pos (pos_mat_ layout, 3D corners from
// _t_pos = TargetMeshData::pos_mat_) and fill the per-layout-edge endpoint
// rows of _seg_a/_seg_b (callers size them n_layout_edges x 2).
void leaf_forward(LeafCtx const& _ctx,
                  Eigen::MatrixX2d const& _bary,
                  Eigen::MatrixX3d const& _t_pos,
                  Eigen::MatrixX3d& _pos,
                  Eigen::MatrixX2d& _seg_a,
                  Eigen::MatrixX2d& _seg_b);

// Adjoint of leaf_forward: routes the node rows of _d_pos and the rows of
// _d_seg_a/_d_seg_b into _d_bary (caller-zeroed, n_pn_v x 2). The node rows'
// adjoints are consumed here by construction (production overwrites them,
// their pre-write values are dead); all other _d_pos rows are constants'.
void leaf_backward(LeafCtx const& _ctx,
                   Eigen::MatrixX3d const& _t_pos,
                   Eigen::MatrixX3d const& _d_pos,
                   Eigen::MatrixX2d const& _d_seg_a,
                   Eigen::MatrixX2d const& _d_seg_b,
                   Eigen::MatrixX2d& _d_bary);

} // namespace LayoutOpt
