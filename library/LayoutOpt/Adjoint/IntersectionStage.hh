#pragma once
// S3 of the adjoint chain (see documentation/adjointDifferentiation.md): the
// stage that OVERWRITES the overlay position rows of the interior path-network
// vertices of each arc with the 3D point implied by the 2D strip-flattening
// intersection parameter.
//
// Per layout edge (= one TriangleStrip), with A/B the flattened 2D positions
// of the arc's endpoint surface points (strip.pn_vhs.front()/back()):
//   for each interior pn vertex i = 1 .. pn_vhs.size()-2 (an EdgePoint whose
//   heh_idx names the crossed target halfedge hh):
//     from_2D = strip.heh_pos_2d[hh]        to_2D = strip.heh_pos_2d[hh.opposite()]
//     closed-form 2D line-line intersection parameter
//     (line_a = (to_2D, from_2D), line_b = (A, B)):
//       a       = from_2D - to_2D
//       b       = B - A
//       cross   = a.x*b.y - a.y*b.x
//       b0_a0   = A - to_2D
//       t_numer = b0_a0.x*b.y - b0_a0.y*b.x
//       t       = t_numer / cross
//     pos[o_row] = from_3D * t + to_3D * (1 - t)   (3D target edge endpoints)
//
// Differentiation boundary (remove-autodiff plan, "Phase 2 result"): the strip
// flattening heh_pos_2d is CONSTANT, so a, to_2D and the 3D target positions
// from_3D/to_3D carry no gradient — the only differentiable inputs are A and B
// (which S1 derives from the layout-node face-point barycentrics). Hence
//   d_t       = d_pos[o_row] . (from_3D - to_3D)
//   d_t_numer = d_t / cross
//   d_cross   = -d_t * t_numer / cross^2
//   d_b0_a0   = ( d_t_numer * b.y, -d_t_numer * b.x)
//   d_b       = (-d_t_numer * b0_a0.y - d_cross * a.y,
//                 d_t_numer * b0_a0.x + d_cross * a.x)
//   d_A      += d_b0_a0 - d_b
//   d_B      += d_b
//
// Faithfulness notes:
//  * the original torch implementation guarded |cross| < 1e-8 by returning an
//    EMPTY tensor, which crashed one line later — i.e. a hard stop, never hit
//    in practice (strips are validity-checked). The forward asserts instead.
//  * the interior overlay rows are OVERWRITTEN, so their entire incoming
//    adjoint is consumed here (routed to A/B); the rows' pre-write values are
//    dead. Layout-node rows and untouched target rows are not this stage's
//    business (S1 / constants).
//  * each interior pn vertex belongs to exactly one arc, so the written rows
//    are disjoint across strips.

#include <vector>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/TriangleStrip.hh>
#include <LayoutOpt/DataStructures/Types.hh>

namespace LayoutOpt
{

// One record per interior pn vertex, in production write order. Everything
// backward needs is copied in, so backward touches neither strips nor pnd.
struct IntersectionRec
{
    int o_row = -1;             // overlay position row that was overwritten
    int t_from = -1, t_to = -1; // target vertex rows of the crossed halfedge (from_3D / to_3D, constants)
    vec2d a = vec2d::Zero();    // from_2D - to_2D (constant)
    vec2d b = vec2d::Zero();    // B - A
    vec2d b0_a0 = vec2d::Zero(); // A - to_2D
    double cross = 0.0, t_numer = 0.0, t = 0.0;
};

struct StripIntersectCtx
{
    int l_edge = -1; // layout edge index (row of _strip_end_a/_strip_end_b)
    std::vector<IntersectionRec> recs;
};

struct IntersectCtx
{
    std::vector<StripIntersectCtx> strips;
};

// Plain port of the per-strip intersection loop. _strip_end_a/_strip_end_b are
// the flattened 2D endpoint positions per layout edge (row = l_eh.idx.value);
// _pos is the overlay position matrix (pos_mat_ layout) whose interior rows
// are overwritten IN PLACE; the 3D constants come from _tmd.pos_mat_.
void intersections_forward(std::vector<TriangleStrip> const& _strips,
                           Eigen::MatrixX2d const& _strip_end_a,
                           Eigen::MatrixX2d const& _strip_end_b,
                           TargetMeshData const& _tmd,
                           LayoutData const& _ld,
                           PathNetworkData const& _pnd,
                           IntersectCtx& _ctx,
                           Eigen::MatrixX3d& _pos);

// Adjoint of intersections_forward. READS the written rows of _d_pos (the
// total loss gradient w.r.t. the post-write overlay positions, as produced by
// the S4..S7 + curvature backwards) and ACCUMULATES into _d_strip_end_a/_b
// (caller-zeroed, n_layout_edges x 2). _d_pos is not modified: the written
// rows' adjoints are consumed here by construction (their pre-write values are
// dead), and all other rows are upstream's business.
void intersections_backward(IntersectCtx const& _ctx,
                            Eigen::MatrixX3d const& _d_pos,
                            TargetMeshData const& _tmd,
                            Eigen::MatrixX2d& _d_strip_end_a,
                            Eigen::MatrixX2d& _d_strip_end_b);

} // namespace LayoutOpt
