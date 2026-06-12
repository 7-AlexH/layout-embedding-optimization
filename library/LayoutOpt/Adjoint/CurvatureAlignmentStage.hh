#pragma once
// Curvature-alignment branch of the adjoint chain (see
// documentation/adjointDifferentiation.md): hand-rolled forward +
// reverse-mode adjoint of the principal-curvature alignment loss.
// This is an INDEPENDENT loss branch: it reads only
// the overlay 3D positions — the direction-field basis/dir/confidence are
// precomputed from the FIXED target mesh and carry no gradient — so its
// backward accumulates straight into d_overlay_pos alongside the S4..S7 chain
// (no ordering constraint w.r.t. the other stages).
//
// Math per pn edge, in pn-edge iteration order (preserved from the original
// torch implementation):
//   vec   = pos[om_a] - pos[om_b]      (overlay endpoints of the pn edge)
//   n     = |vec|                      (bare norm, no eps)
//   v2    = basis * vec                (constant 2x3 tangent basis of the
//                                       target face under the edge)
//   th    = atan2(v2.y, v2.x)
//   rosy  = (cos(4 th), sin(4 th))
//   align = |dir - rosy|^2             (constant 2-vec 4-rosy field direction)
//   loss  = sum_e(n_e * align_e) / sum_e(n_e)
//
// Faithfulness notes:
//  * the original loop carried a commented-out `* direction_data.confidence`
//    factor (dead code); it is NOT mirrored here.
//  * n appears in numerator and denominator; both quotient paths feed d_n.
//  * at vec = 0 (degenerate segment) or v2 = 0 (segment orthogonal to the
//    tangent plane) the derivative is 0/0 = NaN — the original torch
//    implementation produced NaN here too, so no epsilon is added.
//  * summation is sequential in edge order (the original summed via
//    stack().sum(); values agreed to ~ulp, not bitwise).

#include <vector>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>

namespace LayoutOpt
{

// One record per pn edge, in pn-edge iteration order. basis/dir are copied in
// (constants), so backward needs no mesh or field-data access; vec is
// recomputed from _pos via om_a/om_b.
struct CurvAlignEdge
{
    int om_a = -1, om_b = -1;          // overlay vertex rows of pn_eh.vertexA()/vertexB()
    Eigen::Matrix<double, 2, 3> basis; // tangent basis of the target face under the edge
    vec2d dir = vec2d::Zero();         // 4-rosy field direction
    double norm = 0.0;                 // |vec|
    vec2d v2 = vec2d::Zero();          // basis * vec
    double cos4 = 0.0, sin4 = 0.0;     // the rosy vector
    double align = 0.0;                // |dir - rosy|^2
};

struct CurvAlignCtx
{
    double num = 0.0; // sum_e norm_e * align_e
    double den = 0.0; // sum_e norm_e
    std::vector<CurvAlignEdge> edges;
};

// Plain port of principal_curvature_alignment_loss: returns num/den and fills
// the ctx for backward. _pos is the overlay position matrix (pos_mat_ layout);
// _tmd.direction_field_data_ must be set (reads the Eigen mirrors).
double curvature_alignment_forward(Eigen::MatrixX3d const& _pos,
                                   TargetMeshData const& _tmd,
                                   PathNetworkData const& _pnd,
                                   OverlayMeshData const& _omd,
                                   CurvAlignCtx& _ctx);

// Adjoint of curvature_alignment_forward, seeded with _d_loss (the caller
// applies the w_curvature_alignment_loss weight there). With N = num, D = den:
//   d_num = _d_loss / D ;  d_den = -_d_loss * N / D^2
//   d_n     = d_num * align + d_den
//   d_align = d_num * n
//   d_rosy  = -2 (dir - rosy) * d_align
//   d_th4   = -sin4 * d_rosy.x + cos4 * d_rosy.y ;  d_th = 4 * d_th4
//   d_v2    = (-v2.y, v2.x) / |v2|^2 * d_th         (atan2 adjoint)
//   d_vec   = basis^T * d_v2 + (d_n / n) * vec
//   d_pos[om_a] += d_vec ;  d_pos[om_b] -= d_vec
// Accumulates into _d_pos (sized like _pos).
void curvature_alignment_backward(CurvAlignCtx const& _ctx,
                                  Eigen::MatrixX3d const& _pos,
                                  double _d_loss,
                                  Eigen::MatrixX3d& _d_pos);

} // namespace LayoutOpt
