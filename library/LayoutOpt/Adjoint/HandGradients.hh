#pragma once
// Phase 4 / S1 of the remove-autodiff plan: the hand-rolled replacement of
// compute_gradients (ObjectiveFunctions.cc) — the full forward loss + reverse
// pass over the validated Phase 4 stage pairs, torch-free except for the
// constant strip-flattening reads inside leaf_collect / intersections_forward
// (those turn plain in Phase 5/6).
//
// Forward chain (mirrors eval() in Optimization.cc + harmonic_distortion_loss):
//   leaf_forward            bary -> 3D layout-node overlay rows + 2D strip
//                           endpoints A/B                          (S1)
//   intersections_forward   A/B -> overwritten interior overlay rows (S3)
//   if w_harmonic > 0:
//     edge_lengths/pn_t/pn_uvs_forward                              (S4)
//     cotans_forward                                                (S5)
//     per patch: prepare_param -> harmonic_param_forward            (S6)
//                -> patch_distortion_forward                        (S7)
//     harmonic = distortion_loss_forward
//   if w_curvature > 0:
//     curvature = curvature_alignment_forward
//   loss = w_harmonic * harmonic + w_curvature * curvature
//
// Backward runs the adjoints in reverse with the established ordering
// constraints: pn_uvs_backward and pn_t_backward both feed d_lengths, so
// edge_lengths_backward runs last of the S4 trio; every d_pos accumulator
// (S7 direct path, S4, S5, curvature) runs before intersections_backward and
// leaf_backward, which only READ d_pos.
//
// compute_gradients_hand mirrors the production collection exactly: a
// zero-initialized per-pn-vertex attribute, filled only for FacePoint surface
// points (production reads .grad() only for those; gradient reaching any
// other type is discarded there too).

#include <vector>

#include <LayoutOpt/Adjoint/BoundaryUvStage.hh>
#include <LayoutOpt/Adjoint/CotanStage.hh>
#include <LayoutOpt/Adjoint/CurvatureAlignmentStage.hh>
#include <LayoutOpt/Adjoint/DistortionStage.hh>
#include <LayoutOpt/Adjoint/HarmonicStage.hh>
#include <LayoutOpt/Adjoint/IntersectionStage.hh>
#include <LayoutOpt/Adjoint/LeafStage.hh>
#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/TriangleStrip.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/OptimizationOptions.hh>

namespace LayoutOpt
{

// Everything one forward pass produces that the backward pass (or a caller
// comparing intermediates) needs.
struct HandLossCtx
{
    // forward values
    Eigen::MatrixX3d pos;          // overlay positions after the S1 + S3 writes
    Eigen::MatrixX2d seg_a, seg_b; // per-layout-edge 2D strip endpoints
    Eigen::VectorXd lengths, t, cotans;
    Eigen::MatrixX2d uvs;        // per pn halfedge
    std::vector<int> pn_of_oheh; // overlay heh -> pn heh (boundary-UV scatter)

    // stage contexts
    IntersectCtx ictx;
    EdgeLengthCtx el_ctx;
    TCtx t_ctx;
    UvCtx uv_ctx;
    CotanCtx cot_ctx;
    std::vector<PatchHarmonicCtx> hctx;
    std::vector<PatchDistortionCtx> sctx;
    std::vector<std::vector<vec2d>> bnd_uvs;
    std::vector<std::vector<int>> bnd_src;
    CurvAlignCtx curv_ctx;

    // loss pieces (harmonic / curvature are unweighted)
    double harmonic = 0.0;
    double curvature = 0.0;
    double loss = 0.0;
};

// Full hand forward from explicit bary values [n_pn_v x 2] (see collect_bary).
// _omd is read-only (const everywhere; attribute creation via mesh_ pointer).
// Returns the weighted total loss (also stored in _ctx.loss).
double hand_loss_forward(LeafCtx const& _leaf,
                         Eigen::MatrixX2d const& _bary,
                         std::vector<TriangleStrip> const& _strips,
                         TargetMeshData const& _tmd,
                         LayoutData const& _ld,
                         PathNetworkData const& _pnd,
                         OverlayMeshData const& _omd,
                         OptimizationOptions const& _opts,
                         HandLossCtx& _ctx);

// Full hand backward: accumulates the total-loss leaf gradient into _d_bary
// (caller-zeroed, n_pn_v x 2).
void hand_loss_backward(HandLossCtx const& _ctx,
                        LeafCtx const& _leaf,
                        TargetMeshData const& _tmd,
                        OptimizationOptions const& _opts,
                        Eigen::MatrixX2d& _d_bary);

// The compute_gradients replacement: leaf collection + forward + backward in
// one call. Returns the per-pn-vertex bary gradient (vec2d::Zero() default,
// FacePoint entries filled — production's collection convention). If
// _loss_out is non-null it receives the hand forward loss.
pm::vertex_attribute<vec2d> compute_gradients_hand(std::vector<TriangleStrip> const& _strips,
                                                   TargetMeshData const& _tmd,
                                                   LayoutData const& _ld,
                                                   PathNetworkData const& _pnd,
                                                   OverlayMeshData const& _omd,
                                                   OptimizationOptions const& _opts,
                                                   double* _loss_out = nullptr);

} // namespace LayoutOpt
