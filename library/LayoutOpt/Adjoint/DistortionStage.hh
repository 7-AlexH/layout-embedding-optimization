#pragma once
// S7 of the adjoint chain (see documentation/adjointDifferentiation.md):
// hand-rolled forward + reverse-mode adjoint of the distortion stage. Per
// overlay face: rebuild the 2D reference/parameter triangles from the corner
// positions (3D overlay positions / 2D patch UVs), J = M_param * M_ref^-1,
// closed-form 2x2 singular values, weighted distortion branches; per-patch
// summation; loss assembly across patches.
//
// Mirrors the original torch implementation exactly: same op structure, same
// early-outs, same subgradient choices —
//   * early-out faces (ref/param area < EPS, or NaN areas) contribute zero
//     value AND zero gradient (the original returned fresh disconnected zero
//     tensors);
//   * the AIAP min/max selection is by the same value comparison (s0 < s1),
//     and gradient flows only through the selected singular values;
//   * the (dead, const-false) `normalized` per-patch-area branch is kept for
//     parity in the loss assembly.
//
// Stage boundary (inputs treated as leaves for validation):
//   inputs  : per-patch inner UVs [n_inner x 2] (S6 output), per-patch
//             boundary UVs (S4 output), overlay 3D positions [n_overlay x 3]
//             (S1/S3 output)
//   output  : scalar loss
//   adjoints: d_inner_uvs, d_boundary_uvs, d_overlay_pos

#include <vector>

#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/ObjectiveFunctions.hh> // MappingIndex
#include <LayoutOpt/OptimizationOptions.hh>

namespace LayoutOpt
{

// All forward intermediates one face's backward pass needs. Stored rather
// than recomputed so forward and backward see bitwise-identical values.
struct FaceDistortionCtx
{
    // corner identity for the backward scatter; order (A, B, C) =
    // (hehA.from, hehB.from, hehC.from) with hehA = o_fh.any_halfedge()
    int o_vh[3] = {-1, -1, -1};
    MappingIndex uv[3];

    // outputs
    double distortion = 0.0;
    double param_area = 0.0; // as returned (forced to 0 when skipped)
    bool skipped = false;    // early-out: zero contribution, zero gradient

    // intermediates (everything below `ref_area` is valid only when !skipped)
    vec3d ab3, ac3, u_ab3, u_ac3, unit_axis;
    vec2d ab2, ac2, u_ab2, u_ac2;
    double l_ab3 = 0, l_ac3 = 0, l_ab2 = 0, l_ac2 = 0;
    double num3 = 0, den3 = 0, angle3 = 0;
    double num2 = 0, den2 = 0, angle2 = 0;
    vec2d ref_B, ref_C, param_B, param_C; // rebuilt corners, A at the origin
    double ref_area = 0;
    mat2d P, K; // M_param and M_ref^-1
    double e = 0, f = 0, g = 0, h = 0, q = 0, r = 0, s0 = 0, s1 = 0;
};

// Forward for one face from its corner positions (2D patch UVs, 3D overlay
// positions). Corner identity fields (o_vh / uv) are left for the caller.
FaceDistortionCtx face_distortion_forward(vec2d const& _a2, vec2d const& _b2, vec2d const& _c2,
                                          vec3d const& _a3, vec3d const& _b3, vec3d const& _c3,
                                          HarmonicOptions const& _opt);

struct FaceDistortionGrad
{
    vec2d d_a2 = vec2d::Zero(), d_b2 = vec2d::Zero(), d_c2 = vec2d::Zero();
    vec3d d_a3 = vec3d::Zero(), d_b3 = vec3d::Zero(), d_c3 = vec3d::Zero();
};

// Adjoint of face_distortion_forward. _d_distortion / _d_param_area are the
// upstream adjoints of the two face outputs.
FaceDistortionGrad face_distortion_backward(FaceDistortionCtx const& _ctx,
                                            double _d_distortion,
                                            double _d_param_area,
                                            HarmonicOptions const& _opt);

struct PatchDistortionCtx
{
    double sde = 0.0;        // sum of face distortions (sequential, patch face order)
    double param_area = 0.0; // sum of face param areas
    std::vector<FaceDistortionCtx> faces;
};

// Gather (via _map_to_vec, as produced by prepare_param) + per-face forward
// over one patch.
PatchDistortionCtx patch_distortion_forward(std::vector<FH> const& _patch_fhs,
                                            pm::vertex_attribute<MappingIndex> const& _map_to_vec,
                                            Eigen::MatrixX2d const& _inner_uvs,
                                            std::vector<vec2d> const& _boundary_uvs,
                                            Eigen::MatrixX3d const& _overlay_pos,
                                            HarmonicOptions const& _opt);

// Adjoint of patch_distortion_forward: scatters into the per-patch UV adjoint
// buffers (sized like the forward inputs) and accumulates rows of
// _d_overlay_pos (sized like the full overlay position matrix).
void patch_distortion_backward(PatchDistortionCtx const& _ctx,
                               double _d_sde,
                               double _d_param_area,
                               HarmonicOptions const& _opt,
                               Eigen::MatrixX2d& _d_inner_uvs,
                               std::vector<vec2d>& _d_boundary_uvs,
                               Eigen::MatrixX3d& _d_overlay_pos);

// Loss assembly across patches: sum_p sde_p, or sum_p sde_p / param_area_p
// when _normalized (dead code today — HarmonicOptions::normalized is const
// false — kept for parity with the original implementation).
double distortion_loss_forward(std::vector<PatchDistortionCtx> const& _patches, bool _normalized);

// Adjoint of distortion_loss_forward: per-patch (d_sde, d_param_area) seeds.
void distortion_loss_backward(std::vector<PatchDistortionCtx> const& _patches,
                              bool _normalized,
                              double _d_loss,
                              std::vector<double>& _d_sde,
                              std::vector<double>& _d_param_area);

} // namespace LayoutOpt
