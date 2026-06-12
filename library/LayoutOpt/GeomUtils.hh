#pragma once

#include <optional>
#include "LayoutOpt/DataStructures/Types.hh"

namespace LayoutOpt
{
// remove-autodiff Phase 5: Eigen/plain-double ports of the TorchUtils geometry
// helpers used by the detached (non-differentiable) pipeline stages. Each port
// mirrors the torch implementation's arithmetic, including torch's normalize
// semantics v / max(|v|, 1e-12).

vec3d normalized_eps(vec3d const& _v);
vec2d normalized_eps(vec2d const& _v);

/// Eigen counterpart of torch_to_pos3 for the detached stages.
inline pos3 eigen_to_pos3(vec3d const& _v) { return pos3(_v.x(), _v.y(), _v.z()); }

/**
 * @brief Computes a 2D embedding for the vertices of a given face.
 *
 * The vertex referenced by `_heh.next().vertex_to()` is set as the origin of the
 * embedding and `_heh.vertex_from()` lies on the positive x-axis.
 *
 * @param _heh A half-edge handle representing the face.
 * @param _3D_pos The 3D vertex positions (|V| x 3, indexed by vertex idx).
 * @return Rows: vertex_from, vertex_to, next().vertex_to() (= origin, (0,0)).
 */
Eigen::Matrix<double, 3, 2> compute_2D_face_embedding(HEH _heh, Eigen::MatrixX3d const& _3D_pos);

/**
 * @brief 2D embedding of the face of _heh where _3D_origin becomes the origin and
 * the point _3D_x_axis will be lying on the x-axis.
 * @param assumes that _3D_origin and _3D_x_axis are in the plane of the face
 * @param _positive tells if the vector should be aligned along the positive or negative direction
 * @return Rows: vertex_from, vertex_to, next().vertex_to().
 */
Eigen::Matrix<double, 3, 2> compute_2D_face_embedding(
    HEH _heh, vec3d const& _3D_origin, vec3d const& _3D_x_axis, bool _positive, Eigen::MatrixX3d const& _3D_pos);

/// Barycentric coordinates of _point wrt the (2D-embedded) face corners of _heh
/// (vertex_from, vertex_to, next().vertex_to()), via the same homogeneous 3x3
/// partial-pivot LU solve as the torch path (torch::linalg_solve).
vec3d compute_bary_coords_2D(vec2d const& _point, HEH const& _heh, Eigen::MatrixX2d const& _pos);

vec3d face_normal(FH const& _fh, Eigen::MatrixX3d const& _pos);

/// Intersection of line a (_a0,_a1) with line b (_b0,_b1): returns (1-t, t) with
/// intersection = a0 + t*(a1-a0); nullopt if (near-)parallel (|cross| < 1e-8).
std::optional<vec2d> compute_intersection_parameter(vec2d const& _a0, vec2d const& _a1, vec2d const& _b0, vec2d const& _b1);

} // namespace LayoutOpt
