#include "GeomUtils.hh"

#include <cmath>

namespace LayoutOpt
{
namespace
{
// torch::nn::functional::normalize default eps
constexpr double NORMALIZE_EPS = 1e-12;
constexpr double PI = 3.14159265358979323846; // M_PI
}

vec3d normalized_eps(vec3d const& _v) { return _v / std::max(_v.norm(), NORMALIZE_EPS); }
vec2d normalized_eps(vec2d const& _v) { return _v / std::max(_v.norm(), NORMALIZE_EPS); }

Eigen::Matrix<double, 3, 2> compute_2D_face_embedding(HEH _heh, Eigen::MatrixX3d const& _3D_pos)
{
    vec3d const p0 = _3D_pos.row(_heh.vertex_from().idx.value).transpose();
    vec3d const p1 = _3D_pos.row(_heh.vertex_to().idx.value).transpose();
    vec3d const p2 = _3D_pos.row(_heh.next().vertex_to().idx.value).transpose(); // this will be the new origin

    vec3d const e0 = p0 - p2;
    vec3d const e1 = p1 - p2;

    double const length_e0 = e0.norm();
    double const length_e1 = e1.norm();

    vec3d const e0_normalized = normalized_eps(e0);
    vec3d const e1_normalized = normalized_eps(e1);

    double const angle = std::atan2(e0_normalized.cross(e1_normalized).norm(), e0_normalized.dot(e1_normalized));

    Eigen::Matrix<double, 3, 2> res;
    res.row(0) = vec2d(length_e0 * 1.0, length_e0 * 0.0); // x-axis
    res.row(1) = vec2d(length_e1 * std::cos(angle), length_e1 * std::sin(angle));
    res.row(2) = vec2d::Zero();
    return res;
}

Eigen::Matrix<double, 3, 2> compute_2D_face_embedding(
    HEH _heh, vec3d const& _3D_origin, vec3d const& _3D_x_axis, bool _positive, Eigen::MatrixX3d const& _3D_pos)
{
    vec3d const A = _3D_pos.row(_heh.vertex_from().idx.value).transpose();
    vec3d const B = _3D_pos.row(_heh.vertex_to().idx.value).transpose();
    vec3d const C = _3D_pos.row(_heh.next().vertex_to().idx.value).transpose();

    vec3d const& O = _3D_origin; // renaming
    vec3d const& X = _3D_x_axis; // renaming

    // face normal
    vec3d const AC = C - A;
    vec3d const AB = B - A;
    vec3d const normal = normalized_eps(AB.cross(AC));

    // computing all lengths and angles in 3D
    // vectors
    vec3d const OA = A - O;
    vec3d const OB = B - O;
    vec3d const OC = C - O;
    vec3d const OX = X - O;
    // lengths
    double const length_OA = OA.norm();
    double const length_OB = OB.norm();
    double const length_OC = OC.norm();
    // normalized
    vec3d const OA_normalized = normalized_eps(OA);
    vec3d const OB_normalized = normalized_eps(OB);
    vec3d const OC_normalized = normalized_eps(OC);
    vec3d const OX_normalized = normalized_eps(OX);
    // angles
    double angle_XOA = std::atan2(OX_normalized.cross(OA_normalized).dot(normal), OA_normalized.dot(OX_normalized));
    double angle_XOB = std::atan2(OX_normalized.cross(OB_normalized).dot(normal), OB_normalized.dot(OX_normalized));
    double angle_XOC = std::atan2(OX_normalized.cross(OC_normalized).dot(normal), OC_normalized.dot(OX_normalized));

    // correct for alignment along neg x direction
    if (!_positive)
    {
        angle_XOA += PI;
        angle_XOB += PI;
        angle_XOC += PI;
    }

    Eigen::Matrix<double, 3, 2> res;
    res.row(0) = vec2d(length_OA * std::cos(angle_XOA), length_OA * std::sin(angle_XOA));
    res.row(1) = vec2d(length_OB * std::cos(angle_XOB), length_OB * std::sin(angle_XOB));
    res.row(2) = vec2d(length_OC * std::cos(angle_XOC), length_OC * std::sin(angle_XOC));
    return res;
}

vec3d compute_bary_coords_2D(vec2d const& _point, HEH const& _heh, Eigen::MatrixX2d const& _pos)
{
    vec2d const p0 = _pos.row(_heh.vertex_from().idx.value).transpose();
    vec2d const p1 = _pos.row(_heh.vertex_to().idx.value).transpose();
    vec2d const p2 = _pos.row(_heh.next().vertex_to().idx.value).transpose();

    // The torch path solves T^t * lambda = P (homogeneous 3x3 system) via
    // torch::linalg_solve, i.e. LAPACK partial-pivot LU; PartialPivLU matches.
    Eigen::Matrix3d T_t;
    T_t << p0.x(), p1.x(), p2.x(), //
        p0.y(), p1.y(), p2.y(),    //
        1.0, 1.0, 1.0;

    return T_t.partialPivLu().solve(vec3d(_point.x(), _point.y(), 1.0));
}

vec3d face_normal(FH const& _fh, Eigen::MatrixX3d const& _pos)
{
    auto const [vh0, vh1, vh2] = _fh.vertices().to_array<3>();
    vec3d const p0 = _pos.row(vh0.idx.value).transpose();
    vec3d const p1 = _pos.row(vh1.idx.value).transpose();
    vec3d const p2 = _pos.row(vh2.idx.value).transpose();
    return normalized_eps(vec3d(p1 - p0).cross(p2 - p0));
}

std::optional<vec2d> compute_intersection_parameter(vec2d const& _a0, vec2d const& _a1, vec2d const& _b0, vec2d const& _b1)
{
    // Compute direction vectors.
    vec2d const a = _a1 - _a0;
    vec2d const b = _b1 - _b0;

    // Compute the cross product (scalar in 2D)
    double const cross = a.x() * b.y() - a.y() * b.x();
    if (std::abs(cross) < 1e-8)
    {
        DEBUG_OUT("Lines are parallel or nearly parallel.")
        return std::nullopt;
    }

    // Compute parameter t such that intersection = a0 + t*(a1 - a0)
    vec2d const b0_a0 = _b0 - _a0;
    double const t = (b0_a0.x() * b.y() - b0_a0.y() * b.x()) / cross;

    // Barycentric coordinates relative to line_a are (1-t, t)
    return vec2d(1.0 - t, t);
}

} // namespace LayoutOpt
