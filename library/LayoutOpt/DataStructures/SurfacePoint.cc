#include "SurfacePoint.hh"
namespace LayoutOpt
{

SurfacePoint SurfacePoint::copy() const { return *this; }

FH SurfacePoint::fh(polymesh::Mesh const& _mesh) const { return _mesh.handle_of(heh_idx).face(); }

bool SurfacePoint::is_inside_element()
{
    if (type == SurfacePointType::VertexPoint)
    {
        return true;
    }
    if (type == SurfacePointType::EdgePoint)
    {
        double a = bary_params.x();
        double b = 1.0 - a;

        return (a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0);
    }
    if (type == SurfacePointType::FacePoint)
    {
        // FacePoint: Compute c = 1 - a - b
        double a = bary_params.x();
        double b = bary_params.y();
        double c = 1.0 - a - b;

        return (a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0 && c >= 0.0 && c <= 1.0);
    }
    return false;
}

vec3d SurfacePoint::bary_full() const
{
    switch (type)
    {
    case SurfacePointType::VertexPoint:
        return vec3d(1.0, 0.0, 0.0);
    case SurfacePointType::EdgePoint:
    {
        double const a = bary_params.x();
        return vec3d(a, 1.0 - a, 0.0);
    }
    case SurfacePointType::FacePoint:
    {
        double const a = bary_params.x();
        double const b = bary_params.y();
        return vec3d(a, b, 1.0 - a - b);
    }
    default:
        return vec3d(0.0, 0.0, 0.0);
    }
}

vec3d SurfacePoint::get_pos(Eigen::MatrixX3d const& _pos, polymesh::Mesh const& _mesh) const
{
    vec3d const b = bary_full();
    auto const hh = _mesh.handle_of(heh_idx);

    // The three face corners. For vertex/edge points the unused corners carry a
    // zero weight in b, so a single weighted sum covers all surface point types.
    Eigen::RowVector3d const A = _pos.row(hh.vertex_from().idx.value);
    Eigen::RowVector3d const B = _pos.row(hh.vertex_to().idx.value);
    Eigen::RowVector3d const C = _pos.row(hh.next().vertex_to().idx.value);

    return (b[0] * A + b[1] * B + b[2] * C).transpose();
}

vec2d SurfacePoint::get_pos(pm::halfedge_attribute<std::optional<vec2d>> const& _pos) const
{
    auto const hh = _pos.mesh().handle_of(heh_idx);
    vec3d const b = bary_full();

    // Corner convention of the halfedge-attribute overload: the position is
    // associated with vertex_from, so corners are read off successive halfedges
    // (NOT vertex indices): vertex -> hh; edge -> hh, hh.opposite(); face -> hh,
    // hh.next(), hh.next().next(). The unused corners carry zero weight in b.
    if (type == SurfacePointType::VertexPoint)
    {
        return _pos[hh].value();
    }
    else if (type == SurfacePointType::EdgePoint)
    {
        return b[0] * _pos[hh].value() + b[1] * _pos[hh.opposite()].value();
    }
    else if (type == SurfacePointType::FacePoint)
    {
        return b[0] * _pos[hh].value() + b[1] * _pos[hh.next()].value() + b[2] * _pos[hh.next().next()].value();
    }

    assert(false && "should not be reached");
    return vec2d::Zero();
}

vec2d SurfacePoint::get_pos(Eigen::MatrixX2d const& _pos, polymesh::Mesh const& _mesh) const
{
    vec3d const b = bary_full();
    auto const hh = _mesh.handle_of(heh_idx);

    Eigen::RowVector2d const A = _pos.row(hh.vertex_from().idx.value);
    Eigen::RowVector2d const B = _pos.row(hh.vertex_to().idx.value);
    Eigen::RowVector2d const C = _pos.row(hh.next().vertex_to().idx.value);

    return (b[0] * A + b[1] * B + b[2] * C).transpose();
}

// special values
SurfacePoint const SurfacePoint::invalid = SurfacePoint(vec2d::Zero(), pm::halfedge_index::invalid, SurfacePointType::Invalid);

} // namespace LayoutOpt
