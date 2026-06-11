#include "SurfacePoint.hh"
#include "../TorchUtils.hh"
namespace LayoutOpt
{

SurfacePoint SurfacePoint::copy() const { return {bary_coords.clone().detach(), heh_idx, type}; }

FH SurfacePoint::fh(polymesh::Mesh const& _mesh) const { return _mesh.handle_of(heh_idx).face(); }

bool SurfacePoint::is_inside_element()
{
    if (type == SurfacePointType::VertexPoint)
    {
        return true;
    }
    if (type == SurfacePointType::EdgePoint)
    {
        // FacePoint: Compute c = 1 - a - b
        double a = bary_coords[0].item<double>();
        double b = 1.0 - a;

        return (a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0);
    }
    if (type == SurfacePointType::FacePoint)
    {
        // FacePoint: Compute c = 1 - a - b
        double a = bary_coords[0].item<double>();
        double b = bary_coords[1].item<double>();
        double c = 1.0 - a - b;

        return (a >= 0.0 && a <= 1.0 && b >= 0.0 && b <= 1.0 && c >= 0.0 && c <= 1.0);
    }
    return false;
}

void SurfacePoint::is_tensor_valid() const
{
    if (type == SurfacePointType::VertexPoint)
    {
        // there are no bary coords, so no need to check dimensions
        return;
    }
    if (type == SurfacePointType::EdgePoint)
    {
        // Assert tensor is scalar (0-dimensional)
        assert(bary_coords.dim() == 1 && bary_coords.size(0) == 1 && "Tensor must be scalar for EdgePoint");
    }
    else if (type == SurfacePointType::FacePoint)
    {
        // Assert tensor is a vector with 2 entries
        assert(bary_coords.dim() == 1 && bary_coords.size(0) == 2 && "Tensor must be a vector with 2 entries for FacePoint");
    }
    else
    {
        throw std::invalid_argument("Invalid SurfacePointType");
    }
}


at::Tensor LayoutOpt::SurfacePoint::get_pos(at::Tensor const& _pos, polymesh::Mesh const& _mesh) const
{
    // assertion
    is_tensor_valid();

    if (type == SurfacePointType::VertexPoint)
    {
        auto hh = _mesh.handle_of(heh_idx);
        auto vh_A = hh.vertex_from();

        auto A = get_row(_pos, vh_A.idx.value);

        return get_pos_intern(A);
    }
    else if (type == SurfacePointType::EdgePoint)
    {
        auto hh = _mesh.handle_of(heh_idx);
        auto vh_A = hh.vertex_from();
        auto vh_B = hh.vertex_to();

        auto A = get_row(_pos, vh_A.idx.value);
        auto B = get_row(_pos, vh_B.idx.value);

        return get_pos_intern(A, B);
    }
    else if (type == SurfacePointType::FacePoint)
    {
        auto hh = _mesh.handle_of(heh_idx);
        auto vh_A = hh.vertex_from();
        auto vh_B = hh.vertex_to();
        auto vh_C = hh.next().vertex_to();

        auto A = get_row(_pos, vh_A.idx.value);
        auto B = get_row(_pos, vh_B.idx.value);
        auto C = get_row(_pos, vh_C.idx.value);

        auto alpha = bary_coords[0];
        auto beta = bary_coords[1];

        return get_pos_intern(A, B, C);
    }

    assert(false && "should not be reached");
    return {};
}

at::Tensor SurfacePoint::get_pos(pm::vertex_attribute<at::Tensor> const& _pos) const
{
    // assertion
    is_tensor_valid();

    if (type == SurfacePointType::VertexPoint)
    {
        auto hh = _pos.mesh().handle_of(heh_idx);
        auto vh_A = hh.vertex_from();

        auto A = _pos[vh_A];

        return get_pos_intern(A);
    }
    else if (type == SurfacePointType::EdgePoint)
    {
        auto hh = _pos.mesh().handle_of(heh_idx);
        auto vh_A = hh.vertex_from();
        auto vh_B = hh.vertex_to();

        auto A = _pos[vh_A];
        auto B = _pos[vh_B];

        return get_pos_intern(A, B);
    }
    else if (type == SurfacePointType::FacePoint)
    {
        auto hh = _pos.mesh().handle_of(heh_idx);
        auto vh_A = hh.vertex_from();
        auto vh_B = hh.vertex_to();
        auto vh_C = hh.next().vertex_to();

        auto A = _pos[vh_A];
        auto B = _pos[vh_B];
        auto C = _pos[vh_C];

        return get_pos_intern(A, B, C);
    }

    assert(false && "should not be reached");
    return {};
}

at::Tensor SurfacePoint::get_pos(pm::halfedge_attribute<at::Tensor> const& _pos) const
{
    // assertion
    is_tensor_valid();

    if (type == SurfacePointType::VertexPoint)
    {
        auto hh = _pos.mesh().handle_of(heh_idx);
        auto A = _pos[hh];

        return get_pos_intern(A);
    }
    else if (type == SurfacePointType::EdgePoint)
    {
        auto hh = _pos.mesh().handle_of(heh_idx);
        auto A = _pos[hh];
        auto B = _pos[hh.opposite()];

        return get_pos_intern(A, B);
    }
    else if (type == SurfacePointType::FacePoint)
    {
        auto hh = _pos.mesh().handle_of(heh_idx);

        auto A = _pos[hh];
        auto B = _pos[hh.next()];
        auto C = _pos[hh.next().next()];

        return get_pos_intern(A, B, C);
    }

    assert(false && "should not be reached");
    return {};
}

vec3d SurfacePoint::bary_full() const
{
    switch (type)
    {
    case SurfacePointType::VertexPoint:
        return vec3d(1.0, 0.0, 0.0);
    case SurfacePointType::EdgePoint:
    {
        double const a = bary_coords[0].item<double>();
        return vec3d(a, 1.0 - a, 0.0);
    }
    case SurfacePointType::FacePoint:
    {
        double const a = bary_coords[0].item<double>();
        double const b = bary_coords[1].item<double>();
        return vec3d(a, b, 1.0 - a - b);
    }
    default:
        return vec3d(0.0, 0.0, 0.0);
    }
}

vec3d SurfacePoint::get_pos(Eigen::MatrixX3d const& _pos, polymesh::Mesh const& _mesh) const
{
    // assertion (tensor shape is still authoritative during Phases 3-5)
    is_tensor_valid();

    vec3d const b = bary_full();
    auto const hh = _mesh.handle_of(heh_idx);

    // The three face corners. For vertex/edge points the unused corners carry a
    // zero weight in b, so a single weighted sum reproduces every get_pos_intern overload.
    Eigen::RowVector3d const A = _pos.row(hh.vertex_from().idx.value);
    Eigen::RowVector3d const B = _pos.row(hh.vertex_to().idx.value);
    Eigen::RowVector3d const C = _pos.row(hh.next().vertex_to().idx.value);

    return (b[0] * A + b[1] * B + b[2] * C).transpose();
}

at::Tensor SurfacePoint::get_pos_intern(at::Tensor const& A) const { return A; }
at::Tensor SurfacePoint::get_pos_intern(at::Tensor const& A, at::Tensor const& B) const { return bary_coords * A + (1.0 - bary_coords) * B; }
at::Tensor SurfacePoint::get_pos_intern(at::Tensor const& A, at::Tensor const& B, at::Tensor const& C) const
{
    auto alpha = bary_coords[0];
    auto beta = bary_coords[1];

    return alpha * A + beta * B + (1.0 - alpha - beta) * C;
}
// special values
SurfacePoint const SurfacePoint::invalid = SurfacePoint(torch::Tensor(), pm::halfedge_index::invalid, SurfacePointType::Invalid);

} // namespace LayoutOpt
