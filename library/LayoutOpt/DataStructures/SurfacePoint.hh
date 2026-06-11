#pragma once

#include <torch/serialize/input-archive.h>
#include <polymesh/Mesh.hh>
#include <polymesh/attributes.hh>
#include "LayoutOpt/DataStructures/Types.hh"

namespace LayoutOpt
{

enum class SurfacePointType
{
    VertexPoint,
    EdgePoint,
    FacePoint,
    Invalid
};

class SurfacePoint
{
public:
    SurfacePoint() = default;
    SurfacePoint(torch::Tensor _bary_coords, pm::halfedge_index _heh_idx, SurfacePointType _type)
      : bary_coords(std::move(_bary_coords)), heh_idx(_heh_idx), type(_type)
    {
    }

    // copies the Surface Point such that torch::Tensor is not part of computation graph
    SurfacePoint copy() const;

    // for vertex: hh.from
    // for edge: alpha * hh.from + hh.beta * to
    // for face: alpha * hh.from + beta * hh.to + gamma * hh.next.to
    torch::Tensor bary_coords = torch::Tensor();

    pm::halfedge_index heh_idx = pm::halfedge_index::invalid;
    SurfacePointType type = SurfacePointType::Invalid;

    // --- remove-autodiff Phase 3: plain-double mirror of bary_coords ---
    // Full barycentric weights (alpha, beta, gamma) over the three corners
    // (hh.from, hh.to, hh.next.to), with the implicit coordinate made explicit:
    //   vertex -> (1, 0, 0); edge -> (alpha, 1-alpha, 0); face -> (alpha, beta, 1-alpha-beta).
    // While the torch gradient oracle exists (Phases 3-5) the tensor stays
    // authoritative and this is derived from it on demand; the flip to a stored
    // field happens with the torch removal (Phase 5/6, deferred from S1).
    vec3d bary_full() const;

    bool is_vertex_sp() const { return type == SurfacePointType::VertexPoint; }
    bool is_edge_sp() const { return type == SurfacePointType::EdgePoint; }
    bool is_face_sp() const { return type == SurfacePointType::FacePoint; }
    bool is_valid() const { return type != SurfacePointType::Invalid; }

    /// @param _pos position in matrix format, either |V| x 3 or |V| x 2
    /// @param _mesh mesh connectivity the surfacepoint lives on
    /// @return resulting position in ambiant space R³ or R²
    torch::Tensor get_pos(torch::Tensor const& _pos, pm::Mesh const& _mesh) const;
    torch::Tensor get_pos(pm::vertex_attribute<torch::Tensor> const& _pos) const;
    torch::Tensor get_pos(pm::halfedge_attribute<torch::Tensor> const& _pos) const;

    // remove-autodiff Phase 3: plain-double counterpart of get_pos(torch::Tensor [|V|x3], mesh).
    // Mirrors the torch path exactly via bary_full(); used by the ported (torch-free) stages.
    vec3d get_pos(Eigen::MatrixX3d const& _pos, pm::Mesh const& _mesh) const;


    FH fh(polymesh::Mesh const& _mesh) const;

    /// @return true, if the surface point is inside the element (either edge or face)
    /// @note for VertexPoints, this is always true
    bool is_inside_element();

    // special values
    static SurfacePoint const invalid;

private:
    void is_tensor_valid() const;
    torch::Tensor get_pos_intern(torch::Tensor const& A) const;
    torch::Tensor get_pos_intern(torch::Tensor const& A, torch::Tensor const& B) const;
    torch::Tensor get_pos_intern(torch::Tensor const& A, torch::Tensor const& B, torch::Tensor const& C) const;
};


} // namespace LayoutOpt
