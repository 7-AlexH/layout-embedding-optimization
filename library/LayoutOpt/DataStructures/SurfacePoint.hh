#pragma once

#include <optional>

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
    SurfacePoint(vec2d _bary_params, pm::halfedge_index _heh_idx, SurfacePointType _type)
      : bary_params(_bary_params), heh_idx(_heh_idx), type(_type)
    {
    }

    SurfacePoint copy() const;

    // The authoritative barycentric parameters over the anchor halfedge's face
    // corners (hh.from, hh.to, hh.next.to):
    //   vertex -> unused; edge -> (alpha, 0); face -> (alpha, beta).
    vec2d bary_params = vec2d::Zero();

    pm::halfedge_index heh_idx = pm::halfedge_index::invalid;
    SurfacePointType type = SurfacePointType::Invalid;

    // Full barycentric weights (alpha, beta, gamma) over the three corners
    // (hh.from, hh.to, hh.next.to), with the implicit coordinate made explicit:
    //   vertex -> (1, 0, 0); edge -> (alpha, 1-alpha, 0); face -> (alpha, beta, 1-alpha-beta).
    vec3d bary_full() const;

    bool is_vertex_sp() const { return type == SurfacePointType::VertexPoint; }
    bool is_edge_sp() const { return type == SurfacePointType::EdgePoint; }
    bool is_face_sp() const { return type == SurfacePointType::FacePoint; }
    bool is_valid() const { return type != SurfacePointType::Invalid; }

    /// @param _pos position in matrix format, |V| x 3
    /// @param _mesh mesh connectivity the surfacepoint lives on
    /// @return resulting position in ambient space R³
    vec3d get_pos(Eigen::MatrixX3d const& _pos, pm::Mesh const& _mesh) const;

    // same, for 2D position matrices (|V| x 2), e.g. the unfolded-strip frame in do_step
    vec2d get_pos(Eigen::MatrixX2d const& _pos, pm::Mesh const& _mesh) const;

    // Halfedge-attribute overload (TriangleStrip::heh_pos_2d). NOTE the corner
    // convention differs from the |V|-indexed overloads: vertex -> pos[hh];
    // edge -> pos[hh], pos[hh.opposite()]; face -> pos[hh], pos[hh.next()],
    // pos[hh.next().next()].
    vec2d get_pos(pm::halfedge_attribute<std::optional<vec2d>> const& _pos) const;


    FH fh(polymesh::Mesh const& _mesh) const;

    /// @return true, if the surface point is inside the element (either edge or face)
    /// @note for VertexPoints, this is always true
    bool is_inside_element();

    // special values
    static SurfacePoint const invalid;
};


} // namespace LayoutOpt
