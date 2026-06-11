#pragma once
#include <polymesh/Mesh.hh>
#include "LayoutOpt/PrincipalCurvature.hh"
#include "SurfacePoint.hh"
#include "Types.hh"

namespace LayoutOpt
{

// stores all information about the input mesh -- once initialized, this information is constant
class TargetMeshData
{
public:
    TargetMeshData() = delete;
    TargetMeshData(pm::vertex_attribute<pos3> const& _pos);

    pm::unique_ptr<pm::Mesh> mesh_;
    pm::vertex_attribute<pos3> pos_;

    // optional
    std::optional<pm::face_attribute<double>> scalar_field_;
    std::optional<pm::face_attribute<double>> point_wise_gauss_curvature_;
    std::optional<pm::face_attribute<double>> point_wise_mean_curvature_;

    //semi torch
    std::optional<pm::face_attribute<DirectionFieldData>> direction_field_data_;

    // torch
    torch::Tensor torch_pos_;                                       // torch format of pos_
    // remove-autodiff Phase 3: plain-double mirror of torch_pos_ (|V| x 3), set once in ctor.
    Eigen::MatrixX3d pos_mat_;
    std::optional<torch::Tensor> torch_scalar_field_;               // per face
    std::optional<torch::Tensor> torch_point_wise_gauss_curvature_; // per face
    std::optional<torch::Tensor> torch_point_wise_mean_curvature_;  // per face
};

// stores all information about the layout
class LayoutData
{
public:
    LayoutData() = delete;
    LayoutData(pm::vertex_attribute<pos3> const& _l_pos);

    pm::unique_ptr<pm::Mesh> mesh_;
    pm::vertex_attribute<pos3> pos_;

    std::optional<pm::vertex_attribute<pm::vertex_index>> map_to_overlay_vertices_;

    // used for harmonic param, in a differentiable way
    std::optional<torch::Tensor> torch_embedded_edge_length_;
    // remove-autodiff Phase 3: plain-double mirror of torch_embedded_edge_length_,
    // indexed by layout-edge idx (same layout as the torch tensor).
    std::optional<Eigen::VectorXd> embedded_edge_length_;
};

// stores information about the embedded layout, more precisely the boundary between the embedded patches
class PathNetworkData
{
public:
    PathNetworkData(pm::vertex_attribute<pos3> const& _l_pos);
    pm::unique_ptr<pm::Mesh> mesh_;
    pm::vertex_attribute<pm::vertex_index> map_to_layout_vertices_; // the correspondence is intrinsic, this is used to recognize unmapped
    pm::edge_attribute<pm::edge_index> map_to_layout_edges_;

    std::optional<pm::vertex_attribute<SurfacePoint>> sp_on_target_;
    std::optional<pm::vertex_attribute<pm::vertex_index>> map_to_overlay_vertices_;

    // used for harmonic param, t and uvs are assumed to be const per iter
    std::optional<pm::halfedge_attribute<double>> t_;
    std::optional<pm::halfedge_attribute<pos2>> uvs_;

    // used for harmonic param, t and uvs are computed in a differentiable way
    std::optional<torch::Tensor> torch_t_;
    std::optional<torch::Tensor> torch_uvs_;

    // remove-autodiff Phase 3: plain-double mirrors of torch_t_/torch_uvs_, flat
    // and indexed by path-network halfedge idx (same layout as the torch tensors).
    std::optional<Eigen::VectorXd> t_flat_;     // [#pn_heh]
    std::optional<Eigen::MatrixX2d> uvs_flat_;  // [#pn_heh x 2]

    void reset(polymesh::Mesh const& _l); // resets to topology of _l
};

// stores information about the modified mesh obtaines when the layout is embedded into the target mesh
class OverlayMeshData
{
public:
    OverlayMeshData() = delete;
    OverlayMeshData(pm::vertex_attribute<pos3> const& _pos);

    // Delete copy constructor and copy assignment operator
    OverlayMeshData(OverlayMeshData const&) = delete;
    OverlayMeshData& operator=(OverlayMeshData const&) = delete;

    pm::unique_ptr<pm::Mesh> mesh_;
    pm::vertex_attribute<pos3> pos_;

    // true for all edges that are part of the boundary between patches
    std::optional<pm::edge_attribute<bool>> patch_boundary_mask_;
    std::optional<pm::vertex_attribute<pm::vertex_index>> map_to_pn_vertices_;
    std::optional<pm::face_attribute<pm::face_index>> map_to_target_faces_;
    std::optional<pm::face_attribute<pm::face_index>> map_to_layout_faces_; // corresponding to patch labels

    // torch
    std::optional<torch::Tensor> torch_pos_;
    // remove-autodiff Phase 3: plain-double mirror of torch_pos_ (|V| x 3),
    // refreshed in sync_tg_and_torch alongside pos_.
    std::optional<Eigen::MatrixX3d> pos_mat_;

    //============================================================================
    // reset o_pos and o_mesh to _pos and _pos.mesh()
    void reset_mesh_and_pos(pm::vertex_attribute<pos3> const& _pos);
};

} // namespace LayoutOpt
