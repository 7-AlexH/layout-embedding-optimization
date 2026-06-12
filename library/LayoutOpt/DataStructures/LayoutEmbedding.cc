#include "LayoutEmbedding.hh"
#include <LayoutOpt/Utils.hh>
#include <LayoutOpt/Visualization/Viewing.hh>

#include <cassert>

namespace LayoutOpt
{

TargetMeshData::TargetMeshData(pm::vertex_attribute<pos3> const& _pos)
{
    mesh_ = pm::Mesh::create();
    mesh_->copy_from(_pos.mesh());

    pos_ = mesh_->vertices().make_attribute<pos3>();
    pos_.copy_from(_pos);

    pos_mat_.resize(mesh_->vertices().size(), 3);
    for (auto vh : mesh_->vertices())
    {
        auto const& p = pos_[vh];
        pos_mat_.row(vh.idx.value) = Eigen::RowVector3d(p.x, p.y, p.z);
    }
}

LayoutData::LayoutData(pm::vertex_attribute<pos3> const& _l_pos)
{
    mesh_ = pm::Mesh::create();
    mesh_->copy_from(_l_pos.mesh());

    pos_ = mesh_->vertices().make_attribute<pos3>();
    pos_.copy_from(_l_pos);
}

PathNetworkData::PathNetworkData(pm::vertex_attribute<pos3> const& _l_pos)
{
    mesh_ = pm::Mesh::create();
    mesh_->copy_from(_l_pos.mesh());

    // init inital identity mapping between layout and pn
    map_to_layout_vertices_ = mesh_->vertices().make_attribute<pm::vertex_index>(pm::vertex_index::invalid);
    map_to_layout_vertices_.compute([](VH vh) { return vh.idx; });

    map_to_layout_edges_ = mesh_->edges().make_attribute<pm::edge_index>(pm::edge_index::invalid);
    map_to_layout_edges_.compute([](EH eh) { return eh.idx; });
}

void PathNetworkData::reset(pm::Mesh const& _l)
{
    mesh_->copy_from(_l);

    map_to_layout_vertices_ = mesh_->vertices().make_attribute<pm::vertex_index>(pm::vertex_index::invalid);
    map_to_layout_vertices_.compute([](VH vh) { return vh.idx; });

    map_to_layout_edges_ = mesh_->edges().make_attribute<pm::edge_index>(pm::edge_index::invalid);
    map_to_layout_edges_.compute([](EH eh) { return eh.idx; });

    sp_on_target_.reset();
    map_to_overlay_vertices_.reset();
    t_.reset();
}

//============================================================================================================
//============================================================================================================
OverlayMeshData::OverlayMeshData(pm::vertex_attribute<pos3> const& _pos)
{
    mesh_ = pm::Mesh::create();
    mesh_->copy_from(_pos.mesh());

    pos_ = mesh_->vertices().make_attribute<pos3>();
    pos_.copy_from(_pos);
}

void OverlayMeshData::reset_mesh_and_pos(pm::vertex_attribute<pos3> const& _pos)
{
    mesh_->copy_from(_pos.mesh());
    pos_ = mesh_->vertices().make_attribute<pos3>();
    pos_.copy_from(_pos);

    patch_boundary_mask_.reset();
    map_to_pn_vertices_.reset();
    map_to_target_faces_.reset();
    map_to_layout_faces_.reset();
}


} // namespace LayoutOpt
