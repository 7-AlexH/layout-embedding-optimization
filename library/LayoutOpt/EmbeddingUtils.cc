#include "EmbeddingUtils.hh"
#include <geometrycentral/surface/exact_geodesics.h>
#include "LayoutOpt/Adjoint/IntersectionStage.hh"
#include "LayoutOpt/Adjoint/LeafStage.hh"
#include "LayoutOpt/DataStructures/GCMesh.hh"
#include "LayoutOpt/GeomUtils.hh"
#include "LayoutOpt/Resample.hh"
#include "LayoutOpt/Utils.hh"
#include "LayoutOpt/Visualization/ColorGenerator.hh"
#include "LayoutOpt/Visualization/Colors.hh"
#include "LayoutOpt/Visualization/Viewing.hh"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "polymesh/algorithms/triangulate.hh"
#include "polymesh/formats.hh"
#include "polymesh/formats/obj.hh"
#include "polymesh/properties.hh"

#include <filesystem>
#include <glow-extras/viewer/view.hh>
#include "glow-extras/viewer/canvas.hh"

#include <cassert>

namespace LayoutOpt
{
// annonymous ns for aux funcitons
namespace
{
bool contains_patch_corners(pm::face_handle _layout_patch, std::vector<pm::face_handle> const& _faces, LayoutData const& _lmd)
{
    bool all_corners_found = true;
    for (auto corner : _layout_patch.vertices())
    {
        bool found = false;
        for (auto fh : _faces)
        {
            for (auto vh : fh.vertices())
            {
                if (_lmd.map_to_overlay_vertices_.value()[corner] == vh.idx)
                {
                    found = true;
                    break;
                }
            }
            if (found)
                break;
        }
        if (!found)
        {
            all_corners_found = false;
            break;
        }
    }
    return all_corners_found;
}

std::vector<std::vector<pm::face_handle>> get_faces_per_patch(OverlayMeshData& _omd)
{
    auto cd = gv::canvas_data();
    ColorGenerator cg;

    std::vector<std::vector<pm::face_handle>> res;
    auto visited_faces = _omd.mesh_->faces().make_attribute<bool>(false);
    for (auto start_fh : _omd.mesh_->faces())
    {
        if (visited_faces[start_fh])
            continue;

        auto color = cg.generate_next_color();

        auto& patch = res.emplace_back();
        // start a dfs
        std::vector<pm::face_handle> q;
        q.push_back(start_fh);
        visited_faces[start_fh] = true;
        patch.push_back(start_fh);
        while (!q.empty())
        {
            auto fh = q.back();
            q.pop_back();

            cd.add_face(fh, _omd.pos_, color);

            for (auto heh : fh.halfedges())
            {
                auto eh = heh.edge();
                auto opp_fh = heh.opposite_face();
                if (!_omd.patch_boundary_mask_.value()[eh] && !visited_faces[opp_fh])
                {
                    q.push_back(opp_fh);
                    visited_faces[opp_fh] = true;
                    patch.push_back(opp_fh);
                }
            }
        }
        // {
        //     auto c = gv::canvas();
        //     c.add_data(cd);
        // }
    }
    // all faces should be visited
    for (auto fh : _omd.mesh_->faces())
    {
        POLYMESH_ASSERT(visited_faces[fh]);
    }
    // all patches should have at least one face
    for (auto const& patch : res)
    {
        POLYMESH_ASSERT(patch.size() > 0);
    }

    return res;
}

void map_faces_to_patches(std::vector<std::vector<pm::face_handle>> const& _faces_per_patch, OverlayMeshData& _omd, LayoutData const& _ld)
{
    _omd.map_to_layout_faces_.emplace(_omd.mesh_->faces().make_attribute<pm::face_index>(pm::face_index::invalid));

    POLYMESH_ASSERT(_faces_per_patch.size() == _ld.mesh_->faces().size());
    // DEBUG_VAR(_ld.mesh_->faces().size())
    // DEBUG_VAR(_faces_per_patch.size())

    for (auto patch : _ld.mesh_->faces())
    {
        bool patch_mapped = false;
        for (auto const& _face_vector : _faces_per_patch)
        {
            if (contains_patch_corners(patch, _face_vector, _ld))
            {
                for (auto fh : _face_vector)
                {
                    _omd.map_to_layout_faces_.value()[fh] = patch.idx;
                }
                patch_mapped = true;
                break;
            }
        }
        POLYMESH_ASSERT(patch_mapped);
    }
}

} // empty namespace


void compute_path_network(TargetMeshData const& _tmd, PathNetworkData& _pnd)
{
    assert(_pnd.sp_on_target_.has_value());

    // init mesh for path shortening
    // here the face-surfacepoints must be contained in the mesh, for dijkstra init
    // compute flip_out_mesh
    pm::Mesh flip_out_mesh;
    flip_out_mesh.copy_from(*_tmd.mesh_.get());
    pm::vertex_attribute<pos3> flip_out_pos(flip_out_mesh);
    flip_out_pos.copy_from(_tmd.pos_);

    pm::vertex_attribute<VH> map_to_flip_out_mesh(*_pnd.mesh_.get());
    pm::edge_attribute<bool> blocked = _tmd.mesh_->edges().make_attribute<bool>(false);

    for (auto vh : _pnd.mesh_->vertices())
    {
        auto sp = _pnd.sp_on_target_.value()[vh];
        auto target_handle = sp.fh(*_tmd.mesh_.get());

        for (auto eh : target_handle.edges())
        {
            blocked[eh] = true;
        }

        auto flip_out_handle = sp.fh(flip_out_mesh);
        auto tm_handle = sp.fh(*_tmd.mesh_.get());

        assert(flip_out_handle.is_valid() && "currently only one sp per face is allowed");

        auto vh_new = flip_out_mesh.faces().split(flip_out_handle);
        flip_out_pos[vh_new] = eigen_to_pos3(sp.get_pos(_tmd.pos_mat_, *_tmd.mesh_.get()));
        map_to_flip_out_mesh[vh] = vh_new;
    }
    flip_out_mesh.compactify();

    // init gc mesh and flipnetwork
    GCMesh gcmesh = to_gc_mesh(flip_out_pos);
    std::unique_ptr<FlipEdgeNetwork> flipNetwork(new FlipEdgeNetwork(*gcmesh.mesh, *gcmesh.positionGeometry, {}));
    // flipNetwork->EPS_ANGLE = 1e-9;
    flipNetwork->supportRewinding = true;
    flipNetwork->posGeom = gcmesh.positionGeometry.get();

    auto edges = _pnd.mesh_->edges().to_vector();
    for (EH eh : edges)
    {
        auto layout_idx = _pnd.map_to_layout_edges_[eh];
        auto heh = eh.halfedgeA();
        auto A = map_to_flip_out_mesh[heh.vertex_from()];
        auto B = map_to_flip_out_mesh[heh.vertex_to()];

        Vertex startVert = gcmesh.mesh->vertex(A.idx.value);
        Vertex endVert = gcmesh.mesh->vertex(B.idx.value);

        // Get an initial dijkstra path
        std::vector<Halfedge> dijkstraPath = shortestEdgePath(*gcmesh.positionGeometry, startVert, endVert);

        flipNetwork->reinitializePath({dijkstraPath});
        flipNetwork->iterativeShorten();

        auto path_intrinsic = flipNetwork->getPathPolyline().front();

        for (size_t i = 1; i < path_intrinsic.size() - 1; ++i)
        {
            auto intrinsic_point = path_intrinsic[i];

            if (intrinsic_point.type == geometrycentral::surface::SurfacePointType::Edge)
            {
                auto idxA = intrinsic_point.edge.firstVertex().getIndex();
                auto idxB = intrinsic_point.edge.secondVertex().getIndex();

                if (idxA >= _tmd.mesh_->vertices().size() || idxB >= _tmd.mesh_->vertices().size())
                {
                    DEBUG_OUT("skipped")
                    continue;
                }

                auto vhA = _tmd.mesh_->vertices()[idxA];
                auto vhB = _tmd.mesh_->vertices()[idxB];

                auto heh_emb = pm::halfedge_from_to(vhB, vhA);

                auto alpha = intrinsic_point.tEdge;

                // DEBUG_VAR(alpha)

                SurfacePoint sp = {vec2d(alpha, 0.0), heh_emb, SurfacePointType::EdgePoint};

                // DEBUG_VAR(heh_emb.face())
                // DEBUG_VAR(heh_emb.opposite_face())

                auto vh_new = _pnd.mesh_->halfedges().split(heh);
                _pnd.sp_on_target_.value()[vh_new] = sp;

                heh = heh.next();
                _pnd.map_to_layout_edges_[heh.edge()] = layout_idx;
            }
            else if (intrinsic_point.type == geometrycentral::surface::SurfacePointType::Vertex)
            {
                auto ip_vh = _tmd.mesh_->vertices()[intrinsic_point.vertex.getIndex()];

                assert(i - 1 >= 0 && i + 1 < path_intrinsic.size() && "for this construction, there must be a SP before and after this point.");

                auto ip_before = path_intrinsic[i - 1];
                VH ip_before_vhA = VH::invalid;
                VH ip_before_vhB = VH::invalid;

                assert(ip_before.type == geometrycentral::surface::SurfacePointType::Edge || ip_before.type == geometrycentral::surface::SurfacePointType::Vertex);

                if (ip_before.type == geometrycentral::surface::SurfacePointType::Edge)
                {
                    ip_before_vhA = _tmd.mesh_->vertices()[ip_before.edge.firstVertex().getIndex()];
                    ip_before_vhB = _tmd.mesh_->vertices()[ip_before.edge.secondVertex().getIndex()];
                }
                else
                {
                    if (ip_before.vertex.getIndex() < _tmd.mesh_->vertices().size())
                        ip_before_vhA = _tmd.mesh_->vertices()[ip_before.vertex.getIndex()];
                }

                auto ip_after = path_intrinsic[i + 1];

                assert(ip_after.type == geometrycentral::surface::SurfacePointType::Edge || ip_after.type == geometrycentral::surface::SurfacePointType::Vertex);

                VH ip_after_vhA = VH::invalid;
                VH ip_after_vhB = VH::invalid;

                if (ip_after.type == geometrycentral::surface::SurfacePointType::Edge)
                {
                    ip_after_vhA = _tmd.mesh_->vertices()[ip_after.edge.firstVertex().getIndex()];
                    ip_after_vhB = _tmd.mesh_->vertices()[ip_after.edge.secondVertex().getIndex()];
                }
                else
                {
                    if (ip_after.vertex.getIndex() < _tmd.mesh_->vertices().size())
                        ip_after_vhA = _tmd.mesh_->vertices()[ip_after.vertex.getIndex()];
                }

                // find first heh:
                auto iter_heh = HEH::invalid;
                for (auto o_heh : ip_vh.outgoing_halfedges())
                {
                    if (o_heh.vertex_to() == ip_before_vhA || o_heh.vertex_to() == ip_before_vhB)
                    {
                        if (o_heh.next().vertex_to() != ip_before_vhA && o_heh.next().vertex_to() != ip_before_vhB)
                        {
                            iter_heh = o_heh;
                            break;
                        }
                    }
                }
                if (iter_heh.is_invalid())
                {
                    for (auto o_heh : ip_vh.outgoing_halfedges())
                    {
                        if (blocked[o_heh.edge()] && !blocked[o_heh.next().edge()])
                        {
                            iter_heh = o_heh;
                            break;
                        }
                    }
                }
                assert(iter_heh.is_valid());
                // c.add_line(iter_heh, _tmd.pos_, RED).size(10);
                // c.add_line(pm::edge_between(ip_before_vhA, ip_before_vhB), _tmd.pos_, GREEN).size(8);
                // c.add_line(pm::edge_between(ip_after_vhA, ip_after_vhB), _tmd.pos_, MAY_GREEN).size(8);

                std::vector<HEH> detour_hehs;

                if (ip_before.type != geometrycentral::surface::SurfacePointType::Vertex || ip_before_vhA.is_invalid())
                {
                    detour_hehs.push_back(iter_heh);
                }

                iter_heh = iter_heh.next().next().opposite();
                while (iter_heh.vertex_to() != ip_after_vhA && iter_heh.vertex_to() != ip_after_vhB && !blocked[iter_heh.edge()])
                {
                    detour_hehs.push_back(iter_heh);
                    iter_heh = iter_heh.next().next().opposite();
                    // c.add_line(iter_heh, _tmd.pos_, PETROL).size(7);
                }
                if (ip_after.type != geometrycentral::surface::SurfacePointType::Vertex || ip_after_vhA.is_invalid())
                {
                    detour_hehs.push_back(iter_heh);
                }

                for (auto heh_emb : detour_hehs)
                {
                    SurfacePoint sp = {vec2d(0.95, 0.0), heh_emb, SurfacePointType::EdgePoint};

                    auto vh_new = _pnd.mesh_->halfedges().split(heh);
                    _pnd.sp_on_target_.value()[vh_new] = sp;

                    heh = heh.next();
                    _pnd.map_to_layout_edges_[heh.edge()] = layout_idx;
                }
            }
        }
    }
    _pnd.mesh_->compactify();
}

void compute_path_network_robust(TargetMeshData const& _tmd, PathNetworkData& _pnd)
{
    assert(_pnd.sp_on_target_.has_value());

    // init mesh for path shortening
    // here the face-surfacepoints must be contained in the mesh, for dijkstra init
    // compute flip_out_mesh
    pm::Mesh flip_out_mesh;
    flip_out_mesh.copy_from(*_tmd.mesh_.get());
    pm::vertex_attribute<pos3> flip_out_pos(flip_out_mesh);
    flip_out_pos.copy_from(_tmd.pos_);

    // insert suface points
    pm::vertex_attribute<VH> map_to_flip_out_mesh(*_pnd.mesh_.get());
    for (auto vh : _pnd.mesh_->vertices())
    {
        auto sp = _pnd.sp_on_target_.value()[vh];
        auto target_handle = sp.fh(*_tmd.mesh_.get());

        FH flip_out_handle = sp.fh(flip_out_mesh);

        assert(flip_out_handle.is_valid() && !flip_out_handle.is_removed() && "currently only one sp per face is allowed");

        auto vh_new = flip_out_mesh.faces().split(flip_out_handle);
        flip_out_pos[vh_new] = eigen_to_pos3(sp.get_pos(_tmd.pos_mat_, *_tmd.mesh_.get()));
        map_to_flip_out_mesh[vh] = vh_new;
    }
    flip_out_mesh.compactify();

    // init gc mesh and flipnetwork
    GCMesh gcmesh = to_gc_mesh(flip_out_pos);
    std::unique_ptr<FlipEdgeNetwork> flipNetwork(new FlipEdgeNetwork(*gcmesh.mesh, *gcmesh.positionGeometry, {}));
    flipNetwork->supportRewinding = true;
    flipNetwork->posGeom = gcmesh.positionGeometry.get();

    // insert edges
    auto edges = _pnd.mesh_->edges().to_vector();
    for (EH eh : edges)
    {
        auto layout_edge_idx = _pnd.map_to_layout_edges_[eh];
        auto heh = eh.halfedgeA();
        auto A = map_to_flip_out_mesh[heh.vertex_from()];
        auto B = map_to_flip_out_mesh[heh.vertex_to()];

        Vertex startVert = gcmesh.mesh->vertex(A.idx.value);
        Vertex endVert = gcmesh.mesh->vertex(B.idx.value);

        // Get an initial dijkstra path
        std::vector<Halfedge> dijkstraPath = shortestEdgePath(*gcmesh.positionGeometry, startVert, endVert);

        flipNetwork->reinitializePath({dijkstraPath});
        flipNetwork->iterativeShorten();

        auto path_intrinsic = flipNetwork->getPathPolyline().front();

        for (size_t i = 1; i < path_intrinsic.size() - 1; ++i)
        {
            auto intrinsic_point = path_intrinsic[i];

            // add new point to pn
            switch (intrinsic_point.type)
            {
            case geometrycentral::surface::SurfacePointType::Edge:
            {
                auto idxA = intrinsic_point.edge.firstVertex().getIndex();
                auto idxB = intrinsic_point.edge.secondVertex().getIndex();

                assert(idxA < _tmd.mesh_->vertices().size() || idxB < _tmd.mesh_->vertices().size() && "idx out of range");

                auto vhA = _tmd.mesh_->vertices()[idxA];
                auto vhB = _tmd.mesh_->vertices()[idxB];

                auto heh_emb = pm::halfedge_from_to(vhB, vhA);

                auto alpha = intrinsic_point.tEdge;

                SurfacePoint sp = {vec2d(alpha, 0.0), heh_emb, SurfacePointType::EdgePoint};

                auto vh_new = _pnd.mesh_->halfedges().split(heh);
                _pnd.sp_on_target_.value()[vh_new] = sp;

                heh = heh.next();
                _pnd.map_to_layout_edges_[heh.edge()] = layout_edge_idx;

                break;
            }
            case geometrycentral::surface::SurfacePointType::Vertex:
            {
                auto idx = intrinsic_point.vertex.getIndex();
                assert(idx < _tmd.mesh_->vertices().size() && "idx out of range");

                auto vhA = _tmd.mesh_->vertices()[idx];
                auto heh_emb = vhA.any_outgoing_halfedge();

                SurfacePoint sp = {vec2d::Zero(), heh_emb, SurfacePointType::VertexPoint};
                auto vh_new = _pnd.mesh_->halfedges().split(heh);
                _pnd.sp_on_target_.value()[vh_new] = sp;

                heh = heh.next();
                _pnd.map_to_layout_edges_[heh.edge()] = layout_edge_idx;
                break;
            }
            case geometrycentral::surface::SurfacePointType::Face:
                assert(false && "the point can never be a face point");
                break;
            }
        }
    }
    _pnd.mesh_->compactify();
}


void compute_path_network_detour(TargetMeshData const& _tmd, PathNetworkData& _pnd)
{
    assert(_pnd.sp_on_target_.has_value());
    std::vector<VH> vertex_points;
    vertex_points.reserve(_pnd.mesh_->vertices().size());

    for (auto vh : _pnd.mesh_->vertices())
    {
        if (_pnd.sp_on_target_.value()[vh].type == SurfacePointType::VertexPoint)
        {
            vertex_points.push_back(vh);
        }
    }

    if (vertex_points.empty())
        return;


    auto cd = gv::canvas_data();

    for (auto pn_vh : vertex_points)
    {
        auto sp_vh = _pnd.sp_on_target_.value()[pn_vh];
        auto t_vh = _tmd.mesh_->handle_of(sp_vh.heh_idx).vertex_from();

        vec3d const pos_init = _pnd.sp_on_target_.value()[pn_vh].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
        cd.add_point(eigen_to_pos3(pos_init), GREEN).size(12);

        assert(pn_vh.adjacent_vertices().size() == 2 && "expected this to be an inner vertex");

        auto pn_incoming_heh = pn_vh.any_incoming_halfedge();
        auto layout_edge_idx = _pnd.map_to_layout_edges_[pn_incoming_heh.edge()];

        // find previous and next sp
        auto pn_vh_l = pn_incoming_heh.vertex_from();
        auto sp_vh_l = _pnd.sp_on_target_.value()[pn_vh_l];

        auto pn_vh_r = pn_incoming_heh.next().vertex_to();
        auto sp_vh_r = _pnd.sp_on_target_.value()[pn_vh_r];

        // collapse pn_vh
        _pnd.mesh_->halfedges().collapse(pn_incoming_heh.next());

        HEH iter_heh = HEH::invalid;
        if (sp_vh_l.type == SurfacePointType::EdgePoint)
        {
            auto sp_l_eh = _tmd.mesh_->handle_of(sp_vh_l.heh_idx).edge();
            for (auto heh : t_vh.outgoing_halfedges())
            {
                if (heh.opposite().prev().edge() == sp_l_eh)
                {
                    iter_heh = heh;
                    break;
                }
            }
        }
        else if (sp_vh_l.type == SurfacePointType::VertexPoint)
        {
            auto sp_l_vh = _tmd.mesh_->handle_of(sp_vh_l.heh_idx).vertex_from();
            for (auto heh : t_vh.outgoing_halfedges())
            {
                if (heh.opposite().prev().vertex_from() == sp_l_vh)
                {
                    iter_heh = heh;
                    break;
                }
            }
        }
        else if (sp_vh_l.type == SurfacePointType::FacePoint)
        {
            auto sp_l_fh = sp_vh_l.fh(*_tmd.mesh_.get());
            for (auto heh : t_vh.outgoing_halfedges())
            {
                if (heh.opposite_face() == sp_l_fh)
                {
                    iter_heh = heh;
                    break;
                }
            }
        }

        HEH end_heh = HEH::invalid;
        if (sp_vh_r.type == SurfacePointType::EdgePoint)
        {
            auto sp_r_eh = _tmd.mesh_->handle_of(sp_vh_r.heh_idx).edge();
            for (auto heh : t_vh.outgoing_halfedges())
            {
                if (heh.next().edge() == sp_r_eh)
                {
                    end_heh = heh;
                    break;
                }
            }
        }
        else if (sp_vh_r.type == SurfacePointType::VertexPoint)
        {
            auto sp_r_vh = _tmd.mesh_->handle_of(sp_vh_r.heh_idx).vertex_from();
            for (auto heh : t_vh.outgoing_halfedges())
            {
                if (heh.vertex_to() == sp_r_vh)
                {
                    end_heh = heh;
                    break;
                }
            }
        }
        else if (sp_vh_r.type == SurfacePointType::FacePoint)
        {
            auto sp_r_fh = sp_vh_r.fh(*_tmd.mesh_.get());
            for (auto heh : t_vh.outgoing_halfedges())
            {
                if (heh.face() == sp_r_fh)
                {
                    end_heh = heh;
                    break;
                }
            }
        }

        if (iter_heh.is_invalid() || end_heh.is_invalid())
        {
            auto v = gv::view();
            auto c = gv::canvas();

            gv::view(_tmd.pos_, BLUE_50);
            gv::view(gv::lines(_tmd.pos_), BLUE_75);
            for (auto pn_eh : _pnd.mesh_->edges())
            {
                vec3d const posA = _pnd.sp_on_target_.value()[pn_eh.vertexA()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
                vec3d const posB = _pnd.sp_on_target_.value()[pn_eh.vertexB()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());

                cd.add_line(eigen_to_pos3(posA), eigen_to_pos3(posB), MAGENTA).size(8);
                cd.add_point(eigen_to_pos3(posA), RED).size(15);
                cd.add_point(eigen_to_pos3(posB), RED).size(15);
            }
            c.add_data(cd);
        }

        assert(iter_heh.is_valid());
        assert(end_heh.is_valid());
        iter_heh = iter_heh.opposite().next();

        assert(iter_heh.vertex_from() == end_heh.vertex_from());
        while (iter_heh != end_heh)
        {
            SurfacePoint sp = {vec2d(0.98, 0.0), iter_heh, SurfacePointType::EdgePoint};
            auto pn_vh_new = _pnd.mesh_->halfedges().split(pn_incoming_heh);
            _pnd.sp_on_target_.value()[pn_vh_new] = sp;
            pn_incoming_heh = pn_incoming_heh.next();
            _pnd.map_to_layout_edges_[pn_incoming_heh.edge()] = layout_edge_idx;

            iter_heh = iter_heh.opposite().next();
        }
    }

    _pnd.mesh_->compactify();
}


void compute_path_boundary_mask(PathNetworkData const& _pnd, OverlayMeshData& _omd)
{
    _omd.patch_boundary_mask_.emplace(_omd.mesh_->edges().make_attribute<bool>(false));
    for (auto eh : _omd.mesh_->edges())
    {
        auto pn_A = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[eh.vertexA()]);
        auto pn_B = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[eh.vertexB()]);
        if (pn_A.is_valid() && pn_B.is_valid())
        {
            auto pn_eh = pm::edge_between(pn_A, pn_B);
            if (pn_eh.is_valid())
            {
                _omd.patch_boundary_mask_.value()[eh] = true;
            }
        }
    }
}

void compute_mapping_layout_to_overlay(LayoutData& _ld, PathNetworkData const& _pnd)
{
    assert(_pnd.map_to_overlay_vertices_.has_value());
    // update the layout graph mapping to overlay mesh
    _ld.map_to_overlay_vertices_.emplace(_ld.mesh_->vertices().make_attribute<pm::vertex_index>(pm::vertex_index::invalid));
    for (auto vh : _pnd.mesh_->vertices())
    {
        if (_pnd.map_to_layout_vertices_[vh].is_valid())
        {
            _ld.map_to_overlay_vertices_.value()[_pnd.map_to_layout_vertices_[vh]] = _pnd.map_to_overlay_vertices_.value()[vh];
        }
    }
}

void compute_mapping_overlay_to_layout(OverlayMeshData& _omd, LayoutData const& _ld)
{
    assert(_omd.map_to_target_faces_.has_value());

    for (auto fh : _omd.mesh_->faces())
    {
        POLYMESH_ASSERT(_omd.map_to_target_faces_.value()[fh].is_valid());
    }

    // compute patch_label_ by assigning each face to a patch of the layout graph
    auto faces_per_patch = get_faces_per_patch(_omd);
    map_faces_to_patches(faces_per_patch, _omd, _ld);
}

void collapse_pn_to_layout(PathNetworkData& _pnd)
{
    bool collapsed = true;
    while (collapsed)
    {
        collapsed = false;
        for (auto pn_heh : _pnd.mesh_->halfedges())
        {
            if (pn_heh.is_invalid())
                continue;

            if (_pnd.map_to_layout_vertices_[pn_heh.vertex_from()].is_invalid())
            {
                _pnd.mesh_->halfedges().collapse(pn_heh);
                collapsed = true;
            }
        }
    }
    _pnd.mesh_->compactify();
}


void reset_embedding_data(LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    // reset layout
    _ld.map_to_overlay_vertices_.reset();
    _ld.embedded_edge_length_.reset();

    // reset path network
    _pnd.map_to_overlay_vertices_.reset();
    _pnd.t_.reset();
    _pnd.uvs_.reset();
    _pnd.t_flat_.reset();
    _pnd.uvs_flat_.reset();

    // reset overlay mesh
    _omd.patch_boundary_mask_.reset();
    _omd.map_to_pn_vertices_.reset();
    _omd.map_to_target_faces_.reset();
    _omd.map_to_layout_faces_.reset();
}


void compute_overlay_positions(std::vector<TriangleStrip> const& _strips, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData& _omd)
{
    assert(_ld.map_to_overlay_vertices_.has_value());
    assert(_pnd.map_to_overlay_vertices_.has_value());

    // start from what insert_path_network left in pos_: the target rows are
    // final; the path-network rows are overwritten below
    Eigen::MatrixX3d pos((Eigen::Index)_omd.mesh_->vertices().size(), 3);
    for (auto o_vh : _omd.mesh_->vertices())
    {
        auto const& p = _omd.pos_[o_vh];
        pos.row(o_vh.idx.value) = Eigen::RowVector3d(p.x, p.y, p.z);
    }

    // S1: layout-node rows (bary interpolation on the target faces). The 2D
    // strip-endpoint outputs feed S3 below.
    LeafCtx leaf;
    leaf_collect(_strips, _tmd, _ld, _pnd, leaf);
    Eigen::MatrixX2d const bary = collect_bary(_pnd);

    int const n_l_edges = (int)_ld.mesh_->edges().size();
    Eigen::MatrixX2d seg_a = Eigen::MatrixX2d::Zero(n_l_edges, 2);
    Eigen::MatrixX2d seg_b = Eigen::MatrixX2d::Zero(n_l_edges, 2);
    leaf_forward(leaf, bary, _tmd.pos_mat_, pos, seg_a, seg_b);

    // S3: interior arc rows (3D point implied by the 2D strip intersection)
    IntersectCtx ictx;
    intersections_forward(_strips, seg_a, seg_b, _tmd, _ld, _pnd, ictx, pos);

    for (auto o_vh : _omd.mesh_->vertices())
    {
        Eigen::RowVector3d const r = pos.row(o_vh.idx.value);

        assert(!tg::is_nan(r.x()));
        assert(!tg::is_nan(r.y()));
        assert(!tg::is_nan(r.z()));

        _omd.pos_[o_vh] = pos3(r.x(), r.y(), r.z());
    }
    _omd.pos_mat_.emplace(std::move(pos));
}

pm::edge_attribute<int> compute_layout_edge_arc_idx(LayoutData const& _ld)
{
    pm::edge_attribute<int> arc_idx = _ld.mesh_->edges().make_attribute(-1); // -1 corresponds to not assigned yet

    int current_idx = 0;

    for (auto l_hh : _ld.mesh_->halfedges())
    {
        auto l_eh = l_hh.edge();
        auto l_vh = l_hh.vertex_from();

        if (l_vh.adjacent_vertices().size() <= 2)
            continue; // inner vertex
        if (arc_idx[l_eh] != -1)
            continue; // already labeled


        auto iter_hh = l_hh;
        while (iter_hh.vertex_to().adjacent_vertices().size() == 2)
        {
            arc_idx[iter_hh.edge()] = current_idx;
            iter_hh = iter_hh.next();
        }
        arc_idx[iter_hh.edge()] = current_idx;
        current_idx = current_idx + 1;
    }
    return arc_idx;
}

pm::edge_attribute<double> compute_embedded_length_per_layout_edge(TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd)
{
    assert(_pnd.sp_on_target_.has_value());
    pm::edge_attribute<double> embedded_length = _ld.mesh_->edges().make_attribute(0.0);

    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto l_eh = _pnd.map_to_layout_edges_[pn_eh];

        vec3d const posA = _pnd.sp_on_target_.value()[pn_eh.vertexA()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
        vec3d const posB = _pnd.sp_on_target_.value()[pn_eh.vertexB()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());

        double length = (posB - posA).norm();
        embedded_length[l_eh] += length;
    }

    return embedded_length;
}


pm::edge_attribute<std::vector<FH>> compute_target_face_strips_per_layout_edge(TargetMeshData const& _tmd,
                                                                               LayoutData const& _ld,
                                                                               PathNetworkData const& _pnd,
                                                                               pm::edge_attribute<std::vector<VH>> const& _map_l_eh_to_pn_vh)
{
    pm::edge_attribute<std::vector<FH>> l_target_face_strips(*_ld.mesh_.get());

    for (auto l_eh : _ld.mesh_->edges())
    {
        FH last_fh = FH::invalid;
        for (size_t i = 0; i < _map_l_eh_to_pn_vh[l_eh].size() - 1; ++i)
        {
            auto pn_vh = _map_l_eh_to_pn_vh[l_eh][i];
            FH this_face = _pnd.sp_on_target_.value()[pn_vh].fh(*_tmd.mesh_.get());
            HEH this_heh = _tmd.mesh_->handle_of(_pnd.sp_on_target_.value()[pn_vh].heh_idx);

            if (last_fh.is_invalid() || last_fh != this_face)
            {
                l_target_face_strips[l_eh].push_back(this_face);
            }
            else
            {
                l_target_face_strips[l_eh].push_back(this_heh.opposite_face());
            }
            last_fh = l_target_face_strips[l_eh].back();
        }
    }
    return l_target_face_strips;
}

pm::edge_attribute<std::vector<VH>> compute_map_l_eh_to_pn_vh(LayoutData const& _ld, PathNetworkData const& _pnd)
{
    pm::edge_attribute<std::vector<VH>> l_pn_vertices(*_ld.mesh_.get());

    for (auto l_eh : _ld.mesh_->edges())
    {
        // find first heh
        HEH heh_iter = HEH::invalid;
        for (auto pn_heh : _pnd.mesh_->halfedges())
        {
            // from vertex is not inner vertex
            if (_pnd.map_to_layout_vertices_[pn_heh.vertex_from()].is_valid())
            {
                // halfedge_maps to right layout_edge
                if (_pnd.map_to_layout_edges_[pn_heh.edge()] == l_eh)
                {
                    heh_iter = pn_heh;
                    break;
                }
            }
        }
        assert(heh_iter.is_valid() && "could not find valid starting point");

        while (_pnd.map_to_layout_edges_[heh_iter] == l_eh)
        {
            l_pn_vertices[l_eh].push_back(heh_iter.vertex_from());
            heh_iter = heh_iter.next();
        }
        l_pn_vertices[l_eh].push_back(heh_iter.vertex_from());
    }
    return l_pn_vertices;
}


void set_layout_pos_based_on_pn_sp(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData const& _pnd)
{
    assert(_ld.mesh_->vertices().size() == _pnd.mesh_->vertices().size() && "to call this function, ld and pnd must have the same topology");
    _ld.pos_.compute(
        [&](VH l_vh)
        {
            VH pn_vh = _pnd.mesh_->vertices()[l_vh.idx.value];
            return eigen_to_pos3(_pnd.sp_on_target_.value()[pn_vh].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get()));
        });
}

bool pn_contains_degenerate_edges(TargetMeshData const& _tmd, PathNetworkData const& _pnd)
{
    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto spA = _pnd.sp_on_target_.value()[pn_eh.vertexA()];
        auto spB = _pnd.sp_on_target_.value()[pn_eh.vertexB()];

        vec3d const posA = spA.get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
        vec3d const posB = spB.get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());

        double const length = (posA - posB).norm();
        if (length < MEDIUM_EPS)
            return true;
    }
    return false;
}


bool omd_contains_flipped_triangles(TargetMeshData const& _tmd, OverlayMeshData const& _omd)
{
    assert(_omd.map_to_target_faces_.has_value());

    auto t_normals = pm::face_normals(_tmd.pos_);
    auto o_normals = pm::face_normals(_omd.pos_);

    for (auto o_fh : _omd.mesh_->faces())
    {
        assert(o_fh.vertices().size() != 2);

        // only checking for triangles
        if (o_fh.vertices().size() > 3)
            continue;

        auto t_fh = _omd.map_to_target_faces_.value()[o_fh];

        auto const& t_normal = t_normals[t_fh];
        auto const& o_normal = o_normals[o_fh];

        auto dot = tg::dot(o_normal, t_normal);
        if (dot < 0)
            return true;
    }
    return false;
}
bool omd_contains_degenerate_triangles(OverlayMeshData const& _omd)
{
    auto areas = _omd.mesh_->faces().map([&](FH fh) { return pm::face_area(fh, _omd.pos_); });
    for (auto fh : _omd.mesh_->faces())
    {
        if (areas[fh] < SMALL_EPS)
        {
            DEBUG_VAR(fh)
            DEBUG_VAR(areas[fh])
            return true;
        }
    }
    return false;
}


} // namespace LayoutOpt
