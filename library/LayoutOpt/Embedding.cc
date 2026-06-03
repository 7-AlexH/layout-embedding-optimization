
#include "Embedding.hh"
#include <clean-core/format.hh>
#include <glow-extras/viewer/view.hh>
#include "LayoutOpt/DataStructures/GCMesh.hh"
#include "LayoutOpt/DataStructures/TriangleStrip.hh"
#include "LayoutOpt/DifferentiableIntersection.hh"
#include "LayoutOpt/EmbeddingOverlay.hh"
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/FlattenTriangleStrip.hh"
#include "LayoutOpt/IO.hh"
#include "LayoutOpt/Resample.hh"
#include "LayoutOpt/TorchUtils.hh"
#include "LayoutOpt/Utils.hh"
#include "LayoutOpt/Visualization/Colors.hh"
#include "LayoutOpt/Visualization/Viewing.hh"
#include "geometrycentral/surface/exact_geodesics.h"
#include "glow-extras/viewer/canvas.hh"
#include "polymesh/properties.hh"

#include <filesystem>
namespace fs = std::filesystem;

namespace LayoutOpt
{
namespace
{

void compute_layout_embedding_shared_part(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    // view_path_network(_tmd, _pnd);
    // DEBUG_OUT("compute 2D embedding for each strip")

    std::vector<TriangleStrip> strips;
    compute_2D_embeddig_per_strip(_tmd, _ld, _pnd, strips);

    // view_path_network(_tmd, _pnd);

    // DEBUG_OUT("insert path network")
    insert_path_network(_tmd, _pnd, _omd);

    // DEBUG_OUT("compute mask")
    compute_path_boundary_mask(_pnd, _omd);

    compute_mapping_layout_to_overlay(_ld, _pnd);

    // DEBUG_OUT("flood fill")
    compute_mapping_overlay_to_layout(_omd, _ld);

    compute_differentiable_intersections_for_overlay_stable(strips, _tmd, _ld, _pnd, _omd);

    sync_tg_and_torch(_omd);
}

HEH find_halfedge_from_corner0(const FH& fh, PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    HEH heh_from_corner = HEH::invalid;

    for (auto heh : fh.halfedges())
    {
        auto vhA = heh.vertex_to();
        auto vhB = heh.prev().vertex_from();

        auto pn_A = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[vhA]);
        auto pn_B = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[vhB]);

        if (pn_A.is_invalid() || pn_B.is_invalid())
            continue;

        auto pn_eh = pm::edge_between(pn_A, pn_B);

        if (pn_eh.is_valid())
        {
            heh_from_corner = heh;
            break;
        }
    }
    return heh_from_corner;
}

HEH find_halfedge_from_corner1(const FH& fh, const HEH& heh_from_corner0, PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    HEH heh_from_corner1 = HEH::invalid;
    for (auto heh : fh.halfedges())
    {
        if (heh == heh_from_corner0)
            continue;

        auto vhA = heh.vertex_to();
        auto vhB = heh.prev().vertex_from();

        auto pn_A = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[vhA]);
        auto pn_B = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[vhB]);

        if (pn_A.is_invalid() || pn_B.is_invalid())
            continue;

        auto pn_eh = pm::edge_between(pn_A, pn_B);

        if (pn_eh.is_valid())
        {
            heh_from_corner1 = heh;
            break;
        }
    }
    return heh_from_corner1;
}
} // empty namespace

void compute_layout_embedding_init(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, bool _allow_fallback_for_stability)
{
    // inits the path network
    // path network has same connect as layout now and surface points on the target are computed for the available vertices.
    init_path_network_surface_points(_tmd, _ld, _pnd, _allow_fallback_for_stability);
    compute_path_network_exact(_tmd, _ld, _pnd);

    compute_layout_embedding_shared_part(_tmd, _ld, _pnd, _omd);
}

void compute_layout_embedding_update(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    reset_embedding_data(_ld, _pnd, _omd);
    set_layout_pos_based_on_pn_sp(_tmd, _ld, _pnd);

    compute_path_network_exact(_tmd, _ld, _pnd);
    if (down_sample(_tmd, _ld, _pnd, _omd, 0.01))
    {
        init_path_network_surface_points(_tmd, _ld, _pnd, false);
        compute_path_network_exact(_tmd, _ld, _pnd);
    }

    compute_layout_embedding_shared_part(_tmd, _ld, _pnd, _omd);
}

void recover(TargetMeshData& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    init_path_network_surface_points(_tmd, _ld, _pnd);

    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        assert(_pnd.sp_on_target_.value()[pn_vh].type == SurfacePointType::FacePoint);
        _pnd.sp_on_target_.value()[pn_vh].bary_coords = torch::tensor({0.33, 0.33}, torch::dtype(torch::kFloat64).requires_grad(false));
    }

    reset_embedding_data(_ld, _pnd, _omd);
    compute_layout_embedding_update(_tmd, _ld, _pnd, _omd);
}

void init_path_network_surface_points(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, bool _allow_fallback_for_stability)
{
    // initally the layout and pn have the same vertices
    _pnd.reset(*_ld.mesh_.get());

    // compute surface points
    _pnd.sp_on_target_.emplace(_pnd.mesh_->vertices().make_attribute<SurfacePoint>(SurfacePoint::invalid));

    auto pn_vertices = _pnd.mesh_->vertices().to_vector();

    for (size_t i = 0; i < pn_vertices.size(); ++i)
    {
        auto pn_vh = pn_vertices[i];
        // borrow layout pos for pn for init
        auto l_vh = _ld.mesh_->handle_of(_pnd.map_to_layout_vertices_[pn_vh]);
        auto pn_p = _ld.pos_[l_vh];

        auto [p_proj, fh] = project_to_faces(pn_p, _tmd.pos_);
        assert(fh.is_valid() && "there must be a best face");

        auto heh = fh.any_halfedge();
        auto bc = compute_barycentric_coordinates(p_proj, heh, _tmd.pos_);

        if ((bc.alpha > 0.0 + LARGER_EPS && bc.beta > 0.0 + LARGER_EPS && bc.gamma() > 0.0 + LARGER_EPS && bc.alpha < 1.0 - LARGER_EPS
             && bc.beta < 1.0 - LARGER_EPS && bc.gamma() < 1.0 - LARGER_EPS)
            || !_allow_fallback_for_stability)
        {
            _pnd.sp_on_target_.value()[pn_vh]
                = SurfacePoint(torch::tensor({bc.alpha, bc.beta}, torch::dtype(torch::kFloat64).requires_grad(false)), heh, SurfacePointType::FacePoint);
        }
        else // for stability
        {
            bc = stabilize(bc, LARGER_EPS);
            _pnd.sp_on_target_.value()[pn_vh]
                = SurfacePoint(torch::tensor({bc.alpha, bc.beta}, torch::dtype(torch::kFloat64).requires_grad(false)), heh, SurfacePointType::FacePoint);

            _ld.pos_[l_vh] = torch_to_pos3(_pnd.sp_on_target_.value()[pn_vh].get_pos(_tmd.torch_pos_, *_tmd.mesh_.get()));
        }
    }

    // since, they may have been updated
    set_layout_pos_based_on_pn_sp(_tmd, _ld, _pnd);
}

void compute_path_network_exact(TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData& _pnd)
{
    //auto cd = gv::canvas_data();
    assert(_pnd.sp_on_target_.has_value());
    auto const& sp_on_target = _pnd.sp_on_target_.value();

    // 1. create edge attribute to store new points
    auto edge_new_points = _ld.mesh_->edges().make_attribute<std::vector<SurfacePoint>>();

    // 2. convert edges to vector for indexing in OpenMP
    auto l_ehs = _ld.mesh_->edges().to_vector();

#pragma omp parallel
    {
        // Thread-local copy
        GCMesh gcmesh = to_gc_mesh(_tmd.pos_);
        GeodesicAlgorithmExact gae(*(gcmesh.mesh.get()), *gcmesh.positionGeometry.get());

#pragma omp for
        for (size_t i = 0; i < l_ehs.size(); ++i)
        {
            auto l_eh = l_ehs[i];

            auto pn_vhA = _pnd.mesh_->handle_of(l_eh.vertexA().idx);
            auto pn_vhB = _pnd.mesh_->handle_of(l_eh.vertexB().idx);

            auto pn_heh = pm::halfedge_from_to(pn_vhA, pn_vhB);

            auto l_e_idx = _pnd.map_to_layout_edges_[pn_heh];
            assert(l_eh.idx == l_e_idx);

            // 3. get surfacepoints that should be connected
            auto const& sp_A = sp_on_target[pn_vhA];
            auto t_fh_A = sp_A.fh(*_tmd.mesh_.get());
            double alphaA = sp_A.bary_coords[0].item<double>();
            double betaA = sp_A.bary_coords[1].item<double>();
            double gammaA = 1.0 - alphaA - betaA;
            geometrycentral::surface::SurfacePoint spA(gcmesh.mesh->face(t_fh_A.idx.value), {betaA, gammaA, alphaA});

            // cd.add_point(torch_to_pos3(sp_A.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), RED).size(5);
            // auto pos = spA.interpolate(gcmesh.positionGeometry->inputVertexPositions);
            // cd.add_point(pos3(pos.x, pos.y, pos.z), GREEN).size(7);

            auto const& sp_B = sp_on_target[pn_vhB];
            auto t_fh_B = sp_B.fh(*_tmd.mesh_.get());
            double alphaB = sp_B.bary_coords[0].item<double>();
            double betaB = sp_B.bary_coords[1].item<double>();
            double gammaB = 1.0 - alphaB - betaB;
            geometrycentral::surface::SurfacePoint spB(gcmesh.mesh->face(t_fh_B.idx.value), {betaB, gammaB, alphaB});

            // cd.add_point(torch_to_pos3(sp_B.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), RED).size(5);
            // pos = spB.interpolate(gcmesh.positionGeometry->inputVertexPositions);
            // cd.add_point(pos3(pos.x, pos.y, pos.z), GREEN).size(7);

            gae.propagate({spA}, GEODESIC_INF, {spB});
            auto path_intrinsic = gae.traceBack(spB);
            for (size_t i = 1; i < path_intrinsic.size() - 1; ++i)
            {
                auto intrinsic_point = path_intrinsic[path_intrinsic.size() - 1 - i];
                auto pos_intrinsic_point = intrinsic_point.interpolate(gcmesh.positionGeometry->vertexPositions);

                // auto display_pos = pos3(pos_intrinsic_point.x, pos_intrinsic_point.y, pos_intrinsic_point.z);
                // cd.add_point(display_pos, RED).size(10);

                // add new point to list
                switch (intrinsic_point.type)
                {
                case geometrycentral::surface::SurfacePointType::Edge:
                {
                    auto idxA = intrinsic_point.edge.firstVertex().getIndex();
                    auto idxB = intrinsic_point.edge.secondVertex().getIndex();

                    assert(idxA < _tmd.mesh_->vertices().size() && idxB < _tmd.mesh_->vertices().size() && "idx out of range");

                    auto vhA = _tmd.mesh_->vertices()[idxA];
                    auto vhB = _tmd.mesh_->vertices()[idxB];
                    auto heh_emb = pm::halfedge_from_to(vhB, vhA);

                    auto alpha = intrinsic_point.tEdge;

                    SurfacePoint sp = {torch::tensor({alpha}, torch::dtype(torch::kFloat64)), heh_emb, SurfacePointType::EdgePoint};
                    edge_new_points[l_eh].push_back(sp);
                    break;
                }
                case geometrycentral::surface::SurfacePointType::Vertex:
                {
                    auto idx = intrinsic_point.vertex.getIndex();
                    assert(idx < _tmd.mesh_->vertices().size() && "idx out of range");

                    auto vhA = _tmd.mesh_->vertices()[idx];
                    auto heh_emb = vhA.any_outgoing_halfedge();

                    SurfacePoint sp = {torch::tensor({}, torch::dtype(torch::kFloat64)), heh_emb, SurfacePointType::VertexPoint};
                    edge_new_points[l_eh].push_back(sp);
                    break;
                }
                case geometrycentral::surface::SurfacePointType::Face:
                    assert(false && "the point can never be a face point");
                    break;
                }
            }
        }
    }

    // 4. serially apply edge points to the path network mesh
    for (auto l_eh : _ld.mesh_->edges())
    {
        auto pn_vhA = _pnd.mesh_->handle_of(l_eh.vertexA().idx);
        auto pn_vhB = _pnd.mesh_->handle_of(l_eh.vertexB().idx);
        auto pn_heh = pm::halfedge_from_to(pn_vhA, pn_vhB);

        auto& points = edge_new_points[l_eh];
        for (auto& sp : points)
        {
            auto vh_new = _pnd.mesh_->halfedges().split(pn_heh);
            _pnd.sp_on_target_.value()[vh_new] = sp;
            pn_heh = pn_heh.next();
            _pnd.map_to_layout_edges_[pn_heh.edge()] = l_eh.idx;
        }
    }
    _pnd.mesh_->compactify();
}


void compute_2D_embeddig_per_strip(TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData& _pnd, std::vector<TriangleStrip>& _strips)
{
    _strips.clear();
    _strips.reserve(_ld.mesh_->edges().size());

    for (auto _l_eh : _ld.mesh_->edges())
    {
        auto& strip = _strips.emplace_back();
        compute_triangle_strip(strip, _l_eh, _tmd, _ld, _pnd);
        assert(is_strip_valid(strip, _tmd, _pnd));
    }
}
}
