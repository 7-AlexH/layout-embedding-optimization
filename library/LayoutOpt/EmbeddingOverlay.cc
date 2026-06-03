#include "EmbeddingOverlay.hh"
#include <glow-extras/viewer/view.hh>
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/TorchUtils.hh"
#include "LayoutOpt/Utils.hh"
#include "LayoutOpt/Visualization/Colors.hh"
#include "LayoutOpt/Visualization/Viewing.hh"
#include "glow-extras/viewer/canvas.hh"

#include <cassert>

namespace LayoutOpt
{
// annonymous ns for aux funcitons
namespace
{
auto overlay_cd = gv::canvas_data();

// for filling faces with face surface points
bool keep_traversing_o(HEH o_cycle_iter_heh, FH t_fh, TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    auto pn_v_idx = _omd.map_to_pn_vertices_.value()[o_cycle_iter_heh.vertex_to()];

    // keep traversing if the vertex is not a pn vertex
    if (pn_v_idx.is_invalid())
        return true;

    // otherwise check if it is a clipped vertex or a vertex connecting back to a surface point in this face
    auto pn_vh = _pnd.mesh_->handle_of(pn_v_idx);

    for (auto pn_vh_adj : pn_vh.adjacent_vertices())
    {
        auto& sp = _pnd.sp_on_target_.value()[pn_vh_adj];
        // if it is not connected to a surfacepoint, it is a clippling vertex
        if (!sp.is_face_sp())
            continue;

        FH fh = sp.fh(*_tmd.mesh_.get());

        if (fh == t_fh)
            return false; // it is connecting back to a face surface point
    }

    return true;
}
bool keep_traversing_pn(HEH pn_cycle_iter_heh, VH o_cycle_start_vh, PathNetworkData& _pnd)
{
    // check if we completed the cycle
    if (_pnd.map_to_overlay_vertices_.value()[pn_cycle_iter_heh.vertex_to()] == o_cycle_start_vh)
        return false;

    // check if we would leave the face
    if (_pnd.sp_on_target_.value()[pn_cycle_iter_heh.vertex_to()].is_edge_sp())
        return false;

    // otherwise
    return true;
}
// find corners of original triangle that can be cut, the corner that can be cut is heh.vertex_from
std::vector<HEH> find_cutting_corners_cc(const FH& fh, PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    std::vector<HEH> cutting_corners;
    cutting_corners.reserve(3);

    HEH iter_heh_stop = fh.any_halfedge();
    HEH iter_heh = iter_heh_stop;

    do
    {
        auto vhA = iter_heh.prev().vertex_from();
        auto vhO = iter_heh.vertex_from();
        auto vhB = iter_heh.vertex_to();

        auto pnA = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[vhA]);
        auto pnO = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[vhO]);
        auto pnB = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[vhB]);

        // early out
        if (pnO.is_valid()) //  the corner which should be cut, needs to be a target vertex
        {
            iter_heh = iter_heh.next();
            continue;
        }
        // it is cut by pn vertices
        if (pnA.is_invalid() || pnB.is_invalid())
        {
            iter_heh = iter_heh.next();
            continue;
        }

        // auto pn_eh = pm::edge_between(pnA, pnB);
        // // they should be connected
        // if (pn_eh.is_valid())
        // {
        //     cutting_corners.push_back(iter_heh);
        // }
        cutting_corners.push_back(iter_heh);

        iter_heh = iter_heh.next();

    } while (iter_heh != iter_heh_stop);

    return cutting_corners;
}
} // empty namespace

void insert_path_network(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    assert(_pnd.sp_on_target_.has_value());

    // 1. reset overlay mesh to target mesh, they now share a graph structure
    _omd.reset_mesh_and_pos(_tmd.pos_);

    // 1.1 keep some correspondence between omd and others
    _omd.map_to_target_faces_.emplace(_omd.mesh_->faces().make_attribute<pm::face_index>(pm::face_index::invalid));
    _omd.map_to_target_faces_->compute([](FH fh) { return fh.idx; });

    _pnd.map_to_overlay_vertices_.emplace(_pnd.mesh_->vertices().make_attribute<pm::vertex_index>(pm::vertex_index::invalid));
    _omd.map_to_pn_vertices_.emplace(_omd.mesh_->vertices().make_attribute<pm::vertex_index>(pm::vertex_index::invalid));

    // 2. insert edge_surface points
    insert_edge_surface_points(_tmd, _pnd, _omd);

    // 3. add face_surface_points
    insert_face_surface_points(_tmd, _pnd, _omd);

    // 4. insert edges
    // 4.1 not considering surface points
    insert_edge_edge_connections(_tmd, _pnd, _omd);

    // {
    // auto g  =gv::grid();
    // {
    //     view_path_network(_tmd, _pnd);
    // }
    // {
    //     view_overlay(_omd);
    // }
    // }
    // 4.2 triangulate overlay faces containing at least one surface point
    insert_surface_point_edge_connections(_tmd, _pnd, _omd);

    // 5. triangulate
    triangulate_by_ear_clipping(_tmd, _omd);
}

void triangulate_by_ear_clipping(TargetMeshData const& _tmd, OverlayMeshData& _omd)
{
    if (pm::is_triangle_mesh(*_omd.mesh_.get()))
    {
        _omd.mesh_->compactify();
        return;
    }
    assert(pm::is_triangle_mesh(*_tmd.mesh_.get()));

    auto target_normals = pm::triangle_normals(_tmd.pos_);

    std::vector<VH> vs;
    auto const old_faces = _omd.mesh_->faces().to_vector();
    for (auto const o_fh : old_faces)
    {
        vs.clear();
        o_fh.vertices().into_vector(vs);

        // if o_fh is a triangle, we got nothing to do
        if (vs.size() <= 3)
            continue;

        // process face
        auto t_f_idx = _omd.map_to_target_faces_.value()[o_fh];
        auto t_f_normal = target_normals[t_f_idx];

        // store which vertices have been clipped away
        std::vector<bool> clipped_away(vs.size(), false);
        // if this corner contains another vertex, is must be masked
        std::vector<bool> masked(vs.size(), false);

        // remove original face
        _omd.mesh_->faces().remove(o_fh);

        // ==================================Helpers====================================
        auto find_next_and_previous = [](int i, std::vector<bool> const& clipped)
        {
            int const n = static_cast<int>(clipped.size());
            if (n == 0)
                return std::pair<int, int>{-1, -1};

            int prev = (i - 1 + n) % n;
            int iter_count = 0;
            while (clipped[prev])
            {
                prev = (prev - 1 + n) % n;
                if (++iter_count > n)
                    return std::pair<int, int>{-1, -1};
            }

            int next = (i + 1) % n;
            iter_count = 0;
            while (clipped[next])
            {
                next = (next + 1) % n;
                if (++iter_count > n)
                    return std::pair<int, int>{-1, -1};
            }

            return std::pair<int, int>{prev, next};
        };

        struct Corner
        {
            int prev, current, next;
        };

        auto find_next_convex_corner = [&](int current_corner) -> std::optional<Corner>
        {
            int n = (int)vs.size();
            Corner corner;
            int start = (current_corner + 1) % n;

            for (int iter = 0; iter < n; ++iter)
            {
                corner.current = (start + iter) % n;
                std::tie(corner.prev, corner.next) = find_next_and_previous(corner.current, clipped_away);

                if (corner.prev == -1 || corner.next == -1)
                    continue;

                auto const& pos_prev = _omd.pos_[vs[corner.prev]];
                auto const& pos_current = _omd.pos_[vs[corner.current]];
                auto const& pos_next = _omd.pos_[vs[corner.next]];

                if (is_convex_corner_3d(pos_prev, pos_current, pos_next, t_f_normal))
                    return corner;
            }

            return std::nullopt;
        };

        auto find_best_convex_corner = [&](int current_corner) -> std::optional<Corner>
        {
            int n = (int)vs.size();
            Corner best_corner;
            float best_angle = std::numeric_limits<float>::max();
            bool found = false;

            int start = (current_corner + 1) % n;

            for (int iter = 0; iter < n; ++iter)
            {
                Corner corner;
                corner.current = (start + iter) % n;

                if (clipped_away[corner.current] || masked[corner.current])
                    continue;

                std::tie(corner.prev, corner.next) = find_next_and_previous(corner.current, clipped_away);

                if (corner.prev == -1 || corner.next == -1)
                    continue;

                auto const& pos_prev = _omd.pos_[vs[corner.prev]];
                auto const& pos_current = _omd.pos_[vs[corner.current]];
                auto const& pos_next = _omd.pos_[vs[corner.next]];

                auto [is_convex, angle] = corner_convexity_and_angle_3d(pos_prev, pos_current, pos_next, t_f_normal);

                if (is_convex && angle < best_angle)
                {
                    best_angle = angle;
                    best_corner = corner;
                    found = true;
                }
            }

            if (found)
                return best_corner;
            else
                return std::nullopt;
        };

        // ================================End Helpers==================================
        // Ear clipping loop
        int last_index = 0;
        while (true)
        {
            auto corner_opt = find_best_convex_corner(last_index);

            if (!corner_opt.has_value())
                break;

            // DEBUG_VAR(corner_opt->prev)
            // DEBUG_VAR(corner_opt->current)
            // DEBUG_VAR(corner_opt->next)

            auto corner = corner_opt.value();

            auto& posA = _omd.pos_[vs[corner.prev]];
            auto& posB = _omd.pos_[vs[corner.current]];
            auto& posC = _omd.pos_[vs[corner.next]];

            tg::dtriangle3 t(posA, posB, posC);

            // assert that the normal of all faces are consistent with tmd normal
            auto normal = tg::normal_of(t);
            auto dot = tg::dot(normal, t_f_normal);
            assert(dot > 0);
            if (dot < 0)
            {
                DEBUG_OUT("normals are in different directions")
                auto c = gv::canvas();
                c.add_lines(_omd.pos_, GREEN).size(2);
                c.add_face(_tmd.mesh_->handle_of(t_f_idx), _tmd.pos_, T3_BLUE_75);
                c.add_line(pm::face_centroid(_tmd.mesh_->handle_of(t_f_idx), _tmd.pos_), t_f_normal, BLUE);
                c.add_face(t, MAGENTA);
                c.add_line(tg::centroid_of(t), vec3(normal), MAGENTA);
            }

            // check if any other vertex is inside this triangle
            bool has_inside = false;
            for (int i = 0; i < vs.size(); ++i)
            {
                if (clipped_away[i] || i == corner.prev || i == corner.current || i == corner.next)
                    continue;

                auto& p = _omd.pos_[vs[i]];

                if (tg::contains(t, p, EPS))
                {
                    has_inside = true;
                    masked[corner.current] = true;
                    break;
                }
            }

            if (!has_inside)
            {
                if (!_omd.mesh_->faces().can_add(vs[corner.prev], vs[corner.current], vs[corner.next]))
                {
                    DEBUG_OUT("can not add face")
                    auto c = gv::canvas();
                    c.add_faces(_omd.pos_, GREEN_25);
                    c.add_lines(_omd.pos_, GREEN).size(2);
                    c.add_points(_omd.pos_, GREEN).size(7);

                    c.add_point(posA, RED).size(15);
                    c.add_point(posB, GREEN).size(15);
                    c.add_point(posC, BLUE).size(15);
                }


                assert(_omd.mesh_->faces().can_add(vs[corner.prev], vs[corner.current], vs[corner.next]));
                // Add triangle
                auto o_fh_new = _omd.mesh_->faces().add(vs[corner.prev], vs[corner.current], vs[corner.next]);

                // DEBUG_VAR(o_fh_new)

                _omd.map_to_target_faces_.value()[o_fh_new] = t_f_idx;

                // it got clipped away
                clipped_away[corner.current] = true;
                // need to check containment again
                masked.assign(vs.size(), false);

                last_index = corner.current;

                // if (omd_contains_flipped_triangles(_tmd, _omd))
                // {
                //     DEBUG_OUT("contains flipped triangle")

                //     auto v = gv::view();
                //     auto c = gv::canvas();
                //     view_overlay(_omd);

                //     c.add_face(_tmd.mesh_->handle_of(t_f_idx), _tmd.pos_, MAGENTA_10);

                //     c.add_point(posA, RED).size(15);
                //     c.add_point(posB, GREEN).size(15);
                //     c.add_point(posC, BLUE).size(15);
                // }

                // if (omd_contains_degenerate_triangles(_omd))
                //{
                //     DEBUG_OUT("contains degenerate triangle")

                //    auto v = gv::view();
                //    auto c = gv::canvas();
                //    view_overlay(_omd);

                //    auto f_idx_of_interest = pm::face_index(1412);
                //    auto fh_of_interest = _omd.mesh_->handle_of(f_idx_of_interest);

                //    DEBUG_VAR(o_fh_new)
                //    for (auto vh : fh_of_interest.vertices())
                //    {
                //        c.add_point(_omd.pos_[vh], RED).size(10);
                //    }

                //    DEBUG_VAR(corner_convexity_and_angle_3d(posA, posB, posC, t_f_normal));

                //    c.add_point(posA, RED).size(15);
                //    c.add_point(posB, GREEN).size(15);
                //    c.add_point(posC, BLUE).size(15);
                //}
            }
            else
            {
                // Move to next vertex
                last_index = corner.current;
            }

            // Stop if only 3 vertices remain unclipped
            int remaining = (int)std::count(clipped_away.begin(), clipped_away.end(), false);
            if (remaining <= 3)
            {
                std::vector<int> idxs;
                for (size_t i = 0; i < clipped_away.size(); ++i)
                    if (!clipped_away[i])
                        idxs.push_back((int)i);
                if (idxs.size() == 3)
                {
                    auto o_fh_new = _omd.mesh_->faces().add(vs[idxs[0]], vs[idxs[1]], vs[idxs[2]]);
                    // DEBUG_VAR(o_fh_new)
                    _omd.map_to_target_faces_.value()[o_fh_new] = t_f_idx;


                    // if (omd_contains_degenerate_triangles(_omd))
                    // {
                    //     DEBUG_OUT("contains degenerate triangle")

                    //     auto v = gv::view();
                    //     auto c = gv::canvas();
                    //     view_overlay(_omd);

                    //     DEBUG_VAR(o_fh_new)
                    //     for (auto vh : o_fh_new.vertices())
                    //     {
                    //         c.add_point(_omd.pos_[vh], RED).size(10);
                    //     }

                    //     DEBUG_VAR(corner_convexity_and_angle_3d(posA, posB, posC, t_f_normal));

                    //     c.add_point(posA, RED).size(15);
                    //     c.add_point(posB, GREEN).size(15);
                    //     c.add_point(posC, BLUE).size(15);
                    // }
                }
                break;
            }
        } // while ears can be clipped
    } // for all faces
    _omd.mesh_->compactify();

    assert(!omd_contains_flipped_triangles(_tmd, _omd));
}

void insert_edge_surface_points(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    pm::edge_attribute<std::vector<EH>> candidat_edges_on_overlay(*_tmd.mesh_.get());

    // init the edges
    for (auto eh : _tmd.mesh_->edges())
    {
        candidat_edges_on_overlay[eh].push_back(_omd.mesh_->handle_of(eh));
    }

    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto sp = _pnd.sp_on_target_.value()[pn_vh];

        if (!sp.is_edge_sp())
            continue;

        assert(!sp.is_vertex_sp());

        auto pos_new = torch_to_pos3(sp.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get()));

        if (tg::is_nan(pos_new.x) || tg::is_nan(pos_new.y) || tg::is_nan(pos_new.z))
        {
            DEBUG_VAR(sp.bary_coords[0].item<double>())
        }

        auto t_eh = _tmd.mesh_->handle_of(sp.heh_idx).edge();
        auto [pos_proj, eh] = project_to_edges(pos_new, _omd.pos_, candidat_edges_on_overlay[t_eh]);
        assert(eh.is_valid() && "heh must be valid to split it");

        auto vh_new = _omd.mesh_->halfedges().split(eh.halfedgeA());
        _omd.pos_[vh_new] = pos_new;
        candidat_edges_on_overlay[t_eh].push_back(eh.halfedgeA().next().edge());

        _pnd.map_to_overlay_vertices_.value()[pn_vh] = vh_new.idx;
        _omd.map_to_pn_vertices_.value()[vh_new] = pn_vh.idx;
    }
}

void insert_face_surface_points(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    auto containing_sp = _omd.mesh_->faces().make_attribute(false);
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto sp = _pnd.sp_on_target_.value()[pn_vh];

        if (!sp.is_face_sp())
            continue;

        auto pos_new = torch_to_pos3(sp.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get()));
        auto vh_new = _omd.mesh_->vertices().add();

        _pnd.map_to_overlay_vertices_.value()[pn_vh] = vh_new.idx;
        _omd.map_to_pn_vertices_.value()[vh_new] = pn_vh.idx;

        _omd.pos_[vh_new] = pos_new;
        auto fh = sp.fh(*_omd.mesh_.get());
        containing_sp[fh] = true;
    }

    // 2. add pn_faces if completly inside target face
    for (auto pn_face : _pnd.mesh_->faces())
    {
        bool should_add_face = true;
        FH t_fh = FH::invalid;
        std::vector<VH> o_face;
        o_face.reserve(pn_face.vertices().size());

        for (auto pn_heh : pn_face.halfedges())
        {
            auto pn_vh0 = pn_heh.vertex_from();
            auto pn_vh1 = pn_heh.vertex_to();

            auto const& sp0 = _pnd.sp_on_target_.value()[pn_vh0];
            auto const& sp1 = _pnd.sp_on_target_.value()[pn_vh1];

            if (!sp0.is_face_sp() || !sp1.is_face_sp())
            {
                should_add_face = false;
                continue;
            }

            auto t_fh0 = sp0.fh(*_tmd.mesh_.get());
            auto t_fh1 = sp1.fh(*_tmd.mesh_.get());

            if (t_fh0 != t_fh1)
            {
                should_add_face = false;
                continue;
            }
            t_fh = t_fh0;

            // at this point, we know that we can add at least the edge
            auto o_vh0 = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_vh0]);
            auto o_vh1 = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_vh1]);

            // add for potential face
            o_face.push_back(o_vh0);
        }
        if (should_add_face)
        {
            assert(_omd.mesh_->faces().can_add(o_face));
            auto o_new_face = _omd.mesh_->faces().add(o_face);
            _omd.map_to_target_faces_.value()[o_new_face] = t_fh;
        }
    }
}

void insert_edge_edge_connections(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    overlay_cd.clear();

    // 1. subdivide all faces for edge points
    pm::vertex_attribute<bool> is_directly_connected = _omd.mesh_->vertices().make_attribute(false); // tracks if a vertex should be connected in current face
    for (auto o_fh : _omd.mesh_->faces())
    {
        // check if something needs to be inserted
        if (o_fh.vertices().size() == 3)
        {
            continue;
        }

        // start proccesing face
        auto t_f_idx = _omd.map_to_target_faces_.value()[o_fh];
        is_directly_connected.clear(false);
        bool found_directly_connected = false;
        for (auto o_vh : o_fh.vertices())
        {
            if (is_directly_connected[o_vh])
                continue; // already seen before

            auto pn_vh = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[o_vh]);
            if (pn_vh.is_invalid())
                continue; // this is a target vertex

            auto const& sp = _pnd.sp_on_target_.value()[pn_vh];
            for (auto pn_vh_n : pn_vh.adjacent_vertices())
            {
                auto const& sp_n = _pnd.sp_on_target_.value()[pn_vh_n];
                if (sp_n.is_face_sp())
                    continue;

                auto t_fh = shared_face(sp, sp_n, *_tmd.mesh_.get());
                if (t_fh.idx == t_f_idx)
                {
                    is_directly_connected[o_vh] = true;
                    is_directly_connected[_pnd.map_to_overlay_vertices_.value()[pn_vh_n]] = true;
                    found_directly_connected = true;
                }
            }
        }
        if (!found_directly_connected)
            continue; // will be handeled later

        auto cutting_corner_hehs = find_cutting_corners_cc(o_fh, _pnd, _omd);
        assert(cutting_corner_hehs.size() < 4);

        std::vector<std::vector<VH>> new_faces;
        new_faces.reserve(o_fh.vertices().size());

        overlay_cd.clear();
        for (auto heh : cutting_corner_hehs)
        {
            overlay_cd.add_line(heh, _omd.pos_, BLACK).size(5);
            overlay_cd.add_line(heh.prev(), _omd.pos_, BLACK).size(5);

            auto heh_iter_next = heh;
            auto heh_iter_prev = heh.prev();

            // first corner
            std::deque<VH> face = {heh_iter_next.vertex_from()};
            bool should_break = false;

            // add faces for as long as possible
            while (heh_iter_prev != heh_iter_next)
            {
                do // iterate until we find a vertex that should be connected
                {
                    face.emplace_front(heh_iter_prev.vertex_from());
                    overlay_cd.add_point(_omd.pos_[heh_iter_prev.vertex_from()], PURPLE).size(12);
                    overlay_cd.add_line(heh_iter_prev, _omd.pos_, PURPLE).size(6);

                    heh_iter_prev = heh_iter_prev.prev();
                    if (_omd.map_to_pn_vertices_.value()[heh_iter_prev.vertex_to()].is_invalid()) // encountered target vertex
                    {
                        should_break = true;
                    }
                } while (!is_directly_connected(heh_iter_prev.vertex_to()) && !should_break);

                do // iterate until we find a vertex that should be connected
                {
                    face.emplace_back(heh_iter_next.vertex_to());
                    overlay_cd.add_point(_omd.pos_[heh_iter_next.vertex_to()], LILAC).size(12);
                    overlay_cd.add_line(heh_iter_next, _omd.pos_, LILAC).size(6);

                    heh_iter_next = heh_iter_next.next();
                    if (_omd.map_to_pn_vertices_.value()[heh_iter_next.vertex_from()].is_invalid()) // encountered target vertex
                    {
                        should_break = true;
                    }

                } while (!is_directly_connected(heh_iter_next.vertex_from()) && !should_break);

                auto pn_from = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[heh_iter_prev.vertex_to()]);
                auto pn_to = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[heh_iter_next.vertex_from()]);

                if (pn_from.is_valid() && pn_to.is_valid())
                {
                    auto pn_heh = pm::halfedge_from_to(pn_from, pn_to);
                    if (pn_heh.is_invalid())
                    {
                        should_break = true;
                    }
                    else if (pn_heh.is_valid())
                    {
                        new_faces.push_back({face.begin(), face.end()});
                        face.clear();
                        face.emplace_front(heh_iter_prev.vertex_to());
                        face.emplace_back(heh_iter_next.vertex_from());
                    }
                }

                // if (o_fh.idx.value == 305)
                // {
                //     auto c = gv::canvas();
                //     DEBUG_VAR(o_fh)

                //     c.add_points(_omd.pos_, PETROL).size(10);
                //     c.add_lines(_omd.pos_, GREEN).size(3);
                //     c.add_faces(_omd.pos_, GREEN_25);
                //     c.add_data(overlay_cd);
                //     overlay_cd.clear();
                // }

                if (should_break)
                {
                    break;
                }
            }
        } // for each cutting corner

        std::vector<HEH> old_hehs;
        if (new_faces.size() > 0)
        {
            o_fh.halfedges().into_vector(old_hehs);
            _omd.mesh_->faces().remove(o_fh);
        }

        // cutting corner polygons can always be added
        for (auto const& vhs : new_faces)
        {
            // if (!_omd.mesh_->faces().can_add(vhs))
            // {
            //     DEBUG_VAR(o_fh)
            //     overlay_cd.add_point(_omd.pos_[vhs.front()], RED).size(20);
            //     for (auto vh : vhs)
            //     {
            //         overlay_cd.add_point(_omd.pos_[vh], GREEN).size(20);
            //     }

            //     auto c = gv::canvas();
            //     c.add_points(_omd.pos_, PETROL).size(10);
            //     c.add_lines(_omd.pos_, GREEN).size(3);
            //     c.add_faces(_omd.pos_, GREEN_25);
            //     c.add_data(overlay_cd);
            // }

            assert(_omd.mesh_->faces().can_add(vhs));

            auto t_new_face = _omd.mesh_->faces().add(vhs);
            _omd.map_to_target_faces_.value()[t_new_face] = t_f_idx;
        }

        // add inner polygon for now, some may be removed later again
        if (new_faces.size() > 0)
        {
            std::vector<VH> vhs_inner_polygon;
            vhs_inner_polygon.reserve(10);

            HEH o_heh_iter = HEH::invalid;
            for (auto o_heh : old_hehs)
            {
                if (o_heh.is_boundary())
                {
                    o_heh_iter = o_heh;
                    break;
                }
            }
            if (o_heh_iter.is_valid())
            {
                HEH o_heh_start = o_heh_iter;
                do
                {
                    vhs_inner_polygon.push_back(o_heh_iter.vertex_to());
                    o_heh_iter = o_heh_iter.next();
                } while (o_heh_iter != o_heh_start);

                if (!_omd.mesh_->faces().can_add(vhs_inner_polygon))
                {
                    DEBUG_VAR(o_fh)
                    DEBUG_VAR("inner polygon")
                    auto c = gv::canvas();
                    c.add_point(_omd.pos_[vhs_inner_polygon.front()], RED).size(20);

                    if (o_heh_start.is_valid())
                    {
                        c.add_line(o_heh_start, _omd.pos_, ORANGE).size(10);
                    }

                    for (auto vh : vhs_inner_polygon)
                    {
                        c.add_point(_omd.pos_[vh], GREEN).size(20);
                    }

                    c.add_points(_omd.pos_, PETROL).size(5);
                    // c.add_lines(_omd.pos_, GREEN).size(3);
                    c.add_faces(_omd.pos_, GREEN_25);
                }

                assert(_omd.mesh_->faces().can_add(vhs_inner_polygon));
                auto t_new_face = _omd.mesh_->faces().add(vhs_inner_polygon);
                _omd.map_to_target_faces_.value()[t_new_face] = t_f_idx;
            }
        } // last face


        // {
        //     DEBUG_VAR(cutting_corner_hehs.size())
        //     auto c = gv::canvas();
        //     c.add_points(_omd.pos_, PETROL).size(5);
        //     c.add_lines(_omd.pos_, GREEN).size(3);
        //     c.add_faces(_omd.pos_, MAGENTA_25);

        //     if (o_inner_polygon_start_heh.is_valid())
        //     {
        //         c.add_line(o_inner_polygon_start_heh, _omd.pos_, ORANGE).size(6);
        //     }

        //     c.add_data(overlay_cd);
        // }

    } // for all faces
}

void insert_surface_point_edge_connections(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    auto pn_already_visited = _pnd.mesh_->vertices().make_attribute(false); // keeping track
    auto o_already_visited = _omd.mesh_->edges().make_attribute(false);     // keeping track
    for (auto const pn_vh : _pnd.mesh_->vertices())
    {
        // early outs
        if (!_pnd.sp_on_target_.value()[pn_vh].is_face_sp()) // edge surface points are already done
            continue;
        if (pn_already_visited[pn_vh]) // worked on it before
            continue;

        // start processing
        overlay_cd.clear();

        pn_already_visited[pn_vh] = true;
        o_already_visited.clear(false);

        const FH t_fh = _pnd.sp_on_target_.value()[pn_vh].fh(*_tmd.mesh_.get());

        // we know pn_vh is a face surface point
        // find the first outgoing halfedge pointing towards edge-surface point
        const HEH pn_boundary_start_heh
            = pn_vh.outgoing_halfedges().where([&](HEH pn_heh) { return _pnd.sp_on_target_.value()[pn_heh.vertex_to()].is_edge_sp(); }).first();

        // the surface point is only connected to other surface points
        if (pn_boundary_start_heh.is_invalid())
            continue;

        // to fill the "hole" (there is currently a face but not connected with pn_vh), we need to cycle around the overlay "boundary" once
        // keep starting vertex
        const VH o_boundary_start_vh = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_boundary_start_heh.vertex_to()]);
        overlay_cd.add_point(_omd.pos_[o_boundary_start_vh], BLUE).size(20);


        // we have to find all the faces (cycles) that fill the "hole"
        VH o_cycle_start_vh = o_boundary_start_vh;
        auto o_cycle_iter_heh
            = o_cycle_start_vh.outgoing_halfedges().where([&](HEH o_heh) { return _omd.map_to_target_faces_.value()[o_heh.face()] == t_fh; }).first();
        assert(o_cycle_iter_heh.is_valid());
        o_already_visited[o_cycle_iter_heh] = true;
        overlay_cd.add_line(o_cycle_iter_heh, _omd.pos_, MAGENTA).size(6);

        const FH o_fh = o_cycle_iter_heh.face();


        std::vector<std::vector<VH>> o_faces;
        std::vector<std::vector<int>> debug_o_faces_where;

        o_faces.reserve(3 * pn_vh.adjacent_vertices().size());

        // work on current face
        do // cycle this face until we have reached the beginning o_boundary_start_vh
        {
            o_cycle_start_vh = o_cycle_iter_heh.vertex_from();

            // collect all the overlay vertices for a cycle
            std::vector<VH>& o_face = o_faces.emplace_back();
            o_face.reserve(3 * pn_vh.adjacent_vertices().size());
            o_face.push_back(o_cycle_start_vh);

            std::vector<int>& debug_o_face_where = debug_o_faces_where.emplace_back();
            debug_o_face_where.push_back(0);

            // also keep track on where to start next
            HEH o_cycle_iter_candidat_heh = HEH::invalid;

            // we need to cycle on the om and the pn alternatly
            // start with om
            bool cycle_on_om = true;
            bool keep_cycling = true;

            while (keep_cycling) // work on current cycle
            {
                if (cycle_on_om)
                {
                    // traverse the boundary as long as the to vertex is not a valid pn vertex or a pn vertex connected to surface point in this face
                    while (keep_traversing_o(o_cycle_iter_heh, t_fh, _tmd, _pnd, _omd))
                    {
                        o_face.push_back(o_cycle_iter_heh.vertex_to());
                        o_already_visited[o_cycle_iter_heh] = true;

                        o_cycle_iter_heh = o_cycle_iter_heh.next();
                    }
                    o_face.push_back(o_cycle_iter_heh.vertex_to());
                    o_already_visited[o_cycle_iter_heh] = true;
                    debug_o_face_where.push_back(0);

                    // check if we need to keep track for the start of next cycle
                    if (o_cycle_iter_candidat_heh.is_invalid())
                    {
                        o_cycle_iter_candidat_heh = o_cycle_iter_heh.next();
                    }

                    // don't cycle on overlay anymore
                    cycle_on_om = false;
                }
                else // the path network as long as we do not leave the face
                {
                    VH pn_cycle_vh = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[o_cycle_iter_heh.vertex_to()]);
                    assert(pn_cycle_vh.is_valid());

                    // find the heh that points to a face surface point in the same face
                    HEH pn_cycle_iter_heh = pn_cycle_vh.outgoing_halfedges()
                                                .where(
                                                    [&](HEH pn_heh)
                                                    {
                                                        auto sp_to = _pnd.sp_on_target_.value()[pn_heh.vertex_to()];
                                                        if (!sp_to.is_face_sp())
                                                            return false;
                                                        if (sp_to.fh(*_tmd.mesh_.get()) == t_fh)
                                                            return true;
                                                        return false;
                                                    })
                                                .first();

                    if (pn_cycle_iter_heh.is_invalid())
                    {
                        DEBUG_OUT("invalid pn_cycle_heh")
                        auto c = gv::canvas();
                        c.add_point(_omd.pos_[o_cycle_start_vh], BLUE).size(15);

                        c.add_line(tg::pos3{_omd.pos_[o_face[0]]}, tg::pos3{_omd.pos_[o_face[1]]}, tg::color3::cyan).size(8);
                        for (int i = 0; i < int(o_face.size()) - 1; ++i)
                        {
                            c.add_line(_omd.pos_[o_face[i]], _omd.pos_[o_face[(i + 1) % o_face.size()]]).color(tg::color3::red).size(8);
                        }
                    }
                    assert(pn_cycle_iter_heh.is_valid());

                    // now, we need to cycle in the pn until closure or hitting the boundary (ie pn vertex is edge point)
                    while (keep_traversing_pn(pn_cycle_iter_heh, o_cycle_start_vh, _pnd))
                    {
                        o_face.push_back(_omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_cycle_iter_heh.vertex_to()]));
                        pn_already_visited[pn_cycle_iter_heh.vertex_to()] = true;

                        debug_o_face_where.push_back(1);
                        pn_cycle_iter_heh = pn_cycle_iter_heh.next();
                    }

                    // if we do not close the cycle, push the to-vertex back
                    if (_pnd.map_to_overlay_vertices_.value()[pn_cycle_iter_heh.vertex_to()] != o_cycle_start_vh)
                    {
                        o_face.push_back(_omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_cycle_iter_heh.vertex_to()]));
                        pn_already_visited[pn_cycle_iter_heh.vertex_to()] = true;

                        debug_o_face_where.push_back(1);
                    }


                    auto const cycle_to = _pnd.map_to_overlay_vertices_.value()[pn_cycle_iter_heh.vertex_to()];

                    if (cycle_to == o_cycle_start_vh)
                    {
                        keep_cycling = false;
                    }
                    else
                    {
                        cycle_on_om = true;
                        VH o_vh = _omd.mesh_->handle_of(cycle_to);

                        auto const expr = o_vh.outgoing_halfedges().where(
                            [&](HEH o_heh)
                            {
                                // has to have the same face idx
                                if (_omd.map_to_target_faces_.value()[o_heh.face()] != t_fh)
                                    return false;
                                // check whether point exists only in omd
                                auto pn_idx = _omd.map_to_pn_vertices_.value()[o_heh.vertex_to()];
                                if (pn_idx.is_invalid())
                                    return true;
                                // has to be a valid e-sp
                                auto pn_vh = _pnd.mesh_->handle_of(pn_idx);
                                auto sp = _pnd.sp_on_target_.value()[pn_vh];
                                return sp.is_edge_sp();
                            });
                        o_cycle_iter_heh = expr.first();
                        assert(o_cycle_iter_heh.is_valid());
                    }
                }
            } // loop for current cycle


            // if (t_fh.idx.value == 5949)
            // {
            //     DEBUG_VAR(pn_vh);

            //     auto v = gv::view();
            //     view_path_network(_tmd, _pnd);
            //     auto c = gv::canvas();
            //     c.add_point(_omd.pos_[o_cycle_start_vh], GREEN).size(18);

            //     for (auto heh_ : pn_vh.incoming_halfedges())
            //     {
            //         auto o_vh_from = _pnd.map_to_overlay_vertices_.value()[heh_.vertex_from()];
            //         auto o_vh_to = _pnd.map_to_overlay_vertices_.value()[heh_.vertex_to()];

            //         c.add_line(_omd.pos_[o_vh_from], _omd.pos_[o_vh_to]).color(heh_.is_boundary() ? RED : ORANGE).scale_size(heh_.is_boundary() ? 3
            //         : 2); c.add_point(_omd.pos_[o_vh_to]).color(heh_.is_boundary() ? RED : ORANGE).scale_size(heh_.is_boundary() ? 3 : 2);

            //         auto o_vh_from_next = _pnd.map_to_overlay_vertices_.value()[heh_.next().vertex_from()];
            //         auto o_vh_to_next = _pnd.map_to_overlay_vertices_.value()[heh_.next().vertex_to()];

            //         c.add_line(_omd.pos_[o_vh_from], _omd.pos_[o_vh_to_next]).color(heh_.next().is_boundary() ? RED : GREEN).scale_size(heh_.next().is_boundary() ? 3 : 2);
            //         c.add_point(_omd.pos_[o_vh_to_next]).color(heh_.next().is_boundary() ? RED : GREEN).scale_size(heh_.next().is_boundary() ? 3 : 2);
            //     }

            // for (int i = 0; i < int(o_face.size()); ++i)
            // {
            //     tg::color3 color = tg::color3::red;

            //     if (debug_o_face_where[i] == 1)
            //         color = tg::color3::magenta;
            //     c.add_line(_omd.pos_[o_face[i]], _omd.pos_[o_face[(i + 1) % o_face.size()]]).color(color).size(8);
            //     auto avr_pos = tg::lerp(_omd.pos_[o_face[i]], _omd.pos_[o_face[(i + 1) % o_face.size()]], 0.5);
            //     c.add_label(avr_pos, std::to_string(i).c_str());
            // }
            //}

            // prepare next cycle
            o_cycle_iter_heh = o_cycle_iter_candidat_heh;

            while (o_already_visited[o_cycle_iter_heh])
            {
                o_cycle_iter_heh = o_cycle_iter_heh.next();
                if (o_cycle_iter_heh.vertex_from() == o_boundary_start_vh)
                {
                    break;
                }
            }
        } while (o_cycle_iter_heh.vertex_from() != o_boundary_start_vh); // loop for current face


        // adding the new faces
        _omd.mesh_->faces().remove(o_fh);

        int count = 0;
        for (auto const& o_face : o_faces)
        {
            if (!_omd.mesh_->faces().can_add(o_face))
            {
                auto& debug_o_face_where = debug_o_faces_where[count];
                DEBUG_VAR(t_fh)
                DEBUG_OUT("cannot add face")
                auto g = gv::grid();
                {
                    auto c = gv::canvas();
                    c.add_point(_omd.pos_[o_cycle_start_vh], BLUE).size(20);

                    c.add_point(tg::pos3{_omd.pos_[o_face[0]]}, tg::color3::cyan).size(15);
                    for (int i = 0; i < int(o_face.size()); ++i)
                    {
                        tg::color3 color = tg::color3::red;

                        if (debug_o_face_where[i] == 1)
                            color = tg::color3::magenta;

                        c.add_line(_omd.pos_[o_face[i]], _omd.pos_[o_face[(i + 1) % o_face.size()]]).color(color).size(8);
                    }
                }
                {
                    auto v = gv::view();
                    // view_path_network(_tmd, _pnd);
                    auto c = gv::canvas();
                    c.add_points(_omd.pos_, MAGENTA).size(7);
                    c.add_lines(_omd.pos_, BLUE_75);
                    c.add_faces(_omd.pos_, BLUE_25);
                }
                {
                    auto c = gv::canvas();
                    c.add_points(_tmd.pos_, BLUE).size(7);
                    c.add_lines(_tmd.pos_, BLUE_75);
                    c.add_faces(_tmd.pos_, BLUE_25);
                }
            }


            assert(_omd.mesh_->faces().can_add(o_face));


            auto o_new_face = _omd.mesh_->faces().add(o_face);
            _omd.map_to_target_faces_.value()[o_new_face] = t_fh;

            count = count + 1;
        }
    }
}

} // namespace LayoutOpt
