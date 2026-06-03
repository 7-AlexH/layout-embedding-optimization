#include "LoopSubdivision.hh"
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/Visualization/ColorGenerator.hh"
#include "LayoutOpt/Visualization/Viewing.hh"
#include "glow-extras/viewer/canvas.hh"

namespace LayoutOpt
{

void choose_loop_subdivisions(LoopSubdivData& _lsd, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd, double const _target_length, int const _max)
{
    init_loopSubdivData(_lsd, _tmd, _ld, _pnd);

    _lsd.arc_subdivs_ = _lsd.arc_mesh_->edges().make_attribute<int>(0);

    // choose loop subdivion (following Born et al. LayoutEmbedding)
    // Iterate over loops
    auto visited = _lsd.arc_mesh_->edges().make_attribute<bool>(false);
    for (auto l_e : _lsd.arc_mesh_->edges())
    {
        if (visited[l_e])
            continue;

        // Trace loop and sum objective function
        auto calc_loop_objective = [&](auto h_seed, int subdiv_add)
        {
            auto h = h_seed;
            auto obj = 0.0;
            do
            {
                double const length_sub = _lsd.arc_lengths_(h) / (_lsd.arc_subdivs_[h.edge()] + subdiv_add + 1);
                obj += pow(length_sub - _target_length, 2);

                visited[h.edge()] = true;
                h = h.next().next().opposite();
            } while (h != h_seed);
            return obj;
        };

        // Increase subdivision as long as it reduces objective
        while (calc_loop_objective(l_e.halfedgeA(), 1) < calc_loop_objective(l_e.halfedgeA(), 0))
        {
            auto h = l_e.halfedgeA();
            do
            {
                _lsd.arc_subdivs_[h.edge()] += 1;
                h = h.next().next().opposite();
            } while (h != l_e.halfedgeA());
        }
    }

    _lsd.arc_subdivs_ = _lsd.arc_subdivs_.map([&](int s) { return std::min(_max, s); });
}


void init_loopSubdivData(LoopSubdivData& _lsd, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd)
{
    // 1. compute arc ids
    _lsd.l_arc_ids_ = compute_layout_edge_arc_idx(_ld);

    // 2. compute arc mesh
    _lsd.arc_mesh_ = pm::Mesh::create();
    _lsd.arc_mesh_->copy_from(*_ld.mesh_.get());
    _lsd.arc_ids_ = _lsd.l_arc_ids_.copy_to(*_lsd.arc_mesh_.get());
    _lsd.arc_pos_ = _ld.pos_.copy_to(*_lsd.arc_mesh_.get());
    _lsd.map_a_vh_to_layout_ = _lsd.arc_mesh_->vertices().make_attribute<VH>();
    _lsd.map_a_vh_to_layout_.compute([&_ld](VH a_vh) { return _ld.mesh_->handle_of(a_vh); });


    for (auto heh : _lsd.arc_mesh_->halfedges())
    {
        if (heh.is_removed())
            continue;

        if (heh.vertex_from().adjacent_vertices().size() == 2)
        {
            _lsd.arc_mesh_->halfedges().collapse(heh);
        }
    }
    _lsd.arc_mesh_->compactify();

    // 3. compute arc embedded length
    auto l_embedded_length_ = compute_embedded_length_per_layout_edge(_tmd, _ld, _pnd);

    // 3.1 accumulate total length for arcs
    int n_arcs = _lsd.l_arc_ids_.max();
    std::vector<double> arc_lengths(n_arcs + 1, 0);
    for (auto l_eh : _ld.mesh_->edges())
    {
        auto arc_id = _lsd.l_arc_ids_[l_eh];
        arc_lengths[arc_id] += l_embedded_length_[l_eh];
    }
    // 3.2 transport to arc mesh
    _lsd.arc_lengths_ = _lsd.arc_mesh_->edges().make_attribute(0.);
    for (auto arc_edge : _lsd.arc_mesh_->edges())
    {
        auto arc_id = _lsd.arc_ids_[arc_edge];
        _lsd.arc_lengths_[arc_edge] = arc_lengths[arc_id];
    }

    {
        auto g = gv::grid();
        ColorGenerator cg;
        auto colors = cg.generate_next_colors(_lsd.arc_ids_.max() + 1);
        {
            auto v = gv::view();
            auto c = gv::canvas();
            for (auto eh : _ld.mesh_->edges())
                c.add_line(eh, _ld.pos_, colors[_lsd.l_arc_ids_[eh]]).size(7);

            view_mesh(_ld.pos_, vo_layout);
        }
        {
            auto v = gv::view();
            auto c = gv::canvas();
            for (auto eh : _lsd.arc_mesh_->edges())
                c.add_line(eh, _lsd.arc_pos_, colors[_lsd.arc_ids_[eh]]).size(7);
            view_mesh(_lsd.arc_pos_, vo_layout);
        }
    }
}

void init_map_arc_to_overlay(LoopSubdivData& _lsd, LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    pm::halfedge_attribute<bool> visited = _lsd.arc_mesh_->halfedges().make_attribute(false); // map to_overlay
    _lsd.map_arc_to_overlay_ = _lsd.arc_mesh_->halfedges().make_attribute<std::vector<VH>>();
    for (auto l_heh : _ld.mesh_->halfedges())
    {
        if (l_heh.vertex_from().outgoing_halfedges().size() == 2)
            continue; // inner heh

        auto arc_heh
            = _lsd.arc_mesh_->halfedges()
                  .filter(
                      [&_lsd, &l_heh](HEH heh)
                      { return _lsd.arc_ids_[heh] == _lsd.l_arc_ids_[l_heh] && _lsd.map_a_vh_to_layout_[heh.vertex_from()] == l_heh.vertex_from(); })
                  .first();

        if (visited[arc_heh])
            continue; // already encountered

        visited[arc_heh] = true;

        auto& map_arc_to_overlay = _lsd.map_arc_to_overlay_[arc_heh];

        // auto cd = gv::canvas_data();


        auto pn_vertex_from = _pnd.mesh_->handle_of(l_heh.vertex_from()); // there is a one to one map from layout to pn
        HEH pn_iter_heh
            = pn_vertex_from.outgoing_halfedges().filter([&](HEH pn_heh) { return _pnd.map_to_layout_edges_[pn_heh] == l_heh.edge().idx; }).first();
        assert(pn_iter_heh.is_valid());

        do
        {
            map_arc_to_overlay.push_back(_omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_iter_heh.vertex_from()]));
            // cd.add_point(_omd.pos_[map_arc_to_overlay.back()], RED).size(10);

            pn_iter_heh = pn_iter_heh.next();
        } while (pn_iter_heh.vertex_from().adjacent_vertices().size() == 2);

        map_arc_to_overlay.push_back(_omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_iter_heh.vertex_from()]));
        // cd.add_point(_omd.pos_[map_arc_to_overlay.back()], RED).size(10);

        // {
        //     auto c = gv::canvas();
        //     c.add_faces(_omd.pos_, GREEN_25);
        //     c.add_lines(_omd.pos_, GREEN);
        //     c.add_data(cd);
        // }
    }
}


}
