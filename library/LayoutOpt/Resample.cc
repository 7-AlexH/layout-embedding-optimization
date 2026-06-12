#include "Resample.hh"
#include <glow-extras/viewer/canvas.hh>

#include "LayoutOpt/Embedding.hh"
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/GeomUtils.hh"
#include "LayoutOpt/IO.hh"
#include "LayoutOpt/Visualization/Colors.hh"
#include "LayoutOpt/Visualization/Viewing.hh"
namespace LayoutOpt
{

void init_resample_data(ResampleData& _rd, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd)
{
    // 1. compute arc ids
    _rd.l_arc_ids_ = compute_layout_edge_arc_idx(_ld);

    // 2. compute lengths per layout edge
    _rd.l_embedded_length_ = compute_embedded_length_per_layout_edge(_tmd, _ld, _pnd);

    DEBUG_VAR(_rd.l_embedded_length_.min());

    // 3. accumulate total length for arcs
    int n_arcs = _rd.l_arc_ids_.max();
    _rd.arc_lengths_.resize(n_arcs + 1, 0);
    for (auto l_eh : _ld.mesh_->edges())
    {
        auto arc_id = _rd.l_arc_ids_[l_eh];
        _rd.arc_lengths_[arc_id] += _rd.l_embedded_length_[l_eh];
    }
}

void resample_arc(int _arc_id, ResampleData const& _rd, TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, double _target_edge_length)
{
    // 1. accumulate total length for this arc
    double accumulated_length = _rd.arc_lengths_[_arc_id];

    // 2. compute optimal number of segments
    int num_target_segments = std::max(2, (int)std::lround(accumulated_length / _target_edge_length)); // TODO: make the min number input
    resample_arc(_arc_id, _rd, _tmd, _ld, _pnd, num_target_segments);
}


void resample_layout(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, double _target_edge_length)
{
    ResampleData rd;
    init_resample_data(rd, _tmd, _ld, _pnd);

    auto n_arcs = rd.l_arc_ids_.max() + 1;

    for (int arc_id = 0; arc_id < n_arcs; ++arc_id)
    {
        resample_arc(arc_id, rd, _tmd, _ld, _pnd, _target_edge_length);
    }
    _ld.mesh_->compactify();

    // invalidate all data
    reset_embedding_data(_ld, _pnd, _omd);
    _pnd.sp_on_target_.reset();

    assert(!_ld.mesh_->faces().any([](FH fh) { return fh.is_boundary(); }));
}

void resample_layout(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, int _num_of_segments)
{
    ResampleData rd;
    init_resample_data(rd, _tmd, _ld, _pnd);

    auto n_arcs = rd.l_arc_ids_.max() + 1;

    for (int arc_id = 0; arc_id < n_arcs; ++arc_id)
    {
        resample_arc(arc_id, rd, _tmd, _ld, _pnd, _num_of_segments);
    }
    _ld.mesh_->compactify();

    // invalidate all data
    reset_embedding_data(_ld, _pnd, _omd);
    _pnd.sp_on_target_.reset();

    assert(!_ld.mesh_->faces().any([](FH fh) { return fh.is_boundary(); }));
}

void resample_arc(int _arc_id, ResampleData const& _rd, TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, int _num_of_segments)
{
    // find heh where from vertex is singularity and the edge is labeled as this arc
    HEH l_heh_start = _ld.mesh_->halfedges()
                          .filter([&](HEH l_heh) { return l_heh.vertex_from().adjacent_vertices().size() > 2 && _rd.l_arc_ids_[l_heh] == _arc_id; })
                          .first();
    EH l_eh_start = l_heh_start.edge();

    // 1. accumulate total length for this arc
    double accumulated_length = _rd.arc_lengths_[_arc_id];

    // 2. compute the length of a segment for this arc
    double segment_length = accumulated_length / _num_of_segments;

    // 3. find all l_hehs of this arc
    std::vector<HEH> l_hehs;
    l_hehs.reserve(_ld.mesh_->halfedges().size());

    HEH l_heh_iter = l_heh_start;
    l_hehs.push_back(l_heh_iter);

    while (l_heh_iter.vertex_to().adjacent_vertices().size() == 2)
    {
        l_heh_iter = l_heh_iter.next();

        l_hehs.push_back(l_heh_iter);
    }

    // 4. redistribute surface points
    std::vector<pos3> new_positions;
    new_positions.reserve(_num_of_segments - 1);

    std::vector<HEH> pn_hehs;
    pn_hehs.reserve(_pnd.mesh_->halfedges().size());

    double current_length = 0;

    // 4.1 find start on pn
    auto pn_vertex_from = _pnd.mesh_->handle_of(l_heh_start.vertex_from());
    HEH pn_iter_heh = pn_vertex_from.outgoing_halfedges().filter([&](HEH pn_heh) { return _pnd.map_to_layout_edges_[pn_heh] == l_eh_start.idx; }).first();
    assert(pn_iter_heh.is_valid());

    pn_hehs.push_back(pn_iter_heh);

    vec3d posA = _pnd.sp_on_target_.value()[pn_iter_heh.vertex_from()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
    vec3d posB = _pnd.sp_on_target_.value()[pn_iter_heh.vertex_to()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
    current_length += (posB - posA).norm();

    for (int i = 0; i < _num_of_segments - 1; ++i)
    {
        while (current_length < segment_length)
        {
            assert(pn_iter_heh.vertex_to().adjacent_vertices().size() == 2);
            pn_iter_heh = pn_iter_heh.next();
            pn_hehs.push_back(pn_iter_heh);
            vec3d const posA = _pnd.sp_on_target_.value()[pn_iter_heh.vertex_from()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
            vec3d const posB = _pnd.sp_on_target_.value()[pn_iter_heh.vertex_to()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());

            current_length += (posB - posA).norm();
        }

        // 4.2. new point on this edge
        posA = _pnd.sp_on_target_.value()[pn_iter_heh.vertex_from()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());
        posB = _pnd.sp_on_target_.value()[pn_iter_heh.vertex_to()].get_pos(_tmd.pos_mat_, *_tmd.mesh_.get());

        double const edge_length = (posB - posA).norm();
        double const remaining_length = current_length - segment_length; // current_length is more than segment length
        double alpha = remaining_length / edge_length;

        vec3d const pos_new = alpha * posA + (1.0 - alpha) * posB;
        new_positions.push_back(eigen_to_pos3(pos_new));
        current_length = remaining_length;
    }

    // 5. delete old l_ehs except for first.
    for (int i = 1; i < l_hehs.size(); ++i)
    {
        auto l_heh = l_hehs[i];
        _ld.mesh_->halfedges().collapse(l_heh);
    }

    // 6. add newly computed positons
    l_heh_iter = l_hehs.front();
    for (auto new_pos : new_positions)
    {
        auto vh_new = _ld.mesh_->halfedges().split(l_heh_iter);
        _ld.pos_[vh_new] = new_pos;
        l_heh_iter = l_heh_iter.next();
    }
}

bool down_sample(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, double _min_edge_length)
{
    _ld.mesh_->assert_consistency();

    ResampleData rd;
    init_resample_data(rd, _tmd, _ld, _pnd);

    bool removed_edge = true;
    // 1. remove edges that are too small
    removed_edge = false;
    for (auto l_eh : _ld.mesh_->edges())
    {
        // if both vertices are singuarities, don't collapse
        if (l_eh.vertexA().adjacent_vertices().size() > 2 && l_eh.vertexB().adjacent_vertices().size() > 2)
            continue;

        // if there is a singularity, make it the to-vertex
        HEH l_heh = l_eh.halfedgeA().vertex_to().adjacent_vertices().size() > 2 ? l_eh.halfedgeA() : l_eh.halfedgeB();
        if (rd.l_embedded_length_[l_heh] < _min_edge_length)
        {
            removed_edge = true;
            _ld.mesh_->halfedges().collapse(l_heh);
            _ld.mesh_->assert_consistency();
        }
    }
    if (removed_edge == false)
        return removed_edge;

    _ld.mesh_->compactify();
    // 2. invalidate all data
    reset_embedding_data(_ld, _pnd, _omd);
    _pnd.sp_on_target_.reset();

    _ld.mesh_->assert_consistency();
    return removed_edge;
}


} // namespace LayoutOpt
