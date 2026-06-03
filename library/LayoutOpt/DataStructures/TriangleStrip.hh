#pragma once
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/Types.hh"
namespace LayoutOpt
{

struct TriangleStrip
{
    TriangleStrip() = default;
    // delete copy operations
    TriangleStrip(TriangleStrip const&) = delete;
    TriangleStrip& operator=(TriangleStrip const&) = delete;
    // allow move
    TriangleStrip(TriangleStrip&&) = default;
    TriangleStrip& operator=(TriangleStrip&&) = default;

    EH l_eh;
    std::vector<VH> pn_vhs;

    pm::halfedge_attribute<torch::Tensor> heh_pos_2d; // position is always associated with vertex_from

    void set_vertex_pos(VH _t_vh, torch::Tensor _pos)
    {
        for (auto t_heh_outgoing : _t_vh.outgoing_halfedges())
        {
            heh_pos_2d[t_heh_outgoing] = _pos;
        }
    }

    void set_edge_pos(HEH _t_heh, torch::Tensor _pos_from, torch::Tensor _pos_to)
    {
        heh_pos_2d[_t_heh] = _pos_from;
        heh_pos_2d[_t_heh.next()] = _pos_to;

        auto t_heh_opp = _t_heh.opposite();
        heh_pos_2d[t_heh_opp] = _pos_to;
        heh_pos_2d[t_heh_opp.next()] = _pos_from;
    }

    void set_edge_pos_by_copy_from_opposite(HEH _t_heh)
    {
        auto heh_opp = _t_heh.opposite();
        auto pos_from = heh_pos_2d[heh_opp];
        auto pos_to = heh_pos_2d[heh_opp.next()];

        // only set if not set already and if the value is defined
        if (pos_to.defined() && !heh_pos_2d[_t_heh].defined())
            heh_pos_2d[_t_heh] = pos_to;

        if (pos_from.defined() && !heh_pos_2d[_t_heh.next()].defined())
            heh_pos_2d[_t_heh.next()] = pos_from;
    }

    void set_half_edge_by_copy_from_prev_opp(HEH _t_heh) { heh_pos_2d[_t_heh] = heh_pos_2d[_t_heh.prev().opposite()]; }

    std::vector<FH> target_fhs;

    double embedded_length = 0.0;
};


} // namespace LayoutOpt
