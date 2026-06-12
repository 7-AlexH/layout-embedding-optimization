#pragma once
#include <optional>
#include <vector>
#include <polymesh/Mesh.hh>
#include <polymesh/attributes.hh>
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

    // position is always associated with vertex_from; nullopt = not set
    pm::halfedge_attribute<std::optional<vec2d>> heh_pos_2d;

    void set_vertex_pos(VH _t_vh, vec2d const& _pos)
    {
        for (auto t_heh_outgoing : _t_vh.outgoing_halfedges())
        {
            heh_pos_2d[t_heh_outgoing] = _pos;
        }
    }

    void set_edge_pos(HEH _t_heh, vec2d const& _pos_from, vec2d const& _pos_to)
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
        if (pos_to.has_value() && !heh_pos_2d[_t_heh].has_value())
            heh_pos_2d[_t_heh] = pos_to;

        if (pos_from.has_value() && !heh_pos_2d[_t_heh.next()].has_value())
            heh_pos_2d[_t_heh.next()] = pos_from;
    }

    void set_half_edge_by_copy_from_prev_opp(HEH _t_heh) { heh_pos_2d[_t_heh] = heh_pos_2d[_t_heh.prev().opposite()]; }

    std::vector<FH> target_fhs;

    double embedded_length = 0.0;
};


} // namespace LayoutOpt
