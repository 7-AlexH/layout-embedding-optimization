#include "QuadMeshingUtils.hh"
#include "polymesh/properties.hh"

namespace LayoutOpt
{
pm::face_attribute<double> max_inner_angle(pm::vertex_attribute<pos3> _qm_pos)
{
    auto max_inner_angle_attr = _qm_pos.mesh().faces().make_attribute(0.0);
    for(auto fh: _qm_pos.mesh().faces())
    {
        double max_angle = 0.0;

        for(auto heh: fh.halfedges())
        {
            auto v1 = pm::edge_vector(heh.opposite(), _qm_pos);
            auto v2 = pm::edge_vector(heh.next(), _qm_pos);

            auto angle = tg::angle_between(v1, v2);
            max_angle = tg::max(max_angle, angle.radians());
        }
        max_inner_angle_attr[fh] = max_angle;
    }
    return max_inner_angle_attr;
}

pm::face_attribute<double> scaled_jacobian(pm::vertex_attribute<pos3> _qm_pos)
{
    auto sj_attr = _qm_pos.mesh().faces().make_attribute(0.0);

    for (auto fh : _qm_pos.mesh().faces())
    {
        auto quad_pos = fh.vertices().to_array<4>(_qm_pos);

        double min_sj = std::numeric_limits<double>::max();

        for(auto heh: fh.halfedges())
        {
            auto v1 = pm::edge_vector(heh.opposite(), _qm_pos);
            auto v2 = pm::edge_vector(heh.next(), _qm_pos);

            auto e1 = tg::normalize_safe(v1);
            auto e2 = tg::normalize_safe(v2);

            auto n = tg::cross(e1, e2);

            auto signed_sj = tg::dot(tg::cross(v1, v2), n)/(tg::length(v1) * tg::length(v2));
            min_sj = tg::min(min_sj, signed_sj);

        }
        sj_attr[fh] = min_sj;
    }
    return sj_attr;
}

}
