#include "Utils.hh"
#include "polymesh/properties.hh"
namespace LayoutOpt
{

pm::vertex_attribute<pos3> project_from_to(pm::vertex_attribute<pos3> const& _from_pos, pm::vertex_attribute<pos3> const& _to_pos)
{
    return _from_pos.map([&](pos3 pos) { return project_to_faces(pos, _to_pos).first; });
}

std::pair<pos3, FH> project_to_faces(pos3 _p, pm::vertex_attribute<pos3> const& _t_pos)
{
    FH best_t_fh = FH::invalid;
    pos3 best_pos;
    double best_sqr_dist = std::numeric_limits<double>::max();

    for (auto t_fh : _t_pos.mesh().faces())
    {
        tg::dtriangle3 t(t_fh.vertices().to_array<3>(_t_pos));
        auto p_proj = tg::project(_p, t);

        auto current_sqr_dist = tg::length_sqr(p_proj - _p);
        if (current_sqr_dist < best_sqr_dist)
        {
            best_t_fh = t_fh;
            best_pos = p_proj;
            best_sqr_dist = current_sqr_dist;
        }
    }

    return std::make_pair(best_pos, best_t_fh);
}

std::pair<pos3, EH> project_to_edges(pos3 _p, pm::vertex_attribute<pos3> const& _t_pos)
{
    return project_to_edges(_p, _t_pos, _t_pos.mesh().edges().to_vector());
}


std::pair<pos3, EH> project_to_edges(pos3 _p, pm::vertex_attribute<pos3> const& _t_pos, std::vector<EH> const& _candidat_edges)
{
    EH best_t_eh = EH::invalid;
    pos3 best_pos;
    double best_sqr_dist = std::numeric_limits<double>::max();

    for (auto e : _candidat_edges)
    {
        tg::dsegment3 seg(_t_pos[e.vertexA()], _t_pos[e.vertexB()]);
        auto p_proj = tg::project(_p, seg);

        auto current_sqr_dist = tg::length_sqr(p_proj - _p);
        if (current_sqr_dist < best_sqr_dist)
        {
            best_t_eh = e;
            best_pos = p_proj;
            best_sqr_dist = current_sqr_dist;
        }
    }
    return std::make_pair(best_pos, best_t_eh);
}

BarycentricCoordinates compute_barycentric_coordinates(pos3 _pos, HEH _heh, pm::vertex_attribute<pos3> _positions)
{
    auto const& p0 = _positions[_heh.vertex_from()];
    auto const& p1 = _positions[_heh.vertex_to()];
    auto const& p2 = _positions[_heh.next().vertex_to()];

    auto bary_coord_transform = tg::to_barycoord_matrix_of(tg::dtriangle3(p0, p1, p2));
    auto res = bary_coord_transform * tg::dvec4(_pos.x, _pos.y, _pos.z, 1.0);
    return {res.x, res.y};
}

FH shared_face(SurfacePoint const& _spA, SurfacePoint const& _spB, pm::Mesh const& _m)
{
    if (_spA.type == SurfacePointType::VertexPoint && _spB.type == SurfacePointType::VertexPoint)
        return FH::invalid; // in this case, the face is not unique

    if (_spA.type == SurfacePointType::FacePoint)
    {
        auto fh = _spA.fh(_m);
        return fh;
    }

    if (_spB.type == SurfacePointType::FacePoint)
    {
        auto fh = _spB.fh(_m);
        return fh;
    }

    if (_spA.type == SurfacePointType::EdgePoint && _spB.type == SurfacePointType::EdgePoint)
    {
        auto hhA = _m.handle_of(_spA.heh_idx);
        auto hhB = _m.handle_of(_spB.heh_idx);
        HEH hh = HEH::invalid;
        if (hhA.face() == hhB.face() || hhA.face() == hhB.opposite_face())
            hh = hhA;
        if (hh.is_invalid() && (hhA.opposite().face() == hhB.face() || hhA.opposite().face() == hhB.opposite_face()))
            hh = hhA.opposite();
        return hh.face();
    }

    // one is a vertex point the other one an edge point
    VH vh;
    HEH hh;
    if (_spA.type == SurfacePointType::VertexPoint && _spB.type == SurfacePointType::EdgePoint)
    {
        vh = _m.handle_of(_spA.heh_idx).vertex_from();
        hh = _m.handle_of(_spB.heh_idx);
    }
    else if (_spA.type == SurfacePointType::EdgePoint && _spB.type == SurfacePointType::VertexPoint)
    {
        vh = _m.handle_of(_spB.heh_idx).vertex_from();
        hh = _m.handle_of(_spA.heh_idx);
    }

    auto filtered_range = vh.faces().filter([&hh](FH fh) { return hh.face() == fh || hh.opposite_face() == fh; }).to_vector();

    if (filtered_range.size() > 0)
        return filtered_range.front();
    else
        return FH::invalid;
}

HEH shared_halfedge(SurfacePoint const& _sp_from, SurfacePoint const& _sp_to, polymesh::Mesh const& _m)
{
    if (_sp_from.type != SurfacePointType::VertexPoint)
        return HEH::invalid;
    if (_sp_to.type != SurfacePointType::VertexPoint)
        return HEH::invalid;

    auto vh_from = _m.handle_of(_sp_from.heh_idx).vertex_from();
    auto vh_to = _m.handle_of(_sp_to.heh_idx).vertex_from();
    return pm::halfedge_from_to(vh_from, vh_to);
}

bool is_convex_corner_3d(pos3 const& a, pos3 const& b, pos3 const& c, vec3 const& face_normal)
{
    // Compute edge vectors from the corner vertex
    // a - b points from b to a, c - b points from b to c
    vec3 ab = b - a;
    vec3 bc = c - b;

    // Cross product gives orientation of the corner in 3D
    vec3 cross = tg::cross(ab, bc);

    // If cross aligns with face normal, the corner is convex
    return tg::dot(cross, face_normal) > 0;
}

std::pair<bool, float> corner_convexity_and_angle_3d(pos3 const& a, pos3 const& b, pos3 const& c, vec3 const& face_normal)
{
    // Edge vectors from corner vertex b
    vec3 ba = a - b;
    vec3 bc = c - b;

    // Unsigned angle in [0, π]
    auto angle = tg::angle_between(ba, bc);

    // Cross product determines convex vs concave
    vec3 cross = tg::cross(-ba, bc);
    bool is_convex = tg::dot(cross, face_normal) > 0;

    return {is_convex, angle.radians()};
}

double angle_sum(HEH _heh_start, HEH _heh_end, pm::vertex_attribute<pos3> const& _pos)
{
    assert(_heh_start.vertex_from() == _heh_end.vertex_from());
    double angle_sum = 0;

    if (_heh_start == _heh_end)
        return angle_sum;

    HEH heh_iter = _heh_start;
    do
    {
        double angle = pm::angle_to_prev(heh_iter, _pos);
        angle_sum += angle;
        heh_iter = heh_iter.next().next().opposite();

    } while (heh_iter != _heh_end);
    return angle_sum;
}

BarycentricCoordinates stabilize(BarycentricCoordinates const& _bc, double _eps)
{
    // Collect values with their original indices
    std::array<std::pair<double, int>, 3> v = {{{_bc.alpha, 0}, {_bc.beta, 1}, {_bc.gamma(), 2}}};

    // Sort by value
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first < b.first; });

    // Clamp smallest and largest
    v[0].first = std::max(v[0].first, _eps);
    v[2].first = std::min(v[2].first, 1.0 - _eps);

    // Middle = remainder so sum=1
    v[1].first = 1.0 - (v[0].first + v[2].first);

    // --- Fix if middle falls outside [_eps, 1-_eps] ---
    if (v[1].first < _eps)
    {
        double deficit = _eps - v[1].first;
        v[1].first = _eps;
        v[2].first -= deficit; // take from largest (always >= _eps already)
        if (v[2].first < _eps)
            v[2].first = _eps; // safeguard
    }
    else if (v[1].first > 1.0 - _eps)
    {
        double excess = v[1].first - (1.0 - _eps);
        v[1].first = 1.0 - _eps;
        v[0].first -= excess; // take from smallest
        if (v[0].first < _eps)
            v[0].first = _eps; // safeguard
    }

    // Renormalize once at the end
    double sum = v[0].first + v[1].first + v[2].first;
    for (auto& p : v)
        p.first /= sum;

    // Restore original order
    std::array<double, 3> result;
    for (auto& p : v)
        result[p.second] = p.first;

    return {result[0], result[1]};
}
} // namespace LayoutOpt
