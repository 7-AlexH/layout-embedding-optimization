#include "FlattenTriangleStrip.hh"
#include "LayoutOpt/GeomUtils.hh"

#include <glow-extras/viewer/canvas.hh>
#include "LayoutOpt/Utils.hh"
#include "LayoutOpt/Visualization/Colors.hh"
#include "LayoutOpt/Visualization/Viewing.hh"
#include "Visualization/ColorGenerator.hh"

namespace LayoutOpt
{

long g_vertex_sp_conversion_count = 0;

namespace
{
// remove-autodiff Phase 5c: strip 2D positions are plain doubles (std::optional<vec2d>);
// nullopt replaces the undefined-tensor sentinel. The strip is constant w.r.t. the
// autograd graph (Phase 2 audit), so nothing here needs to be differentiable.

pos3 pos3_of_2D(vec2d const& _v) { return pos3(_v.x(), _v.y(), 0.0); }

// DEBUG
auto cd_2D = gv::canvas_data();
auto cd_3D = gv::canvas_data();
void init_canvas()
{
    cd_2D.clear();
    cd_2D.add_point(tg::pos3::zero, BLACK).size(5);
    cd_2D.add_line(tg::pos3::zero, 5 * tg::vec3::unit_x, BLACK).size(0.5);
    cd_2D.add_line(tg::pos3::zero, tg::vec3::unit_y, BLACK).size(0.5);
    cd_2D.add_line(tg::pos3::zero, tg::vec3::unit_z, BLACK).size(0.5);

    cd_3D.clear();
}

void init_3D_view(TargetMeshData const& _tmd) { cd_3D.add_lines(_tmd.pos_, MAGENTA).size(2); }

void init_canvas_current_strip_based_on_fhs(TriangleStrip const& _strip)
{
    ColorGenerator cg;
    for (auto fh : _strip.target_fhs)
    {
        std::vector<pos3> positions;
        for (auto heh : fh.halfedges())
        {
            auto pos = pos3_of_2D(_strip.heh_pos_2d[heh].value());
            positions.push_back(pos);
        }
        auto color = cg.generate_next_color();
        cd_2D.add_face(positions[0], positions[1], positions[2], color);
    }
}

void init_canvas_current_2D_pos(TriangleStrip const& _strip)
{
    ColorGenerator cg;
    for (auto fh : _strip.heh_pos_2d.mesh().faces())
    {
        std::vector<pos3> positions;
        bool can_view = true;
        for (auto heh : fh.halfedges())
        {
            if (!_strip.heh_pos_2d[heh].has_value())
            {
                can_view = false;
                break;
            }
            auto pos = pos3_of_2D(_strip.heh_pos_2d[heh].value());
            positions.push_back(pos);
        }

        if (!can_view)
            continue;

        auto color = cg.generate_next_color();
        cd_2D.add_face(positions[0], positions[1], positions[2], color);
    }

    for (auto eh : _strip.heh_pos_2d.mesh().edges())
    {
        bool can_view = true;
        std::vector<pos3> positions;
        if (_strip.heh_pos_2d[eh.halfedgeA()].has_value() && _strip.heh_pos_2d[eh.halfedgeB().next()].has_value())
        {
            auto pos = pos3_of_2D(_strip.heh_pos_2d[eh.halfedgeA()].value());
            positions.push_back(pos);
            cd_2D.add_point(pos, BLUE).size(7);
        }
        else
        {
            can_view = false;
        }
        if (_strip.heh_pos_2d[eh.halfedgeB()].has_value() && _strip.heh_pos_2d[eh.halfedgeA().next()].has_value())
        {
            auto pos = pos3_of_2D(_strip.heh_pos_2d[eh.halfedgeB()].value());
            positions.push_back(pos);
            cd_2D.add_point(pos, BLUE).size(7);
        }
        else
        {
            can_view = false;
        }

        if (!can_view)
            continue;

        auto color = cg.generate_next_color();
        cd_2D.add_line(positions[0], positions[1], color).size(5);
    }
}

// END

enum class PointSide
{
    Undefined,
    Above,
    Below
};

double compute_strip_segment_length(TriangleStrip const& _strip, size_t _idx_A, size_t _idx_B, TargetMeshData const& _tmd, PathNetworkData const& _pnd)
{
    auto pn_A = _strip.pn_vhs[_idx_A];
    auto pn_B = _strip.pn_vhs[_idx_B];

    auto const& sp_A = _pnd.sp_on_target_.value()[pn_A];
    auto const& sp_B = _pnd.sp_on_target_.value()[pn_B];

    vec3d const pos_A = sp_A.get_pos(_tmd.pos_mat_, *_tmd.mesh_);
    vec3d const pos_B = sp_B.get_pos(_tmd.pos_mat_, *_tmd.mesh_);

    return (pos_B - pos_A).norm();
}

void set_triangle_pos(TriangleStrip& _strip,
                      size_t _idx_O,
                      size_t _idx_X,
                      bool positive,
                      TargetMeshData const& _tmd,
                      PathNetworkData const& _pnd,
                      vec2d _displacement = vec2d::Zero())
{
    auto pn_O = _strip.pn_vhs[_idx_O];
    auto pn_X = _strip.pn_vhs[_idx_X];

    auto t_heh_O = _tmd.mesh_->handle_of(_pnd.sp_on_target_.value()[pn_O].heh_idx);

    vec3d const pos_O = _pnd.sp_on_target_.value()[pn_O].get_pos(_tmd.pos_mat_, *_tmd.mesh_);
    vec3d const pos_X = _pnd.sp_on_target_.value()[pn_X].get_pos(_tmd.pos_mat_, *_tmd.mesh_);

    auto face_2D_coords = compute_2D_face_embedding(t_heh_O, pos_O, pos_X, positive, _tmd.pos_mat_);

    auto t_heh_iter = t_heh_O;

    // set face
    auto const& sp_X = _pnd.sp_on_target_.value()[pn_X];
    for (size_t i = 0; i < 3; ++i)
    {
        _strip.heh_pos_2d[t_heh_iter] = vec2d(face_2D_coords.row(i).transpose()) + _displacement;
        // special handeling for vertex point
        if (sp_X.is_vertex_sp() && t_heh_iter.vertex_from() == _tmd.mesh_->handle_of(sp_X.heh_idx).vertex_from())
        {
            _strip.set_vertex_pos(t_heh_iter.vertex_from(), _strip.heh_pos_2d[t_heh_iter].value());
            // {
            //     DEBUG_OUT("here")
            //     auto c = gv::canvas();
            //     init_canvas_current_2D_pos(_strip);
            //     c.add_data(cd_2D);
            // }
        }
        t_heh_iter = t_heh_iter.next();
    }
}

void puzzle_sp_prev_vertex(SurfacePoint const& _sp_prev, SurfacePoint const& _sp_curr, TriangleStrip& _strip, TargetMeshData const& _tmd)
{
    auto t_vh = _tmd.mesh_->handle_of(_sp_prev.heh_idx).vertex_from();
    auto t_hh = _tmd.mesh_->handle_of(_sp_curr.heh_idx);

    HEH t_hh_iter = t_hh.face().halfedges().filter([t_vh](auto heh) { return heh.vertex_from() == t_vh; }).first();
    if (t_hh_iter.is_invalid())
        t_hh_iter = t_hh.opposite_face().halfedges().filter([t_vh](auto heh) { return heh.vertex_from() == t_vh; }).first();
    assert(t_hh_iter.is_valid() && t_hh_iter.vertex_from() == t_vh);

    vec3d const A = _tmd.pos_mat_.row(t_hh_iter.vertex_from().idx.value).transpose();
    assert(_strip.heh_pos_2d[t_hh_iter].has_value());
    vec2d const A_2D = _strip.heh_pos_2d[t_hh_iter].value();

    vec3d const B = _tmd.pos_mat_.row(t_hh_iter.vertex_to().idx.value).transpose();
    vec3d const C = _tmd.pos_mat_.row(t_hh_iter.next().vertex_to().idx.value).transpose();

    vec3d const& O = A;
    vec3d const X = _sp_curr.get_pos(_tmd.pos_mat_, *_tmd.mesh_);

    vec3d const AB = B - A;
    vec3d const AC = C - A;

    vec3d const normal = normalized_eps(AB.cross(AC));

    vec3d const OB = B - O;
    vec3d const OC = C - O;
    vec3d const OX = X - O;

    double const length_OB = OB.norm();
    double const length_OC = OC.norm();

    vec3d const OB_normalized = normalized_eps(OB);
    vec3d const OC_normalized = normalized_eps(OC);
    vec3d const OX_normalized = normalized_eps(OX);

    double const angle_XOB = std::atan2(OX_normalized.cross(OB_normalized).dot(normal), OB_normalized.dot(OX_normalized));
    double const angle_XOC = std::atan2(OX_normalized.cross(OC_normalized).dot(normal), OC_normalized.dot(OX_normalized));

    vec2d const OB_2D = A_2D + length_OB * vec2d(std::cos(angle_XOB), std::sin(angle_XOB));
    vec2d const OC_2D = A_2D + length_OC * vec2d(std::cos(angle_XOC), std::sin(angle_XOC));

    _strip.set_edge_pos(t_hh_iter.next(), OB_2D, OC_2D);
    _strip.heh_pos_2d[t_hh_iter.next().next()] = OC_2D;

    cd_2D.add_point(pos3_of_2D(OB_2D), GREEN).size(9);
    cd_2D.add_point(pos3_of_2D(OC_2D), GREEN).size(9);
}

void puzzle_face(HEH _t_hh_iter, TriangleStrip& _strip, TargetMeshData const& _tmd)
{
    cd_3D.add_line(_t_hh_iter, _tmd.pos_, GREEN).size(4);

    // copy to other side if not already done
    _strip.set_edge_pos_by_copy_from_opposite(_t_hh_iter);

    vec3d const A = _tmd.pos_mat_.row(_t_hh_iter.vertex_from().idx.value).transpose();
    vec2d const A_2D = _strip.heh_pos_2d[_t_hh_iter].value();

    vec3d const B = _tmd.pos_mat_.row(_t_hh_iter.vertex_to().idx.value).transpose();
    vec2d const B_2D = _strip.heh_pos_2d[_t_hh_iter.next()].value();

    HEH hehC = _t_hh_iter.next().next();
    VH vhC = hehC.vertex_from();
    vec3d const C = _tmd.pos_mat_.row(vhC.idx.value).transpose();

    vec3d const AB = B - A;
    vec3d const AC = C - A;

    vec3d const normal = normalized_eps(AB.cross(AC));

    vec3d const AB_normalized = normalized_eps(AB);
    vec3d const height_vector = normal.cross(AB_normalized);
    vec3d const height_vector_normalized = normalized_eps(height_vector);

    vec3d const BC = C - B;
    double const height = BC.dot(height_vector_normalized);

    vec3d const P = C - height * height_vector_normalized;
    vec3d const AP = P - A;
    double const length_AP = AP.dot(AB_normalized);

    vec2d const AB_2D = B_2D - A_2D;
    vec2d const AB_2D_normalized = normalized_eps(AB_2D);
    vec2d const height_vector_normalized_2D(-AB_2D_normalized.y(), AB_2D_normalized.x());

    vec2d const C_2D = A_2D + length_AP * AB_2D_normalized + height * height_vector_normalized_2D;

    _strip.heh_pos_2d[hehC] = C_2D;
}

void puzzle_face(SurfacePoint const& _sp_prev, SurfacePoint const& _sp_curr, TriangleStrip& _strip, TargetMeshData const& _tmd)
{
    auto t_hh_prev = _tmd.mesh_->handle_of(_sp_prev.heh_idx);
    auto t_hh_curr = _tmd.mesh_->handle_of(_sp_curr.heh_idx);

    HEH t_hh_iter = HEH::invalid;
    if (t_hh_prev.face() == t_hh_curr.face() || t_hh_prev.face() == t_hh_curr.opposite_face())
        t_hh_iter = t_hh_prev;
    else if (t_hh_prev.opposite().face() == t_hh_curr.face() || t_hh_prev.opposite().face() == t_hh_curr.opposite_face())
        t_hh_iter = t_hh_prev.opposite();

    assert(t_hh_iter.is_valid());

    puzzle_face(t_hh_iter, _strip, _tmd);
}
/// @param _start_heh and _end_heh are outgoing heh of the same vertex
void puzzle_triangle_fan(HEH _start_heh, HEH _end_heh, TriangleStrip& _strip, TargetMeshData const& _tmd)
{
    assert(_start_heh.vertex_from() == _end_heh.vertex_from());

    if (_start_heh == _end_heh)
        return; // nothing to do

    HEH iter_heh = _start_heh;

    // get apex of triangle fan
    auto vh_O = _start_heh.vertex_from();
    vec3d const O = _tmd.pos_mat_.row(_start_heh.vertex_from().idx.value).transpose();
    vec2d const O_2D = _strip.heh_pos_2d[_start_heh].value();

    // total angle around apex from start to end in 3D
    double total_angle_start_end_3D = angle_sum(_start_heh, _end_heh, _tmd.pos_); // radians

    // total angle around apex from start to end in 2D
    vec2d const start_to_2D = _strip.heh_pos_2d[_start_heh.opposite()].value();
    vec2d const end_to_2D = _strip.heh_pos_2d[_end_heh.next()].value();

    double const lenght_O_start_to = (start_to_2D - O_2D).norm();
    double const length_O_end_to = (end_to_2D - O_2D).norm();
    double const length_start_to_end_to = (start_to_2D - end_to_2D).norm();

    // Law of cosines
    double total_angle_start_end_2D_cos = (lenght_O_start_to * lenght_O_start_to + length_O_end_to * length_O_end_to - length_start_to_end_to * length_start_to_end_to)
                                          / (2.0 * lenght_O_start_to * length_O_end_to);
    total_angle_start_end_2D_cos = std::clamp(total_angle_start_end_2D_cos, -1.0, 1.0);
    double total_angle_start_end_2D = std::acos(total_angle_start_end_2D_cos);

    // DEBUG_OUT("===================================================")
    // DEBUG_VAR(total_angle_start_end_3D / M_PI * 180.0)
    // DEBUG_VAR(total_angle_start_end_2D / M_PI * 180.0)

    while (iter_heh != _end_heh)
    {
        auto hehOA = iter_heh;
        auto hehAB = iter_heh.next();
        auto hehOB = iter_heh.next().next().opposite();

        auto vhA = hehAB.vertex_from();
        auto vhB = hehAB.vertex_to();

        vec3d const A = _tmd.pos_mat_.row(vhA.idx.value).transpose(); // 3D pos of A
        vec3d const B = _tmd.pos_mat_.row(vhB.idx.value).transpose(); // 3D pos of B

        // Vector lengths
        double const length_OA = (A - O).norm();
        double const length_OB = (B - O).norm();
        double const length_AB = (A - B).norm();

        // angles
        // Law of cosines for angle at O
        double cosAOB = (length_OA * length_OA + length_OB * length_OB - length_AB * length_AB) / (2.0 * length_OA * length_OB);
        cosAOB = std::clamp(cosAOB, -1.0, 1.0);
        double AOB = std::acos(cosAOB);

        // DEBUG_VAR(AOB / M_PI * 180.0);

        // account for gauss curvature
        auto AOB_2D = total_angle_start_end_2D * (AOB / total_angle_start_end_3D); // frac of available angle

        // 2D pos
        vec2d const A_2D = _strip.heh_pos_2d[hehAB].value();
        vec2d const OA_2D = A_2D - O_2D;
        vec2d const OA_2D_norm = OA_2D / OA_2D.norm();

        // Rotation matrix components
        auto cos_theta = std::cos(AOB_2D);
        auto sin_theta = std::sin(AOB_2D);

        // Assuming CCW rotation, the rotated vector is:
        vec2d OB_2D(cos_theta * OA_2D_norm.x() - sin_theta * OA_2D_norm.y(), //
                    sin_theta * OA_2D_norm.x() + cos_theta * OA_2D_norm.y());
        // Scale by OB length to preserve edge length
        OB_2D = OB_2D * length_OB;

        // Translate relative to O_2D
        vec2d const B_2D = O_2D + OB_2D;

        // Store
        _strip.heh_pos_2d[hehAB.next()] = B_2D;
        _strip.set_edge_pos_by_copy_from_opposite(hehOB);

        cd_2D.add_face(pos3_of_2D(A_2D), pos3_of_2D(B_2D), pos3_of_2D(O_2D), PETROL_25);
        iter_heh = hehOB;
    }
}

void adapt_metric(size_t _idx_current, HEH _t_hh, std::vector<VH> const& _old_pn_vhs, TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData& _pnd)
{
    auto const& pn_vh_prev = _strip.pn_vhs.back();
    auto const& pn_vh_next = _old_pn_vhs[_idx_current + 1];

    auto const& sp_prev = _pnd.sp_on_target_.value()[pn_vh_prev];
    auto const& sp_next = _pnd.sp_on_target_.value()[pn_vh_next];

    // 3. slightly adapt metric
    if (_idx_current == 1 || _idx_current == _old_pn_vhs.size() - 2) // by rotation
    {
        vec2d rotation_center = vec2d::Zero();
        FH f_to_rotate;
        double theta = 0.2 * M_PI / 180.0; // 0.1 degrees in radians

        if (_idx_current == 1)
        {
            f_to_rotate = sp_prev.fh(*_tmd.mesh_.get());
        }
        else if (_idx_current == _old_pn_vhs.size() - 2)
        {
            theta = -theta;
            f_to_rotate = sp_next.fh(*_tmd.mesh_.get());
            rotation_center = rotation_center + _strip.embedded_length * vec2d(1.0, 0.0);
        }

        // --- Rotation matrix ---
        auto c = std::cos(theta);
        auto s = std::sin(theta);
        mat2d R;
        R << c, -s, s, c;

        for (auto t_heh : f_to_rotate.halfedges())
        {
            vec2d const p = _strip.heh_pos_2d[t_heh].value() - rotation_center;
            vec2d const rotated = R * p + rotation_center;
            _strip.heh_pos_2d[t_heh] = rotated;

            if (t_heh.vertex_from() == _t_hh.vertex_from())
            {
                // make consistent with other triangles
                _strip.set_vertex_pos(t_heh.vertex_from(), rotated);
            }
        }
    }
    else // by translation
    {
        // NOTE: the torch version's += was an in-place tensor add; hehs that shared one
        // tensor handle (set_vertex_pos aliases) were shifted once per alias. Value
        // semantics shifts every heh exactly once, which is the intended perturbation.
        vec2d const displacement = LARGE_EPS * vec2d(0.0, 1.0);
        for (auto t_heh_outgoing : _t_hh.vertex_from().outgoing_halfedges())
        {
            _strip.heh_pos_2d[t_heh_outgoing] = _strip.heh_pos_2d[t_heh_outgoing].value() + displacement;
        }
    }
};

void transform_single_v_sp(
    size_t idx_current, std::vector<VH> const& _old_pn_vhs, TriangleStrip& strip, PathNetworkData& pnd, TargetMeshData const& tmd, vec2d const& layout_A, vec2d const& layout_B)
{
    init_canvas();

    auto const& pn_vh_prev = strip.pn_vhs.back();
    auto const& pn_vh_curr = _old_pn_vhs[idx_current];
    auto const& pn_vh_next = _old_pn_vhs[idx_current + 1];

    auto const& sp_prev = pnd.sp_on_target_.value()[pn_vh_prev];
    auto& sp_curr = pnd.sp_on_target_.value()[pn_vh_curr];
    auto const& sp_next = pnd.sp_on_target_.value()[pn_vh_next];

    // 1. find the fan that needs to be unfolded
    // 1.1 find the starting heh
    auto t_face_pc = shared_face(sp_prev, sp_curr, *tmd.mesh_);
    auto t_vh_curr = tmd.mesh_->handle_of(sp_curr.heh_idx).vertex_from();
    HEH t_start_hh = t_face_pc.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_to() == t_vh_curr; }).first().opposite();
    assert(t_start_hh.is_valid());
    strip.heh_pos_2d[t_start_hh] = strip.heh_pos_2d[t_start_hh.opposite().next()];

    // make consistent
    strip.heh_pos_2d[t_start_hh.next()] = strip.heh_pos_2d[t_start_hh.opposite()];

    // 1.2 find final heh
    assert(!sp_next.is_vertex_sp());
    auto t_face_cn = shared_face(sp_curr, sp_next, *tmd.mesh_);
    HEH t_end_hh = t_face_cn.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_from() == t_vh_curr; }).first();
    assert(t_end_hh.is_valid());

    cd_2D.add_line(pos3_of_2D(strip.heh_pos_2d[t_start_hh].value()), pos3_of_2D(strip.heh_pos_2d[t_start_hh.next()].value()), GREEN).size(12);

    if (!strip.heh_pos_2d[t_end_hh].has_value() || !strip.heh_pos_2d[t_end_hh.next()].has_value())
    {
        auto c = gv::canvas_data();
        init_canvas();
        init_canvas_current_2D_pos(strip);
        c.add_data(cd_2D);
    }
    cd_2D.add_line(pos3_of_2D(strip.heh_pos_2d[t_end_hh].value()), pos3_of_2D(strip.heh_pos_2d[t_end_hh.next()].value()), RED).size(10);

    // 2. puzzle triangle fan and make last vertex consistent
    auto final_pos = strip.heh_pos_2d[t_end_hh.next()];
    puzzle_triangle_fan(t_start_hh, t_end_hh, strip, tmd);

    // make sure the last vertex is not touched
    strip.heh_pos_2d[t_end_hh.opposite()] = final_pos;

    // 3. adapt the metric
    adapt_metric(idx_current, t_start_hh, _old_pn_vhs, strip, tmd, pnd);

    if (idx_current == 1)
    {
        strip.heh_pos_2d[t_start_hh.next()] = strip.heh_pos_2d[t_start_hh.opposite()];
    }
    else if (idx_current == _old_pn_vhs.size() - 2)
    {
        strip.heh_pos_2d[t_end_hh.opposite()] = strip.heh_pos_2d[t_end_hh.next()];
    }

    // 4. adapt pnd
    HEH t_iter_heh = t_start_hh;
    auto pn_heh = pm::halfedge_from_to(pn_vh_prev, pn_vh_curr);
    auto pn_to_le_label = pnd.map_to_layout_edges_[pn_heh];

    while (t_iter_heh != t_end_hh.next().next().opposite())
    {
        // gather information if not already there
        strip.set_edge_pos_by_copy_from_opposite(t_iter_heh);
        strip.set_edge_pos_by_copy_from_opposite(t_iter_heh.opposite());

        vec2d const from_2D = strip.heh_pos_2d[t_iter_heh].value();
        vec2d const to_2D = strip.heh_pos_2d[t_iter_heh.next()].value();

        cd_2D.add_line(pos3_of_2D(from_2D), pos3_of_2D(to_2D), MAGENTA).size(5);

        auto params = compute_intersection_parameter(to_2D, from_2D, layout_A, layout_B);

        auto vh_new = pn_heh.vertex_to();
        strip.pn_vhs.push_back(vh_new);

        pnd.sp_on_target_.value()[vh_new]
            = SurfacePoint(vec2d(params.value().y(), 0.0), t_iter_heh.idx, SurfacePointType::EdgePoint);

        // DEBUG
        auto pos = pnd.sp_on_target_.value()[vh_new].get_pos(strip.heh_pos_2d);
        cd_2D.add_point(pos3_of_2D(pos), MAGENTA).size(20);
        //  END

        t_iter_heh = t_iter_heh.next().next().opposite();

        if (t_iter_heh != t_end_hh.next().next().opposite())
        {
            pn_heh = pn_heh.next();
            pnd.mesh_->halfedges().split(pn_heh);
            pnd.map_to_layout_edges_[pn_heh.next().edge()] = pn_to_le_label;
        }
    }
}

// return the last index that was processed
int transform_multiple_v_sp(size_t const idx_current,
                            std::vector<VH> const& _old_pn_vhs,
                            TriangleStrip& strip,
                            PathNetworkData& pnd,
                            TargetMeshData const& tmd,
                            vec2d const& layout_A,
                            vec2d const& layout_B)
{
    auto const& pn_vh_prev = strip.pn_vhs.back();
    auto const& pn_vh_curr = _old_pn_vhs[idx_current];

    auto const& sp_prev = pnd.sp_on_target_.value()[pn_vh_prev];
    auto& sp_curr = pnd.sp_on_target_.value()[pn_vh_curr];

    // 1. find the strip that needs to be unfolded
    // 1.1 find the starting heh
    auto t_face_pc = shared_face(sp_prev, sp_curr, *tmd.mesh_);
    auto t_vh_curr = tmd.mesh_->handle_of(sp_curr.heh_idx).vertex_from();
    HEH t_iter_hh = t_face_pc.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_to() == t_vh_curr; }).first().opposite();
    assert(t_iter_hh.is_valid());

    // 1.2 adapt metric
    adapt_metric(idx_current, t_iter_hh, _old_pn_vhs, strip, tmd, pnd);

    size_t offset = 1;
    auto pn_vh_next = _old_pn_vhs[idx_current + offset];
    auto sp_next = pnd.sp_on_target_.value()[pn_vh_next];

    auto t_vh_next = tmd.mesh_->handle_of(sp_next.heh_idx).vertex_from();
    HEH t_end_hh = HEH::invalid;

    std::vector<HEH> hehs_strip;
    do
    {
        hehs_strip.push_back(t_iter_hh);

        if (idx_current + offset >= _old_pn_vhs.size())
            break; // avoid out-of-bounds

        if (t_iter_hh.next().vertex_to() == t_vh_next)
        {
            // adapt metric
            adapt_metric(idx_current + offset, t_iter_hh.next().opposite(), _old_pn_vhs, strip, tmd, pnd);

            pn_vh_next = _old_pn_vhs[idx_current + offset + 1];
            sp_next = pnd.sp_on_target_.value()[pn_vh_next];

            if (sp_next.is_vertex_sp())
            {
                t_vh_next = tmd.mesh_->handle_of(sp_next.heh_idx).vertex_from();
                offset = offset + 1;
            }
            else
            {
                auto pn_vh = _old_pn_vhs[idx_current + offset];
                auto sp = pnd.sp_on_target_.value()[pn_vh];

                VH t_vh = tmd.mesh_->handle_of(sp.heh_idx).vertex_from();

                auto t_face_cn = shared_face(sp, sp_next, *tmd.mesh_);

                t_end_hh = t_face_cn.halfedges().filter([&](HEH heh) { return heh.vertex_from() == t_vh; }).first();
                assert(t_end_hh.is_valid());
            }
            t_iter_hh = t_iter_hh.next().opposite();
        }
        else
        {
            t_iter_hh = t_iter_hh.next().next().opposite();
        }


    } while (t_end_hh.is_invalid() || t_iter_hh.edge() != t_end_hh.edge());
    hehs_strip.push_back(t_iter_hh);

    // 2. compute embedded total length
    double total_length_3D = 0;
    for (int i = 0; i < hehs_strip.size() - 1; ++i)
    {
        total_length_3D = total_length_3D + tg::length(tmd.pos_[hehs_strip[i].vertex_to()] - tmd.pos_[hehs_strip[i + 1].vertex_to()]);
    }

    // 3. compute available length:
    strip.heh_pos_2d[hehs_strip.front().next()] = strip.heh_pos_2d[hehs_strip.front().opposite()];
    vec2d const pos_from = strip.heh_pos_2d[hehs_strip.front().next()].value();
    vec2d const pos_to = strip.heh_pos_2d[hehs_strip.back().next()].value();

    // 4. distribute along available length and adapt metric
    double accumulated_length = 0;
    for (int i = 1; i < hehs_strip.size(); ++i)
    {
        auto const heh_prev = hehs_strip[i - 1];
        auto const heh_curr = hehs_strip[i];

        if (heh_curr.vertex_to() == heh_prev.vertex_to())
        {
            strip.heh_pos_2d[heh_curr.next()] = strip.heh_pos_2d[heh_prev.next()];
        }
        else
        {
            auto length = tg::length(tmd.pos_[heh_prev.vertex_to()] - tmd.pos_[heh_curr.vertex_to()]);
            accumulated_length += length;
            auto t = 1.0 - (accumulated_length / total_length_3D);
            vec2d const pos = t * pos_from + (1.0 - t) * pos_to;

            strip.heh_pos_2d[heh_curr.opposite()] = pos;
            strip.heh_pos_2d[heh_curr.next()] = pos;
        }
    }

    auto pn_heh = pm::halfedge_from_to(pn_vh_prev, pn_vh_curr);
    auto pn_to_le_label = pnd.map_to_layout_edges_[pn_heh];

    for (int i = 0; i < hehs_strip.size(); ++i)
    {
        auto const t_heh = hehs_strip[i];

        vec2d const from_2D = strip.heh_pos_2d[t_heh].value();
        vec2d const to_2D = strip.heh_pos_2d[t_heh.next()].value();

        auto params = compute_intersection_parameter(to_2D, from_2D, layout_A, layout_B);

        auto vh_new = pn_heh.vertex_to();
        strip.pn_vhs.push_back(vh_new);

        pnd.sp_on_target_.value()[vh_new]
            = SurfacePoint(vec2d(params.value().y(), 0.0), t_heh.idx, SurfacePointType::EdgePoint);

        // DEBUG
        auto pos = pnd.sp_on_target_.value()[vh_new].get_pos(strip.heh_pos_2d);
        cd_2D.add_point(pos3_of_2D(pos), MAGENTA).size(20);
        //  END

        if ((i + 1) < hehs_strip.size())
        {
            auto const t_heh_next = hehs_strip[i + 1];
            if (t_heh.vertex_from() == t_heh_next.vertex_from())
            {
                pn_heh = pn_heh.next();
                pnd.mesh_->halfedges().split(pn_heh);
                pnd.map_to_layout_edges_[pn_heh.next().edge()] = pn_to_le_label;
            }
            else
            {
                pn_heh = pn_heh.next();
            }
        }
    }
    return idx_current + offset;
}


void transform_v_sp_to_e_sp(
    size_t idx_current, std::vector<VH> const& _old_pn_vhs, TriangleStrip& strip, PathNetworkData& pnd, TargetMeshData const& tmd, vec2d const& layout_A, vec2d const& layout_B)
{
    auto pn_vh_prev = strip.pn_vhs.back();
    auto pn_vh_curr = _old_pn_vhs[idx_current];
    auto pn_vh_next = _old_pn_vhs[idx_current + 1];

    auto const& sp_prev = pnd.sp_on_target_.value()[pn_vh_prev];
    auto& sp_curr = pnd.sp_on_target_.value()[pn_vh_curr];
    auto const& sp_next = pnd.sp_on_target_.value()[pn_vh_next];

    assert(sp_prev.type != SurfacePointType::VertexPoint && sp_curr.type == SurfacePointType::VertexPoint);

    // 1. find the fan that needs to be unfolded
    // 1.1 find the starting heh
    auto t_face_pc = shared_face(sp_prev, sp_curr, *tmd.mesh_);
    assert(t_face_pc.is_valid());

    for (auto t_heh : t_face_pc.halfedges())
    {
        strip.set_edge_pos_by_copy_from_opposite(t_heh);
    }

    auto t_vh_curr = tmd.mesh_->handle_of(sp_curr.heh_idx).vertex_from();
    auto t_vh_curr_heh = t_face_pc.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_to() == t_vh_curr; }).first().opposite();
    strip.set_edge_pos_by_copy_from_opposite(t_vh_curr_heh);

    auto t_start_hh = t_vh_curr_heh;

    if (t_start_hh.is_invalid())
    {
        auto pos_prev = eigen_to_pos3(sp_prev.get_pos(tmd.pos_mat_, *tmd.mesh_.get()));
        auto pos_curr = eigen_to_pos3(sp_curr.get_pos(tmd.pos_mat_, *tmd.mesh_.get()));

        auto v = gv::view();
        auto c = gv::canvas();

        c.add_label(pos_prev, "prev");
        c.add_point(pos_curr, "current");
    }
    assert(t_start_hh.is_valid());

    cd_2D.add_line(pos3_of_2D(strip.heh_pos_2d[t_start_hh].value()), pos3_of_2D(strip.heh_pos_2d[t_start_hh.next()].value()), GREEN);


    // 1.2 find final heh, we need to distinguish if the sp_next is edge or vertex
    HEH t_end_hh;
    if (!sp_next.is_vertex_sp())
    {
        auto t_face_cn = shared_face(sp_curr, sp_next, *tmd.mesh_);
        t_end_hh = t_vh_curr.outgoing_halfedges()
                       .filter([&](HEH t_heh) { return t_heh.face() == t_face_cn && strip.heh_pos_2d[t_heh.next()].value().y() < 0; })
                       .first();

        auto t_hehs2 = t_face_cn.halfedges().to_vector();
        cd_2D.add_face(pos3_of_2D(strip.heh_pos_2d[t_hehs2[0]].value()), pos3_of_2D(strip.heh_pos_2d[t_hehs2[1]].value()),
                       pos3_of_2D(strip.heh_pos_2d[t_hehs2[2]].value()), GREEN_50);
    }
    else // sp_next is vertex_surface point
    {
        auto t_vh_next = tmd.mesh_->handle_of(sp_next.heh_idx).vertex_from();
        auto t_hh_cn = pm::halfedge_from_to(t_vh_curr, t_vh_next);
        t_end_hh = t_hh_cn;
    }

    cd_2D.add_line(pos3_of_2D(strip.heh_pos_2d[t_end_hh].value()), pos3_of_2D(strip.heh_pos_2d[t_end_hh.next()].value()), RED);

    // 2. puzzle triangle fan
    puzzle_triangle_fan(t_start_hh, t_end_hh, strip, tmd);
    // make consistent
    strip.set_vertex_pos(t_end_hh.vertex_to(), strip.heh_pos_2d[t_end_hh.next()].value());

    // 3. slightly adapt metric by translation
    // (value semantics; see the note in adapt_metric about the torch in-place +=)
    vec2d const displacement = LARGE_EPS * vec2d(0.0, 1.0);
    for (auto t_heh_outgoing : t_vh_curr.outgoing_halfedges())
    {
        strip.heh_pos_2d[t_heh_outgoing] = strip.heh_pos_2d[t_heh_outgoing].value() + displacement;
    }

    cd_2D.add_point(pos3_of_2D(strip.heh_pos_2d[t_vh_curr_heh].value()), MAY_GREEN).size(15);

    // 4. adapt pnd
    HEH t_iter_heh = t_start_hh;
    auto pn_heh = pm::halfedge_from_to(pn_vh_prev, pn_vh_curr);
    auto pn_to_le_label = pnd.map_to_layout_edges_[pn_heh];

    bool should_continue = true;
    if (sp_next.is_vertex_sp())
    {
        if (t_iter_heh == t_end_hh)
            should_continue = false;
    }
    else
    {
        if (t_iter_heh == t_end_hh.next().next().opposite())
            should_continue = false;
    }

    init_canvas();

    while (should_continue)
    {
        // gather information if not already there
        strip.set_edge_pos_by_copy_from_opposite(t_iter_heh);
        strip.set_edge_pos_by_copy_from_opposite(t_iter_heh.opposite());

        vec2d const from_2D = strip.heh_pos_2d[t_iter_heh].value();
        vec2d const to_2D = strip.heh_pos_2d[t_iter_heh.next()].value();

        cd_2D.add_line(pos3_of_2D(from_2D), pos3_of_2D(to_2D), MAGENTA).size(5);

        auto params = compute_intersection_parameter(to_2D, from_2D, layout_A, layout_B);

        auto vh_new = pn_heh.vertex_to();
        strip.pn_vhs.push_back(vh_new);

        pnd.sp_on_target_.value()[vh_new]
            = SurfacePoint(vec2d(params.value().y(), 0.0), t_iter_heh.idx, SurfacePointType::EdgePoint);

        // DEBUG
        auto pos = pnd.sp_on_target_.value()[vh_new].get_pos(strip.heh_pos_2d);
        cd_2D.add_point(pos3_of_2D(pos), MAGENTA).size(20);
        //  END

        t_iter_heh = t_iter_heh.next().next().opposite();

        if (sp_next.is_vertex_sp())
        {
            if (t_iter_heh == t_end_hh)
                should_continue = false;
        }
        else
        {
            if (t_iter_heh == t_end_hh.next().next().opposite())
                should_continue = false;
        }

        if (should_continue)
        {
            pn_heh = pn_heh.next();
            pnd.mesh_->halfedges().split(pn_heh);
            pnd.map_to_layout_edges_[pn_heh.next().edge()] = pn_to_le_label;
        }
    }
}

void transform_v_sp_to_e_sp_front_back(
    size_t idx_current, std::vector<VH> const& _old_pn_vhs, TriangleStrip& strip, PathNetworkData& pnd, TargetMeshData const& tmd, vec2d const& layout_A, vec2d const& layout_B)
{
    assert(idx_current == 1 || idx_current == _old_pn_vhs.size() - 2);
    auto pn_vh_prev = strip.pn_vhs.back();
    auto pn_vh_curr = _old_pn_vhs[idx_current];
    auto pn_vh_next = _old_pn_vhs[idx_current + 1];

    auto const& sp_prev = pnd.sp_on_target_.value()[pn_vh_prev];
    auto& sp_curr = pnd.sp_on_target_.value()[pn_vh_curr];
    auto const& sp_next = pnd.sp_on_target_.value()[pn_vh_next];

    assert(sp_prev.type == SurfacePointType::FacePoint || sp_next.type == SurfacePointType::FacePoint);
    assert(sp_curr.type == SurfacePointType::VertexPoint);
    assert(sp_prev.type != SurfacePointType::VertexPoint);

    // 1. find the fan that needs to be unfolded
    // 1.1 find the starting heh
    auto t_face_pc = shared_face(sp_prev, sp_curr, *tmd.mesh_);
    auto t_vh_curr = tmd.mesh_->handle_of(sp_curr.heh_idx).vertex_from();

    HEH t_start_hh = t_face_pc.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_to() == t_vh_curr; }).first().opposite();

    assert(t_start_hh.is_valid());
    strip.heh_pos_2d[t_start_hh] = strip.heh_pos_2d[t_start_hh.opposite().next()];
    strip.heh_pos_2d[t_start_hh.next()] = strip.heh_pos_2d[t_start_hh.opposite()];

    cd_2D.add_line(pos3_of_2D(strip.heh_pos_2d[t_start_hh].value()), pos3_of_2D(strip.heh_pos_2d[t_start_hh.next()].value()), GREEN);

    // 1.2 find final heh, we need to distinguish if the sp_next is edge/face or vertex
    HEH t_end_hh;
    if (!sp_next.is_vertex_sp())
    {
        auto t_face_cn = shared_face(sp_curr, sp_next, *tmd.mesh_);
        t_end_hh = t_face_cn.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_from() == t_vh_curr; }).first();
    }
    else // sp_next is vertex_surface point
    {
        auto t_vh_next = tmd.mesh_->handle_of(sp_next.heh_idx).vertex_from();
        auto t_hh_cn = pm::halfedge_from_to(t_vh_curr, t_vh_next);
        t_end_hh = t_hh_cn;
    }
    assert(t_end_hh.is_valid());
    cd_2D.add_line(pos3_of_2D(strip.heh_pos_2d[t_end_hh].value()), pos3_of_2D(strip.heh_pos_2d[t_end_hh.next()].value()), RED).size(5);

    // 2. puzzle triangle fan and make last vertex consistent
    auto final_pos = strip.heh_pos_2d[t_end_hh.next()];
    puzzle_triangle_fan(t_start_hh, t_end_hh, strip, tmd);
    strip.set_vertex_pos(t_end_hh.next().vertex_from(), final_pos.value());

    // 3. slightly adapt metric by rotation
    vec2d rotation_center = vec2d::Zero();
    FH f_to_rotate;
    double theta = 0.5 * M_PI / 180.0; // 0.1 degrees in radians

    if (idx_current == 1)
    {
        f_to_rotate = sp_prev.fh(*tmd.mesh_.get());
    }
    else if (sp_next.is_face_sp())
    {
        theta = -theta;
        f_to_rotate = sp_next.fh(*tmd.mesh_.get());
        rotation_center = rotation_center + strip.embedded_length * vec2d(1.0, 0.0);
    }

    // --- Rotation matrix ---
    auto c = std::cos(theta);
    auto s = std::sin(theta);
    mat2d R;
    R << c, -s, s, c;

    for (auto t_heh : f_to_rotate.halfedges())
    {
        vec2d const p = strip.heh_pos_2d[t_heh].value() - rotation_center;
        vec2d const rotated = R * p + rotation_center;
        strip.heh_pos_2d[t_heh] = rotated;
    }

    // make consistent with other triangles
    if (idx_current == 1)
    {
        strip.set_vertex_pos(t_start_hh.vertex_from(), strip.heh_pos_2d[t_start_hh.opposite().next()].value());
        strip.heh_pos_2d[t_start_hh.next()] = strip.heh_pos_2d[t_start_hh.opposite()];
    }
    else
    {
        strip.set_vertex_pos(t_start_hh.vertex_from(), strip.heh_pos_2d[t_end_hh].value());
        strip.heh_pos_2d[t_end_hh.opposite()] = strip.heh_pos_2d[t_end_hh.next()];
    }

    // 4. adapt pnd
    HEH t_iter_heh = t_start_hh;
    auto pn_heh = pm::halfedge_from_to(pn_vh_prev, pn_vh_curr);
    auto pn_to_le_label = pnd.map_to_layout_edges_[pn_heh];

    bool should_continue = true;
    if (sp_next.is_vertex_sp())
    {
        if (t_iter_heh == t_end_hh)
            should_continue = false;
    }
    else
    {
        if (t_iter_heh == t_end_hh.next().next().opposite())
            should_continue = false;
    }

    while (should_continue)
    {
        vec2d const from_2D = strip.heh_pos_2d[t_iter_heh].value();
        vec2d const to_2D = strip.heh_pos_2d[t_iter_heh.next()].value();

        cd_2D.add_line(pos3_of_2D(from_2D), pos3_of_2D(to_2D), BLUE).size(2);

        auto params = compute_intersection_parameter(to_2D, from_2D, layout_A, layout_B);

        auto vh_new = pn_heh.vertex_to();
        strip.pn_vhs.push_back(vh_new);

        pnd.sp_on_target_.value()[vh_new]
            = SurfacePoint(vec2d(params.value().y(), 0.0), t_iter_heh.idx, SurfacePointType::EdgePoint);

        // DEBUG
        auto pos = pnd.sp_on_target_.value()[vh_new].get_pos(strip.heh_pos_2d);
        cd_2D.add_point(pos3_of_2D(pos), BLUE).size(20);
        //  END

        t_iter_heh = t_iter_heh.next().next().opposite();

        if (sp_next.is_vertex_sp())
        {
            if (t_iter_heh == t_end_hh)
                should_continue = false;
        }
        else
        {
            if (t_iter_heh == t_end_hh.next().next().opposite())
                should_continue = false;
        }

        if (should_continue)
        {
            pn_heh = pn_heh.next();
            pnd.mesh_->halfedges().split(pn_heh);
            pnd.map_to_layout_edges_[pn_heh.next().edge()] = pn_to_le_label;
        }
    }
}

// from is facepoint, current is vertexpoint and next is facepoint
void transform_special_case(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData& _pnd, vec2d const& layout_A, vec2d const& layout_B)
{
    auto pn_vh_prev = _strip.pn_vhs[0];
    auto pn_vh_curr = _strip.pn_vhs[1];
    auto pn_vh_next = _strip.pn_vhs[2];

    _strip.pn_vhs.clear();
    _strip.pn_vhs.push_back(pn_vh_prev);

    auto const& sp_prev = _pnd.sp_on_target_.value()[pn_vh_prev];
    auto& sp_curr = _pnd.sp_on_target_.value()[pn_vh_curr];
    auto const& sp_next = _pnd.sp_on_target_.value()[pn_vh_next];

    auto t_face_pc = shared_face(sp_prev, sp_curr, *_tmd.mesh_);
    auto t_face_cn = shared_face(sp_curr, sp_next, *_tmd.mesh_);

    double const l_pc = (sp_curr.get_pos(_strip.heh_pos_2d) - sp_prev.get_pos(_strip.heh_pos_2d)).norm();
    double const l_cn = (sp_next.get_pos(_strip.heh_pos_2d) - sp_curr.get_pos(_strip.heh_pos_2d)).norm();

    auto t_vh_curr = _tmd.mesh_->handle_of(sp_curr.heh_idx).vertex_from();

    HEH t_start_hh = t_face_pc.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_to() == t_vh_curr; }).first().opposite();
    assert(t_start_hh.is_valid());
    _strip.heh_pos_2d[t_start_hh] = _strip.heh_pos_2d[t_start_hh.opposite().next()];
    _strip.heh_pos_2d[t_start_hh.next()] = _strip.heh_pos_2d[t_start_hh.opposite()];

    cd_2D.add_line(pos3_of_2D(_strip.heh_pos_2d[t_start_hh].value()), pos3_of_2D(_strip.heh_pos_2d[t_start_hh.next()].value()), GREEN);

    HEH t_end_hh = t_face_cn.halfedges().filter([&](HEH t_heh) { return t_heh.vertex_from() == t_vh_curr; }).first();
    assert(t_end_hh.is_valid());
    cd_2D.add_line(pos3_of_2D(_strip.heh_pos_2d[t_end_hh].value()), pos3_of_2D(_strip.heh_pos_2d[t_end_hh.next()].value()), RED).size(5);


    puzzle_triangle_fan(t_start_hh, t_end_hh, _strip, _tmd);
    // make consistent
    _strip.set_vertex_pos(t_end_hh.vertex_to(), _strip.heh_pos_2d[t_end_hh.next()].value());

    // 3. slightly adapt metric by rotation
    vec2d const rotation_center1 = _strip.embedded_length * vec2d(1.0, 0.0);

    double alpha = 0.2 * M_PI / 180.0; // 0.1 degrees in radians
    double beta = -std::atan(std::tan(alpha) * l_pc / l_cn);

    // --- Rotation matrix ---
    auto c = std::cos(alpha);
    auto s = std::sin(alpha);
    mat2d R0;
    R0 << c, -s, s, c;

    c = std::cos(beta);
    s = std::sin(beta);
    mat2d R1;
    R1 << c, -s, s, c;

    for (auto t_heh : t_face_pc.halfedges())
    {
        vec2d const p = _strip.heh_pos_2d[t_heh].value();
        vec2d const rotated = R0 * p;
        _strip.heh_pos_2d[t_heh] = rotated;
    }

    for (auto t_heh : t_face_cn.halfedges())
    {
        vec2d const p = _strip.heh_pos_2d[t_heh].value() - rotation_center1;
        vec2d const rotated = R1 * p + rotation_center1;
        _strip.heh_pos_2d[t_heh] = rotated;
    }

    // make consistent with other triangles
    _strip.set_vertex_pos(t_start_hh.vertex_from(), _strip.heh_pos_2d[t_start_hh.opposite().next()].value());
    _strip.heh_pos_2d[t_start_hh.next()] = _strip.heh_pos_2d[t_start_hh.opposite()];
    _strip.heh_pos_2d[t_end_hh.opposite()] = _strip.heh_pos_2d[t_end_hh.next()];

    // 4. adapt pnd
    HEH t_iter_heh = t_start_hh;
    auto pn_heh = pm::halfedge_from_to(pn_vh_prev, pn_vh_curr);
    auto pn_to_le_label = _pnd.map_to_layout_edges_[pn_heh];

    bool should_continue = true;
    if (t_iter_heh == t_end_hh.next().next().opposite())
        should_continue = false;

    while (should_continue)
    {
        vec2d const from_2D = _strip.heh_pos_2d[t_iter_heh].value();
        vec2d const to_2D = _strip.heh_pos_2d[t_iter_heh.next()].value();

        cd_2D.add_line(pos3_of_2D(from_2D), pos3_of_2D(to_2D), BLUE).size(2);

        auto params = compute_intersection_parameter(to_2D, from_2D, layout_A, layout_B);

        auto vh_new = pn_heh.vertex_to();
        _strip.pn_vhs.push_back(vh_new);

        _pnd.sp_on_target_.value()[vh_new]
            = SurfacePoint(vec2d(params.value().y(), 0.0), t_iter_heh.idx, SurfacePointType::EdgePoint);

        // DEBUG
        auto pos = _pnd.sp_on_target_.value()[vh_new].get_pos(_strip.heh_pos_2d);
        cd_2D.add_point(pos3_of_2D(pos), BLUE).size(20);
        //  END

        t_iter_heh = t_iter_heh.next().next().opposite();

        if (t_iter_heh == t_end_hh.next().next().opposite())
            should_continue = false;

        if (should_continue)
        {
            pn_heh = pn_heh.next();
            _pnd.mesh_->halfedges().split(pn_heh);
            _pnd.map_to_layout_edges_[pn_heh.next().edge()] = pn_to_le_label;
        }
    }
    // pushback surfacepoint
    _strip.pn_vhs.push_back(pn_vh_next);
}

}

void compute_triangle_strip(TriangleStrip& _strip, const EH _l_eh, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData& _pnd)
{
    _strip.l_eh = _l_eh;
    init_triangle_strip_pn_vhs(_strip, _l_eh, _pnd);

    // if (_strip.l_eh.idx.value == 209)
    // {
    //     auto v = gv::view();
    //     view_path_network(_tmd, _pnd);
    //     auto c = gv::canvas();
    //     c.add_faces(_tmd.pos_, BLUE_25);
    //     c.add_lines(_tmd.pos_, BLUE);
    //     for(int i = 0; i < _strip.pn_vhs.size() - 1; ++i)
    //     {
    //         auto const sp_curr = _pnd.sp_on_target_.value()(_strip.pn_vhs[i]);
    //         c.add_point(torch_to_pos3(sp_curr.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), RED).size(20);

    //         auto const sp_next = _pnd.sp_on_target_.value()(_strip.pn_vhs[i + 1]);
    //         c.add_point(torch_to_pos3(sp_next.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), MAGENTA).size(20);

    //         c.add_line(torch_to_pos3(sp_curr.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), torch_to_pos3(sp_next.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), GREEN).size(10);

    //         if(!sp_curr.is_vertex_sp() || !sp_next.is_vertex_sp())
    //         {
    //             auto fh = shared_face(sp_curr, sp_next, *_tmd.mesh_.get());
    //             DEBUG_VAR(fh)
    //         }
    //     }
    // }

    init_embedded_length(_strip, _tmd, _pnd);

    init_2D_strip_pos(_strip, _tmd, _ld, _pnd);

    convert_to_snake_refactor(_strip, _tmd, _pnd);

    // if (_strip.l_eh.idx.value == 209)
    // {
    //     GLOW_VIEWER_CONFIG(glow::viewer::camera_transform(tg::pos3(0.003340f, 0.008365f, 0.123197f), tg::pos3(0.042787f, -0.000368f, 0.000838f)));
    //     init_canvas();
    //     auto c = gv::canvas();
    //     init_canvas_current_2D_pos(_strip);
    //     c.add_data(cd_2D);
    // }

    init_triangle_strip_t_fhs(_strip, _tmd, _pnd);

    // if (_strip.l_eh.idx.value == 47)
    // {
    //     init_canvas();
    //     auto c = gv::canvas();
    //     init_canvas_current_strip_based_on_fhs(_strip);
    //     c.add_data(cd_2D);
    // }
}

void init_triangle_strip_pn_vhs(TriangleStrip& _strip, const EH _l_eh, PathNetworkData const& _pnd)
{
    auto l_heh = _l_eh.halfedgeA();
    auto pn_vertex_from = _pnd.mesh_->vertices()[l_heh.vertex_from().idx];

    // find first heh
    HEH pn_heh_iter
        = pn_vertex_from.outgoing_halfedges()
              .filter(
                  [&](HEH pn_heh)
                  { return (_pnd.map_to_layout_vertices_[pn_heh.vertex_from()].is_valid()) && (_pnd.map_to_layout_edges_[pn_heh.edge()] == _l_eh); })
              .first();
    assert(pn_heh_iter.is_valid() && "could not find valid starting point");

    while (_pnd.map_to_layout_edges_[pn_heh_iter] == _l_eh)
    {
        _strip.pn_vhs.push_back(pn_heh_iter.vertex_from());
        pn_heh_iter = pn_heh_iter.next();
    }
    _strip.pn_vhs.push_back(pn_heh_iter.vertex_from());
}

void init_embedded_length(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd)
{
    assert(!_strip.pn_vhs.empty());

    double length = 0.0;
    for (size_t i = 0; i < _strip.pn_vhs.size() - 1; ++i)
    {
        length += compute_strip_segment_length(_strip, i, i + 1, _tmd, _pnd);
    }
    _strip.embedded_length = length;
}


void init_2D_strip_pos(TriangleStrip& _strip, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd)
{
    assert(!_strip.pn_vhs.empty());

    init_canvas();
    init_3D_view(_tmd);

    // 1. init attribute to be computed
    _strip.heh_pos_2d = _tmd.mesh_->halfedges().make_attribute<std::optional<vec2d>>(std::nullopt);

    // 2. the embedding will be such that the geodesic corresponds to the positive x-axis. To determine the end point
    //    we need the total embedded length
    double total_length = _strip.embedded_length;

    // we compute 2D such that
    // 3. the first point is in (0,0) and the second point is (l, 0), aligned with x-axis
    set_triangle_pos(_strip, 0, 1, true, _tmd, _pnd);

    // 4. the last point is at (l,0) and the second to last is aligned with x-axis in negative direction
    vec2d const displacement = total_length * vec2d(1.0, 0.0);
    set_triangle_pos(_strip, _strip.pn_vhs.size() - 1, _strip.pn_vhs.size() - 2, false, _tmd, _pnd, displacement);

    // 5. find 2D positions of inner vertices
    // since we start at vertex 2 we need to count the segment between 0-1 as traveled
    double traveled_distance = compute_strip_segment_length(_strip, 0, 1, _tmd, _pnd);
    for (size_t i = 2; i < _strip.pn_vhs.size() - 1; ++i) // the second and second to last point are already handeled by embedding the first and last triangle
    {
        // asserts
        auto pn_vh_prev = _strip.pn_vhs[i - 1];
        auto const& sp_prev = _pnd.sp_on_target_.value()[pn_vh_prev];
        assert(sp_prev.type != SurfacePointType::FacePoint); // since we skip the first and the last face, this should never happen

        auto pn_vh_curr = _strip.pn_vhs[i];
        auto const& sp_curr = _pnd.sp_on_target_.value()[pn_vh_curr];
        assert(sp_curr.type != SurfacePointType::FacePoint); // since we skip the first and the last face, this should never happen

        // start computation
        double current_segment_length = compute_strip_segment_length(_strip, i - 1, i, _tmd, _pnd);
        traveled_distance += current_segment_length;

        // if the point is a vertex point, place it at the right distance along the X axis
        if (sp_curr.type == SurfacePointType::VertexPoint)
        {
            vec2d const position = traveled_distance * vec2d(1.0, 0.0);
            auto t_vh = _tmd.mesh_->handle_of(sp_curr.heh_idx).vertex_from();
            _strip.set_vertex_pos(t_vh, position);
            auto t_fh = shared_face(sp_prev, sp_curr, *_tmd.mesh_.get());

            if (t_fh.is_invalid())
                continue;

            for (auto t_heh : t_fh.halfedges())
            {
                if (!_strip.heh_pos_2d[t_heh].has_value())
                {
                    if (t_heh.vertex_to() == t_vh) // pointing to vertex
                    {
                        _strip.heh_pos_2d[t_heh] = _strip.heh_pos_2d[t_heh.prev().opposite()];
                    }
                    else // opposite to vertex
                    {
                        _strip.set_edge_pos_by_copy_from_opposite(t_heh);
                    }
                }
            }
        }
        else if (sp_prev.type == SurfacePointType::VertexPoint && sp_curr.type == SurfacePointType::EdgePoint)
        {
            // DEBUG_OUT("curr is edgepoint, prev is vertexpoint")
            puzzle_sp_prev_vertex(sp_prev, sp_curr, _strip, _tmd);
        }
        else if (sp_prev.type == SurfacePointType::EdgePoint && sp_curr.type == SurfacePointType::EdgePoint)
        {
            // DEBUG_OUT("curr is edgepoint, prev is edgepoint")
            puzzle_face(sp_prev, sp_curr, _strip, _tmd);
        }
        else
        {
            assert(false && "this case should not occure!");
        }
    }
}

void convert_to_snake_refactor(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData& _pnd)
{
    // early outs
    //  then those must be two surface points in the same face and than there is nothing to do
    if (_strip.pn_vhs.size() == 2)
        return;

    bool needs_conversion = false;
    for (int i = 1; i < _strip.pn_vhs.size() - 1; ++i)
    {
        auto intersect_pn_vh = _strip.pn_vhs[i];
        if (_pnd.sp_on_target_.value()[intersect_pn_vh].type == SurfacePointType::VertexPoint)
        {
            needs_conversion = true;
        }
    }

    if (!needs_conversion)
        return;

    // start converting
    auto sp_from = _pnd.sp_on_target_.value()[_strip.pn_vhs.front()];
    auto sp_to = _pnd.sp_on_target_.value()[_strip.pn_vhs.back()];

    vec2d const A = sp_from.get_pos(_strip.heh_pos_2d);
    vec2d const B = sp_to.get_pos(_strip.heh_pos_2d);

    // special handeling
    if (_strip.pn_vhs.size() == 3)
    {
        if (_pnd.sp_on_target_.value()[_strip.pn_vhs[1]].is_vertex_sp())
        {
            ++g_vertex_sp_conversion_count;
            transform_special_case(_strip, _tmd, _pnd, A, B);
            // DEBUG_OUT("3 vertex case");
            //  {
            //      init_canvas();
            //      auto c = gv::canvas();
            //      init_canvas_current_2D_pos(_strip);
            //      c.add_data(cd_2D);
            //  }
            assert(_pnd.mesh_->is_compact());
            return;
        }
    }

    auto pn_vhs = std::move(_strip.pn_vhs);
    _strip.pn_vhs.clear();
    _strip.pn_vhs.push_back(pn_vhs.front());

    // middle vertex surface points
    for (size_t i = 1; i < pn_vhs.size() - 1; ++i)
    {
        auto pn_vh_curr = pn_vhs[i];
        auto pn_vh_next = pn_vhs[i + 1];
        if (_pnd.sp_on_target_.value()[pn_vh_curr].is_vertex_sp() && !_pnd.sp_on_target_.value()[pn_vh_next].is_vertex_sp())
        {
            // DEBUG_OUT("single");
            ++g_vertex_sp_conversion_count;
            transform_single_v_sp(i, pn_vhs, _strip, _pnd, _tmd, A, B);
        }
        else if (_pnd.sp_on_target_.value()[pn_vh_curr].is_vertex_sp() && _pnd.sp_on_target_.value()[pn_vh_next].is_vertex_sp())
        {
            // DEBUG_OUT("multiple");
            ++g_vertex_sp_conversion_count;
            auto j = transform_multiple_v_sp(i, pn_vhs, _strip, _pnd, _tmd, A, B);
            i = j;
        }
        else
        {
            _strip.pn_vhs.push_back(pn_vh_curr);
        }
    }

    // // special handeling for last face
    // auto intersect_pn_vh_sec_to_last = pn_vhs[pn_vhs.size() - 2];
    // if (_pnd.sp_on_target_.value()[intersect_pn_vh_sec_to_last].is_vertex_sp())
    // {
    //     transform_v_sp_to_e_sp_front_back(pn_vhs.size() - 2, pn_vhs, _strip, _pnd, _tmd, A, B);
    // }

    _strip.pn_vhs.push_back(pn_vhs.back());
    assert(_pnd.mesh_->is_compact());
}

void init_triangle_strip_t_fhs(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd)
{
    // auto cd = gv::canvas_data();
    // cd.add_faces(_tmd.pos_, BLUE_25);
    // cd.add_lines(_tmd.pos_, BLUE_25);

    // DEBUG_VAR(_strip.l_eh);

    _strip.target_fhs.clear();
    for (size_t i = 1; i < _strip.pn_vhs.size(); i++)
    {
        auto vh_prev = _strip.pn_vhs[i - 1];
        auto vh_curr = _strip.pn_vhs[i];

        auto const& sp_prev = _pnd.sp_on_target_.value()[vh_prev];
        auto const& sp_curr = _pnd.sp_on_target_.value()[vh_curr];

        assert(!sp_prev.is_vertex_sp());
        assert(!sp_curr.is_vertex_sp());

        // cd.add_point(torch_to_pos3(sp_prev.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), MAGENTA);
        // cd.add_point(torch_to_pos3(sp_curr.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get())), MAGENTA);

        // if (_strip.l_eh.idx.value == 34)
        // {
        //     auto c = gv::canvas();
        //     c.add_data(cd);
        // }

        auto shared_fh = shared_face(sp_prev, sp_curr, *_tmd.mesh_.get());
        _strip.target_fhs.push_back(shared_fh);
    }
}


bool is_strip_valid(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd)
{
    init_canvas();

    auto sp_from = _pnd.sp_on_target_.value()[_strip.pn_vhs.front()];
    auto sp_to = _pnd.sp_on_target_.value()[_strip.pn_vhs.back()];

    vec2d const A = sp_from.get_pos(_strip.heh_pos_2d);
    vec2d const B = sp_to.get_pos(_strip.heh_pos_2d);

    bool valid = true;
    for (size_t i = 1; i < _strip.pn_vhs.size() - 1; ++i) // skip the first and the last as they are face points
    {
        auto intersect_pn_vh = _strip.pn_vhs[i];
        auto const& sp = _pnd.sp_on_target_.value()[intersect_pn_vh];

        if (!sp.is_edge_sp())
        {
            valid = false;
            cd_2D.add_point(pos3_of_2D(sp.get_pos(_strip.heh_pos_2d)), MAGENTA).size(15);
            continue;
        }

        auto intersect_hh = _tmd.mesh_->handle_of(sp.heh_idx);

        auto const& from_2D_opt = _strip.heh_pos_2d[intersect_hh];

        if (!from_2D_opt.has_value())
        {
            init_canvas_current_2D_pos(_strip);
            auto c = gv::canvas();
            c.add_data(cd_2D);
            valid = false;
            continue;
        }

        auto const& to_2D_opt = _strip.heh_pos_2d[intersect_hh.next()];

        if (!to_2D_opt.has_value())
        {
            init_canvas_current_2D_pos(_strip);
            auto c = gv::canvas();
            c.add_data(cd_2D);
            valid = false;
            continue;
        }

        vec2d const from_2D = from_2D_opt.value();
        vec2d const to_2D = to_2D_opt.value();

        auto tg_from = pos3_of_2D(from_2D);
        auto tg_to = pos3_of_2D(to_2D);
        cd_2D.add_line(tg_from, tg_to, BLACK).size(4.0);


        auto params = compute_intersection_parameter(to_2D, from_2D, A, B);

        if (!params.has_value() || (params.value().x() < 0) || (params.value().x() > 1.0))
        {
            auto tg_from = pos3_of_2D(from_2D);
            auto tg_to = pos3_of_2D(to_2D);
            cd_2D.add_line(tg_from, tg_to, RED).size(5.0);

            if (params.has_value())
            {
                DEBUG_VAR(_strip.l_eh)
                DEBUG_VAR(params.value().x())
                DEBUG_VAR(params.value().y())

                auto alpha = params.value().y();

                auto intersection_point = alpha * tg_from + (1.0 - alpha) * tg_to;
                cd_2D.add_point(intersection_point, RED).size(15);
            }
            valid = false;
        }
    }

    // if (!valid)
    // {
    //     GLOW_VIEWER_CONFIG(glow::viewer::camera_transform(tg::pos3(0.103661f, 0.006799f, 0.364353f), tg::pos3(0.144128f, 0.003635f, 0.000008f)));

    //     DEBUG_VAR(_strip.pn_vhs.size())

    //     init_canvas_current_2D_pos(_strip);
    //     auto c = gv::canvas();
    //     c.add_data(cd_2D);

    //     auto tg_from_2D = tg::pos3(torch_to_pos2(A));
    //     auto tg_to_2D = tg::pos3(torch_to_pos2(B));
    //     c.add_line(tg_from_2D, tg_to_2D, MAGENTA);
    // }

    return valid;
}


} // namespace LayoutOpt
