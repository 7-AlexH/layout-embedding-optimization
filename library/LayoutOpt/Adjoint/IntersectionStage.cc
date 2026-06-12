#include "IntersectionStage.hh"

#include <cassert>
#include <cmath>

namespace LayoutOpt
{

void intersections_forward(std::vector<TriangleStrip> const& _strips,
                           Eigen::MatrixX2d const& _strip_end_a,
                           Eigen::MatrixX2d const& _strip_end_b,
                           TargetMeshData const& _tmd,
                           LayoutData const& _ld,
                           PathNetworkData const& _pnd,
                           IntersectCtx& _ctx,
                           Eigen::MatrixX3d& _pos)
{
    assert(_pnd.sp_on_target_.has_value());
    assert(_pnd.map_to_overlay_vertices_.has_value());

    Eigen::MatrixX3d const& t_pos = _tmd.pos_mat_;

    _ctx.strips.clear();
    _ctx.strips.reserve(_ld.mesh_->edges().size());

    for (auto l_eh : _ld.mesh_->edges())
    {
        TriangleStrip const& strip = _strips[l_eh.idx.value];

        StripIntersectCtx sctx;
        sctx.l_edge = l_eh.idx.value;

        vec2d const A = _strip_end_a.row(l_eh.idx.value).transpose();
        vec2d const B = _strip_end_b.row(l_eh.idx.value).transpose();
        vec2d const b = B - A;

        for (size_t i = 1; i < strip.pn_vhs.size() - 1; ++i)
        {
            auto intersect_pn_vh = strip.pn_vhs[i];
            auto intersect_hh = _tmd.mesh_->handle_of(_pnd.sp_on_target_.value()[intersect_pn_vh].heh_idx);

            // the crossed target edge, flattened (constant per the Phase 2 audit)
            vec2d const from_2D = strip.heh_pos_2d[intersect_hh].value();
            vec2d const to_2D = strip.heh_pos_2d[intersect_hh.opposite()].value();

            IntersectionRec rec;
            rec.o_row = _pnd.map_to_overlay_vertices_.value()[intersect_pn_vh].value;
            rec.t_from = intersect_hh.vertex_from().idx.value;
            rec.t_to = intersect_hh.vertex_to().idx.value;

            // intersection parameter of line_a = (to_2D, from_2D) with
            // line_b = (A, B); operation order preserved from the original
            // torch implementation
            rec.a = from_2D - to_2D;
            rec.b = b;
            rec.b0_a0 = A - to_2D;
            rec.cross = rec.a.x() * rec.b.y() - rec.a.y() * rec.b.x();
            // never hit on valid strips (the original implementation crashed
            // here too, via an empty-result read)
            assert(std::abs(rec.cross) >= 1e-8 && "lines are (nearly) parallel");
            rec.t_numer = rec.b0_a0.x() * rec.b.y() - rec.b0_a0.y() * rec.b.x();
            rec.t = rec.t_numer / rec.cross;

            // intersect_pos = from_3D * t + to_3D * (1 - t)
            double const s = 1.0 - rec.t;
            for (int c = 0; c < 3; ++c)
                _pos(rec.o_row, c) = t_pos(rec.t_from, c) * rec.t + t_pos(rec.t_to, c) * s;

            sctx.recs.push_back(rec);
        }

        _ctx.strips.push_back(std::move(sctx));
    }
}

void intersections_backward(IntersectCtx const& _ctx,
                            Eigen::MatrixX3d const& _d_pos,
                            TargetMeshData const& _tmd,
                            Eigen::MatrixX2d& _d_strip_end_a,
                            Eigen::MatrixX2d& _d_strip_end_b)
{
    Eigen::MatrixX3d const& t_pos = _tmd.pos_mat_;

    for (auto s_it = _ctx.strips.rbegin(); s_it != _ctx.strips.rend(); ++s_it)
    {
        StripIntersectCtx const& sctx = *s_it;

        vec2d d_A = vec2d::Zero();
        vec2d d_B = vec2d::Zero();

        for (auto r_it = sctx.recs.rbegin(); r_it != sctx.recs.rend(); ++r_it)
        {
            IntersectionRec const& rec = *r_it;

            // pos[o_row] = from_3D * t + to_3D * (1 - t)
            double d_t = 0.0;
            for (int c = 0; c < 3; ++c)
                d_t += _d_pos(rec.o_row, c) * (t_pos(rec.t_from, c) - t_pos(rec.t_to, c));

            // t = t_numer / cross
            double const d_t_numer = d_t / rec.cross;
            double const d_cross = -d_t * rec.t_numer / (rec.cross * rec.cross);

            // t_numer = b0_a0.x*b.y - b0_a0.y*b.x ; cross = a.x*b.y - a.y*b.x
            // (a, to_2D, from_3D, to_3D are constants)
            vec2d const d_b0_a0(d_t_numer * rec.b.y(), -d_t_numer * rec.b.x());
            vec2d const d_b(-d_t_numer * rec.b0_a0.y() - d_cross * rec.a.y(),
                            d_t_numer * rec.b0_a0.x() + d_cross * rec.a.x());

            // b0_a0 = A - to_2D ; b = B - A
            d_A += d_b0_a0 - d_b;
            d_B += d_b;
        }

        _d_strip_end_a.row(sctx.l_edge) += d_A.transpose();
        _d_strip_end_b.row(sctx.l_edge) += d_B.transpose();
    }
}

} // namespace LayoutOpt
