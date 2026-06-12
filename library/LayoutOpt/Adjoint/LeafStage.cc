#include "LeafStage.hh"

#include <cassert>

namespace LayoutOpt
{

void leaf_collect(std::vector<TriangleStrip> const& _strips,
                  TargetMeshData const& _tmd,
                  LayoutData const& _ld,
                  PathNetworkData const& _pnd,
                  LeafCtx& _ctx)
{
    assert(_pnd.sp_on_target_.has_value());
    assert(_pnd.map_to_overlay_vertices_.has_value());

    _ctx.nodes.clear();
    _ctx.ends.clear();

    auto const& sps = _pnd.sp_on_target_.value();

    // (a) layout-node rows, in compute_differentiable_surface_points order
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        if (_pnd.map_to_layout_vertices_[pn_vh].is_invalid())
            continue;
        SurfacePoint const& sp = sps[pn_vh];
        auto hh = _tmd.mesh_->handle_of(sp.heh_idx);

        LeafNodeRec rec;
        rec.pn_v = pn_vh.idx.value;
        rec.o_row = _pnd.map_to_overlay_vertices_.value()[pn_vh].value;
        rec.type = sp.type;
        rec.rA = hh.vertex_from().idx.value;
        if (sp.type != SurfacePointType::VertexPoint)
            rec.rB = hh.vertex_to().idx.value;
        if (sp.type == SurfacePointType::FacePoint)
            rec.rC = hh.next().vertex_to().idx.value;
        _ctx.nodes.push_back(rec);
    }

    // (b) strip endpoints, per layout edge (constant 2D corners copied out of
    // the strip flattening; get_pos(halfedge_attribute) corner convention)
    for (auto l_eh : _ld.mesh_->edges())
    {
        int const le = l_eh.idx.value;
        TriangleStrip const& strip = _strips[le];

        for (int side = 0; side < 2; ++side)
        {
            auto pn_vh = side == 0 ? strip.pn_vhs.front() : strip.pn_vhs.back();
            SurfacePoint const& sp = sps[pn_vh];
            auto hh = _tmd.mesh_->handle_of(sp.heh_idx);

            LeafEndpointRec rec;
            rec.l_edge = le;
            rec.side = side;
            rec.pn_v = pn_vh.idx.value;
            rec.type = sp.type;

            auto load = [&](pm::halfedge_handle h) { return strip.heh_pos_2d[h].value(); };
            if (sp.type == SurfacePointType::VertexPoint)
                rec.A2 = load(hh);
            else if (sp.type == SurfacePointType::EdgePoint)
            {
                rec.A2 = load(hh);
                rec.B2 = load(hh.opposite());
            }
            else
            {
                rec.A2 = load(hh);
                rec.B2 = load(hh.next());
                rec.C2 = load(hh.next().next());
            }
            _ctx.ends.push_back(rec);
        }
    }
}

Eigen::MatrixX2d collect_bary(PathNetworkData const& _pnd)
{
    assert(_pnd.sp_on_target_.has_value());
    auto const& sps = _pnd.sp_on_target_.value();

    Eigen::MatrixX2d bary = Eigen::MatrixX2d::Zero((Eigen::Index)_pnd.mesh_->vertices().size(), 2);
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        SurfacePoint const& sp = sps[pn_vh];
        if (sp.type == SurfacePointType::FacePoint)
        {
            vec3d const b = sp.bary_full();
            bary(pn_vh.idx.value, 0) = b[0];
            bary(pn_vh.idx.value, 1) = b[1];
        }
        else if (sp.type == SurfacePointType::EdgePoint)
        {
            bary(pn_vh.idx.value, 0) = sp.bary_full()[0];
        }
    }
    return bary;
}

void leaf_forward(LeafCtx const& _ctx,
                  Eigen::MatrixX2d const& _bary,
                  Eigen::MatrixX3d const& _t_pos,
                  Eigen::MatrixX3d& _pos,
                  Eigen::MatrixX2d& _seg_a,
                  Eigen::MatrixX2d& _seg_b)
{
    for (auto const& n : _ctx.nodes)
    {
        switch (n.type)
        {
        case SurfacePointType::VertexPoint:
            _pos.row(n.o_row) = _t_pos.row(n.rA);
            break;
        case SurfacePointType::EdgePoint:
        {
            double const a = _bary(n.pn_v, 0);
            _pos.row(n.o_row) = a * _t_pos.row(n.rA) + (1.0 - a) * _t_pos.row(n.rB);
            break;
        }
        case SurfacePointType::FacePoint:
        {
            double const a = _bary(n.pn_v, 0);
            double const b = _bary(n.pn_v, 1);
            double const g = 1.0 - a - b;
            _pos.row(n.o_row) = a * _t_pos.row(n.rA) + b * _t_pos.row(n.rB) + g * _t_pos.row(n.rC);
            break;
        }
        default:
            assert(false && "invalid leaf surface point");
        }
    }

    for (auto const& e : _ctx.ends)
    {
        vec2d p;
        switch (e.type)
        {
        case SurfacePointType::VertexPoint:
            p = e.A2;
            break;
        case SurfacePointType::EdgePoint:
        {
            double const a = _bary(e.pn_v, 0);
            p = a * e.A2 + (1.0 - a) * e.B2;
            break;
        }
        case SurfacePointType::FacePoint:
        {
            double const a = _bary(e.pn_v, 0);
            double const b = _bary(e.pn_v, 1);
            p = a * e.A2 + b * e.B2 + (1.0 - a - b) * e.C2;
            break;
        }
        default:
            assert(false && "invalid endpoint surface point");
            p = vec2d::Zero();
        }
        (e.side == 0 ? _seg_a : _seg_b).row(e.l_edge) = p.transpose();
    }
}

void leaf_backward(LeafCtx const& _ctx,
                   Eigen::MatrixX3d const& _t_pos,
                   Eigen::MatrixX3d const& _d_pos,
                   Eigen::MatrixX2d const& _d_seg_a,
                   Eigen::MatrixX2d const& _d_seg_b,
                   Eigen::MatrixX2d& _d_bary)
{
    for (auto const& n : _ctx.nodes)
    {
        if (n.type == SurfacePointType::EdgePoint)
        {
            double d_a = 0.0;
            for (int c = 0; c < 3; ++c)
                d_a += _d_pos(n.o_row, c) * (_t_pos(n.rA, c) - _t_pos(n.rB, c));
            _d_bary(n.pn_v, 0) += d_a;
        }
        else if (n.type == SurfacePointType::FacePoint)
        {
            double d_a = 0.0, d_b = 0.0;
            for (int c = 0; c < 3; ++c)
            {
                d_a += _d_pos(n.o_row, c) * (_t_pos(n.rA, c) - _t_pos(n.rC, c));
                d_b += _d_pos(n.o_row, c) * (_t_pos(n.rB, c) - _t_pos(n.rC, c));
            }
            _d_bary(n.pn_v, 0) += d_a;
            _d_bary(n.pn_v, 1) += d_b;
        }
        // VertexPoint: constant, no gradient
    }

    for (auto const& e : _ctx.ends)
    {
        Eigen::Matrix<double, 1, 2> const d = (e.side == 0 ? _d_seg_a : _d_seg_b).row(e.l_edge);
        if (e.type == SurfacePointType::EdgePoint)
        {
            _d_bary(e.pn_v, 0) += d(0) * (e.A2.x() - e.B2.x()) + d(1) * (e.A2.y() - e.B2.y());
        }
        else if (e.type == SurfacePointType::FacePoint)
        {
            _d_bary(e.pn_v, 0) += d(0) * (e.A2.x() - e.C2.x()) + d(1) * (e.A2.y() - e.C2.y());
            _d_bary(e.pn_v, 1) += d(0) * (e.B2.x() - e.C2.x()) + d(1) * (e.B2.y() - e.C2.y());
        }
    }
}

} // namespace LayoutOpt
