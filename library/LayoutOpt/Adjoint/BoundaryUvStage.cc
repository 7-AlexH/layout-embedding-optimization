#include "BoundaryUvStage.hh"

#include <algorithm>
#include <cassert>

namespace LayoutOpt
{

//=== sub-stage 1: per-layout-edge embedded arc lengths ======================

void edge_lengths_forward(Eigen::MatrixX3d const& _pos,
                          LayoutData const& _ld,
                          PathNetworkData const& _pnd,
                          EdgeLengthCtx& _ctx,
                          Eigen::VectorXd& _lengths)
{
    assert(_pnd.map_to_overlay_vertices_.has_value());

    int64_t const L = (int64_t)_ld.mesh_->edges().size();
    size_t const n_seg = _pnd.mesh_->edges().size();
    _ctx.n_layout_edges = L;
    _ctx.l_edge.clear();
    _ctx.l_edge.reserve(n_seg);
    _ctx.om_a.clear();
    _ctx.om_a.reserve(n_seg);
    _ctx.om_b.clear();
    _ctx.om_b.reserve(n_seg);
    _ctx.seg_norm.clear();
    _ctx.seg_norm.reserve(n_seg);

    _lengths = Eigen::VectorXd::Zero(L);

    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto const om_A = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexA()];
        auto const om_B = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexB()];
        auto const l_eh = _pnd.map_to_layout_edges_[pn_eh];

        double const n = (_pos.row((int)om_A.value) - _pos.row((int)om_B.value)).norm();

        _ctx.l_edge.push_back((int)l_eh.value);
        _ctx.om_a.push_back((int)om_A.value);
        _ctx.om_b.push_back((int)om_B.value);
        _ctx.seg_norm.push_back(n);

        _lengths((int)l_eh.value) += n;
    }
}

void edge_lengths_backward(EdgeLengthCtx const& _ctx,
                           Eigen::MatrixX3d const& _pos,
                           Eigen::VectorXd const& _d_lengths,
                           Eigen::MatrixX3d& _d_pos)
{
    for (size_t k = 0; k < _ctx.l_edge.size(); ++k)
    {
        int const a = _ctx.om_a[k];
        int const b = _ctx.om_b[k];
        double const d_n = _d_lengths(_ctx.l_edge[k]);

        // n = |pa - pb|  =>  d_pa += d_n * (pa - pb) / n
        Eigen::RowVector3d const g = (d_n / _ctx.seg_norm[k]) * (_pos.row(a) - _pos.row(b));
        _d_pos.row(a) += g;
        _d_pos.row(b) -= g;
    }
}

//=== sub-stage 2: arc-length parameter t per pn halfedge ====================

void pn_t_forward(Eigen::MatrixX3d const& _pos,
                  Eigen::VectorXd const& _lengths,
                  LayoutData const& _ld,
                  PathNetworkData const& _pnd,
                  TCtx& _ctx,
                  Eigen::VectorXd& _t)
{
    int64_t const H = (int64_t)_pnd.mesh_->halfedges().size();
    _ctx.n_pn_halfedges = H;
    _ctx.chains.clear();
    _t = Eigen::VectorXd::Zero(H);

    auto l_heh_visited = _ld.mesh_->halfedges().make_attribute<bool>(false);

    for (auto l_heh : _ld.mesh_->halfedges())
    {
        if (l_heh_visited[l_heh])
            continue;

        if (l_heh.vertex_from().outgoing_halfedges().size() == 2)
            continue; // the from vertex is an inner subdivision vertex

        l_heh_visited[l_heh] = true;

        _ctx.chains.emplace_back();
        TChain& chain = _ctx.chains.back();

        auto l_eh = l_heh.edge();
        auto pn_vertex_from = _pnd.mesh_->handle_of(l_heh.vertex_from());

        HEH pn_iter_heh
            = pn_vertex_from.outgoing_halfedges().filter([&](HEH pn_heh) { return _pnd.map_to_layout_edges_[pn_heh] == l_eh.idx; }).first();
        assert(pn_iter_heh.is_valid());

        auto record_step = [&](HEH h) -> TStep
        {
            TStep s;
            s.heh = h.idx.value;
            s.om_a = (int)_pnd.map_to_overlay_vertices_.value()[h.vertex_from()].value;
            s.om_b = (int)_pnd.map_to_overlay_vertices_.value()[h.vertex_to()].value;
            s.seg_norm = (_pos.row(s.om_a) - _pos.row(s.om_b)).norm();
            return s;
        };

        // first segment: contributes to the accumulator only, its t stays 0
        TStep s0 = record_step(pn_iter_heh);
        s0.acc_before = 0.0;
        double acc = s0.seg_norm;
        chain.steps.push_back(s0);

        // total arc length over the layout-edge chain (same walk order)
        chain.chain_edges.push_back(l_eh.idx.value);
        double total = _lengths(l_eh.idx.value);
        auto l_heh_iter = l_heh;
        while (l_heh_iter.vertex_to().outgoing_halfedges().size() == 2)
        {
            l_heh_iter = l_heh_iter.next();
            l_heh_visited[l_heh_iter] = true;
            total = total + _lengths(l_heh_iter.edge().idx.value);
            chain.chain_edges.push_back(l_heh_iter.edge().idx.value);
        }
        chain.total = total;

        pn_iter_heh = pn_iter_heh.next();
        while (pn_iter_heh.vertex_from().outgoing_halfedges().size() == 2)
        {
            TStep s = record_step(pn_iter_heh);
            s.acc_before = acc;
            _t(s.heh) = 1.0 - (acc / total);
            acc = acc + s.seg_norm;
            chain.steps.push_back(s);
            pn_iter_heh = pn_iter_heh.next();
        }
        _t(pn_iter_heh.idx.value) = 0.0; // explicit zero at the chain exit, preserved from the original
    }
}

void pn_t_backward(TCtx const& _ctx,
                   Eigen::MatrixX3d const& _pos,
                   Eigen::VectorXd const& _d_t,
                   Eigen::MatrixX3d& _d_pos,
                   Eigen::VectorXd& _d_lengths)
{
    for (auto const& chain : _ctx.chains)
    {
        double const total = chain.total;
        double d_total = 0.0;
        double d_acc = 0.0; // adjoint of the running prefix sum

        auto scatter_seg = [&](TStep const& s, double d_n)
        {
            Eigen::RowVector3d const g = (d_n / s.seg_norm) * (_pos.row(s.om_a) - _pos.row(s.om_b));
            _d_pos.row(s.om_a) += g;
            _d_pos.row(s.om_b) -= g;
        };

        // forward per step k >= 1: t[h_k] = 1 - acc_before/total, then
        // acc += s_k. Reverse replays the steps backward, undoing the acc
        // update first, then the t assignment.
        for (size_t k = chain.steps.size(); k-- > 1;)
        {
            TStep const& s = chain.steps[k];

            scatter_seg(s, d_acc); // acc_k = acc_{k-1} + s_k => d_s_k = d_acc
            // d_acc carries to acc_{k-1} unchanged, plus the t contribution
            double const dt = _d_t(s.heh);
            d_acc += -dt / total;
            d_total += dt * s.acc_before / (total * total);
        }
        // step 0: acc_0 = 0 + s_0
        scatter_seg(chain.steps[0], d_acc);

        for (int e : chain.chain_edges)
            _d_lengths(e) += d_total;
    }
}

//=== sub-stage 3: boundary UVs per pn halfedge ==============================

namespace
{
// the constant corner factors of the unit parameter domain (the original
// implementation's float literals 0/1, promoted to double — exact)
constexpr double kFactors[4][2] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
} // namespace

void pn_uvs_forward(Eigen::VectorXd const& _lengths,
                    Eigen::VectorXd const& _t,
                    LayoutData const& _ld,
                    PathNetworkData const& _pnd,
                    bool _fixed_parameter_domain,
                    UvCtx& _ctx,
                    Eigen::MatrixX2d& _uvs)
{
    int64_t const H = (int64_t)_pnd.mesh_->halfedges().size();
    _ctx.n_pn_halfedges = H;
    _ctx.fixed_domain = _fixed_parameter_domain;
    _ctx.faces.clear();
    _ctx.faces.reserve(_ld.mesh_->faces().size());

    _uvs = Eigen::MatrixX2d::Zero(H, 2);

    for (auto l_fh : _ld.mesh_->faces())
    {
        _ctx.faces.emplace_back();
        UvFaceRec& rec = _ctx.faces.back();

        HEH l_heh = l_fh.halfedges().filter([](HEH _heh) { return _heh.vertex_from().outgoing_halfedges().size() != 2; }).first();
        assert(l_heh.is_valid());

        double length = 0.0;
        double height = 0.0;

        if (_fixed_parameter_domain)
        {
            length = 2.0;
            height = 2.0;
        }
        else
        {
            // sum the layout-edge chains of the 4 sides (original walk order)
            auto l_heh_iter = l_heh;
            auto sum_side = [&](std::vector<int>& _edges) -> double
            {
                double s = _lengths(l_heh_iter.edge().idx.value);
                _edges.push_back(l_heh_iter.edge().idx.value);
                while (l_heh_iter.vertex_to().outgoing_halfedges().size() == 2)
                {
                    l_heh_iter = l_heh_iter.next();
                    s += _lengths(l_heh_iter.edge().idx.value);
                    _edges.push_back(l_heh_iter.edge().idx.value);
                }
                return s;
            };

            double const length_0 = sum_side(rec.side_edges[0]);
            l_heh_iter = l_heh_iter.next();
            double const height_0 = sum_side(rec.side_edges[1]);
            l_heh_iter = l_heh_iter.next();
            double const length_1 = sum_side(rec.side_edges[2]);
            l_heh_iter = l_heh_iter.next();
            double const height_1 = sum_side(rec.side_edges[3]);

            rec.length_res = 0.5 * (length_0 + length_1);
            rec.height_res = 0.5 * (height_0 + height_1);
            length = std::max(0.02, rec.length_res);
            height = std::max(0.02, rec.height_res);
        }
        rec.length = length;
        rec.height = height;

        auto pn_vertex_from = _pnd.mesh_->handle_of(l_heh.vertex_from());
        HEH pn_iter_heh = pn_vertex_from.outgoing_halfedges()
                              .filter([&](HEH pn_heh) { return _pnd.map_to_layout_edges_[pn_heh] == l_heh.edge().idx; })
                              .first();
        assert(pn_iter_heh.is_valid());

        for (size_t i = 0; i < 4; ++i)
        {
            double const ax = kFactors[i][0] * length;
            double const ay = kFactors[i][1] * height;
            double const bx = kFactors[(i + 1) % 4][0] * length;
            double const by = kFactors[(i + 1) % 4][1] * height;

            rec.sides[i].start_heh = pn_iter_heh.idx.value;
            _uvs(pn_iter_heh.idx.value, 0) = ax;
            _uvs(pn_iter_heh.idx.value, 1) = ay;
            pn_iter_heh = pn_iter_heh.next();

            while (pn_iter_heh.vertex_from().adjacent_vertices().size() == 2)
            {
                double const t = _t(pn_iter_heh.idx.value);
                _uvs(pn_iter_heh.idx.value, 0) = t * ax + (1.0 - t) * bx;
                _uvs(pn_iter_heh.idx.value, 1) = t * ay + (1.0 - t) * by;
                rec.sides[i].inner_heh.push_back(pn_iter_heh.idx.value);
                pn_iter_heh = pn_iter_heh.next();
            }
            // the exit halfedge is the next side's start; assign B here (as the
            // original did) — it is overwritten by the next side's A with the
            // same value (after side 3 it re-writes side 0's start with (0,0))
            _uvs(pn_iter_heh.idx.value, 0) = bx;
            _uvs(pn_iter_heh.idx.value, 1) = by;
        }
    }
}

void pn_uvs_backward(UvCtx const& _ctx,
                     Eigen::VectorXd const& _t,
                     Eigen::MatrixX2d const& _d_uvs,
                     Eigen::VectorXd& _d_lengths,
                     Eigen::VectorXd& _d_t)
{
    for (auto const& rec : _ctx.faces)
    {
        double d_L = 0.0;
        double d_H = 0.0;

        for (int i = 0; i < 4; ++i)
        {
            double const fax = kFactors[i][0];
            double const fay = kFactors[i][1];
            double const fbx = kFactors[(i + 1) % 4][0];
            double const fby = kFactors[(i + 1) % 4][1];
            double const ax = fax * rec.length;
            double const ay = fay * rec.height;
            double const bx = fbx * rec.length;
            double const by = fby * rec.height;

            // start halfedge: final value is A_i = (fa.x*L, fa.y*H). (Side 0's
            // start actually holds B_3, whose factors are also (0,0) — the
            // contribution is identically zero either way.)
            d_L += fax * _d_uvs(rec.sides[i].start_heh, 0);
            d_H += fay * _d_uvs(rec.sides[i].start_heh, 1);

            for (int h : rec.sides[i].inner_heh)
            {
                double const t = _t(h);
                double const du = _d_uvs(h, 0);
                double const dv = _d_uvs(h, 1);

                // uv = t*A + (1-t)*B
                _d_t(h) += du * (ax - bx) + dv * (ay - by);
                d_L += fax * (t * du) + fbx * ((1.0 - t) * du);
                d_H += fay * (t * dv) + fby * ((1.0 - t) * dv);
            }
        }

        if (!_ctx.fixed_domain)
        {
            // max(0.02, res), torch max.other convention: grad to res iff 0.02 <= res
            double const d_len_res = rec.length_res >= 0.02 ? d_L : 0.0;
            double const d_hgt_res = rec.height_res >= 0.02 ? d_H : 0.0;

            // length_res = 0.5*(length_0 + length_1), each a sum over its chain
            for (int e : rec.side_edges[0])
                _d_lengths(e) += 0.5 * d_len_res;
            for (int e : rec.side_edges[2])
                _d_lengths(e) += 0.5 * d_len_res;
            for (int e : rec.side_edges[1])
                _d_lengths(e) += 0.5 * d_hgt_res;
            for (int e : rec.side_edges[3])
                _d_lengths(e) += 0.5 * d_hgt_res;
        }
    }
}

} // namespace LayoutOpt
