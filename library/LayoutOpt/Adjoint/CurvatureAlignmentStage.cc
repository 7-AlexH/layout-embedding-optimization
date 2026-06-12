#include "CurvatureAlignmentStage.hh"

#include <cassert>
#include <cmath>

#include <polymesh/properties.hh>

namespace LayoutOpt
{

double curvature_alignment_forward(Eigen::MatrixX3d const& _pos,
                                   TargetMeshData const& _tmd,
                                   PathNetworkData const& _pnd,
                                   OverlayMeshData const& _omd,
                                   CurvAlignCtx& _ctx)
{
    assert(_tmd.direction_field_data_.has_value());

    _ctx = CurvAlignCtx{};
    _ctx.edges.reserve(_pnd.mesh_->edges().size());

    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto o_vhA = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_eh.vertexA()]);
        auto o_vhB = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_eh.vertexB()]);

        auto o_eh = pm::edge_between(o_vhA, o_vhB);
        assert(o_eh.is_valid());

        auto t_fh = _omd.map_to_target_faces_.value()[o_eh.faceA()];
        assert(t_fh.is_valid());

        CurvAlignEdge rec;
        rec.om_a = o_vhA.idx.value;
        rec.om_b = o_vhB.idx.value;

        auto const& direction_data = _tmd.direction_field_data_.value()[t_fh];
        rec.basis = direction_data.basis_eigen;
        rec.dir = direction_data.dir_eigen;

        Eigen::Vector3d const vec = (_pos.row(rec.om_a) - _pos.row(rec.om_b)).transpose();
        rec.norm = vec.norm();
        rec.v2 = rec.basis * vec;
        double const angle = std::atan2(rec.v2.y(), rec.v2.x());
        double const angle_4 = 4.0 * angle;
        rec.cos4 = std::cos(angle_4);
        rec.sin4 = std::sin(angle_4);
        double const dx = rec.dir.x() - rec.cos4;
        double const dy = rec.dir.y() - rec.sin4;
        rec.align = dx * dx + dy * dy;

        _ctx.num += rec.norm * rec.align;
        _ctx.den += rec.norm;
        _ctx.edges.push_back(rec);
    }

    return _ctx.num / _ctx.den;
}

void curvature_alignment_backward(CurvAlignCtx const& _ctx,
                                  Eigen::MatrixX3d const& _pos,
                                  double _d_loss,
                                  Eigen::MatrixX3d& _d_pos)
{
    // loss = num / den
    double const d_num = _d_loss / _ctx.den;
    double const d_den = -_d_loss * _ctx.num / (_ctx.den * _ctx.den);

    for (size_t k = _ctx.edges.size(); k-- > 0;)
    {
        CurvAlignEdge const& rec = _ctx.edges[k];

        // num += n * align ; den += n
        double const d_n = d_num * rec.align + d_den;
        double const d_align = d_num * rec.norm;

        // align = (dir.x - cos4)^2 + (dir.y - sin4)^2
        double const d_cos4 = -2.0 * (rec.dir.x() - rec.cos4) * d_align;
        double const d_sin4 = -2.0 * (rec.dir.y() - rec.sin4) * d_align;

        // rosy = (cos th4, sin th4)
        double const d_th4 = -rec.sin4 * d_cos4 + rec.cos4 * d_sin4;
        double const d_th = 4.0 * d_th4;

        // th = atan2(v2.y, v2.x):  d/dx = -y/(x^2+y^2), d/dy = x/(x^2+y^2)
        double const r2 = rec.v2.x() * rec.v2.x() + rec.v2.y() * rec.v2.y();
        vec2d const d_v2(-rec.v2.y() / r2 * d_th, rec.v2.x() / r2 * d_th);

        // v2 = basis * vec ; n = |vec|
        Eigen::Vector3d const vec = (_pos.row(rec.om_a) - _pos.row(rec.om_b)).transpose();
        Eigen::Vector3d const d_vec = rec.basis.transpose() * d_v2 + (d_n / rec.norm) * vec;

        // vec = pos[a] - pos[b]
        _d_pos.row(rec.om_a) += d_vec.transpose();
        _d_pos.row(rec.om_b) -= d_vec.transpose();
    }
}

} // namespace LayoutOpt
