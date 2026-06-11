#include "CotanStage.hh"

namespace LayoutOpt
{

namespace
{
// The cotan-division regularizer from torch_cotans (TorchUtils.cc). NOT the
// Types EPS (1e-9) — torch_cotans uses 1e-10 literally, and it only appears in
// the division, not in the norm whose derivative is taken.
constexpr double kCotanEps = 1e-10;
} // namespace

void cotans_forward(Eigen::MatrixX3d const& _pos,
                    pm::Mesh const& _mesh,
                    CotanCtx& _ctx,
                    Eigen::VectorXd& _cotans)
{
    int64_t const E = (int64_t)_mesh.edges().size();
    _ctx.n_edges = E;
    _ctx.i_idx.assign((size_t)E, 0);
    _ctx.j_idx.assign((size_t)E, 0);
    _ctx.a_idx.assign((size_t)E, 0);
    _ctx.b_idx.assign((size_t)E, 0);
    _cotans = Eigen::VectorXd::Zero(E);

    for (auto eh : _mesh.edges())
    {
        auto h0 = eh.halfedgeA();
        auto h1 = eh.halfedgeB();

        int const i = h0.vertex_to().idx.value;
        int const j = h1.vertex_to().idx.value;
        int const a = h0.next().vertex_to().idx.value;
        int const b = h1.next().vertex_to().idx.value;

        int const e = eh.idx.value; // consumer (S6) indexes by edge idx
        _ctx.i_idx[(size_t)e] = i;
        _ctx.j_idx[(size_t)e] = j;
        _ctx.a_idx[(size_t)e] = a;
        _ctx.b_idx[(size_t)e] = b;

        vec3d const pi = _pos.row(i).transpose();
        vec3d const pj = _pos.row(j).transpose();
        vec3d const pa = _pos.row(a).transpose();
        vec3d const pb = _pos.row(b).transpose();

        vec3d const e_ia = pi - pa;
        vec3d const e_ja = pj - pa;
        vec3d const e_ib = pi - pb;
        vec3d const e_jb = pj - pb;

        double const denom_a = e_ia.cross(e_ja).norm();
        double const denom_b = e_ib.cross(e_jb).norm();
        double const dot_a = e_ia.dot(e_ja);
        double const dot_b = e_ib.dot(e_jb);

        double const cot_a = dot_a / (denom_a + kCotanEps);
        double const cot_b = dot_b / (denom_b + kCotanEps);

        _cotans(e) = cot_a + cot_b;
    }
}

void cotans_backward(CotanCtx const& _ctx,
                     Eigen::MatrixX3d const& _pos,
                     Eigen::VectorXd const& _d_cotans,
                     Eigen::MatrixX3d& _d_pos)
{
    // accumulate one (i, j, apex) corner triple's adjoint into _d_pos.
    auto scatter_side = [&](int i, int j, int apex, double d_cot)
    {
        vec3d const pi = _pos.row(i).transpose();
        vec3d const pj = _pos.row(j).transpose();
        vec3d const pap = _pos.row(apex).transpose();

        vec3d const e_i = pi - pap; // e_ia / e_ib
        vec3d const e_j = pj - pap; // e_ja / e_jb

        vec3d const cross = e_i.cross(e_j);
        double const denom = cross.norm();
        double const dot = e_i.dot(e_j);
        double const denom_eps = denom + kCotanEps;

        // cot = dot / (denom + eps)
        double const d_dot = d_cot / denom_eps;
        double const d_denom = -d_cot * dot / (denom_eps * denom_eps);

        // dot = e_i . e_j
        vec3d d_e_i = d_dot * e_j;
        vec3d d_e_j = d_dot * e_i;

        // denom = |cross|  =>  d_cross = d_denom * cross / denom (bare norm)
        vec3d const d_cross = d_denom * (cross / denom);

        // cross = e_i x e_j  =>  d_e_i += e_j x d_cross ; d_e_j += d_cross x e_i
        d_e_i += e_j.cross(d_cross);
        d_e_j += d_cross.cross(e_i);

        // e_i = pi - pap ; e_j = pj - pap
        _d_pos.row(i) += d_e_i.transpose();
        _d_pos.row(j) += d_e_j.transpose();
        _d_pos.row(apex) -= (d_e_i + d_e_j).transpose();
    };

    for (int64_t e = 0; e < _ctx.n_edges; ++e)
    {
        double const d_cot = _d_cotans(e);
        // cot[e] = cot_a + cot_b, so both sides receive d_cot.
        scatter_side(_ctx.i_idx[(size_t)e], _ctx.j_idx[(size_t)e], _ctx.a_idx[(size_t)e], d_cot);
        scatter_side(_ctx.i_idx[(size_t)e], _ctx.j_idx[(size_t)e], _ctx.b_idx[(size_t)e], d_cot);
    }
}

} // namespace LayoutOpt
