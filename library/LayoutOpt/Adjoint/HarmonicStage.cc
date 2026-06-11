#include "HarmonicStage.hh"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <iostream>
#include <stdexcept>

#include <polymesh/properties.hh>

namespace LayoutOpt
{

namespace
{

// Assemble A := -L from the flattened structure lists. Same triplet order as
// the Phase 1 SparseHarmonicSolve assembly (all off-diagonal entries, then all
// diagonal entries; setFromTriplets sums duplicates), so the factorization and
// solution are bitwise-equal to the production sparse path.
Eigen::SparseMatrix<double> assemble_system_matrix(Eigen::VectorXd const& _w,
                                                   std::vector<int64_t> const& _off,
                                                   std::vector<int64_t> const& _diag,
                                                   int64_t _n)
{
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(_off.size() / 3 + _diag.size() / 2);
    for (size_t k = 0; k < _off.size(); k += 3)
        triplets.emplace_back((int)_off[k], (int)_off[k + 1], -_w[_off[k + 2]]);
    for (size_t k = 0; k < _diag.size(); k += 2)
        triplets.emplace_back((int)_diag[k], (int)_diag[k], _w[_diag[k + 1]]);

    Eigen::SparseMatrix<double> A(_n, _n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

Eigen::MatrixXd solve_system(Eigen::SparseMatrix<double> const& _A, Eigen::MatrixXd const& _B)
{
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(_A);
    if (ldlt.info() == Eigen::Success)
        return ldlt.solve(_B);

    // Negative cotan weights on badly shaped overlay triangles can make the
    // system indefinite enough for LDLT to fail; fall back to a general LU.
    std::cerr << "harmonic_param: LDLT factorization failed, falling back to SparseLU" << std::endl;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(_A);
    if (lu.info() != Eigen::Success)
        throw std::runtime_error("harmonic_param: sparse factorization failed");
    return lu.solve(_B);
}

} // namespace

void prepare_param(int const _patch_value,
                   PathNetworkData const& _pnd,
                   OverlayMeshData const& _omd,
                   Eigen::MatrixX2d const& _o_uvs,
                   pm::vertex_attribute<MappingIndex>& _map_to_vec,
                   std::vector<VH>& _inner_patch_vhs,
                   std::vector<VH>& _boundary_patch_vhs,
                   std::vector<vec2d>& _boundary_uvs,
                   std::vector<FH>& _patch_fhs,
                   std::vector<int>* _boundary_src_hehs)
{
    // conservative reserve
    _patch_fhs.reserve(_omd.mesh_->faces().size());
    _boundary_patch_vhs.reserve(_omd.mesh_->vertices().size());
    _inner_patch_vhs.reserve(_omd.mesh_->vertices().size());
    _boundary_uvs.reserve(_omd.mesh_->vertices().size());

    // collect all relevant vertices
    for (auto o_fh : _omd.mesh_->faces())
    {
        // the face does not belong to the current patch
        if (_omd.map_to_layout_faces_.value()[o_fh].value != _patch_value)
            continue;

        _patch_fhs.push_back(o_fh);
        for (auto o_heh : o_fh.halfedges())
        {
            auto o_vh_from = o_heh.vertex_from();
            auto o_vh_to = o_heh.vertex_to();

            // already visited
            if (_map_to_vec[o_vh_from].idx >= 0)
                continue;

            auto pn_vh_from = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[o_vh_from]);
            auto pn_vh_to = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[o_vh_to]);

            HEH pn_heh = HEH::invalid;
            if (pn_vh_from.is_valid() && pn_vh_to.is_valid())
            {
                pn_heh = pm::halfedge_from_to(pn_vh_from, pn_vh_to);
            }

            // boundary vertex and uvs are stored on this halfedge
            if (pn_heh.is_valid())
            {
                _boundary_patch_vhs.push_back(o_vh_from);
                int idx = static_cast<int>(_boundary_patch_vhs.size()) - 1;
                _map_to_vec[o_vh_from] = {idx, true};
                _boundary_uvs.push_back(vec2d(_o_uvs.row(o_heh.idx.value).transpose()));
                if (_boundary_src_hehs)
                    _boundary_src_hehs->push_back(o_heh.idx.value);
            }
            else if (pn_vh_from.is_invalid())
            {
                _inner_patch_vhs.push_back(o_vh_from);
                int idx = static_cast<int>(_inner_patch_vhs.size()) - 1;
                _map_to_vec[o_vh_from] = {idx, false};
            }
        }
    }
}

PatchHarmonicCtx harmonic_param_forward(pm::vertex_attribute<MappingIndex> const& _map_to_vec,
                                        std::vector<VH> const& _inner_patch_vhs,
                                        std::vector<vec2d> const& _boundary_uvs,
                                        Eigen::VectorXd const& _cotans)
{
    PatchHarmonicCtx ctx;
    ctx.n_inner = (int)_inner_patch_vhs.size();
    ctx.inner_uvs = Eigen::MatrixX2d::Zero(ctx.n_inner, 2);
    if (ctx.n_inner == 0)
        return ctx;

    // system structure — same traversal as torch_harmonic_param's sparse path
    for (int i = 0; i < ctx.n_inner; ++i)
    {
        auto o_vh = _inner_patch_vhs[i];
        for (HEH o_heh : o_vh.outgoing_halfedges())
        {
            MappingIndex j = _map_to_vec[o_heh.vertex_to()];
            int64_t e = o_heh.edge().idx.value;

            ctx.diag.push_back(i);
            ctx.diag.push_back(e);
            if (j.on_boundary)
            {
                ctx.rhs.push_back(i);
                ctx.rhs.push_back(j.idx);
                ctx.rhs.push_back(e);
            }
            else
            {
                ctx.off.push_back(i);
                ctx.off.push_back(j.idx);
                ctx.off.push_back(e);
            }
        }
    }

    auto A = assemble_system_matrix(_cotans, ctx.off, ctx.diag, ctx.n_inner);

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(ctx.n_inner, 2);
    for (size_t k = 0; k < ctx.rhs.size(); k += 3)
    {
        auto i = ctx.rhs[k];
        auto b = ctx.rhs[k + 1];
        auto e = ctx.rhs[k + 2];
        B(i, 0) += _cotans[e] * _boundary_uvs[b].x();
        B(i, 1) += _cotans[e] * _boundary_uvs[b].y();
    }

    ctx.inner_uvs = solve_system(A, B);
    return ctx;
}

void harmonic_param_backward(PatchHarmonicCtx const& _ctx,
                             std::vector<vec2d> const& _boundary_uvs,
                             Eigen::VectorXd const& _cotans,
                             Eigen::MatrixX2d const& _d_inner_uvs,
                             Eigen::VectorXd& _d_cotans,
                             std::vector<vec2d>& _d_boundary_uvs)
{
    if (_ctx.n_inner == 0)
        return;

    // mu := A^-T d_inner; A is symmetric, so solve with A directly. The
    // factorization is redone here (mirrors the Phase 1 custom function).
    auto A = assemble_system_matrix(_cotans, _ctx.off, _ctx.diag, _ctx.n_inner);
    Eigen::MatrixXd Mu = solve_system(A, _d_inner_uvs);

    Eigen::MatrixX2d const& X = _ctx.inner_uvs;

    // dL/dA = -mu X^T, scattered through A's entry structure:
    // off-diagonal A(i,j) = -w[e]  =>  d_w[e] += mu(i,:)·X(j,:)
    for (size_t k = 0; k < _ctx.off.size(); k += 3)
    {
        auto i = _ctx.off[k];
        auto j = _ctx.off[k + 1];
        auto e = _ctx.off[k + 2];
        _d_cotans[e] += Mu(i, 0) * X(j, 0) + Mu(i, 1) * X(j, 1);
    }
    // diagonal A(i,i) += w[e]  =>  d_w[e] -= mu(i,:)·X(i,:)
    for (size_t k = 0; k < _ctx.diag.size(); k += 2)
    {
        auto i = _ctx.diag[k];
        auto e = _ctx.diag[k + 1];
        _d_cotans[e] -= Mu(i, 0) * X(i, 0) + Mu(i, 1) * X(i, 1);
    }
    // dL/dB = mu; B(i,:) += w[e] * uvs(b,:)
    for (size_t k = 0; k < _ctx.rhs.size(); k += 3)
    {
        auto i = _ctx.rhs[k];
        auto b = _ctx.rhs[k + 1];
        auto e = _ctx.rhs[k + 2];
        _d_cotans[e] += Mu(i, 0) * _boundary_uvs[b].x() + Mu(i, 1) * _boundary_uvs[b].y();
        _d_boundary_uvs[b].x() += _cotans[e] * Mu(i, 0);
        _d_boundary_uvs[b].y() += _cotans[e] * Mu(i, 1);
    }
}

} // namespace LayoutOpt
