#ifdef LAYOUTOPT_WITH_TORCH

#include "SparseHarmonicSolve.hh"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace LayoutOpt
{

namespace
{

// Assemble A := -L from the structure tensors. setFromTriplets sums duplicate
// entries, which is what the diagonal accumulation relies on (one triplet per
// outgoing halfedge of each inner vertex).
Eigen::SparseMatrix<double> assemble_system_matrix(double const* _w,
                                                   at::TensorAccessor<int64_t, 2> const& _off,
                                                   at::TensorAccessor<int64_t, 2> const& _diag,
                                                   int64_t _n)
{
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(_off.size(0) + _diag.size(0));
    for (int64_t k = 0; k < _off.size(0); ++k)
        triplets.emplace_back((int)_off[k][0], (int)_off[k][1], -_w[_off[k][2]]);
    for (int64_t k = 0; k < _diag.size(0); ++k)
        triplets.emplace_back((int)_diag[k][0], (int)_diag[k][0], _w[_diag[k][1]]);

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
    std::cerr << "sparse_harmonic_solve: LDLT factorization failed, falling back to SparseLU" << std::endl;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(_A);
    if (lu.info() != Eigen::Success)
        throw std::runtime_error("sparse_harmonic_solve: sparse factorization failed");
    return lu.solve(_B);
}

struct SparseHarmonicSolveFunction : public torch::autograd::Function<SparseHarmonicSolveFunction>
{
    static at::Tensor forward(torch::autograd::AutogradContext* _ctx,
                              at::Tensor _cotans,
                              at::Tensor _boundary_uvs,
                              at::Tensor _idx_off,
                              at::Tensor _idx_diag,
                              at::Tensor _idx_rhs,
                              int64_t _n_inner)
    {
        auto cotans = _cotans.contiguous();
        auto uvs = _boundary_uvs.contiguous();
        double const* w = cotans.data_ptr<double>();
        auto uv_acc = uvs.accessor<double, 2>();
        auto off = _idx_off.accessor<int64_t, 2>();
        auto diag = _idx_diag.accessor<int64_t, 2>();
        auto rhs = _idx_rhs.accessor<int64_t, 2>();

        auto A = assemble_system_matrix(w, off, diag, _n_inner);

        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(_n_inner, 2);
        for (int64_t k = 0; k < rhs.size(0); ++k)
        {
            auto i = rhs[k][0];
            auto b = rhs[k][1];
            auto e = rhs[k][2];
            B(i, 0) += w[e] * uv_acc[b][0];
            B(i, 1) += w[e] * uv_acc[b][1];
        }

        Eigen::MatrixXd X = solve_system(A, B);

        auto out = torch::empty({_n_inner, 2}, torch::dtype(torch::kFloat64));
        auto out_acc = out.accessor<double, 2>();
        for (int64_t i = 0; i < _n_inner; ++i)
        {
            out_acc[i][0] = X(i, 0);
            out_acc[i][1] = X(i, 1);
        }

        // The factorization cannot be stowed in saved_data, so backward
        // re-assembles and re-factors A (still far cheaper than the dense
        // solve this replaces).
        _ctx->save_for_backward({cotans, uvs, _idx_off, _idx_diag, _idx_rhs, out});
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* _ctx,
                                                   torch::autograd::variable_list _grad_outputs)
    {
        auto saved = _ctx->get_saved_variables();
        auto const& cotans = saved[0];
        auto const& uvs = saved[1];
        auto const& idx_off = saved[2];
        auto const& idx_diag = saved[3];
        auto const& idx_rhs = saved[4];
        auto const& X_t = saved[5];

        if (!_grad_outputs[0].defined())
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};

        auto gbar = _grad_outputs[0].contiguous();
        int64_t n = X_t.size(0);

        double const* w = cotans.data_ptr<double>();
        auto uv_acc = uvs.accessor<double, 2>();
        auto off = idx_off.accessor<int64_t, 2>();
        auto diag = idx_diag.accessor<int64_t, 2>();
        auto rhs = idx_rhs.accessor<int64_t, 2>();
        auto x_acc = X_t.accessor<double, 2>();
        auto g_acc = gbar.accessor<double, 2>();

        // mu := A^-T gbar; A is symmetric, so solve with A directly.
        auto A = assemble_system_matrix(w, off, diag, n);
        Eigen::MatrixXd G(n, 2);
        for (int64_t i = 0; i < n; ++i)
        {
            G(i, 0) = g_acc[i][0];
            G(i, 1) = g_acc[i][1];
        }
        Eigen::MatrixXd Mu = solve_system(A, G);

        auto d_cotans = torch::zeros_like(cotans);
        auto d_uvs = torch::zeros_like(uvs);
        double* dw = d_cotans.data_ptr<double>();
        auto duv_acc = d_uvs.accessor<double, 2>();

        // dL/dA = -mu X^T, scattered through A's entry structure:
        // off-diagonal A(i,j) = -w[e]  =>  d_w[e] += mu(i,:)·X(j,:)
        for (int64_t k = 0; k < off.size(0); ++k)
        {
            auto i = off[k][0];
            auto j = off[k][1];
            auto e = off[k][2];
            dw[e] += Mu(i, 0) * x_acc[j][0] + Mu(i, 1) * x_acc[j][1];
        }
        // diagonal A(i,i) += w[e]  =>  d_w[e] -= mu(i,:)·X(i,:)
        for (int64_t k = 0; k < diag.size(0); ++k)
        {
            auto i = diag[k][0];
            auto e = diag[k][1];
            dw[e] -= Mu(i, 0) * x_acc[i][0] + Mu(i, 1) * x_acc[i][1];
        }
        // dL/dB = mu; B(i,:) += w[e] * uvs(b,:)
        for (int64_t k = 0; k < rhs.size(0); ++k)
        {
            auto i = rhs[k][0];
            auto b = rhs[k][1];
            auto e = rhs[k][2];
            dw[e] += Mu(i, 0) * uv_acc[b][0] + Mu(i, 1) * uv_acc[b][1];
            duv_acc[b][0] += w[e] * Mu(i, 0);
            duv_acc[b][1] += w[e] * Mu(i, 1);
        }

        // one slot per forward arg; index tensors and n_inner are non-differentiable
        return {d_cotans, d_uvs, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

} // namespace

at::Tensor sparse_harmonic_solve(at::Tensor const& _cotans,
                                 at::Tensor const& _boundary_uvs,
                                 at::Tensor const& _idx_off,
                                 at::Tensor const& _idx_diag,
                                 at::Tensor const& _idx_rhs,
                                 int64_t _n_inner)
{
    return SparseHarmonicSolveFunction::apply(_cotans, _boundary_uvs, _idx_off, _idx_diag, _idx_rhs, _n_inner);
}

} // namespace LayoutOpt

#endif // LAYOUTOPT_WITH_TORCH
