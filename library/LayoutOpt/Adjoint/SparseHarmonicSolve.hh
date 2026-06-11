#pragma once
// Phase 1 of the remove-autodiff plan: replace the dense torch::linalg::solve
// in torch_harmonic_param with an Eigen sparse solve wrapped in a custom
// torch::autograd::Function with a hand-derived adjoint backward.
//
// On Windows this also removes the MKL LAPACK dense-solve kernel that violates
// the Win64 ABI (clobbers callee-saved xmm14/xmm15) — see the baseline doc.
#ifdef LAYOUTOPT_WITH_TORCH

#include <torch/torch.h>

namespace LayoutOpt
{

// Solves the per-patch harmonic system A X = B for the inner-vertex UVs,
// where A := -L is the SPD interior cotan Laplacian and B := -R collects the
// boundary terms. The system structure is passed as index tensors so the
// forward/backward passes can assemble A and B (and their adjoints) without
// building an autograd graph through the assembly:
//
//   _cotans        [n_edges]    per-overlay-edge cotan weights (differentiable)
//   _boundary_uvs  [n_b, 2]     patch boundary UVs (differentiable)
//   _idx_off       [n_off, 3]   (i, j, e): A(i,j) = -cotans[e], inner neighbor j
//   _idx_diag      [n_diag, 2]  (i, e):    A(i,i) += cotans[e], one entry per
//                               outgoing halfedge of inner vertex i
//   _idx_rhs       [n_rhs, 3]   (i, b, e): B(i,:) += cotans[e] * uvs(b,:),
//                               boundary neighbor b
//   _n_inner                    number of inner vertices (rows of A / X)
//
// Index tensors are int64 and non-differentiable. Returns X [n_inner, 2]
// (float64) with gradients flowing back into _cotans and _boundary_uvs via
// the solve adjoint: with mu := A^-1 gbar,
//   d_cotans[e] += mu(i,:)·X(j,:)        (off-diagonal entries)
//   d_cotans[e] -= mu(i,:)·X(i,:)        (diagonal entries)
//   d_cotans[e] += mu(i,:)·uvs(b,:)      (rhs entries)
//   d_uvs(b,:)  += cotans[e] * mu(i,:)   (rhs entries)
at::Tensor sparse_harmonic_solve(at::Tensor const& _cotans,
                                 at::Tensor const& _boundary_uvs,
                                 at::Tensor const& _idx_off,
                                 at::Tensor const& _idx_diag,
                                 at::Tensor const& _idx_rhs,
                                 int64_t _n_inner);

} // namespace LayoutOpt

#endif // LAYOUTOPT_WITH_TORCH
