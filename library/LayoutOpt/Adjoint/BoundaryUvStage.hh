#pragma once
// Phase 4 / S4 of the remove-autodiff plan: hand-rolled forward + reverse-mode
// adjoint of the boundary-UV stage — the three "prepare" functions at the top
// of harmonic_distortion_loss:
//
//   torch_compute_embedded_layout_edge_lengths -> edge_lengths_forward/backward
//   torch_compute_t_for_pn_halfedge            -> pn_t_forward/backward
//   torch_compute_pn_uvs                       -> pn_uvs_forward/backward
//
// Stage boundary (validation per plan section 4/S4 — inputs treated as
// leaves):
//   inputs  : overlay 3D positions [n_overlay_v x 3]
//   outputs : per-layout-edge arc lengths [n_layout_e], per-pn-halfedge t
//             [n_pn_heh], per-pn-halfedge boundary UVs [n_pn_heh x 2]
//   adjoints: d_overlay_pos (accumulated), with d_lengths / d_t as
//             intermediates at the sub-stage boundaries
//
// Each forward records the walk order (halfedge indices, segment endpoints,
// prefix sums) in a context; backward replays the recorded arrays in reverse
// — it never re-walks the mesh. Saved forward values (segment norms, chain
// totals, patch length/height) are reused in backward, like autograd does.
//
// Backward call order matters: pn_uvs_backward and pn_t_backward both
// accumulate into d_lengths, so edge_lengths_backward must run last
// (pn_uvs_backward -> pn_t_backward -> edge_lengths_backward).
//
// Faithfulness notes (vs the torch path):
//  - segment norms are bare |a - b| (no epsilon); the adjoint divides by the
//    saved norm, so a degenerate zero-length segment would produce inf/nan —
//    exactly as the torch path would.
//  - torch::max(0.02, res) on the patch length/height (max.other derivative):
//    the gradient flows to res iff 0.02 <= res, else it is dropped.
//  - the torch uv walk assigns each side's exit halfedge B and then overwrites
//    it with the next side's A (bitwise the same value); side 0's start ends up
//    holding B_3 whose corner factors are (0,0), identical in value and
//    gradient (zero) to the overwritten A_0, so backward can treat every side
//    start uniformly as A_i.

#include <cstdint>
#include <vector>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>

namespace LayoutOpt
{

//=== sub-stage 1: per-layout-edge embedded arc lengths ======================

struct EdgeLengthCtx
{
    int64_t n_layout_edges = 0;
    // one record per pn edge, in pn-mesh edge order (a scatter-add; the order
    // only matters for bitwise-matching the forward per-edge sums)
    std::vector<int> l_edge;        // layout edge idx the segment belongs to
    std::vector<int> om_a, om_b;    // overlay vertex idx of the segment endpoints
    std::vector<double> seg_norm;   // saved forward |pos[a] - pos[b]|
};

// lengths[l_e] = sum over its pn segments of |pos[a] - pos[b]|
void edge_lengths_forward(Eigen::MatrixX3d const& _pos,
                          LayoutData const& _ld,
                          PathNetworkData const& _pnd,
                          EdgeLengthCtx& _ctx,
                          Eigen::VectorXd& _lengths);

// d_pos[a] += d_lengths[l_e] * (pos[a] - pos[b]) / seg_norm (and -= for b)
void edge_lengths_backward(EdgeLengthCtx const& _ctx,
                           Eigen::MatrixX3d const& _pos,
                           Eigen::VectorXd const& _d_lengths,
                           Eigen::MatrixX3d& _d_pos);

//=== sub-stage 2: arc-length parameter t per pn halfedge ====================

struct TStep
{
    int heh = -1;             // pn halfedge idx; steps k >= 1 get t assigned,
                              // step 0 only feeds the accumulator
    int om_a = -1, om_b = -1; // segment endpoints (overlay vertex idx)
    double seg_norm = 0.0;    // saved forward segment norm
    double acc_before = 0.0;  // prefix sum of seg norms over steps 0..k-1
                              // (the accumulator value t at this step divides)
};

struct TChain // one walked arc: a layout halfedge chain and its pn halfedges
{
    double total = 0.0;           // saved sum of lengths over chain_edges
    std::vector<int> chain_edges; // layout edge idxs summed into total
    std::vector<TStep> steps;
};

struct TCtx
{
    int64_t n_pn_halfedges = 0;
    std::vector<TChain> chains;
};

// t[h_k] = 1 - (s_0 + ... + s_{k-1}) / total, walked exactly like the torch
// function (both halfedge directions of every arc form their own chain).
// Halfedges whose from-vertex is a layout corner keep t = 0.
void pn_t_forward(Eigen::MatrixX3d const& _pos,
                  Eigen::VectorXd const& _lengths,
                  LayoutData const& _ld,
                  PathNetworkData const& _pnd,
                  TCtx& _ctx,
                  Eigen::VectorXd& _t);

// Replays each chain in reverse, carrying the accumulator adjoint backward:
//   d_s_k     += d_acc                      (acc_k = acc_{k-1} + s_k)
//   d_acc     += -d_t[h_k] / total          (t = 1 - acc_before/total)
//   d_total   += d_t[h_k] * acc_before / total^2
// then scatters each d_s_k through the segment-norm adjoint into _d_pos and
// d_total into _d_lengths over the chain's layout edges.
void pn_t_backward(TCtx const& _ctx,
                   Eigen::MatrixX3d const& _pos,
                   Eigen::VectorXd const& _d_t,
                   Eigen::MatrixX3d& _d_pos,
                   Eigen::VectorXd& _d_lengths);

//=== sub-stage 3: boundary UVs per pn halfedge ==============================

struct UvSide
{
    int start_heh = -1;         // gets the corner UV A_i = (f_i.x*L, f_i.y*H)
    std::vector<int> inner_heh; // get t*A + (1-t)*B
};

struct UvFaceRec
{
    double length = 0.0, height = 0.0;         // saved post-max L/H
    double length_res = 0.0, height_res = 0.0; // pre-max values (gate the max
                                               // adjoint); unused if fixed domain
    std::vector<int> side_edges[4]; // layout edge chains: [0]=length_0,
                                    // [1]=height_0, [2]=length_1, [3]=height_1
    UvSide sides[4];
};

struct UvCtx
{
    int64_t n_pn_halfedges = 0;
    bool fixed_domain = false;
    std::vector<UvFaceRec> faces;
};

// Per layout face: L/H from the four side sums (or the 2.0 constants if
// _fixed_parameter_domain), then walk the patch boundary assigning corner UVs
// and t-interpolated UVs, exactly like torch_compute_pn_uvs.
void pn_uvs_forward(Eigen::VectorXd const& _lengths,
                    Eigen::VectorXd const& _t,
                    LayoutData const& _ld,
                    PathNetworkData const& _pnd,
                    bool _fixed_parameter_domain,
                    UvCtx& _ctx,
                    Eigen::MatrixX2d& _uvs);

// uv = t*A + (1-t)*B with A/B linear in (L, H):
//   d_t[h]  += d_uv . (A - B)
//   d_L     += f_a.x*(t*d_uv.x) + f_b.x*((1-t)*d_uv.x)   (analogous for d_H)
// then d_L/d_H pass the max(0.02, .) gate and scatter 0.5*d into d_lengths
// over the recorded side chains (skipped entirely for a fixed domain).
void pn_uvs_backward(UvCtx const& _ctx,
                     Eigen::VectorXd const& _t,
                     Eigen::MatrixX2d const& _d_uvs,
                     Eigen::VectorXd& _d_lengths,
                     Eigen::VectorXd& _d_t);

} // namespace LayoutOpt
