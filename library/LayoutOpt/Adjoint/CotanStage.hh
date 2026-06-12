#pragma once
// S5 of the adjoint chain (see documentation/adjointDifferentiation.md):
// hand-rolled forward + reverse-mode adjoint of the per-edge cotan-weight
// stage. Per overlay edge, the cotan weight is the sum of the two
// opposite-corner cotangents:
//
//   cot[e] = (e_ia·e_ja)/(|e_ia x e_ja| + eps) + (e_ib·e_jb)/(|e_ib x e_jb| + eps)
//
// where i,j are the edge endpoints and a,b the two opposite apex vertices
// (i = halfedgeA->to, j = halfedgeB->to, a = halfedgeA.next->to,
// b = halfedgeB.next->to). The +eps (1e-10, NOT Types EPS) lives only in the
// cotan division; the norm derivative uses the bare |cross|, following torch's
// norm-backward convention.
//
// Stage boundary (inputs treated as leaves for validation):
//   inputs  : overlay 3D positions [n_overlay x 3] (S1/S3 output)
//   output  : per-edge cotan weights [n_edges] (S6 input)
//   adjoints: d_overlay_pos (accumulated; the cotan path is one of two paths
//             positions feed — the other is the direct 3D distortion path in S7)
//
// NOTE on face areas: the original torch implementation also computed per-face
// areas here, but their result was dead in the live loss path (only the
// experimental Yamabe losses consumed them, and those were deleted with the
// torch removal) — so face areas were deliberately never ported.

#include <cstdint>
#include <vector>

#include <polymesh/Mesh.hh>

#include <LayoutOpt/DataStructures/Types.hh>

namespace LayoutOpt
{

// Per-edge incident vertex indices (i, j, a, b), stored at edge idx so the
// output / ctx line up with the consumer (S6 indexes cotans by edge idx).
// Backward recomputes the edge vectors / crosses from the positions, so only
// the topology needs stowing.
struct CotanCtx
{
    int64_t n_edges = 0;
    std::vector<int> i_idx; // edge endpoint (halfedgeA->to)
    std::vector<int> j_idx; // edge endpoint (halfedgeB->to)
    std::vector<int> a_idx; // apex opposite on side A (halfedgeA.next->to)
    std::vector<int> b_idx; // apex opposite on side B (halfedgeB.next->to)
};

// Forward: fill _cotans [n_edges] (indexed by edge idx) and _ctx. The
// arithmetic per edge matches the original implementation op for op.
void cotans_forward(Eigen::MatrixX3d const& _pos,
                    pm::Mesh const& _mesh,
                    CotanCtx& _ctx,
                    Eigen::VectorXd& _cotans);

// Adjoint of cotans_forward. Given _d_cotans [n_edges], accumulate into
// _d_pos [n_overlay x 3]. Recomputes per-edge intermediates from _pos.
void cotans_backward(CotanCtx const& _ctx,
                     Eigen::MatrixX3d const& _pos,
                     Eigen::VectorXd const& _d_cotans,
                     Eigen::MatrixX3d& _d_pos);

} // namespace LayoutOpt
