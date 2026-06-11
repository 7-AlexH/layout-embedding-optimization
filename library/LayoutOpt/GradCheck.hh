#pragma once
// Gradient-checking harness for the hand-rolled adjoint port.
// All functionality requires LAYOUTOPT_WITH_TORCH (torch oracle path).
#ifdef LAYOUTOPT_WITH_TORCH

#include <string>
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/Types.hh"

namespace LayoutOpt
{

// ---------------------------------------------------------------------------
// Leaf-gradient dump / compare
// ---------------------------------------------------------------------------

// Serialize per-leaf bary_coords.grad() to a text file.
// Only layout-node FacePoint vertices (requires_grad=true) are written.
// File format: header line, then one "pn_vh_idx grad_x grad_y" line per leaf.
// Call after loss.backward().
void gradcheck_dump_leaf_grads(PathNetworkData const& _pnd, std::string const& _path);

// Load leaf grads dumped by gradcheck_dump_leaf_grads and compare against
// _hand_grads (Eigen::Vector2d per pn vertex; zero for non-leaf vertices).
// Prints per-leaf diffs and a summary. Returns the max absolute diff seen.
// Returns -1.0 on file-load failure.
double gradcheck_compare_leaf_grads(PathNetworkData const& _pnd,
                                     pm::vertex_attribute<vec2d> const& _hand_grads,
                                     std::string const& _path,
                                     double _atol = 1e-9,
                                     double _rtol = 1e-7);

// ---------------------------------------------------------------------------
// Intermediate-grad retention (call before the forward pass in gradcheck mode)
// ---------------------------------------------------------------------------
// Adds retain_grad() to tensors at stage-boundary nodes so their .grad() fields
// are populated after backward. The stage-boundary map:
//   _omd.torch_pos_        S2/S3 boundary (overlay vertex positions)
//   _pnd.torch_uvs_        S4   boundary (path-network boundary UVs)
//   _pnd.torch_t_          S4   internal (arc-length t values)
//   _ld.torch_embedded_edge_length_  S4 input (arc lengths per layout edge)
// cotans and per-patch L/rhs tensors need retain_grad() inside
// torch_cotans / torch_harmonic_param respectively (add in Phase 1).
void gradcheck_retain_intermediate_grads(LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd);

// Dump intermediate grads (after backward) alongside leaf grads.
// Appends extra sections to the file written by gradcheck_dump_leaf_grads.
void gradcheck_dump_intermediate_grads(LayoutData const& _ld,
                                        PathNetworkData const& _pnd,
                                        OverlayMeshData const& _omd,
                                        std::string const& _path);

} // namespace LayoutOpt

#endif // LAYOUTOPT_WITH_TORCH
