#pragma once

#include <polymesh/attributes.hh>
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/Types.hh"
#include "LayoutOpt/OptimizationOptions.hh"

namespace LayoutOpt
{

struct EvalInfo
{
    // check if gradients are available
    bool success = false;

    // loss
    double loss;

#ifdef LAYOUTOPT_WITH_TORCH
    // updates
    pm::vertex_attribute<torch::Tensor> pn_sp_grads; // only for facepoints
    pm::vertex_attribute<at::Tensor> pn_sp_update_dirs;
#endif

    // for hamonic loss
    pm::face_attribute<double> o_distortion_harmonic;
    pm::halfedge_attribute<pos2> o_uvs;

#ifdef LAYOUTOPT_WITH_TORCH
    // for variance loss (attic — inactive)
    pm::face_attribute<double> o_local_vars;
    pm::face_attribute<double> l_per_patch_vars;
    // for distorion_loss (attic — inactive)
    pm::vertex_attribute<double> o_scaling_factor;
    pm::vertex_attribute<double> o_scaling_factor_extrinsic;
#endif
};

//=======================================================================================================================================================================

// harmonic_distortion_loss
torch::Tensor harmonic_distortion_loss(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, HarmonicOptions const& _variant, EvalInfo& _eval_info);

// principal curvature alignement
torch::Tensor principal_curvature_alignment_loss(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, EvalInfo& _eval_info);

// --- inactive losses (implementations in ObjectiveFunctionsAttic.cc) ---
#ifdef LAYOUTOPT_WITH_TORCH
// variance loss
torch::Tensor variance_loss(TargetMeshData const& _tmd, LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo& _eval_info);

// distortion minimization loss -> yamabe equation
torch::Tensor distorion_loss(TargetMeshData const& _tmd, LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo& _eval_info);

// distortion minimization loss --> modified yamabe equation with mean curvature
torch::Tensor distorion_loss_extrinsic(TargetMeshData const& _tmd, LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo& _eval_info);

// path network length reduction loss
at::Tensor total_length_loss(PathNetworkData const& _pnd, OverlayMeshData const& _omd);

// regularizer encouraging patches with inner angle of 90deg
at::Tensor inner_angle_loss(PathNetworkData const& _pnd, OverlayMeshData const& _omd);

// regularizer encouraging surface points being equally distant
at::Tensor repel_loss_v1(LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd);
at::Tensor repel_loss_v2(LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd);
#endif // LAYOUTOPT_WITH_TORCH

//=======================================================================================================================================================================

pm::vertex_attribute<torch::Tensor> compute_gradients(torch::Tensor& _loss, PathNetworkData const& _pnd);

//=======================================================================================================================================================================


//=======================================================================================================================================================================
// helper for harmonic distortio loss
void torch_compute_embedded_layout_edge_lengths(LayoutData& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd);
void torch_compute_t_for_pn_halfedge(LayoutData const& _ld, PathNetworkData& _pnd, OverlayMeshData const& _omd);
void torch_compute_pn_uvs(LayoutData const& _ld, PathNetworkData& _pnd, HarmonicOptions const& _opts);
//=======================================================================================================================================================================

struct MappingIndex
{
    int idx = -1;
    bool on_boundary = false;
};

void torch_prepare_param(int const _patch_value,
                         PathNetworkData const& _pnd,
                         OverlayMeshData const& _omd,
                         at::Tensor const& _torch_o_uvs,
                         pm::vertex_attribute<MappingIndex>& _map_to_vec,
                         std::vector<VH>& _inner_patch_vhs,
                         std::vector<VH>& _boundary_patch_vhs,
                         std::vector<at::Tensor>& _boundary_uvs,
                         std::vector<FH>& _patch_fhs);

at::Tensor torch_harmonic_param(pm::vertex_attribute<MappingIndex> const& _map_to_vec,
                                pm::edge_attribute<bool> const& _is_boundary,
                                std::vector<VH> const& _inner_patch_vhs,
                                std::vector<at::Tensor> const& _boundary_uvs,
                                at::Tensor const& _cotans,
                                bool _use_sparse_solve);


struct DistortionInfo
{
    at::Tensor distortion_val;
    at::Tensor param_area;
};

DistortionInfo compute_distortion(at::Tensor const& _ref_A,
                                  at::Tensor const& _ref_B,
                                  at::Tensor const& _ref_C,
                                  at::Tensor const& _param_A,
                                  at::Tensor const& _param_B,
                                  at::Tensor const& _param_C,
                                  HarmonicOptions const& _opt);

// remove-autodiff Phase 4 (S7): the per-overlay-face body of the
// harmonic_distortion_loss patch loop — rebuilds the 2D reference/parameter
// triangles from the corner positions (2D patch UVs / 3D overlay positions)
// and evaluates compute_distortion. Extracted so the S7 gradcheck can drive
// the exact production computation with the stage inputs as leaves.
DistortionInfo torch_face_distortion(at::Tensor const& _posA_2D,
                                     at::Tensor const& _posB_2D,
                                     at::Tensor const& _posC_2D,
                                     at::Tensor const& _posA_3D,
                                     at::Tensor const& _posB_3D,
                                     at::Tensor const& _posC_3D,
                                     HarmonicOptions const& _opt);
} // namespace LayoutOpt
