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

    // updates
    pm::vertex_attribute<torch::Tensor> pn_sp_grads; // only for facepoints
    pm::vertex_attribute<at::Tensor> pn_sp_update_dirs;

    // for variance loss
    pm::face_attribute<double> o_local_vars;     // local variance for overlay mesh
    pm::face_attribute<double> l_per_patch_vars; // variances per patch

    // for distorion_loss
    pm::vertex_attribute<double> o_scaling_factor;
    pm::vertex_attribute<double> o_scaling_factor_extrinsic;

    // for hamonic loss
    pm::face_attribute<double> o_distortion_harmonic;
    pm::halfedge_attribute<pos2> o_uvs;
};

//=======================================================================================================================================================================

// harmonic_distortion_loss
torch::Tensor harmonic_distortion_loss(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, HarmonicOptions _variant, EvalInfo& _eval_info);

// principal curvature alignement
torch::Tensor principal_curvature_alignment_loss(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, EvalInfo& _eval_info);

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
// note: so far not producing expected result
at::Tensor repel_loss_v1(LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd);
at::Tensor repel_loss_v2(LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd);

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
                                at::Tensor const& _cotans);


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
                                  HarmonicOptions _opt);
} // namespace LayoutOpt
