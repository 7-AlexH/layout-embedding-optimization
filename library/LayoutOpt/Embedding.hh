#pragma once
#include <vector>

#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/TriangleStrip.hh"
namespace LayoutOpt
{
//==============================================================================
//=============================Layout Embedding=================================
//==============================================================================
/// @brief Compute layout embedding based on layout node positions (ambient) and initializes surface points (intrinsic).
///        Slightly corrects layout positions if they are not on the surface or were corrected (_allow_fallback_for_stability = true).
/// @param _allow_fallback_for_stability If true, ensures surface points lie inside a triangle by clamping barycentric coordinates to _eps
/// @param _strips_out if non-null, receives the per-layout-edge triangle strips this embedding was
///        built from (eval()'s adjoint chain differentiates through their 2D flattening; otherwise
///        they stay local and are discarded)
void compute_layout_embedding_init(TargetMeshData const& _tmd,
                                   LayoutData& _ld,
                                   PathNetworkData& _pnd,
                                   OverlayMeshData& _omd,
                                   bool _allow_fallback_for_stability = false,
                                   std::vector<TriangleStrip>* _strips_out = nullptr);


/// @brief compute layout embedding based on surface points and set layout positions correctly
/// @param _strips_out see compute_layout_embedding_init
void compute_layout_embedding_update(TargetMeshData const& _tmd,
                                     LayoutData& _ld,
                                     PathNetworkData& _pnd,
                                     OverlayMeshData& _omd,
                                     std::vector<TriangleStrip>* _strips_out = nullptr);

/// @brief tries to recover by slightly changing the positions of the layout in hopes of getting new gradients again
void recover(TargetMeshData& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd);

//==============================================================================
//===============================Path Network===================================
//==============================================================================

/// @brief based on layout points
void init_path_network_surface_points(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, bool _allow_fallback_for_stability = false);

/// @brief the path network defined by surface points is embedded in the target surface by additional edge- and vertex-surface points
// void compute_path_network_exact(TargetMeshData const& _tmd, PathNetworkData& _pnd);
void compute_path_network_exact(TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData& _pnd);

/// @brief removes veretx_surface points by slightly adapting the metric
/// @return _strips contains the local flattening per layout edge
void compute_2D_embeddig_per_strip(TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData& _pnd, std::vector<LayoutOpt::TriangleStrip>& _strips);


}
