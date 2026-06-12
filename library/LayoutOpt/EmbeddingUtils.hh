#pragma once
#include <vector>

#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/TriangleStrip.hh"
#include "LayoutOpt/DataStructures/Types.hh"
namespace LayoutOpt
{

void set_layout_pos_based_on_pn_sp(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData const& _pnd);

void compute_path_boundary_mask(PathNetworkData const& _pnd, OverlayMeshData& _omd);
void compute_mapping_layout_to_overlay(LayoutData& _ld, PathNetworkData const& _pnd);
void compute_mapping_overlay_to_layout(OverlayMeshData& _omd, LayoutData const& _ld);

void collapse_pn_to_layout(PathNetworkData& _pnd);
void reset_embedding_data(LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd);

// Computes the overlay positions: overwrites the layout-node rows (S1 leaf
// interpolation) and the interior arc rows (S3 strip intersection) of pos_,
// then refreshes pos_mat_ in lockstep. Requires the overlay mappings to be
// computed.
void compute_overlay_positions(std::vector<TriangleStrip> const& _strips,
                               TargetMeshData const& _tmd,
                               LayoutData const& _ld,
                               PathNetworkData const& _pnd,
                               OverlayMeshData& _omd);

//=============================================================================================
//=================================addidtional information=====================================
//=============================================================================================

/// @short find the coarsest sturcutre, assigning the same label to layout edges that actually just subdivide (ie where the vertices are not singularities)
pm::edge_attribute<int> compute_layout_edge_arc_idx(LayoutData const& _ld);

pm::edge_attribute<double> compute_embedded_length_per_layout_edge(TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd);

pm::edge_attribute<std::vector<FH>> compute_target_face_strips_per_layout_edge(TargetMeshData const& _tmd,
                                                                               LayoutData const& _ld,
                                                                               PathNetworkData const& _pnd,
                                                                               pm::edge_attribute<std::vector<VH>> const& _map_l_eh_to_pn_vh);
pm::edge_attribute<std::vector<VH>> compute_map_l_eh_to_pn_vh(LayoutData const& _ld, PathNetworkData const& _pnd);


pm::edge_attribute<std::vector<VH>> compute_l_eh_is_intersecting(LayoutData const& _ld, PathNetworkData const& _pnd);

//==============================================================================
//=================================validity=====================================
//==============================================================================
bool pn_contains_degenerate_edges(TargetMeshData const& _tmd, PathNetworkData const& _pnd);
bool omd_contains_flipped_triangles(TargetMeshData const& _tmd, OverlayMeshData const& _omd);
bool omd_contains_degenerate_triangles(OverlayMeshData const& _omd);

} // namespace LayoutOpt
