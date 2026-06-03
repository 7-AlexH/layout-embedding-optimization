#pragma once

#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"

namespace LayoutOpt
{

struct ResampleData
{
    pm::edge_attribute<int> l_arc_ids_;
    pm::edge_attribute<double> l_embedded_length_; // inital embedded_length
    std::vector<double> arc_lengths_;
};


//==============================================================================
//============================resampling methods================================
//==============================================================================

void resample_arc(int _arc_id, ResampleData const& _rd, TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, double _target_edge_length);
void resample_arc(int _arc_id, ResampleData const& _rd, TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, int _num_of_segments);

void resample_layout(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, double _target_edge_length);
void resample_layout(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, int _num_of_segments);

/// @brief Removes all layout edges in `_ld.mesh` shorter than `_min_length`.
/// @details Edges with embedding length smaller than the user-defined threshold
/// `_min_length` are removed from the  layout mesh.
/// @return true if edges were removed
bool down_sample(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, double _min_edge_length);

//==============================================================================
//==================================helper======================================
//==============================================================================

void init_resample_data(ResampleData& _rd, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd);

} // namespace LayoutOpt
