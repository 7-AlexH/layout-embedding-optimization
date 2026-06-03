#pragma once
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/Types.hh"
namespace LayoutOpt
{
/// @brief assumes that the pn is composed of face- and edge-surface points only
/// computes the topology of the embeddding
void insert_path_network(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd);

//==============================================================================
//====================================aux=======================================
//==============================================================================
void insert_edge_surface_points(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd);
void insert_face_surface_points(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd);
void insert_edge_edge_connections(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd);
void insert_surface_point_edge_connections(TargetMeshData const& _tmd, PathNetworkData& _pnd, OverlayMeshData& _omd);

void triangulate_by_ear_clipping(TargetMeshData const& _tmd, OverlayMeshData& _omd);
} // namespace LayoutOpt
