#pragma once
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/Resample.hh"

//=============================================================================================
//=================================adapted from born et al.====================================
//=============================================================================================

namespace LayoutOpt
{
struct LoopSubdivData
{
    pm::edge_attribute<int> l_arc_ids_; //attribute of layout

    pm::unique_ptr<pm::Mesh> arc_mesh_; // coarsest structure encoding connectivity of layout (without subdivisions)
    pm::vertex_attribute<pos3> arc_pos_;  //positions for debugging
    pm::edge_attribute<int> arc_ids_;  //correspondence to layoutData

    pm::vertex_attribute<VH> map_a_vh_to_layout_; //map to layout
    pm::halfedge_attribute<std::vector<VH>> map_arc_to_overlay_; //map to_overlay
    pm::edge_attribute<double> arc_lengths_;
    pm::edge_attribute<int> arc_subdivs_;
};
/// Takes an embedded quad layout and returns the number of
/// subdivisions per edge that best achives the target edge length.
void choose_loop_subdivisions(LoopSubdivData& _lsd,
                              TargetMeshData const& _tmd,
                              LayoutData const& _ld,
                              PathNetworkData const& _pnd,
                              const double _target_length,
                              const int _max);

// helper
void init_loopSubdivData(LoopSubdivData& _lsd, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd);
void init_map_arc_to_overlay(LoopSubdivData& _lsd, LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd);
}
