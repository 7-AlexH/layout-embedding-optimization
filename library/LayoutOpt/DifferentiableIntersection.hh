#pragma once
#include "DataStructures/TriangleStrip.hh"
namespace LayoutOpt
{
void compute_differentiable_intersections_for_overlay_stable(
    std::vector<TriangleStrip> const& _strips, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData& _omd);

// helper
void compute_differentiable_surface_points(TargetMeshData const& _tmd, PathNetworkData const& _pnd, OverlayMeshData& _omd);
void compute_differentiable_intersection_for_overlay(TriangleStrip const& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd, OverlayMeshData& _omd);


} // namespace LayoutOpt
