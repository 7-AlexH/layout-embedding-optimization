#pragma once
#include "DataStructures/LayoutEmbedding.hh"
#include "DataStructures/TriangleStrip.hh"
namespace LayoutOpt
{

/// diagnostic: counts vertex-sp -> edge-sp conversions performed by
/// convert_to_snake_refactor (each conversion perturbs the strip metric by
/// ~LARGE_EPS by design, so trajectories that disagree in conversion counts
/// diverge at the 1e-5 level). Read/reset by validation harnesses.
extern long g_vertex_sp_conversion_count;

void compute_triangle_strip(TriangleStrip& _strip, const EH _l_eh, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData& _pnd);

//=============================================================helper=============================================================================
/// @short compute the pn vhs that belong to this strip, the strip is identified via the layout edge
void init_triangle_strip_pn_vhs(TriangleStrip& _strip, const EH _l_eh, PathNetworkData const& _pnd);
/// @short compute the embedded length of the strip, ie the sum of the segment lengths
void init_embedded_length(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd);

/// @short compute the 2D embedding for most faces
void init_2D_strip_pos(TriangleStrip& _strip, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd);

/// some inner vertices are still vertex-surface points. change the metric slightly and adapt path network
void convert_to_snake_refactor(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData& _pnd);

// void make_strip_stable(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd);

/// @short after conversion to snake
void init_triangle_strip_t_fhs(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd);


// checks if all intersections are between ]0,1[
bool is_strip_valid(TriangleStrip& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd);

} // namespace LayoutOpt
