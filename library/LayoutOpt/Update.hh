#pragma once
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/SurfacePoint.hh"
#include "LayoutOpt/Optimizers.hh"
namespace LayoutOpt
{
/**
 * @brief Traces the update directions on the target mesh.
 *
 * This function updates the variables in `_pnd` based on the given direction updates
 * in barycentric coordinates provided by `_dirs`.
 *
 * @param _tmd The target mesh data on which _dirs will be traced.
 * @param _pnd The variables that should be updated.
 * @param _dirs The updates specified in barycentric coordinates.
 */
pm::vertex_attribute<std::vector<SurfacePoint>> do_step(TargetMeshData const& _tmd,
                                                        PathNetworkData& _pnd,
                                                        pm::vertex_attribute<vec2d> const& _dirs,
                                                        OptimizerData* _od = nullptr);
} // namespace LayoutOpt
