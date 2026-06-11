#pragma once
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/ObjectiveFunctions.hh"
#include "LayoutOpt/Optimizers.hh"
#include "OptimizationOptions.hh"

namespace LayoutOpt
{

/// @brief evaluates the objective function and computes gradients as well as update directions
EvalInfo eval(TargetMeshData& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, OptimizationOptions const& _opts, int i, OptimizerData& _vad);

/// @brief applies the update direcitons and recomputes the embedding
void apply(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, OptimizerData& _od, EvalInfo const& _eval_info);
} // namespace LayoutOpt
