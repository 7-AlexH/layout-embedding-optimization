#pragma once
#include <vector>

#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/DataStructures/TriangleStrip.hh"
#include "LayoutOpt/ObjectiveFunctions.hh"
#include "LayoutOpt/Optimizers.hh"
#include "OptimizationOptions.hh"

namespace LayoutOpt
{

/// @brief evaluates the objective function and computes gradients as well as update directions.
///        _strips are the per-layout-edge triangle strips the current embedding was built from
///        (out-param of compute_layout_embedding_init/update) — the hand-rolled adjoint chain
///        differentiates through their 2D flattening (see Adjoint/HandGradients.hh).
EvalInfo eval(TargetMeshData& _tmd,
              LayoutData& _ld,
              PathNetworkData& _pnd,
              OverlayMeshData& _omd,
              std::vector<TriangleStrip> const& _strips,
              OptimizationOptions const& _opts,
              int i,
              OptimizerData& _vad);

/// @brief applies the update direcitons and recomputes the embedding. If _strips_out is non-null
///        it receives the rebuilt strips (callers feeding the next eval() must pass it).
void apply(TargetMeshData const& _tmd,
           LayoutData& _ld,
           PathNetworkData& _pnd,
           OverlayMeshData& _omd,
           OptimizerData& _od,
           EvalInfo const& _eval_info,
           std::vector<TriangleStrip>* _strips_out = nullptr);
} // namespace LayoutOpt
