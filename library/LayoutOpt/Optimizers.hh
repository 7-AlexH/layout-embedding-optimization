#pragma once

#include <polymesh/Mesh.hh>
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"

namespace LayoutOpt
{

struct OptimizerData
{
    int time_step = 0;

    // vector adam
    double step_size = 0.0;
    double beta1 = 0.0;
    double beta2 = 0.0;
    double epsilon = 0.0;

    pm::vertex_attribute<vec2d> m;  // First moment vector (R²)
    pm::vertex_attribute<double> v; // Second moment vector (R)

    void init_vetor_adam_param(pm::Mesh const& _mesh, double _step_size = 0.01);
};

pm::vertex_attribute<vec2d> gradient_descent(PathNetworkData const& _pnd, pm::vertex_attribute<vec2d> const& _gradients, OptimizerData& _od);
pm::vertex_attribute<vec2d> vector_adam_updates(TargetMeshData const& _tmd,
                                                PathNetworkData const& _pnd,
                                                pm::vertex_attribute<vec2d> const& _gradients,
                                                OptimizerData& _od);

// check if a gradient in world space is very large compared to the others and caps it
void preprocess_gradients(TargetMeshData const& _tmd, PathNetworkData const& _pnd, pm::vertex_attribute<vec2d>& _gradients);

} // namespace LayoutOpt
