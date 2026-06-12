#include "Optimizers.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "LayoutOpt/Utils/Debug.hh"

namespace LayoutOpt
{

namespace
{

// The three corners of the face the surface point lives on (rows of _pos by
// vertex idx, in the order vertex_from, vertex_to, next().vertex_to()).
struct FaceCorners
{
    vec3d A, B, C;
};

FaceCorners face_corners(SurfacePoint const& _sp, TargetMeshData const& _tmd)
{
    auto const hh = _tmd.mesh_->handle_of(_sp.heh_idx);
    return {_tmd.pos_mat_.row(hh.vertex_from().idx.value).transpose(),
            _tmd.pos_mat_.row(hh.vertex_to().idx.value).transpose(),
            _tmd.pos_mat_.row(hh.next().vertex_to().idx.value).transpose()};
}

} // namespace

pm::vertex_attribute<vec2d> gradient_descent(PathNetworkData const& _pnd, pm::vertex_attribute<vec2d> const& _gradients, OptimizerData& _od)
{
    pm::vertex_attribute<vec2d> dirs(_gradients.mesh());
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        dirs[pn_vh] = vec2d::Zero();
        if (_pnd.map_to_layout_vertices_[pn_vh].is_invalid())
            continue;
        dirs[pn_vh] = -_od.step_size * _gradients[pn_vh];
    }
    return dirs;
}

void OptimizerData::init_vetor_adam_param(polymesh::Mesh const& _mesh, double _step_size)
{
    time_step = 0;
    step_size = _step_size;
    beta1 = 0.9;
    beta2 = 0.9;
    epsilon = 1e-8;

    m = _mesh.vertices().make_attribute<vec2d>(vec2d::Zero());
    v = _mesh.vertices().make_attribute<double>(0.0);
}

pm::vertex_attribute<vec2d> vector_adam_updates(TargetMeshData const& _tmd, PathNetworkData const& _pnd, pm::vertex_attribute<vec2d> const& _gradients, OptimizerData& _od)
{
    _od.time_step = _od.time_step + 1;

    pm::vertex_attribute<vec2d> dirs(_gradients.mesh());
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        dirs[pn_vh] = vec2d::Zero();
        if (_pnd.map_to_layout_vertices_[pn_vh].is_invalid())
            continue;

        // world-space step length of the gradient (mapped sps are FacePoints,
        // so the (a, b) bary slots are exactly the gradient components)
        auto const& sp = _pnd.sp_on_target_.value()[pn_vh];
        auto const [A, B, C] = face_corners(sp, _tmd);
        vec3d const bary = sp.bary_full();
        vec2d const& g = _gradients[pn_vh];

        vec3d const pos_from = bary[0] * A + bary[1] * B + bary[2] * C;
        double const a_to = bary[0] + g.x();
        double const b_to = bary[1] + g.y();
        vec3d const pos_to = a_to * A + b_to * B + (1.0 - a_to - b_to) * C;

        // following https://arxiv.org/pdf/2205.13599 Algorithm 1
        _od.m[pn_vh] = _od.beta1 * _od.m[pn_vh] + (1. - _od.beta1) * g;
        double const step_norm = (pos_to - pos_from).norm();
        _od.v[pn_vh] = _od.beta2 * _od.v[pn_vh] + (1. - _od.beta2) * (step_norm * step_norm);

        vec2d const m_avg = _od.m[pn_vh] / (1.0 - tg::pow(_od.beta1, _od.time_step));
        double const v_avg = _od.v[pn_vh] / (1.0 - tg::pow(_od.beta2, _od.time_step));

        dirs[pn_vh] = -_od.step_size * m_avg / (std::sqrt(v_avg) + _od.epsilon);
    }
    return dirs;
}

void preprocess_gradients(TargetMeshData const& _tmd, PathNetworkData const& _pnd, pm::vertex_attribute<vec2d>& _gradients)
{
    std::vector<VH> pn_vhs;
    pn_vhs.reserve(_pnd.mesh_->vertices().size());

    std::vector<double> magnitudes;
    magnitudes.reserve(_pnd.mesh_->vertices().size());

    // Step 1: compute world-space gradient magnitudes
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        if (_pnd.map_to_layout_vertices_[pn_vh].is_invalid())
            continue;

        auto const& sp = _pnd.sp_on_target_.value()[pn_vh];
        auto const [A, B, C] = face_corners(sp, _tmd);
        vec3d const bary = sp.bary_full();
        vec2d const& g = _gradients[pn_vh];

        vec3d const pos_from = bary[0] * A + bary[1] * B + bary[2] * C;
        double const a_to = bary[0] + g.x();
        double const b_to = bary[1] + g.y();
        vec3d const pos_to = a_to * A + b_to * B + (1.0 - a_to - b_to) * C;

        double const mag = (pos_to - pos_from).norm();

        pn_vhs.push_back(pn_vh);
        magnitudes.push_back(mag);
    }

    if (magnitudes.empty())
        return;

    // Step 2: compute median
    std::vector<double> sorted = magnitudes;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];

    DEBUG_VAR(median)

    // Step 3: compute absolute deviations
    std::vector<double> deviations;
    deviations.reserve(magnitudes.size());
    for (double m : magnitudes)
    {
        deviations.push_back(std::abs(m - median));
    }

    // Step 4: MAD
    std::sort(deviations.begin(), deviations.end());
    double mad = deviations[deviations.size() / 2];
    if (mad < 1e-8)
        mad = 1e-8; // avoid div by zero

    DEBUG_VAR(mad)

    double threshold = 85.0;

    // Step 5: threshold & filter
    for (int idx = 0; idx < pn_vhs.size(); ++idx)
    {
        auto pn_vh = pn_vhs[idx];
        double modified_z = 0.6745 * (magnitudes[idx] - median) / mad;

        // threshold=3.5
        if (std::abs(modified_z) > threshold)
        {
            DEBUG_OUT("- filtering gradient!")
            std::cout << "median=" << median << ", mad=" << mad << ", z=" << modified_z << std::endl;

            // zero out gradient
            double scale = 0.1 * threshold / std::abs(modified_z);
            _gradients[pn_vh] = _gradients[pn_vh] * scale;
        }
    }
}


} // namespace LayoutOpt
