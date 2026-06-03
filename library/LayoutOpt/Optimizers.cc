#include "Optimizers.hh"
#include <glow-extras/viewer/canvas.hh>
#include "LayoutOpt/TorchUtils.hh"
#include "LayoutOpt/Utils/Debug.hh"
#include "LayoutOpt/Visualization/Colors.hh"

namespace LayoutOpt
{

pm::vertex_attribute<at::Tensor> gradient_descent(PathNetworkData const& _pnd, pm::vertex_attribute<at::Tensor> const& _gradients, OptimizerData& _od)
{
    pm::vertex_attribute<at::Tensor> dirs(_gradients.mesh());
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        dirs[pn_vh] = torch::tensor({0.0, 0.0}, torch::dtype(torch::kFloat64));
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

    m = _mesh.vertices().make_attribute<torch::Tensor>(torch::zeros({2}));
    v = _mesh.vertices().make_attribute<torch::Tensor>(torch::zeros({}));
}

pm::vertex_attribute<at::Tensor> vector_adam_updates(TargetMeshData const& _tmd, PathNetworkData const& _pnd, pm::vertex_attribute<at::Tensor> const& _gradients, OptimizerData& _od)
{
    _od.time_step = _od.time_step + 1;

    pm::vertex_attribute<at::Tensor> dirs(_gradients.mesh());
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        dirs[pn_vh] = torch::tensor({0.0, 0.0}, torch::dtype(torch::kFloat64));
        if (_pnd.map_to_layout_vertices_[pn_vh].is_invalid())
            continue;

        auto const sp_from = _pnd.sp_on_target_.value()[pn_vh].copy();
        auto pos_from = sp_from.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get());

        auto sp_to = _pnd.sp_on_target_.value()[pn_vh].copy();
        sp_to.bary_coords = sp_to.bary_coords + _gradients[pn_vh];
        auto pos_to = sp_to.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get());

        // following https://arxiv.org/pdf/2205.13599 Algorithm 1
        _od.m[pn_vh] = _od.beta1 * _od.m[pn_vh] + (1. - _od.beta1) * _gradients[pn_vh];
        _od.v[pn_vh] = _od.beta2 * _od.v[pn_vh] + (1. - _od.beta2) * torch::norm(pos_to - pos_from).square();

        auto m_avg = _od.m[pn_vh] / (1.0 - tg::pow(_od.beta1, _od.time_step));
        auto v_avg = _od.v[pn_vh] / (1.0 - tg::pow(_od.beta2, _od.time_step));

        dirs[pn_vh] = -_od.step_size * m_avg / (torch::sqrt(v_avg) + _od.epsilon);
    }
    return dirs;
}

void preprocess_gradients(TargetMeshData const& _tmd, PathNetworkData const& _pnd, pm::vertex_attribute<at::Tensor>& _gradients)
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

        auto sp = _pnd.sp_on_target_.value()[pn_vh].copy();

        auto pos_from = sp.get_pos(_tmd.torch_pos_, *_tmd.mesh_);
        sp.bary_coords = sp.bary_coords + _gradients[pn_vh];
        auto pos_to = sp.get_pos(_tmd.torch_pos_, *_tmd.mesh_);

        auto diff = pos_to - pos_from;
        double mag = torch::norm(diff).item<double>();

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
