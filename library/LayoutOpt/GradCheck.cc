#ifdef LAYOUTOPT_WITH_TORCH

#include "GradCheck.hh"
#include "LayoutOpt/DataStructures/SurfacePoint.hh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

namespace LayoutOpt
{

void gradcheck_dump_leaf_grads(PathNetworkData const& _pnd, std::string const& _path)
{
    std::ofstream f(_path);
    if (!f)
    {
        std::cerr << "[GradCheck] cannot open " << _path << " for writing\n";
        return;
    }

    f << "gradcheck_leaf_v1\n";

    int count = 0;
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto const& sp = _pnd.sp_on_target_.value()[pn_vh];
        if (sp.type != SurfacePointType::FacePoint)
            continue;

        auto const& bc = sp.bary_coords;
        if (!bc.defined() || !bc.requires_grad())
            continue;

        auto g = bc.grad();
        if (!g.defined())
            continue;

        double gx = g[0].item<double>();
        double gy = g[1].item<double>();
        f << pn_vh.idx.value << " " << gx << " " << gy << "\n";
        ++count;
    }

    std::cout << "[GradCheck] wrote " << count << " leaf grads to " << _path << "\n";
}


double gradcheck_compare_leaf_grads(PathNetworkData const& _pnd,
                                     pm::vertex_attribute<vec2d> const& _hand_grads,
                                     std::string const& _path,
                                     double _atol,
                                     double _rtol)
{
    std::ifstream f(_path);
    if (!f)
    {
        std::cerr << "[GradCheck] cannot open " << _path << " for reading\n";
        return -1.0;
    }

    std::string header;
    std::getline(f, header);
    if (header != "gradcheck_leaf_v1")
    {
        std::cerr << "[GradCheck] unexpected header: " << header << "\n";
        return -1.0;
    }

    double max_diff = 0.0;
    int n_ok = 0, n_fail = 0;

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream ss(line);
        int idx;
        double ref_x, ref_y;
        ss >> idx >> ref_x >> ref_y;

        auto pn_vh = _pnd.mesh_->vertices()[idx];
        vec2d hand = _hand_grads[pn_vh];

        double dx = std::abs(hand[0] - ref_x);
        double dy = std::abs(hand[1] - ref_y);
        double abs_diff = std::max(dx, dy);
        double mag = std::max(std::abs(ref_x), std::abs(ref_y));
        double rel_diff = (mag > _rtol) ? (abs_diff / mag) : 0.0;

        max_diff = std::max(max_diff, abs_diff);

        bool ok = (abs_diff < _atol) || (mag > _rtol && rel_diff < _rtol);
        if (ok)
            ++n_ok;
        else
        {
            ++n_fail;
            std::cerr << "[GradCheck] FAIL pn_vh=" << idx
                      << "  ref=(" << ref_x << "," << ref_y << ")"
                      << "  hand=(" << hand[0] << "," << hand[1] << ")"
                      << "  abs_diff=" << abs_diff << "  rel_diff=" << rel_diff << "\n";
        }
    }

    std::cout << "[GradCheck] leaf compare: " << n_ok << " ok, " << n_fail << " fail, max_abs_diff=" << max_diff << "\n";
    return max_diff;
}


void gradcheck_retain_intermediate_grads(LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd)
{
    if (_omd.torch_pos_.has_value() && _omd.torch_pos_.value().requires_grad())
        _omd.torch_pos_.value().retain_grad();

    if (_pnd.torch_uvs_.has_value() && _pnd.torch_uvs_.value().requires_grad())
        _pnd.torch_uvs_.value().retain_grad();

    if (_pnd.torch_t_.has_value() && _pnd.torch_t_.value().requires_grad())
        _pnd.torch_t_.value().retain_grad();

    if (_ld.torch_embedded_edge_length_.has_value() && _ld.torch_embedded_edge_length_.value().requires_grad())
        _ld.torch_embedded_edge_length_.value().retain_grad();
}


void gradcheck_dump_intermediate_grads(LayoutData const& _ld,
                                        PathNetworkData const& _pnd,
                                        OverlayMeshData const& _omd,
                                        std::string const& _path)
{
    std::ofstream f(_path, std::ios::app); // append after leaf section
    if (!f)
    {
        std::cerr << "[GradCheck] cannot open " << _path << " for appending\n";
        return;
    }

    auto dump_tensor = [&](std::string const& name, std::optional<torch::Tensor> const& opt) {
        if (!opt.has_value())
            return;
        auto const& t = opt.value();
        if (!t.defined() || !t.grad().defined())
            return;
        auto g = t.grad().contiguous();
        auto* data = g.data_ptr<double>();
        f << "# " << name << " grad shape";
        for (auto s : g.sizes())
            f << " " << s;
        f << "\n";
        int64_t n = g.numel();
        for (int64_t i = 0; i < n; ++i)
            f << data[i] << "\n";
    };

    dump_tensor("torch_embedded_edge_length", _ld.torch_embedded_edge_length_);
    dump_tensor("torch_t", _pnd.torch_t_);
    dump_tensor("torch_uvs", _pnd.torch_uvs_);
    dump_tensor("torch_pos", _omd.torch_pos_);
}

} // namespace LayoutOpt

#endif // LAYOUTOPT_WITH_TORCH
