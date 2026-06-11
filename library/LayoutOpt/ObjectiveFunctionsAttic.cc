// Inactive loss functions — not used in eval() but kept for reference.
// All depend on LAYOUTOPT_WITH_TORCH; compile to nothing when the flag is OFF.
#ifdef LAYOUTOPT_WITH_TORCH

#include "ObjectiveFunctions.hh"

#include <glow-extras/viewer/canvas.hh>
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/TorchUtils.hh"
#include "LayoutOpt/Visualization/Colors.hh"
#include "LayoutOpt/Visualization/Viewing.hh"

#include <torch/linalg.h>

namespace LayoutOpt
{

torch::Tensor variance_loss(TargetMeshData const& _tmd, LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo& _eval_info)
{
    assert(_omd.map_to_target_faces_.has_value());
    assert(_tmd.torch_scalar_field_.has_value());

    _eval_info.o_local_vars = _omd.mesh_->faces().make_attribute<double>(0.0);
    _eval_info.l_per_patch_vars = _ld.mesh_->faces().make_attribute<double>(0.0);

    std::vector<torch::Tensor> area_list;
    std::vector<torch::Tensor> scalars_list;

    for (auto fh : _omd.mesh_->faces())
    {
        auto [vh0, vh1, vh2] = fh.vertices().to_array<3>();
        auto pos0 = _omd.torch_pos_.value()[vh0.idx.value];
        auto pos1 = _omd.torch_pos_.value()[vh1.idx.value];
        auto pos2 = _omd.torch_pos_.value()[vh2.idx.value];

        auto vec0 = pos1 - pos0;
        auto vec1 = pos2 - pos0;
        auto det = torch::norm(torch::linalg_cross(vec0, vec1));

        area_list.push_back(0.5 * det);

        auto const& target_fh = _omd.map_to_target_faces_.value()[fh];
        scalars_list.push_back(_tmd.torch_scalar_field_.value()[target_fh.value]);
    }
    auto areas = torch::stack(area_list).squeeze();
    auto scalars = torch::stack(scalars_list).squeeze();

    std::vector<torch::Tensor> patch_total_vars_list;
    std::vector<torch::Tensor> patch_total_areas_list;
    std::vector<torch::Tensor> patch_total_scalars_list;
    std::vector<torch::Tensor> patch_means_list;

    int n_patches = _ld.mesh_->faces().size();
    for (int i = 0; i < n_patches; ++i)
    {
        auto cd = gv::canvas_data();

        std::vector<torch::Tensor> patch_areas_list;
        std::vector<torch::Tensor> patch_scalars_list;
        std::vector<FH> fhs;
        for (auto fh : _omd.mesh_->faces())
        {
            if (_omd.map_to_layout_faces_.value()[fh].value == i)
            {
                patch_areas_list.push_back(areas[fh.idx.value]);
                patch_scalars_list.push_back(scalars[fh.idx.value]);
                fhs.push_back(fh);
            }
        }

        auto patch_areas = torch::stack(patch_areas_list).squeeze();
        auto patch_scalars = torch::stack(patch_scalars_list).squeeze();
        auto patch_area = patch_areas.sum();
        auto patch_mean = (patch_areas * patch_scalars).sum() / patch_area;
        auto diff_vec = patch_scalars - patch_mean;
        auto diff_vec_sq = diff_vec * diff_vec;
        auto local_patch_var = (diff_vec_sq * patch_areas) / patch_area;

        auto patch_var = local_patch_var.sum();
        patch_total_vars_list.push_back(patch_var);
        patch_total_areas_list.push_back(patch_area);
        patch_total_scalars_list.push_back((patch_scalars * patch_areas).sum());
        patch_means_list.push_back(patch_mean);

        _eval_info.l_per_patch_vars[_ld.mesh_->faces()[i]] = patch_var.item<double>();

        for (size_t fh_idx = 0; fh_idx < fhs.size(); ++fh_idx)
        {
            _eval_info.o_local_vars[fhs[fh_idx]] = local_patch_var[fh_idx].item<double>();
        }
    }

    auto patch_total_vars = torch::stack(patch_total_vars_list).squeeze();
    auto loss = patch_total_vars.mean();

    return loss;
}


torch::Tensor distorion_loss(TargetMeshData const& _tmd, LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo& _eval_info)
{
    std::vector<torch::Tensor> area_list;
    std::vector<torch::Tensor> gauss_curvature_list;
    for (auto fh : _omd.mesh_->faces())
    {
        auto [vh0, vh1, vh2] = fh.vertices().to_array<3>();
        auto pos0 = _omd.torch_pos_.value()[vh0.idx.value];
        auto pos1 = _omd.torch_pos_.value()[vh1.idx.value];
        auto pos2 = _omd.torch_pos_.value()[vh2.idx.value];

        auto vec0 = pos1 - pos0;
        auto vec1 = pos2 - pos0;
        auto det = torch::norm(torch::linalg_cross(vec0, vec1));

        area_list.push_back(0.5 * det);

        auto const& target_fh = _omd.map_to_target_faces_.value()[fh];
        gauss_curvature_list.push_back(_tmd.torch_point_wise_gauss_curvature_.value()[target_fh.value]);
    }
    auto areas = torch::stack(area_list).squeeze();
    auto gauss_curvatures = torch::stack(gauss_curvature_list).squeeze();

    std::vector<torch::Tensor> cotan_list;
    for (auto eh : _omd.mesh_->edges())
    {
        auto h0 = eh.halfedgeA();
        auto h1 = eh.halfedgeB();

        auto cot_a = torch::tensor(0.0, torch::dtype(torch::kFloat64));
        auto cot_b = torch::tensor(0.0, torch::dtype(torch::kFloat64));

        auto pi = _omd.torch_pos_.value()[h0.vertex_to().idx.value];
        auto pj = _omd.torch_pos_.value()[h1.vertex_to().idx.value];

        auto pa = _omd.torch_pos_.value()[h0.next().vertex_to().idx.value];
        auto e_ia = pi - pa;
        auto e_ja = pj - pa;
        auto denom_a = (e_ia[0] * e_ja[1] - e_ja[0] * e_ia[1]).norm();
        cot_a = e_ia.dot(e_ja) / (denom_a + 1e-10);

        auto pb = _omd.torch_pos_.value()[h1.next().vertex_to().idx.value];
        auto e_ib = pi - pb;
        auto e_jb = pj - pb;
        auto denom_b = (e_ib[0] * e_jb[1] - e_jb[0] * e_ib[1]).norm();
        cot_b = e_ib.dot(e_jb) / (denom_b + 1e-10);

        cotan_list.push_back(cot_a + cot_b);
    }
    auto cotans = torch::stack(cotan_list);

    int n_patches = _ld.mesh_->faces().size();
    std::vector<torch::Tensor> patch_scaling_factors_list;
    _eval_info.o_scaling_factor = _omd.mesh_->vertices().make_attribute<double>(0.0);

    for (int i = 0; i < n_patches; ++i)
    {
        std::vector<VH> inner_patch_vhs;
        pm::vertex_attribute<int> map_to_vec = _omd.mesh_->vertices().make_attribute<int>(-1);

        for (auto fh : _omd.mesh_->faces())
        {
            if (_omd.map_to_layout_faces_.value()[fh].value != i)
                continue;

            for (auto vh : fh.vertices())
            {
                if (_omd.map_to_pn_vertices_.value()[vh].is_valid() || map_to_vec[vh] >= 0)
                    continue;

                inner_patch_vhs.push_back(vh);
                map_to_vec[vh] = inner_patch_vhs.size() - 1;
            }
        }

        int n_inner_vhs = inner_patch_vhs.size();
        auto L = torch::zeros({n_inner_vhs, n_inner_vhs}, torch::dtype(torch::kFloat64));
        auto rhs = torch::zeros({n_inner_vhs}, torch::dtype(torch::kFloat64));

        for (int i = 0; i < (int)inner_patch_vhs.size(); ++i)
        {
            auto vh = inner_patch_vhs[i];

            auto rhs_sum = torch::zeros({}, torch::dtype(torch::kFloat64));
            for (auto fh : vh.faces())
            {
                rhs_sum = rhs_sum + (1. / 3.) * areas[fh.idx.value] * gauss_curvatures[fh.idx.value];
            }
            rhs[i] = rhs_sum.squeeze();

            auto sum = torch::zeros({}, torch::dtype(torch::kFloat64));
            for (HEH heh : vh.outgoing_halfedges())
            {
                auto vh_to = heh.vertex_to();
                auto weight = cotans[heh.edge().idx.value];
                sum = sum + weight;

                int j = map_to_vec[vh_to];
                if (j == -1)
                {
                    rhs[i] = rhs[i] + (-weight * 0.0);
                }
                else
                {
                    L.index_put_({i, j}, weight);
                }
            }
            L.index_put_({i, i}, -sum);
        }
        auto u = torch::linalg::solve(L, rhs, true);

        for (int i = 0; i < (int)inner_patch_vhs.size(); ++i)
        {
            auto vh = inner_patch_vhs[i];
            _eval_info.o_scaling_factor[vh] = tg::pow2(u[i].item<double>());
        }
        patch_scaling_factors_list.push_back(u);
    }
    auto patch_scaling_factors = torch::cat(patch_scaling_factors_list).squeeze();
    return (patch_scaling_factors * patch_scaling_factors).sum();
}


at::Tensor distorion_loss_extrinsic(TargetMeshData const& _tmd, LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo& _eval_info)
{
    std::vector<torch::Tensor> area_list;
    std::vector<torch::Tensor> mean_curvature_list;
    for (auto fh : _omd.mesh_->faces())
    {
        auto [vh0, vh1, vh2] = fh.vertices().to_array<3>();
        auto pos0 = _omd.torch_pos_.value()[vh0.idx.value];
        auto pos1 = _omd.torch_pos_.value()[vh1.idx.value];
        auto pos2 = _omd.torch_pos_.value()[vh2.idx.value];

        auto vec0 = pos1 - pos0;
        auto vec1 = pos2 - pos0;
        auto det = torch::norm(torch::linalg_cross(vec0, vec1));

        area_list.push_back(0.5 * det);

        auto const& target_fh = _omd.map_to_target_faces_.value()[fh];
        mean_curvature_list.push_back(_tmd.torch_point_wise_mean_curvature_.value()[target_fh.value]);
    }
    auto areas = torch::stack(area_list).squeeze();
    auto mean_curvatures = torch::stack(mean_curvature_list).squeeze();

    std::vector<torch::Tensor> cotan_list;
    for (auto eh : _omd.mesh_->edges())
    {
        auto h0 = eh.halfedgeA();
        auto h1 = eh.halfedgeB();

        auto cot_a = torch::tensor(0.0, torch::dtype(torch::kFloat64));
        auto cot_b = torch::tensor(0.0, torch::dtype(torch::kFloat64));

        auto pi = _omd.torch_pos_.value()[h0.vertex_to().idx.value];
        auto pj = _omd.torch_pos_.value()[h1.vertex_to().idx.value];

        auto pa = _omd.torch_pos_.value()[h0.next().vertex_to().idx.value];
        auto e_ia = pi - pa;
        auto e_ja = pj - pa;
        auto denom_a = (e_ia[0] * e_ja[1] - e_ja[0] * e_ia[1]).norm();
        cot_a = e_ia.dot(e_ja) / (denom_a + 1e-10);

        auto pb = _omd.torch_pos_.value()[h1.next().vertex_to().idx.value];
        auto e_ib = pi - pb;
        auto e_jb = pj - pb;
        auto denom_b = (e_ib[0] * e_jb[1] - e_jb[0] * e_ib[1]).norm();
        cot_b = e_ib.dot(e_jb) / (denom_b + 1e-10);

        cotan_list.push_back(cot_a + cot_b);
    }
    auto cotans = torch::stack(cotan_list);

    int n_patches = _ld.mesh_->faces().size();
    std::vector<torch::Tensor> patch_scaling_factors_list;
    _eval_info.o_scaling_factor_extrinsic = _omd.mesh_->vertices().make_attribute<double>(0.0);

    for (int i = 0; i < n_patches; ++i)
    {
        std::vector<VH> inner_patch_vhs;
        pm::vertex_attribute<int> map_to_vec = _omd.mesh_->vertices().make_attribute<int>(-1);

        for (auto fh : _omd.mesh_->faces())
        {
            if (_omd.map_to_layout_faces_.value()[fh].value != i)
                continue;

            for (auto vh : fh.vertices())
            {
                if (_omd.map_to_pn_vertices_.value()[vh].is_valid() || map_to_vec[vh] >= 0)
                    continue;

                inner_patch_vhs.push_back(vh);
                map_to_vec[vh] = inner_patch_vhs.size() - 1;
            }
        }

        int n_inner_vhs = inner_patch_vhs.size();
        auto L = torch::zeros({n_inner_vhs, n_inner_vhs}, torch::dtype(torch::kFloat64));
        auto rhs = torch::zeros({n_inner_vhs}, torch::dtype(torch::kFloat64));

        for (int i = 0; i < (int)inner_patch_vhs.size(); ++i)
        {
            auto vh = inner_patch_vhs[i];

            auto rhs_sum = torch::zeros({}, torch::dtype(torch::kFloat64));
            for (auto fh : vh.faces())
            {
                rhs_sum = rhs_sum + (1. / 3.) * areas[fh.idx.value] * mean_curvatures[fh.idx.value];
            }
            rhs[i] = rhs_sum.squeeze();

            auto sum = torch::zeros({}, torch::dtype(torch::kFloat64));
            for (HEH heh : vh.outgoing_halfedges())
            {
                auto vh_to = heh.vertex_to();
                auto weight = cotans[heh.edge().idx.value];
                sum = sum + weight;

                int j = map_to_vec[vh_to];
                if (j == -1)
                {
                    rhs[i] = rhs[i] + (-weight * 0.0);
                }
                else
                {
                    L.index_put_({i, j}, weight);
                }
            }
            L.index_put_({i, i}, -sum);
        }
        auto u = torch::linalg::solve(L, rhs, true);

        for (int i = 0; i < (int)inner_patch_vhs.size(); ++i)
        {
            auto vh = inner_patch_vhs[i];
            _eval_info.o_scaling_factor_extrinsic[vh] = tg::pow2(u[i].item<double>());
        }
        patch_scaling_factors_list.push_back(u);
    }
    auto patch_scaling_factors = torch::cat(patch_scaling_factors_list).squeeze();
    return (patch_scaling_factors * patch_scaling_factors).sum();
}

at::Tensor total_length_loss(PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    std::vector<torch::Tensor> edge_lengths_list;
    edge_lengths_list.reserve(_pnd.mesh_->edges().size());

    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto o_vhA = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexA()];
        auto o_vhB = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexB()];

        auto posA = _omd.torch_pos_.value()[o_vhA.value];
        auto posB = _omd.torch_pos_.value()[o_vhB.value];

        edge_lengths_list.push_back(torch::norm(posA - posB));
    }
    auto edge_lengths = torch::stack(edge_lengths_list);

    return edge_lengths.sum();
}

at::Tensor inner_angle_loss(PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    std::vector<torch::Tensor> dots_list;
    dots_list.reserve(_pnd.mesh_->vertices().size());
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        if (pn_vh.adjacent_vertices().size() <= 2)
            continue;

        auto om_vh = _pnd.map_to_overlay_vertices_.value()[pn_vh];
        auto torch_pos_from = _omd.torch_pos_.value()[om_vh.value];
        auto pn_outgoing_hehs = pn_vh.outgoing_halfedges().to_vector();

        for (size_t i = 0; i < pn_outgoing_hehs.size(); ++i)
        {
            auto pn_heh0 = pn_outgoing_hehs[i];
            auto pn_vh0_to = pn_heh0.vertex_to();
            auto om_vh0_to = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_vh0_to]);
            auto torch_pos0_to = _omd.torch_pos_.value()[om_vh0_to.idx.value];

            auto pn_heh1 = pn_outgoing_hehs[i];
            auto pn_vh1_to = pn_heh1.vertex_to();
            auto om_vh1_to = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_vh1_to]);
            auto torch_pos1_to = _omd.torch_pos_.value()[om_vh1_to.idx.value];

            auto vec0 = torch_pos0_to - torch_pos_from;
            auto vec1 = torch_pos1_to - torch_pos_from;

            dots_list.push_back(torch::abs(torch::dot(vec0, vec1)));
        }
    }

    auto dots = torch::stack(dots_list);
    return dots.mean();
}

at::Tensor repel_loss_v1(LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    std::vector<torch::Tensor> length_per_layout_edge_list;
    length_per_layout_edge_list.resize(_ld.mesh_->edges().size(), torch::zeros({}, torch::kFloat32));

    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto l_e_idx = _pnd.map_to_layout_edges_[pn_eh];

        auto o_vhA = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexA()];
        auto o_vhB = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexB()];

        auto posA = _omd.torch_pos_.value()[o_vhA.value];
        auto posB = _omd.torch_pos_.value()[o_vhB.value];

        auto length = torch::norm(posA - posB, 2);
        length_per_layout_edge_list[l_e_idx.value] += length;
    }
    auto length_per_layout_edge = torch::stack(length_per_layout_edge_list);

    auto l_arc_idx = compute_layout_edge_arc_idx(_ld);
    auto max_id = l_arc_idx.max();
    std::vector<std::vector<torch::Tensor>> lengths_per_arc;
    lengths_per_arc.resize(max_id + 1);
    for (auto l_eh : _ld.mesh_->edges())
    {
        auto arc_idx = l_arc_idx[l_eh];
        lengths_per_arc[arc_idx].push_back(length_per_layout_edge[l_eh.idx.value]);
    }

    std::vector<torch::Tensor> repel_losses;
    for (auto arc_lengts_list : lengths_per_arc)
    {
        auto const arc_lengths = torch::stack(arc_lengts_list);
        auto const total_arc_length = arc_lengths.sum();
        auto const arc_lengths_sqr = arc_lengths * arc_lengths;
        repel_losses.push_back(arc_lengths_sqr.sum() / total_arc_length);
    }

    assert(length_per_layout_edge.dim() == 1);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t_end - t_start;
    std::cout << "Non-Vectorized arc accumulation took " << ms.count() << " ms\n";

    return torch::stack(repel_losses).sum();
}

at::Tensor repel_loss_v2(LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    auto const& ld_mesh = *_ld.mesh_;
    auto const& pnd_mesh = *_pnd.mesh_;
    auto const& overlay_pos = _omd.torch_pos_.value();
    auto const& map_pn_to_o_vh = _pnd.map_to_overlay_vertices_.value();

    auto const num_layout_edges = ld_mesh.edges().size();

    auto length_per_layout_edge = torch::zeros({num_layout_edges}, torch::kFloat32);

    for (auto pn_eh : pnd_mesh.edges())
    {
        int64_t l_e_idx = _pnd.map_to_layout_edges_[pn_eh].value;

        auto o_vhA = map_pn_to_o_vh[pn_eh.vertexA()];
        auto o_vhB = map_pn_to_o_vh[pn_eh.vertexB()];

        auto length = torch::norm(overlay_pos[o_vhA.value] - overlay_pos[o_vhB.value], 2);
        length_per_layout_edge[l_e_idx] += length;
    }

    auto l_arc_idx = torch_compute_layout_edge_arc_idx(_ld);
    int64_t num_arcs = l_arc_idx.max().item<int64_t>() + 1;

    assert(length_per_layout_edge.dim() == 1);
    assert(l_arc_idx.dim() == 1);
    assert(l_arc_idx.sizes()[0] == length_per_layout_edge.sizes()[0]);

    auto total_arc_length = torch::zeros({num_arcs}, torch::kFloat32).index_add_(0, l_arc_idx, length_per_layout_edge);
    auto total_arc_length_sq = torch::zeros({num_arcs}, torch::kFloat32).index_add_(0, l_arc_idx, length_per_layout_edge * length_per_layout_edge);

    auto repel_per_arc = total_arc_length_sq / total_arc_length;

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t_end - t_start;
    std::cout << "Vectorized arc accumulation took " << ms.count() << " ms\n";

    return repel_per_arc.sum();
}

} // namespace LayoutOpt

#endif // LAYOUTOPT_WITH_TORCH
