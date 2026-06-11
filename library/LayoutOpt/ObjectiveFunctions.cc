#include "ObjectiveFunctions.hh"

#include <LayoutOpt/Adjoint/SparseHarmonicSolve.hh>
#include <LayoutOpt/Visualization/ColorGenerator.hh>
#include <glow-extras/viewer/canvas.hh>
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/TorchUtils.hh"
#include "LayoutOpt/Utils/Timer.hh"
#include "LayoutOpt/Visualization/Colors.hh"
#include "LayoutOpt/Visualization/Viewing.hh"

#include <torch/linalg.h>

namespace LayoutOpt
{

at::Tensor harmonic_distortion_loss(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, HarmonicOptions const& _opt, EvalInfo& _eval_info)
{
    // prepare
    torch_compute_embedded_layout_edge_lengths(_ld, _pnd, _omd);
    torch_compute_t_for_pn_halfedge(_ld, _pnd, _omd);
    torch_compute_pn_uvs(_ld, _pnd, _opt);

    // compute areas
    auto areas = torch_face_areas(_omd.torch_pos_.value(), *_omd.mesh_.get());
    // compute cotans per heh
    auto cotans = torch_cotans(_omd.torch_pos_.value(), *_omd.mesh_.get());

    // store uvs of pn hehs, convention is that value belongs to from vertex of halfedge
    auto heh_count = _omd.mesh_->halfedges().size();
    auto uvs = torch::zeros({(int64_t)heh_count, 2}, torch::dtype(torch::kFloat64));

    // compute uvs
    for (auto pn_heh : _pnd.mesh_->halfedges())
    {
        auto o_vh_from = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_from()]);
        auto o_vh_to = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_to()]);

        auto o_heh = pm::halfedge_from_to(o_vh_from, o_vh_to);


        // Fill the preallocated tensor directly
        uvs[o_heh.idx.value] = _pnd.torch_uvs_.value()[pn_heh.idx.value];
    }

    // compute uvs per patch
    int n_patches = _ld.mesh_->faces().size();
    std::vector<torch::Tensor> SDE_per_patch_list;
    SDE_per_patch_list.reserve(n_patches);
    std::vector<torch::Tensor> param_areas_per_patch_list;
    param_areas_per_patch_list.reserve(n_patches);

    _eval_info.o_distortion_harmonic = _omd.mesh_->faces().make_attribute<double>(0.0);
    _eval_info.o_uvs = _omd.mesh_->halfedges().make_attribute<pos2>();


    //    std::mutex mtx; // for parallel code

    // #pragma omp parallel for
    for (int patch_i = 0; patch_i < n_patches; ++patch_i)
    {
        std::vector<VH> inner_patch_vhs;
        std::vector<VH> boundary_patch_vhs;
        std::vector<torch::Tensor> boundary_uvs;
        std::vector<FH> patch_fhs;

        // mtx.lock();
        pm::vertex_attribute<MappingIndex> map_to_vec = _omd.mesh_->vertices().make_attribute<MappingIndex>({-1, true});
        // mtx.unlock();

        torch_prepare_param(patch_i, _pnd, _omd, uvs, map_to_vec, inner_patch_vhs, boundary_patch_vhs, boundary_uvs, patch_fhs);
        auto inner_uvs = torch_harmonic_param(map_to_vec, _omd.patch_boundary_mask_.value(), inner_patch_vhs, boundary_uvs, cotans,
                                              _opt.use_sparse_harmonic_solve);

        std::vector<torch::Tensor> SDE_list;
        SDE_list.reserve(patch_fhs.size());

        torch::Tensor total_param_area = torch::zeros({}, torch::kFloat64);


        for (auto o_fh : patch_fhs)
        {
            auto hehA = o_fh.any_halfedge();
            auto hehB = hehA.next();
            auto hehC = hehB.next();

            auto i = map_to_vec[hehA.vertex_from()];
            auto j = map_to_vec[hehB.vertex_from()];
            auto k = map_to_vec[hehC.vertex_from()];

            torch::Tensor torch_posA_2D = !i.on_boundary ? inner_uvs[i.idx] : boundary_uvs[i.idx];
            torch::Tensor torch_posA_3D = _omd.torch_pos_.value()[hehA.vertex_from().idx.value];

            torch::Tensor torch_posB_2D = !j.on_boundary ? inner_uvs[j.idx] : boundary_uvs[j.idx];
            torch::Tensor torch_posB_3D = _omd.torch_pos_.value()[hehB.vertex_from().idx.value];

            torch::Tensor torch_posC_2D = !k.on_boundary ? inner_uvs[k.idx] : boundary_uvs[k.idx];
            torch::Tensor torch_posC_3D = _omd.torch_pos_.value()[hehC.vertex_from().idx.value];

            _eval_info.o_uvs[hehA] = torch_to_pos2(torch_posA_2D);
            _eval_info.o_uvs[hehB] = torch_to_pos2(torch_posB_2D);
            _eval_info.o_uvs[hehC] = torch_to_pos2(torch_posC_2D);

            auto SDE_info = torch_face_distortion(torch_posA_2D, torch_posB_2D, torch_posC_2D, torch_posA_3D, torch_posB_3D, torch_posC_3D, _opt);
            SDE_list.push_back(SDE_info.distortion_val);

            total_param_area = total_param_area + SDE_info.param_area;

            _eval_info.o_distortion_harmonic[o_fh] = SDE_info.distortion_val.item<double>();
        }

        // mtx.lock();
        param_areas_per_patch_list.push_back(total_param_area);
        SDE_per_patch_list.push_back(torch::stack(SDE_list).sum());
        // mtx.unlock();

        if (_opt.normalized)
        {
            // mtx.lock();
            DEBUG_OUT("normalizing prooved bad");
            for (auto o_fh : patch_fhs)
            {
                _eval_info.o_distortion_harmonic[o_fh] /= param_areas_per_patch_list.back().item<double>();
            }
            // mtx.unlock();
        }
    }


    // loss is sum over all
    auto loss = torch::stack(SDE_per_patch_list);
    if (_opt.normalized)
    {
        loss = loss / torch::stack(param_areas_per_patch_list);
    }
    return loss.sum();
}

torch::Tensor principal_curvature_alignment_loss(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, EvalInfo& _eval_info)
{
    assert(_tmd.direction_field_data_.has_value());

    std::vector<torch::Tensor> edge_lengths_list;
    edge_lengths_list.reserve(_pnd.mesh_->edges().size());

    std::vector<torch::Tensor> alignment_loss_list;
    alignment_loss_list.reserve(_pnd.mesh_->edges().size());

    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto o_vhA = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_eh.vertexA()]);
        auto o_vhB = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_eh.vertexB()]);

        auto o_eh = pm::edge_between(o_vhA, o_vhB);
        assert(o_eh.is_valid());

        auto t_fh = _omd.map_to_target_faces_.value()[o_eh.faceA()];
        assert(t_fh.is_valid());

        auto posA = _omd.torch_pos_.value()[o_vhA.idx.value];
        auto posB = _omd.torch_pos_.value()[o_vhB.idx.value];

        auto vec = posA - posB;
        auto norm = torch::norm(vec);
        edge_lengths_list.push_back(norm);

        // alignment
        auto const& direction_data = _tmd.direction_field_data_.value()[t_fh];
        auto vec_in_tangent_space = direction_data.basis.matmul(vec);
        auto angle = torch::atan2(vec_in_tangent_space[1], vec_in_tangent_space[0]);
        auto angle_4 = 4.0 * angle;
        auto n_rosy = torch::stack({torch::cos(angle_4), torch::sin(angle_4)});
        auto alignment_loss = (direction_data.dir - n_rosy).pow(2).sum();
        alignment_loss_list.push_back(norm * alignment_loss); //* direction_data.confidence);
    }

    return torch::stack(alignment_loss_list).sum() / torch::stack(edge_lengths_list).sum();
}


// Inactive loss bodies live in ObjectiveFunctionsAttic.cc (LAYOUTOPT_WITH_TORCH-gated).

pm::vertex_attribute<torch::Tensor> compute_gradients(torch::Tensor& _loss, PathNetworkData const& _pnd)
{
    _loss.backward();
    auto pn_bc_grads = _pnd.mesh_->vertices().make_attribute<torch::Tensor>(torch::zeros(2));
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto& sp = _pnd.sp_on_target_.value()[pn_vh];
        if (sp.type == SurfacePointType::FacePoint)
        {
            auto grad = sp.bary_coords.grad();
            pn_bc_grads[pn_vh] = grad;
        }
    }
    return pn_bc_grads;
}


void torch_prepare_param(int const _patch_value,
                         PathNetworkData const& _pnd,
                         OverlayMeshData const& _omd,
                         at::Tensor const& _torch_o_uvs,
                         pm::vertex_attribute<MappingIndex>& _map_to_vec,
                         std::vector<VH>& _inner_patch_vhs,
                         std::vector<VH>& _boundary_patch_vhs,
                         std::vector<at::Tensor>& _boundary_uvs,
                         std::vector<FH>& _patch_fhs)
{
    // conservative reserve
    _patch_fhs.reserve(_omd.mesh_->faces().size());
    _boundary_patch_vhs.reserve(_omd.mesh_->vertices().size());
    _inner_patch_vhs.reserve(_omd.mesh_->vertices().size());
    _boundary_uvs.reserve(_omd.mesh_->vertices().size());

    // collect all relevant vertices
    for (auto o_fh : _omd.mesh_->faces())
    {
        // the face does not belong to the current patch
        if (_omd.map_to_layout_faces_.value()[o_fh].value != _patch_value)
            continue;

        _patch_fhs.push_back(o_fh);
        for (auto o_heh : o_fh.halfedges())
        {
            auto o_vh_from = o_heh.vertex_from();
            auto o_vh_to = o_heh.vertex_to();

            // already visited
            if (_map_to_vec[o_vh_from].idx >= 0)
                continue;

            auto pn_vh_from = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[o_vh_from]);
            auto pn_vh_to = _pnd.mesh_->handle_of(_omd.map_to_pn_vertices_.value()[o_vh_to]);

            HEH pn_heh = HEH::invalid;
            if (pn_vh_from.is_valid() && pn_vh_to.is_valid())
            {
                pn_heh = pm::halfedge_from_to(pn_vh_from, pn_vh_to);
            }

            // boundary vertex and uvs are stored on this halfedge
            if (pn_heh.is_valid())
            {
                _boundary_patch_vhs.push_back(o_vh_from);
                int idx = static_cast<int>(_boundary_patch_vhs.size()) - 1;
                _map_to_vec[o_vh_from] = {idx, true};
                _boundary_uvs.push_back(_torch_o_uvs[o_heh.idx.value]);
            }
            else if (pn_vh_from.is_invalid())
            {
                _inner_patch_vhs.push_back(o_vh_from);
                int idx = static_cast<int>(_inner_patch_vhs.size()) - 1;
                _map_to_vec[o_vh_from] = {idx, false};
            }
        }
    }
}


at::Tensor torch_harmonic_param(pm::vertex_attribute<MappingIndex> const& _map_to_vec,
                                pm::edge_attribute<bool> const& _is_boundary,
                                std::vector<VH> const& _inner_patch_vhs,
                                std::vector<at::Tensor> const& _boundary_uvs,
                                at::Tensor const& _cotans,
                                bool _use_sparse_solve)
{
    int n_inner_vhs = _inner_patch_vhs.size();

    if (_use_sparse_solve)
    {
        if (n_inner_vhs == 0)
            return torch::zeros({0, 2}, torch::dtype(torch::kFloat64));

        // Same traversal as the dense assembly below, but only the system
        // structure is recorded; sparse_harmonic_solve assembles A := -L and
        // B := -R itself (and their adjoints in backward), so no autograd
        // graph is built through the assembly.
        std::vector<int64_t> off;  // flattened (i, j, e): inner neighbor
        std::vector<int64_t> diag; // flattened (i, e): every outgoing halfedge
        std::vector<int64_t> rhs;  // flattened (i, b, e): boundary neighbor
        for (int i = 0; i < n_inner_vhs; ++i)
        {
            auto o_vh = _inner_patch_vhs[i];
            for (HEH o_heh : o_vh.outgoing_halfedges())
            {
                MappingIndex j = _map_to_vec[o_heh.vertex_to()];
                int64_t e = o_heh.edge().idx.value;

                diag.push_back(i);
                diag.push_back(e);
                if (j.on_boundary)
                {
                    rhs.push_back(i);
                    rhs.push_back(j.idx);
                    rhs.push_back(e);
                }
                else
                {
                    off.push_back(i);
                    off.push_back(j.idx);
                    off.push_back(e);
                }
            }
        }

        auto to_idx_tensor = [](std::vector<int64_t>& _v, int64_t _cols)
        {
            if (_v.empty())
                return torch::zeros({0, _cols}, torch::dtype(torch::kInt64));
            return torch::from_blob(_v.data(), {(int64_t)_v.size() / _cols, _cols}, torch::dtype(torch::kInt64)).clone();
        };
        auto idx_off = to_idx_tensor(off, 3);
        auto idx_diag = to_idx_tensor(diag, 2);
        auto idx_rhs = to_idx_tensor(rhs, 3);

        // torch::stack on an empty list throws
        auto boundary_uvs = _boundary_uvs.empty() ? torch::zeros({0, 2}, torch::dtype(torch::kFloat64)) : torch::stack(_boundary_uvs);

        return sparse_harmonic_solve(_cotans, boundary_uvs, idx_off, idx_diag, idx_rhs, n_inner_vhs);
    }

    // dense path (validation oracle)
    // build lse
    auto L = torch::zeros({n_inner_vhs, n_inner_vhs}, torch::dtype(torch::kFloat64));
    auto rhs_u = torch::zeros({n_inner_vhs}, torch::dtype(torch::kFloat64));
    auto rhs_v = torch::zeros({n_inner_vhs}, torch::dtype(torch::kFloat64));

    for (int i = 0; i < _inner_patch_vhs.size(); ++i)
    {
        auto o_vh = _inner_patch_vhs[i];

        // Harmonic param: Laplace u/v = 0
        auto sum = torch::zeros({}, torch::dtype(torch::kFloat64));
        for (HEH o_heh : o_vh.outgoing_halfedges())
        {
            auto o_vh_to = o_heh.vertex_to();
            MappingIndex j = _map_to_vec[o_vh_to];

            auto weight = _cotans[o_heh.edge().idx.value];
            sum = sum + weight;

            if (j.on_boundary) // j belongs to cut
            {
                rhs_u[i] = rhs_u[i] - weight * _boundary_uvs[j.idx][0];
                rhs_v[i] = rhs_v[i] - weight * _boundary_uvs[j.idx][1];
            }
            else
            {
                L.index_put_({i, j.idx}, weight);
            }
        }
        L.index_put_({i, i}, -sum);
    }
    auto u = torch::linalg::solve(L, rhs_u, true);
    auto v = torch::linalg::solve(L, rhs_v, true);

    return torch::stack({u, v}, 1); // Stack along dimension 1
}


DistortionInfo compute_distortion(at::Tensor const& _ref_A,
                                  at::Tensor const& _ref_B,
                                  at::Tensor const& _ref_C,
                                  at::Tensor const& _param_A,
                                  at::Tensor const& _param_B,
                                  at::Tensor const& _param_C,
                                  HarmonicOptions const& _opt)
{
    auto ref_AB = _ref_B - _ref_A;
    auto ref_AC = _ref_C - _ref_A;

    auto param_AB = _param_B - _param_A;
    auto param_AC = _param_C - _param_A;

    auto ref_area = 0.5 * (ref_AB[0] * ref_AC[1] - ref_AB[1] * ref_AC[0]);
    auto param_area = 0.5 * (param_AB[0] * param_AC[1] - param_AB[1] * param_AC[0]);

    if (ref_area.item<double>() < EPS || param_area.item<double>() < EPS)
        return {torch::zeros({}), torch::zeros({})}; // if the ref area is very small it is likely from a detour triangle and than the param area may flip

    if (param_area.item<double>() != param_area.item<double>() || ref_area.item<double>() != ref_area.item<double>())
    {
        DEBUG_VAR(param_area.item<double>())
        DEBUG_VAR(ref_area.item<double>())
        return {torch::zeros({}), torch::zeros({})};
        // auto v = gv::view();
        // pos3 ref_A = pos3(torch_to_pos2(_ref_A)) + 0.0001 * vec3(0., 0., 1.);
        // pos3 ref_B = pos3(torch_to_pos2(_ref_B)) + 0.0001 * vec3(0., 0., 1.);
        // pos3 ref_C = pos3(torch_to_pos2(_ref_C)) + 0.0001 * vec3(0., 0., 1.);

        // pos3 param_A = pos3(torch_to_pos2(_param_A));
        // pos3 param_B = pos3(torch_to_pos2(_param_B));
        // pos3 param_C = pos3(torch_to_pos2(_param_C));

        // DEBUG_VAR(ref_A)
        // DEBUG_VAR(ref_B)
        // DEBUG_VAR(ref_C)

        // DEBUG_VAR(param_A)
        // DEBUG_VAR(param_B)
        // DEBUG_VAR(param_C)

        // auto c = gv::canvas();

        // c.add_face(ref_A, ref_B, ref_C, BLUE_50);
        // c.add_face(param_A, param_B, param_C, BLUE);

        // c.add_line(ref_A, ref_B, GREEN_50);
        // c.add_line(param_A, param_B, GREEN);

        // c.add_line(ref_A, ref_C, MAGENTA_50);
        // c.add_line(param_A, param_C, MAGENTA);
        return {torch::zeros({}), torch::zeros({})};
    }

    TORCH_CHECK(ref_area.item<double>() >= 0, "ref_area is too small: ", ref_area.item<double>())
    TORCH_CHECK(param_area.item<double>() >= 0, "param_area is too small: ", param_area.item<double>(), "but ref area is: ", ref_area.item<double>())

    auto mat_ref = torch::stack({ref_AB, ref_AC}).t();
    auto mat_param = torch::stack({param_AB, param_AC}).t();

    auto jacobian = mat_param.matmul(torch::linalg_inv(mat_ref));
    auto sing_vals = torch_singular_values(jacobian);

    std::vector<torch::Tensor> distortion_list;
    distortion_list.reserve(10);

    if(_opt.w_SDE_DirectComputation > 0)
    {
        // following https://people.engr.tamu.edu/schaefer/research/bijective.pdf
        auto factor1 = (torch::ones({}, torch::dtype(torch::kFloat64)) + (ref_area * ref_area / param_area * param_area));

        auto norm_ref_AB = torch::norm(ref_AB);
        auto norm_ref_AC = torch::norm(ref_AC);

        auto norm_param_AB = torch::norm(param_AB);
        auto norm_param_AC = torch::norm(param_AC);

        auto factor2 = (norm_param_AC * norm_param_AC * norm_ref_AB * norm_ref_AB + norm_param_AB * norm_param_AB * norm_ref_AC * norm_ref_AC)
                       / (torch::tensor({4.0}, torch::dtype(torch::kFloat64)) * ref_area);

        auto factor3 = (torch::dot(param_AC, param_AB) * torch::dot(ref_AC, ref_AB)) / (torch::tensor({2.0}, torch::dtype(torch::kFloat64)) * ref_area);

        distortion_list.push_back(_opt.w_SDE_DirectComputation * (factor1 * (factor2 - factor3)).squeeze());
    }
    if(_opt.w_SDE_SingValDecomp > 0)
    {
        auto val = ref_area * (sing_vals[0] * sing_vals[0] + sing_vals[1] * sing_vals[1])
                    + param_area * (1.0 / (sing_vals[0] * sing_vals[0] + EPS) + 1.0 / (sing_vals[1] * sing_vals[1] + EPS));
        distortion_list.push_back( _opt.w_SDE_SingValDecomp * val);
    }
    if(_opt.w_DE_SingValDecomp > 0)
    {
        auto val = ref_area * (sing_vals[0] * sing_vals[0] + sing_vals[1] * sing_vals[1]);
        distortion_list.push_back( _opt.w_DE_SingValDecomp * val);
    }
    if(_opt.w_AIAP_SingValDecomp > 0)
    {
        torch::Tensor sing_min;
        torch::Tensor sing_max;

        if (sing_vals[0].item<double>() < sing_vals[1].item<double>())
        {
            sing_min = sing_vals[0];
            sing_max = sing_vals[1];
        }
        else
        {
            sing_min = sing_vals[1];
            sing_max = sing_vals[0];
        }
        auto val = ref_area * (sing_max * sing_max + 1.0 / (sing_min * sing_min));
        distortion_list.push_back( _opt.w_AIAP_SingValDecomp * val);
    }
    if(_opt.w_I_DevFrom1_SingValDecomp > 0)
    {
        auto val = ref_area * ((sing_vals[0] - 1.0) * (sing_vals[0] - 1.0) + (sing_vals[1] - 1.0) * (sing_vals[1] - 1.0));
        distortion_list.push_back( _opt.w_I_DevFrom1_SingValDecomp * val);
    }
    if(_opt.w_AreaPreserving_SingValDecomp > 0)
    {
        auto val = ref_area * ((1.0 - sing_vals[0] * sing_vals[1]) * (1.0 - sing_vals[0] * sing_vals[1]));
        distortion_list.push_back( _opt.w_AreaPreserving_SingValDecomp * val);
    }
    return {torch::stack(distortion_list).sum(), param_area};
}


// remove-autodiff Phase 4 (S7): the per-overlay-face body of the
// harmonic_distortion_loss patch loop, extracted verbatim so the S7 gradcheck
// can drive the exact production computation with the stage inputs as leaves.
DistortionInfo torch_face_distortion(at::Tensor const& _posA_2D,
                                     at::Tensor const& _posB_2D,
                                     at::Tensor const& _posC_2D,
                                     at::Tensor const& _posA_3D,
                                     at::Tensor const& _posB_3D,
                                     at::Tensor const& _posC_3D,
                                     HarmonicOptions const& _opt)
{
    torch::Tensor base_unit = torch::tensor({1.0, 0.0}, torch::kFloat64);
    torch::Tensor zero2D = torch::zeros({2}, torch::kFloat64);

    // compute 2D reference triangles to compute the scale factors
    auto torch_vecAB_3D = _posB_3D - _posA_3D;
    auto torch_vecAC_3D = _posC_3D - _posA_3D;

    auto torch_vecAB_2D = _posB_2D - _posA_2D;
    auto torch_vecAC_2D = _posC_2D - _posA_2D;

    auto length_vecAB_3D = torch::norm(torch_vecAB_3D);
    auto torch_unit_vecAB_3D = torch_vecAB_3D / length_vecAB_3D;

    auto length_vecAB_2D = torch::norm(torch_vecAB_2D);
    auto torch_unit_vecAB_2D = torch_vecAB_2D / length_vecAB_2D;

    auto length_vecAC_3D = torch::norm(torch_vecAC_3D);
    auto torch_unit_vecAC_3D = torch_vecAC_3D / length_vecAC_3D;

    auto length_vecAC_2D = torch::norm(torch_vecAC_2D);
    auto torch_unit_vecAC_2D = torch_vecAC_2D / length_vecAC_2D;

    auto axis_3D = torch::linalg_cross(torch_unit_vecAB_3D, torch_unit_vecAC_3D);
    auto axis_norm_3D = torch::norm(axis_3D);
    auto unit_axis_3D = axis_3D / axis_norm_3D;

    auto angle_3D = torch::atan2(torch::dot(torch::linalg_cross(torch_unit_vecAB_3D, torch_unit_vecAC_3D), unit_axis_3D),
                                 torch::dot(torch_unit_vecAB_3D, torch_unit_vecAC_3D));

    auto angle_2D = torch::atan2(torch_unit_vecAB_2D[0] * torch_unit_vecAC_2D[1] - torch_unit_vecAB_2D[1] * torch_unit_vecAC_2D[0],
                                 torch_unit_vecAB_2D[0] * torch_unit_vecAC_2D[0] + torch_unit_vecAB_2D[1] * torch_unit_vecAC_2D[1]); // https://wumbo.net/formulas/angle-between-two-vectors-2d/

    // rebuild triangles
    auto torch_ref_A = zero2D;
    auto torch_param_A = zero2D;

    auto torch_ref_B = torch_ref_A + length_vecAB_3D * base_unit;
    auto torch_param_B = torch_ref_A + length_vecAB_2D * base_unit;

    auto torch_ref_C = zero2D.clone();
    torch_ref_C[0] = length_vecAC_3D * torch::cos(angle_3D);
    torch_ref_C[1] = length_vecAC_3D * torch::sin(angle_3D);

    auto torch_param_C = zero2D.clone();
    torch_param_C[0] = length_vecAC_2D * torch::cos(angle_2D);
    torch_param_C[1] = length_vecAC_2D * torch::sin(angle_2D);

    return compute_distortion(torch_ref_A, torch_ref_B, torch_ref_C, torch_param_A, torch_param_B, torch_param_C, _opt);
}


void torch_compute_embedded_layout_edge_lengths(LayoutData& _ld, PathNetworkData const& _pnd, OverlayMeshData const& _omd)
{
    assert(_pnd.map_to_overlay_vertices_.has_value());

    std::vector<torch::Tensor> edge_lengths_list;
    edge_lengths_list.resize(_ld.mesh_->edges().size());
    for (auto& val : edge_lengths_list)
    {
        val = torch::zeros({}, torch::dtype(torch::kFloat64));
    }

    for (auto pn_eh : _pnd.mesh_->edges())
    {
        auto const om_A = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexA()];
        auto const om_B = _pnd.map_to_overlay_vertices_.value()[pn_eh.vertexB()];

        auto const l_eh = _pnd.map_to_layout_edges_[pn_eh];

        auto const& posA = _omd.torch_pos_.value()[om_A.value];
        auto const& posB = _omd.torch_pos_.value()[om_B.value];

        edge_lengths_list[l_eh.value] += torch::norm(posA - posB);
    }
    _ld.torch_embedded_edge_length_.emplace(torch::stack(edge_lengths_list));

    // remove-autodiff Phase 3: plain-double mirror, derived from the tensor just built
    // (same flat layout: indexed by layout-edge idx). Phase 4 (S4) computes this directly.
    {
        auto const& tel = _ld.torch_embedded_edge_length_.value();
        Eigen::VectorXd mirror(tel.size(0));
        for (int64_t i = 0; i < tel.size(0); ++i)
            mirror(i) = tel[i].item<double>();
        _ld.embedded_edge_length_.emplace(std::move(mirror));
    }
}

void torch_compute_t_for_pn_halfedge(LayoutData const& _ld, PathNetworkData& _pnd, OverlayMeshData const& _omd)
{
    assert(_ld.torch_embedded_edge_length_.has_value());

    std::vector<torch::Tensor> t_list;
    t_list.reserve(_pnd.mesh_->halfedges().size());
    for (auto heh : _pnd.mesh_->halfedges())
    {
        t_list.push_back(torch::zeros({}, torch::dtype(torch::kFloat64)));
    }
    auto l_heh_visited = _ld.mesh_->halfedges().make_attribute<bool>(false); // needed for extension where l_eh are subdivided

    for (auto l_heh : _ld.mesh_->halfedges())
    {
        if (l_heh_visited[l_heh])
            continue; // if it is already visited, we don't need to process again

        if (l_heh.vertex_from().outgoing_halfedges().size() == 2)
            continue; // the from vertex is an inner subdivision vertex.

        l_heh_visited[l_heh] = true; // start processing

        auto l_eh = l_heh.edge();
        auto pn_vertex_from = _pnd.mesh_->handle_of(l_heh.vertex_from());

        HEH pn_iter_heh = pn_vertex_from.outgoing_halfedges().filter([&](HEH pn_heh) { return _pnd.map_to_layout_edges_[pn_heh] == l_eh.idx; }).first();
        assert(pn_iter_heh.is_valid());

        // accumulated length on path network
        auto om_A = _pnd.map_to_overlay_vertices_.value()[pn_iter_heh.vertex_from()];
        auto om_B = _pnd.map_to_overlay_vertices_.value()[pn_iter_heh.vertex_to()];

        torch::Tensor accumulated_length = torch::zeros({}, torch::kFloat64);
        accumulated_length = accumulated_length + torch::norm(_omd.torch_pos_.value()[om_A.value] - _omd.torch_pos_.value()[om_B.value]);

        torch::Tensor total_length = torch::zeros({}, torch::kFloat64);
        total_length = total_length + _ld.torch_embedded_edge_length_.value()[l_eh.idx.value];

        pn_iter_heh = pn_iter_heh.next();
        auto l_heh_iter = l_heh;

        while (l_heh_iter.vertex_to().outgoing_halfedges().size() == 2)
        {
            l_heh_iter = l_heh_iter.next();

            l_heh_visited[l_heh_iter] = true;
            total_length = total_length + _ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value];
        }

        while (pn_iter_heh.vertex_from().outgoing_halfedges().size() == 2)
        {
            t_list[pn_iter_heh.idx.value] = 1.0 - (accumulated_length / total_length);

            auto om_A = _pnd.map_to_overlay_vertices_.value()[pn_iter_heh.vertex_from()];
            auto om_B = _pnd.map_to_overlay_vertices_.value()[pn_iter_heh.vertex_to()];

            accumulated_length = accumulated_length + torch::norm(_omd.torch_pos_.value()[om_A.value] - _omd.torch_pos_.value()[om_B.value]);
            pn_iter_heh = pn_iter_heh.next();
        }
        t_list[pn_iter_heh.idx.value] = torch::zeros({}, torch::dtype(torch::kFloat64));
    }
    _pnd.torch_t_.emplace(torch::stack(t_list));

    // remove-autodiff Phase 3: plain-double mirror (flat, by pn-halfedge idx).
    {
        auto const& tt = _pnd.torch_t_.value();
        Eigen::VectorXd mirror(tt.size(0));
        for (int64_t i = 0; i < tt.size(0); ++i)
            mirror(i) = tt[i].item<double>();
        _pnd.t_flat_.emplace(std::move(mirror));
    }
}

void torch_compute_pn_uvs(LayoutData const& _ld, PathNetworkData& _pnd, HarmonicOptions const& _opts)
{
    assert(_ld.torch_embedded_edge_length_.has_value());
    assert(_pnd.torch_t_.has_value());

    std::vector<torch::Tensor> uvs_list;
    uvs_list.reserve(_pnd.mesh_->halfedges().size());
    for (auto heh : _pnd.mesh_->halfedges())
    {
        uvs_list.push_back(torch::zeros({2}, torch::dtype(torch::kFloat64)));
    }

    for (auto l_fh : _ld.mesh_->faces())
    {
        HEH l_heh = l_fh.halfedges().filter([](HEH _heh) { return _heh.vertex_from().outgoing_halfedges().size() != 2; }).first();

        assert(l_heh.is_valid());

        torch::Tensor length = torch::zeros({}, torch::dtype(torch::kFloat64));
        torch::Tensor height = torch::zeros({}, torch::dtype(torch::kFloat64));

        if (_opts.fixed_parameter_domain)
        {
            length = 2.0 * torch::ones({}, torch::dtype(torch::kFloat64));
            height = 2.0 * torch::ones({}, torch::dtype(torch::kFloat64));
        }
        else
        {
            auto l_heh_iter = l_heh;
            // compute length of the 4 sides:
            std::vector<torch::Tensor> total_length_0_list;
            total_length_0_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);

            while (l_heh_iter.vertex_to().outgoing_halfedges().size() == 2)
            {
                l_heh_iter = l_heh_iter.next();
                total_length_0_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);
            }
            torch::Tensor length_0 = torch::stack(total_length_0_list).sum();

            l_heh_iter = l_heh_iter.next();
            std::vector<torch::Tensor> total_height_0_list;
            total_height_0_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);

            while (l_heh_iter.vertex_to().outgoing_halfedges().size() == 2)
            {
                l_heh_iter = l_heh_iter.next();
                total_height_0_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);
            }
            torch::Tensor height_0 = torch::stack(total_height_0_list).sum();

            l_heh_iter = l_heh_iter.next();

            std::vector<torch::Tensor> total_length_1_list;
            total_length_1_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);

            while (l_heh_iter.vertex_to().outgoing_halfedges().size() == 2)
            {
                l_heh_iter = l_heh_iter.next();
                total_length_1_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);
            }
            torch::Tensor length_1 = torch::stack(total_length_1_list).sum();

            l_heh_iter = l_heh_iter.next();
            std::vector<torch::Tensor> total_height_1_list;
            total_height_1_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);

            while (l_heh_iter.vertex_to().outgoing_halfedges().size() == 2)
            {
                l_heh_iter = l_heh_iter.next();
                total_height_1_list.push_back(_ld.torch_embedded_edge_length_.value()[l_heh_iter.edge().idx.value]);
            }
            torch::Tensor height_1 = torch::stack(total_height_1_list).sum();

            //length = 0.5 * (length_0 + length_1);
            //height = 0.5 * (height_0 + height_1);

            auto length_res = 0.5 * (length_0 + length_1);
            auto height_res = 0.5 * (height_0 + height_1);

            length = torch::max(0.02 * torch::ones({}, torch::dtype(torch::kFloat64)), length_res);
            height = torch::max(0.02 * torch::ones({}, torch::dtype(torch::kFloat64)), height_res);
        }

        torch::Tensor factors = torch::tensor({{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}}, torch::dtype(torch::kFloat64));

        // DEBUG_OUT("")
        // DEBUG_VAR(length.item<double>())
        // DEBUG_VAR(height.item<double>())

        auto cd = gv::canvas_data();

        auto pn_vertex_from = _pnd.mesh_->handle_of(l_heh.vertex_from()); // there is a one to one map from layout to pn
        HEH pn_iter_heh
            = pn_vertex_from.outgoing_halfedges().filter([&](HEH pn_heh) { return _pnd.map_to_layout_edges_[pn_heh] == l_heh.edge().idx; }).first();
        assert(pn_iter_heh.is_valid());

        for (size_t i = 0; i < 4; ++i)
        {
            auto A = torch::stack({factors[i][0] * length, factors[i][1] * height});
            auto B = torch::stack({factors[(i + 1) % 4][0] * length, factors[(i + 1) % 4][1] * height});

            // DEBUG_VAR(A[0].item<double>())
            // DEBUG_VAR(A[1].item<double>())
            // DEBUG_VAR(B[0].item<double>())
            // DEBUG_VAR(B[1].item<double>())

            uvs_list[pn_iter_heh.idx.value] = A; // t is not valid for beginning and end, bc there is no way of storing it for each halfedge of the layout consistenly
            pn_iter_heh = pn_iter_heh.next();

            cd.add_point(pos3(torch_to_pos2(uvs_list[pn_iter_heh.idx.value])));
            cd.add_line(pos3(torch_to_pos2(uvs_list[pn_iter_heh.prev().idx.value])), pos3(torch_to_pos2(uvs_list[pn_iter_heh.idx.value])));

            while (pn_iter_heh.vertex_from().adjacent_vertices().size() == 2) //_pnd.map_to_layout_edges_[pn_iter_heh.edge()] == l_heh.edge().idx)
            {
                auto t = _pnd.torch_t_.value()[pn_iter_heh.idx.value];

                uvs_list[pn_iter_heh.idx.value] = t * A + (1.0 - t) * B;


                cd.add_point(pos3(torch_to_pos2(uvs_list[pn_iter_heh.idx.value])));

                pn_iter_heh = pn_iter_heh.next();

                // DEBUG_VAR(t.item<double>())
                // DEBUG_VAR(uvs_list[pn_iter_heh.idx.value][0].item<double>())
                // DEBUG_VAR(uvs_list[pn_iter_heh.idx.value][1].item<double>())
            }
            uvs_list[pn_iter_heh.idx.value] = B;

            cd.add_point(pos3(torch_to_pos2(uvs_list[pn_iter_heh.idx.value])));

            // auto c = gv::canvas();
            // c.add_data(cd);

            // DEBUG_OUT("")
        }
    }
    _pnd.torch_uvs_.emplace(torch::stack(uvs_list));

    // remove-autodiff Phase 3: plain-double mirror ([#pn_heh x 2], by pn-halfedge idx).
    {
        auto const& tu = _pnd.torch_uvs_.value();
        Eigen::MatrixX2d mirror(tu.size(0), 2);
        for (int64_t i = 0; i < tu.size(0); ++i)
        {
            mirror(i, 0) = tu[i][0].item<double>();
            mirror(i, 1) = tu[i][1].item<double>();
        }
        _pnd.uvs_flat_.emplace(std::move(mirror));
    }
}


} // namespace LayoutOpt
