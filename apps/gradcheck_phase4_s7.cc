// Phase 4 / S7 validation (remove-autodiff plan): the hand-rolled distortion
// stage (Adjoint/DistortionStage.{hh,cc}) must reproduce the torch path's
// forward value and stage-input adjoints.
//
// Stage boundary (inputs treated as leaves, plan section 4/S7): per-patch
// inner UVs (S6 solve output), per-patch boundary UVs (S4 output), overlay 3D
// positions (S1/S3 output) -> scalar loss. Per patch, the production
// torch_prepare_param + torch_harmonic_param re-derive the real input values,
// which are then detached into FRESH leaf tensors -- backward cannot enter the
// solve, so the resulting grads are exactly the S7-only adjoints. The torch
// oracle is the production face loop itself (torch_face_distortion was
// extracted verbatim from harmonic_distortion_loss), so its loss must equal
// the production loss BITWISE -- this also validates that the extraction
// refactor changed nothing.
//
// Two weight configs: the production one (AIAP + AreaPreserving) and an
// all-branches one that exercises every distortion branch's adjoint.
//
// Tolerances: 1e-9 abs / 1e-7 rel (plan section 5). As with the earlier
// gradchecks, in-process FP constants are not trustworthy on Windows
// (torch_cpu.dll clobbers callee-saved xmm14/15), so tolerances are loaded
// from volatiles right before use and all raw maxima are dumped to
// gradcheck_phase4_s7.txt for an external, torch-free check.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>
#include <polymesh/properties.hh>

#include <LayoutOpt/Adjoint/DistortionStage.hh>
#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/Embedding.hh>
#include <LayoutOpt/EmbeddingUtils.hh>
#include <LayoutOpt/Init.hh>
#include <LayoutOpt/ObjectiveFunctions.hh>
#include <LayoutOpt/OptimizationOptions.hh>
#include <LayoutOpt/Resample.hh>
#include <LayoutOpt/ScalarFields.hh>
#include <LayoutOpt/TorchUtils.hh>

using namespace LayoutOpt;
namespace fs = std::filesystem;

namespace
{

struct Cmp
{
    double max_abs = 0.0; // max |hand - oracle|
    double worst = 0.0;   // max |hand - oracle| / max(atol, rtol * |oracle|); pass iff <= 1
    size_t n = 0;
};

struct ConfigResult
{
    std::string name;
    double ref_loss = 0.0;    // production harmonic_distortion_loss
    double oracle_loss = 0.0; // torch oracle on the stage-input leaves
    double hand_loss = 0.0;   // hand-rolled stage pair
    bool oracle_bitwise = false;
    Cmp c_loss, c_inner, c_bnd, c_pos;
    bool pass = false;
};

ConfigResult run_config(std::string const& _name,
                        HarmonicOptions const& _hopt,
                        TargetMeshData& _tmd,
                        LayoutData& _ld,
                        PathNetworkData& _pnd,
                        OverlayMeshData& _omd)
{
    ConfigResult res;
    res.name = _name;

    //=====================================================================
    // production reference (also populates pnd.torch_uvs_ / mirrors)
    //=====================================================================
    EvalInfo info;
    torch::Tensor ref_loss_t = harmonic_distortion_loss(_tmd, _ld, _pnd, _omd, _hopt, info);
    res.ref_loss = ref_loss_t.item<double>();

    //=====================================================================
    // rebuild the stage inputs exactly as production does
    //=====================================================================
    auto cotans = torch_cotans(_omd.torch_pos_.value(), *_omd.mesh_.get());

    auto heh_count = _omd.mesh_->halfedges().size();
    auto uvs = torch::zeros({(int64_t)heh_count, 2}, torch::dtype(torch::kFloat64));
    for (auto pn_heh : _pnd.mesh_->halfedges())
    {
        auto o_vh_from = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_from()]);
        auto o_vh_to = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_to()]);
        auto o_heh = pm::halfedge_from_to(o_vh_from, o_vh_to);
        uvs[o_heh.idx.value] = _pnd.torch_uvs_.value()[pn_heh.idx.value];
    }

    // overlay positions as a single global leaf; the hand path reads the
    // bitwise-identical Phase 3 Eigen mirror
    torch::Tensor pos3_leaf = _omd.torch_pos_.value().detach().clone().requires_grad_(true);
    Eigen::MatrixX3d const& pos_e = _omd.pos_mat_.value();

    int const n_patches = (int)_ld.mesh_->faces().size();
    std::vector<torch::Tensor> SDE_per_patch;
    SDE_per_patch.reserve(n_patches);
    std::vector<torch::Tensor> inner_leaves;
    inner_leaves.reserve(n_patches);
    std::vector<std::vector<torch::Tensor>> bnd_leaves(n_patches);
    std::vector<PatchDistortionCtx> hand_patches;
    hand_patches.reserve(n_patches);

    for (int patch_i = 0; patch_i < n_patches; ++patch_i)
    {
        std::vector<VH> inner_vhs;
        std::vector<VH> bnd_vhs;
        std::vector<torch::Tensor> bnd_uvs;
        std::vector<FH> fhs;
        pm::vertex_attribute<MappingIndex> map = _omd.mesh_->vertices().make_attribute<MappingIndex>({-1, true});

        torch_prepare_param(patch_i, _pnd, _omd, uvs, map, inner_vhs, bnd_vhs, bnd_uvs, fhs);
        auto inner = torch_harmonic_param(map, _omd.patch_boundary_mask_.value(), inner_vhs, bnd_uvs, cotans, _hopt.use_sparse_harmonic_solve);

        // detach into fresh leaves: backward cannot reach the solve -> the
        // grads are the isolated S7 stage-input adjoints
        torch::Tensor inner_leaf = inner.detach().clone().requires_grad_(true);
        std::vector<torch::Tensor>& bls = bnd_leaves[patch_i];
        bls.reserve(bnd_uvs.size());
        for (auto const& b : bnd_uvs)
            bls.push_back(b.detach().clone().requires_grad_(true));

        // torch oracle: the production face loop, on the leaves
        std::vector<torch::Tensor> SDE_list;
        SDE_list.reserve(fhs.size());
        torch::Tensor total_param_area = torch::zeros({}, torch::kFloat64);
        for (auto o_fh : fhs)
        {
            auto hehA = o_fh.any_halfedge();
            auto hehB = hehA.next();
            auto hehC = hehB.next();

            auto i = map[hehA.vertex_from()];
            auto j = map[hehB.vertex_from()];
            auto k = map[hehC.vertex_from()];

            torch::Tensor posA_2D = !i.on_boundary ? inner_leaf[i.idx] : bls[i.idx];
            torch::Tensor posB_2D = !j.on_boundary ? inner_leaf[j.idx] : bls[j.idx];
            torch::Tensor posC_2D = !k.on_boundary ? inner_leaf[k.idx] : bls[k.idx];
            torch::Tensor posA_3D = pos3_leaf[hehA.vertex_from().idx.value];
            torch::Tensor posB_3D = pos3_leaf[hehB.vertex_from().idx.value];
            torch::Tensor posC_3D = pos3_leaf[hehC.vertex_from().idx.value];

            auto SDE_info = torch_face_distortion(posA_2D, posB_2D, posC_2D, posA_3D, posB_3D, posC_3D, _hopt);
            SDE_list.push_back(SDE_info.distortion_val);
            total_param_area = total_param_area + SDE_info.param_area;
        }
        SDE_per_patch.push_back(torch::stack(SDE_list).sum());

        // hand forward on the same (bitwise-identical) leaf values
        Eigen::MatrixX2d inner_e(inner_leaf.size(0), 2);
        if (inner_leaf.size(0) > 0)
        {
            auto acc = inner_leaf.accessor<double, 2>();
            for (int64_t r = 0; r < inner_leaf.size(0); ++r)
            {
                inner_e(r, 0) = acc[r][0];
                inner_e(r, 1) = acc[r][1];
            }
        }
        std::vector<vec2d> bnd_e;
        bnd_e.reserve(bls.size());
        for (auto const& b : bls)
        {
            auto acc = b.accessor<double, 1>();
            bnd_e.emplace_back(acc[0], acc[1]);
        }

        hand_patches.push_back(patch_distortion_forward(fhs, map, inner_e, bnd_e, pos_e, _hopt));
        inner_leaves.push_back(inner_leaf);
    }

    //=====================================================================
    // oracle loss (must equal production BITWISE) + backward
    //=====================================================================
    torch::Tensor oracle_loss_t = torch::stack(SDE_per_patch).sum();
    res.oracle_loss = oracle_loss_t.item<double>();
    res.oracle_bitwise = std::memcmp(&res.oracle_loss, &res.ref_loss, sizeof(double)) == 0;
    oracle_loss_t.backward();

    //=====================================================================
    // extract the oracle grads into plain buffers (all torch traffic ends here)
    //=====================================================================
    Eigen::MatrixX3d g_pos = Eigen::MatrixX3d::Zero(pos_e.rows(), 3);
    {
        torch::Tensor g = pos3_leaf.grad();
        if (g.defined())
        {
            auto acc = g.accessor<double, 2>();
            for (int64_t r = 0; r < g.size(0); ++r)
                for (int c = 0; c < 3; ++c)
                    g_pos(r, c) = acc[r][c];
        }
    }
    std::vector<Eigen::MatrixX2d> g_inner(n_patches);
    std::vector<std::vector<vec2d>> g_bnd(n_patches);
    for (int p = 0; p < n_patches; ++p)
    {
        g_inner[p] = Eigen::MatrixX2d::Zero(inner_leaves[p].size(0), 2);
        torch::Tensor g = inner_leaves[p].grad();
        if (g.defined())
        {
            auto acc = g.accessor<double, 2>();
            for (int64_t r = 0; r < g.size(0); ++r)
            {
                g_inner[p](r, 0) = acc[r][0];
                g_inner[p](r, 1) = acc[r][1];
            }
        }
        g_bnd[p].assign(bnd_leaves[p].size(), vec2d::Zero());
        for (size_t b = 0; b < bnd_leaves[p].size(); ++b)
        {
            torch::Tensor gb = bnd_leaves[p][b].grad();
            if (gb.defined())
            {
                auto acc = gb.accessor<double, 1>();
                g_bnd[p][b] = vec2d(acc[0], acc[1]);
            }
        }
    }

    //=====================================================================
    // hand loss + backward (pure double/Eigen; no torch from here on)
    //=====================================================================
    res.hand_loss = distortion_loss_forward(hand_patches, false);

    std::vector<double> d_sde, d_pa;
    distortion_loss_backward(hand_patches, false, 1.0, d_sde, d_pa);

    Eigen::MatrixX3d d_pos = Eigen::MatrixX3d::Zero(pos_e.rows(), 3);
    std::vector<Eigen::MatrixX2d> d_inner(n_patches);
    std::vector<std::vector<vec2d>> d_bnd(n_patches);
    for (int p = 0; p < n_patches; ++p)
    {
        d_inner[p] = Eigen::MatrixX2d::Zero(g_inner[p].rows(), 2);
        d_bnd[p].assign(g_bnd[p].size(), vec2d::Zero());
        patch_distortion_backward(hand_patches[p], d_sde[p], d_pa[p], _hopt, d_inner[p], d_bnd[p], d_pos);
    }

    //=====================================================================
    // compare (volatile-loaded tolerances; see header comment)
    //=====================================================================
    static volatile double atol_v = 1e-9;
    static volatile double rtol_v = 1e-7;
    static volatile double one_v = 1.0;
    double const atol = atol_v;
    double const rtol = rtol_v;
    double const one = one_v;

    auto add = [&](Cmp& s, double h, double o)
    {
        double const d = std::abs(h - o);
        s.max_abs = std::max(s.max_abs, d);
        s.worst = std::max(s.worst, d / std::max(atol, rtol * std::abs(o)));
        ++s.n;
    };

    add(res.c_loss, res.hand_loss, res.oracle_loss);
    for (int p = 0; p < n_patches; ++p)
    {
        for (Eigen::Index r = 0; r < g_inner[p].rows(); ++r)
            for (int c = 0; c < 2; ++c)
                add(res.c_inner, d_inner[p](r, c), g_inner[p](r, c));
        for (size_t b = 0; b < g_bnd[p].size(); ++b)
            for (int c = 0; c < 2; ++c)
                add(res.c_bnd, d_bnd[p][b](c), g_bnd[p][b](c));
    }
    for (Eigen::Index r = 0; r < g_pos.rows(); ++r)
        for (int c = 0; c < 3; ++c)
            add(res.c_pos, d_pos(r, c), g_pos(r, c));

    res.pass = res.oracle_bitwise && res.c_loss.worst <= one && res.c_inner.worst <= one && res.c_bnd.worst <= one && res.c_pos.worst <= one;
    return res;
}

void report(std::ostream& _f, ConfigResult const& _r)
{
    char buf[256];
    std::snprintf(buf, sizeof buf, "config %s\n", _r.name.c_str());
    _f << buf;
    std::snprintf(buf, sizeof buf, "  ref_loss    %.17g\n", _r.ref_loss);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  oracle_loss %.17g  bitwise_eq %d\n", _r.oracle_loss, _r.oracle_bitwise ? 1 : 0);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  hand_loss   %.17g\n", _r.hand_loss);
    _f << buf;
    Cmp const* cmps[4] = {&_r.c_loss, &_r.c_inner, &_r.c_bnd, &_r.c_pos};
    char const* names[4] = {"loss", "d_inner_uvs", "d_boundary_uvs", "d_overlay_pos"};
    for (int i = 0; i < 4; ++i)
    {
        std::snprintf(buf, sizeof buf, "  %-14s max_abs %.17g  worst %.17g  (n=%zu)\n", names[i], cmps[i]->max_abs, cmps[i]->worst, cmps[i]->n);
        _f << buf;
    }
    std::snprintf(buf, sizeof buf, "  %s\n", _r.pass ? "PASS" : "FAIL");
    _f << buf;
}

} // namespace

int main()
{
    fs::path base_path = fs::path(DATA_PATH) / "Spot/";
    fs::path path = base_path / "bananna.obj";
    fs::path l_path = base_path / "bananna_layout.obj";

    pm::Mesh m;
    pm::vertex_attribute<pos3> pos(m);
    pm::load(path.string().c_str(), m, pos);

    pm::Mesh l;
    pm::vertex_attribute<pos3> l_pos(l);
    pm::load(l_path.string().c_str(), l, l_pos);

    preprocess_target_and_layout(pos, l_pos);

    TargetMeshData tmd(pos);
    tmd.direction_field_data_.emplace(smooth_direction_field(tmd.pos_));
    LayoutData ld(l_pos);
    PathNetworkData pnd(ld.pos_);
    OverlayMeshData omd(tmd.pos_);

    // same embedding state as optimize.cc reaches at its first eval
    compute_layout_embedding_init(tmd, ld, pnd, omd, true);
    resample_layout(tmd, ld, pnd, omd, 0.2);
    compute_layout_embedding_init(tmd, ld, pnd, omd, false);

    // config A: the production weights (optimize.cc / gradcheck_phase3)
    HarmonicOptions prod;
    prod.w_AIAP_SingValDecomp = 0.5;
    prod.w_AreaPreserving_SingValDecomp = 0.5;

    // config B: every distortion branch active -> validates every branch adjoint
    HarmonicOptions all;
    all.w_SDE_DirectComputation = 0.3;
    all.w_SDE_SingValDecomp = 0.2;
    all.w_DE_SingValDecomp = 0.15;
    all.w_AIAP_SingValDecomp = 0.25;
    all.w_I_DevFrom1_SingValDecomp = 0.1;
    all.w_AreaPreserving_SingValDecomp = 0.2;

    std::vector<ConfigResult> results;
    results.push_back(run_config("production(AIAP+AreaPreserving)", prod, tmd, ld, pnd, omd));
    results.push_back(run_config("all-branches", all, tmd, ld, pnd, omd));

    //=== dump raw values (torch-free file write -> trustworthy on Windows) ===
    {
        std::ofstream f("gradcheck_phase4_s7.txt");
        for (auto const& r : results)
            report(f, r);
    }

    bool ok = true;
    for (auto const& r : results)
    {
        report(std::cout, r);
        ok = ok && r.pass;
    }

    std::printf("\n%s\n", ok ? "PHASE 4 / S7 GRADCHECK PASSED" : "PHASE 4 / S7 GRADCHECK FAILED");
    return ok ? 0 : 1;
}
