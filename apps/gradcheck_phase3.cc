// Phase 3 validation (remove-autodiff plan): the plain-double / Eigen mirrors
// added in Phase 3 must reproduce their torch counterparts exactly (forward
// values only -- Phase 3 introduces no gradient math). Checks, on the banana
// model after one forward loss eval:
//
//   1. SurfacePoint::get_pos(Eigen::MatrixX3d, mesh)  vs  the torch get_pos
//      (the only INDEPENDENT recomputation -- bary_full + Eigen interpolation
//       vs torch tensor interpolation),
//   2. OverlayMeshData::pos_mat_                      vs  torch_pos_,
//   3. LayoutData::embedded_edge_length_             vs  torch_embedded_edge_length_,
//   4. PathNetworkData::t_flat_                      vs  torch_t_,
//   5. PathNetworkData::uvs_flat_                    vs  torch_uvs_.
//
// Exits nonzero on mismatch (atol 1e-9 / rtol 1e-7, plan section 5). As with
// gradcheck_phase1, the IN-PROCESS tolerance comparison is not trustworthy on
// Windows (torch_cpu.dll clobbers callee-saved xmm14/15, poisoning FP constants
// MSVC parks in registers), so the raw max-diffs are also dumped to
// gradcheck_phase3.txt for an external, torch-free sanity check.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>

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

int main()
{
    OptimizationOptions opts;
    opts.w_curvature_alignment_loss = 0.1;
    opts.w_harmonic_distorion_loss = 1.0;
    opts.harmonic_options.w_AIAP_SingValDecomp = 0.5;
    opts.harmonic_options.w_AreaPreserving_SingValDecomp = 0.5;
    opts.harmonic_options.w_I_DevFrom1_SingValDecomp = .0;
    opts.harmonic_options.w_DE_SingValDecomp = .0;
    opts.harmonic_options.w_SDE_SingValDecomp = .0;

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

    // one forward loss eval -- populates torch_t_/torch_uvs_/torch_embedded_edge_length_
    // and their Phase 3 Eigen mirrors (no backward needed: Phase 3 is forward-only).
    EvalInfo info;
    torch::Tensor loss = torch::zeros({}, torch::dtype(torch::kFloat64));
    loss = loss + opts.w_harmonic_distorion_loss * harmonic_distortion_loss(tmd, ld, pnd, omd, opts.harmonic_options, info);
    loss = loss + opts.w_curvature_alignment_loss * principal_curvature_alignment_loss(tmd, ld, pnd, omd, info);
    double const loss_val = loss.item<double>();

    //=========================================================================
    // 1. SurfacePoint Eigen get_pos vs torch get_pos (independent recomputation)
    //=========================================================================
    double max_sp = 0.0;
    int n_sp = 0;
    for (auto pn_vh : pnd.mesh_->vertices())
    {
        auto const& sp = pnd.sp_on_target_.value()[pn_vh];
        if (!sp.is_valid())
            continue;
        pos3 const p_torch = torch_to_pos3(sp.get_pos(tmd.torch_pos_, *tmd.mesh_.get()));
        vec3d const p_eig = sp.get_pos(tmd.pos_mat_, *tmd.mesh_.get());
        double const d = std::max({std::abs(p_torch.x - p_eig.x()), std::abs(p_torch.y - p_eig.y()), std::abs(p_torch.z - p_eig.z())});
        max_sp = std::max(max_sp, d);
        ++n_sp;
    }

    //=========================================================================
    // 2. OverlayMeshData::pos_mat_ vs torch_pos_
    //=========================================================================
    double max_opos = 0.0;
    {
        auto const& tp = omd.torch_pos_.value();
        auto const& pm_ = omd.pos_mat_.value();
        for (int64_t i = 0; i < tp.size(0); ++i)
            for (int64_t c = 0; c < 3; ++c)
                max_opos = std::max(max_opos, std::abs(tp[i][c].item<double>() - pm_(i, c)));
    }

    //=========================================================================
    // 3. LayoutData::embedded_edge_length_ vs torch_embedded_edge_length_
    //=========================================================================
    double max_el = 0.0;
    {
        auto const& tel = ld.torch_embedded_edge_length_.value();
        auto const& el = ld.embedded_edge_length_.value();
        for (int64_t i = 0; i < tel.size(0); ++i)
            max_el = std::max(max_el, std::abs(tel[i].item<double>() - el(i)));
    }

    //=========================================================================
    // 4. PathNetworkData::t_flat_ vs torch_t_
    //=========================================================================
    double max_t = 0.0;
    {
        auto const& tt = pnd.torch_t_.value();
        auto const& tf = pnd.t_flat_.value();
        for (int64_t i = 0; i < tt.size(0); ++i)
            max_t = std::max(max_t, std::abs(tt[i].item<double>() - tf(i)));
    }

    //=========================================================================
    // 5. PathNetworkData::uvs_flat_ vs torch_uvs_
    //=========================================================================
    double max_uv = 0.0;
    {
        auto const& tu = pnd.torch_uvs_.value();
        auto const& uf = pnd.uvs_flat_.value();
        for (int64_t i = 0; i < tu.size(0); ++i)
            for (int64_t c = 0; c < 2; ++c)
                max_uv = std::max(max_uv, std::abs(tu[i][c].item<double>() - uf(i, c)));
    }

    //=== dump raw diffs (torch-free file write -> trustworthy on Windows) ===
    {
        std::ofstream f("gradcheck_phase3.txt");
        char buf[160];
        std::snprintf(buf, sizeof buf, "loss %.17g\n", loss_val);
        f << buf;
        std::snprintf(buf, sizeof buf, "sp_get_pos %.17g (n=%d)\n", max_sp, n_sp);
        f << buf;
        std::snprintf(buf, sizeof buf, "overlay_pos %.17g\n", max_opos);
        f << buf;
        std::snprintf(buf, sizeof buf, "embedded_edge_length %.17g\n", max_el);
        f << buf;
        std::snprintf(buf, sizeof buf, "t %.17g\n", max_t);
        f << buf;
        std::snprintf(buf, sizeof buf, "uvs %.17g\n", max_uv);
        f << buf;
    }

    //=== verdict ===
    // volatile, NOT constexpr: torch_cpu.dll clobbers callee-saved xmm14/15
    // (Win64 ABI violation); MSVC parks FP constants used after the torch eval in
    // those registers. volatile forces a memory reload so the tolerance is sound.
    static volatile double atol_v = 1e-9;
    double const atol = atol_v;

    std::printf("loss                 = %.17g\n", loss_val);
    std::printf("sp get_pos  max diff = %.3g  (%d surface points)\n", max_sp, n_sp);
    std::printf("overlay pos max diff = %.3g\n", max_opos);
    std::printf("edge length max diff = %.3g\n", max_el);
    std::printf("t           max diff = %.3g\n", max_t);
    std::printf("uvs         max diff = %.3g\n", max_uv);

    bool const ok = max_sp <= atol && max_opos <= atol && max_el <= atol && max_t <= atol && max_uv <= atol;
    std::printf("\n%s\n", ok ? "PHASE 3 MIRROR CHECK PASSED" : "PHASE 3 MIRROR CHECK FAILED");
    return ok ? 0 : 1;
}
