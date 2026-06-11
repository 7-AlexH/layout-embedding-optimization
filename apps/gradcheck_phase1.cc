// Phase 1 validation (remove-autodiff plan): the sparse Eigen solve +
// hand-derived adjoint in sparse_harmonic_solve must reproduce the dense
// torch::linalg::solve path — same loss, same leaf gradients — on the banana
// model. Exits nonzero on mismatch (atol 1e-9 / rtol 1e-7, plan §5).
//
// Both evals run on the SAME embedding, so the first backward must retain the
// shared graph segment built during embedding computation (bary_coords leaves
// -> overlay positions), and leaf .grad() fields must be zeroed in between
// (leaves persist across evals; grads accumulate).

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>

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

using namespace LayoutOpt;
namespace fs = std::filesystem;

namespace
{

struct EvalResult
{
    double loss = 0.0;
    std::map<int, std::array<double, 2>> leaf_grads; // pn_vh idx -> grad
    double ms_forward = 0.0;
    double ms_backward = 0.0;
};

void zero_leaf_grads(PathNetworkData& _pnd)
{
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto& sp = _pnd.sp_on_target_.value()[pn_vh];
        if (sp.type == SurfacePointType::FacePoint && sp.bary_coords.defined())
            sp.bary_coords.mutable_grad() = torch::Tensor();
    }
}

EvalResult run_eval(TargetMeshData& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, OptimizationOptions& _opts, bool _use_sparse,
                    bool _retain_graph)
{
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

    _opts.harmonic_options.use_sparse_harmonic_solve = _use_sparse;

    EvalResult res;
    EvalInfo info;

    // same loss assembly as eval() in Optimization.cc
    auto t0 = clock::now();
    torch::Tensor loss = torch::zeros({}, torch::dtype(torch::kFloat64));
    loss = loss + _opts.w_harmonic_distorion_loss * harmonic_distortion_loss(_tmd, _ld, _pnd, _omd, _opts.harmonic_options, info);
    loss = loss + _opts.w_curvature_alignment_loss * principal_curvature_alignment_loss(_tmd, _ld, _pnd, _omd, info);
    auto t1 = clock::now();
    res.loss = loss.item<double>();
    res.ms_forward = ms(t0, t1);

    // the embedding's graph segment is shared between the two evals
    auto t2 = clock::now();
    loss.backward({}, _retain_graph);
    auto t3 = clock::now();
    res.ms_backward = ms(t2, t3);

    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto const& sp = _pnd.sp_on_target_.value()[pn_vh];
        if (sp.type != SurfacePointType::FacePoint)
            continue;
        auto grad = sp.bary_coords.grad();
        if (!grad.defined())
            continue;
        res.leaf_grads[pn_vh.idx.value] = {grad[0].item<double>(), grad[1].item<double>()};
    }
    return res;
}

} // namespace

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

    // The in-process tolerance comparison below is NOT trustworthy on Windows:
    // torch_cpu.dll clobbers callee-saved xmm14/15, poisoning any FP constant
    // MSVC parked in registers. The raw grads are therefore also dumped to
    // files (pure memory -> printf) for an external, torch-free comparison.
    auto dump = [](EvalResult const& r, char const* path)
    {
        std::ofstream f(path);
        char buf[128];
        std::snprintf(buf, sizeof buf, "loss %.17g\n", r.loss);
        f << buf;
        for (auto const& [idx, g] : r.leaf_grads)
        {
            std::snprintf(buf, sizeof buf, "%d %.17g %.17g\n", idx, g[0], g[1]);
            f << buf;
        }
    };

    std::printf("=== dense eval (oracle) ===\n");
    auto dense = run_eval(tmd, ld, pnd, omd, opts, /*use_sparse=*/false, /*retain_graph=*/true);
    std::printf("loss    = %.17g\n", dense.loss);
    std::printf("forward = %.1f ms, backward = %.1f ms, leaves = %zu\n", dense.ms_forward, dense.ms_backward, dense.leaf_grads.size());
    dump(dense, "gradcheck_phase1_dense.txt");

    zero_leaf_grads(pnd);

    std::printf("=== sparse eval ===\n");
    auto sparse = run_eval(tmd, ld, pnd, omd, opts, /*use_sparse=*/true, /*retain_graph=*/false);
    std::printf("loss    = %.17g\n", sparse.loss);
    std::printf("forward = %.1f ms, backward = %.1f ms, leaves = %zu\n", sparse.ms_forward, sparse.ms_backward, sparse.leaf_grads.size());
    dump(sparse, "gradcheck_phase1_sparse.txt");

    //=== compare ===
    // volatile, NOT constexpr: torch_cpu.dll clobbers callee-saved xmm14/15
    // (Win64 ABI violation, see torch-xmm notes); MSVC parks FP constants used
    // after the torch evals in exactly those registers. volatile forces a
    // memory reload at every use so the tolerances stay trustworthy.
    static volatile double atol_v = 1e-9;
    static volatile double rtol_v = 1e-7;
    static volatile double rel_floor_v = 1e-300;
    static volatile double diag_thresh_v = 1e-6;
    const double atol = atol_v;
    const double rtol = rtol_v;
    const double rel_floor = rel_floor_v;
    const double diag_thresh = diag_thresh_v;
    bool ok = true;

    double loss_diff = std::abs(dense.loss - sparse.loss);
    double loss_rel = loss_diff / std::max(std::abs(dense.loss), rel_floor);
    std::printf("\nloss abs diff = %.3g, rel diff = %.3g\n", loss_diff, loss_rel);
    if (loss_diff > atol && loss_rel > rtol)
    {
        std::printf("FAIL: loss mismatch\n");
        ok = false;
    }

    if (dense.leaf_grads.size() != sparse.leaf_grads.size())
    {
        std::printf("FAIL: leaf count mismatch (%zu dense vs %zu sparse)\n", dense.leaf_grads.size(), sparse.leaf_grads.size());
        ok = false;
    }

    // diagnostics: per-eval gradient magnitudes (catches accumulation bugs:
    // sparse ~= 2x dense means the leaf grads were not zeroed in between)
    double sum_dense = 0.0, sum_sparse = 0.0;
    for (auto const& [idx, g] : dense.leaf_grads)
        sum_dense += std::abs(g[0]) + std::abs(g[1]);
    for (auto const& [idx, g] : sparse.leaf_grads)
        sum_sparse += std::abs(g[0]) + std::abs(g[1]);
    std::printf("grad L1 norms: dense = %.17g, sparse = %.17g, ratio = %.6f\n", sum_dense, sum_sparse, sum_sparse / std::max(sum_dense, rel_floor));

    double max_abs = 0.0, max_rel = 0.0;
    int n_bad = 0, n_printed_diag = 0;
    for (auto const& [idx, gd] : dense.leaf_grads)
    {
        auto it = sparse.leaf_grads.find(idx);
        if (it == sparse.leaf_grads.end())
        {
            std::printf("FAIL: leaf %d missing in sparse eval\n", idx);
            ok = false;
            continue;
        }
        for (int c = 0; c < 2; ++c)
        {
            double a = gd[c], b = it->second[c];
            double abs_diff = std::abs(a - b);
            double rel_diff = abs_diff / std::max(std::abs(a), rel_floor);
            if (abs_diff > diag_thresh && n_printed_diag < 20)
            {
                std::printf("DIAG leaf %d[%d]: dense %.17g vs sparse %.17g (abs %.3g, rel %.3g)\n", idx, c, a, b, abs_diff, rel_diff);
                ++n_printed_diag;
            }
            max_abs = std::max(max_abs, abs_diff);
            if (abs_diff > atol)
                max_rel = std::max(max_rel, rel_diff);
            if (abs_diff > atol && rel_diff > rtol)
            {
                if (n_bad < 10)
                    std::printf("FAIL leaf %d[%d]: dense %.17g vs sparse %.17g (abs %.3g, rel %.3g)\n", idx, c, a, b, abs_diff, rel_diff);
                ++n_bad;
                ok = false;
            }
        }
    }
    std::printf("leaf grads: max abs diff = %.3g, max rel diff (where abs > atol) = %.3g, failures = %d\n", max_abs, max_rel, n_bad);

    std::printf("\n%s\n", ok ? "PHASE 1 GRADCHECK PASSED" : "PHASE 1 GRADCHECK FAILED");
    return ok ? 0 : 1;
}
