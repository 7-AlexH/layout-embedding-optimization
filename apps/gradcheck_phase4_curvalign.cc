// Phase 4 / curvature-alignment validation (remove-autodiff plan): the
// hand-rolled stage (Adjoint/CurvatureAlignmentStage.{hh,cc}) must reproduce
// the torch principal_curvature_alignment_loss forward value and its gradient
// w.r.t. the overlay positions.
//
// Stage boundary: overlay 3D positions -> loss scalar. The direction-field
// basis/dir are constants (precomputed from the fixed target mesh), so the
// ONLY leaf is the overlay position tensor and the branch is validated
// standalone — composition with the S4..S7 chain is pure accumulation into
// d_overlay_pos (wired in S1).
//
// Oracle: the production principal_curvature_alignment_loss, run with the
// leaf swapped into _omd.torch_pos_ (same technique as the S4 gradcheck).
// The oracle loss must equal the production loss BITWISE (the leaf is a
// bitwise clone of the production positions). Two configs vary the loss
// weight w (the backward seed): leaf.grad of (w * loss) vs hand backward
// seeded with d_loss = w. The raw (unweighted) losses are compared.
//
// Additionally the new DirectionFieldData Eigen mirrors (basis_eigen /
// dir_eigen, filled in smooth_direction_field from the same doubles as the
// tensors) are checked bitwise against the tensors over all target faces.
//
// Like S4, the hand forward is expected to match to ~ulp, not bitwise
// (std::atan2/cos/sin vs torch kernels, summation order).
//
// Tolerances: 1e-9 abs / 1e-7 rel (plan section 5). In-process FP constants
// are not trustworthy on Windows (torch_cpu.dll clobbers callee-saved
// xmm14/15), so tolerances are loaded from volatiles right before use and all
// raw maxima are dumped to gradcheck_phase4_curvalign.txt for an external
// check.

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

#include <LayoutOpt/Adjoint/CurvatureAlignmentStage.hh>
#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/Embedding.hh>
#include <LayoutOpt/EmbeddingUtils.hh>
#include <LayoutOpt/Init.hh>
#include <LayoutOpt/ObjectiveFunctions.hh>
#include <LayoutOpt/OptimizationOptions.hh>
#include <LayoutOpt/PrincipalCurvature.hh>
#include <LayoutOpt/Resample.hh>
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
    double weight = 1.0;
    double ref_loss = 0.0;
    double oracle_loss = 0.0;
    double hand_loss = 0.0;
    bool oracle_bitwise = false;
    Cmp c_loss, c_pos;
    bool pass = false;
};

ConfigResult run_config(std::string const& _name,
                        double const _weight,
                        TargetMeshData& _tmd,
                        LayoutData& _ld,
                        PathNetworkData& _pnd,
                        OverlayMeshData& _omd)
{
    ConfigResult res;
    res.name = _name;
    res.weight = _weight;

    //=====================================================================
    // production reference (original positions tensor)
    //=====================================================================
    EvalInfo info;
    torch::Tensor ref_loss_t = principal_curvature_alignment_loss(_tmd, _ld, _pnd, _omd, info);
    res.ref_loss = ref_loss_t.item<double>();

    //=====================================================================
    // oracle: production loss re-run with the leaf swapped in. The weight is
    // applied outside the loss, exactly as Optimization.cc does, so
    // leaf.grad = w * dloss/dpos.
    //=====================================================================
    torch::Tensor orig_pos = _omd.torch_pos_.value();
    torch::Tensor pos3_leaf = orig_pos.detach().clone().requires_grad_(true);
    Eigen::MatrixX3d const& pos_e = _omd.pos_mat_.value();

    _omd.torch_pos_.emplace(pos3_leaf);
    torch::Tensor oracle_loss_t = principal_curvature_alignment_loss(_tmd, _ld, _pnd, _omd, info);
    _omd.torch_pos_.emplace(orig_pos); // restore; the graph holds its own refs

    res.oracle_loss = oracle_loss_t.item<double>();
    res.oracle_bitwise = std::memcmp(&res.oracle_loss, &res.ref_loss, sizeof(double)) == 0;

    (_weight * oracle_loss_t).backward();

    //=====================================================================
    // extract the oracle grad into a plain buffer (torch traffic ends here)
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

    //=====================================================================
    // hand forward + backward (pure double/Eigen; no torch from here on)
    //=====================================================================
    CurvAlignCtx ctx;
    res.hand_loss = curvature_alignment_forward(pos_e, _tmd, _pnd, _omd, ctx);

    Eigen::MatrixX3d d_pos = Eigen::MatrixX3d::Zero(pos_e.rows(), 3);
    curvature_alignment_backward(ctx, pos_e, _weight, d_pos);

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
    for (Eigen::Index r = 0; r < g_pos.rows(); ++r)
        for (int c = 0; c < 3; ++c)
            add(res.c_pos, d_pos(r, c), g_pos(r, c));

    res.pass = res.oracle_bitwise && res.c_loss.worst <= one && res.c_pos.worst <= one;
    return res;
}

void report(std::ostream& _f, ConfigResult const& _r)
{
    char buf[256];
    std::snprintf(buf, sizeof buf, "config %s\n", _r.name.c_str());
    _f << buf;
    std::snprintf(buf, sizeof buf, "  weight      %.17g (backward seed; losses below are raw)\n", _r.weight);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  ref_loss    %.17g\n", _r.ref_loss);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  oracle_loss %.17g  bitwise_eq %d\n", _r.oracle_loss, _r.oracle_bitwise ? 1 : 0);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  hand_loss   %.17g\n", _r.hand_loss);
    _f << buf;
    Cmp const* cmps[2] = {&_r.c_loss, &_r.c_pos};
    char const* names[2] = {"loss", "d_overlay_pos"};
    for (int i = 0; i < 2; ++i)
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

    // the DirectionFieldData Eigen mirrors must be BITWISE equal to the
    // tensors (both are built from the same doubles in smooth_direction_field)
    double mirror_max_abs = 0.0;
    for (auto t_fh : tmd.mesh_->faces())
    {
        auto const& d = tmd.direction_field_data_.value()[t_fh];
        auto b_acc = d.basis.accessor<double, 2>();
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 3; ++c)
                mirror_max_abs = std::max(mirror_max_abs, std::abs(d.basis_eigen(r, c) - b_acc[r][c]));
        auto dir_acc = d.dir.accessor<double, 1>();
        for (int c = 0; c < 2; ++c)
            mirror_max_abs = std::max(mirror_max_abs, std::abs(d.dir_eigen(c) - dir_acc[c]));
    }
    bool const mirror_ok = mirror_max_abs == 0.0;

    std::vector<ConfigResult> results;
    results.push_back(run_config("w=1.0", 1.0, tmd, ld, pnd, omd));
    results.push_back(run_config("w=0.7", 0.7, tmd, ld, pnd, omd));

    //=== dump raw values (torch-free file write -> trustworthy on Windows) ===
    {
        std::ofstream f("gradcheck_phase4_curvalign.txt");
        char buf[128];
        std::snprintf(buf, sizeof buf, "direction-field mirror max_abs %.17g (bitwise expected)\n", mirror_max_abs);
        f << buf;
        for (auto const& r : results)
            report(f, r);
    }

    {
        char buf[128];
        std::snprintf(buf, sizeof buf, "direction-field mirror max_abs %.17g (bitwise expected)\n", mirror_max_abs);
        std::cout << buf;
    }
    bool ok = mirror_ok;
    for (auto const& r : results)
    {
        report(std::cout, r);
        ok = ok && r.pass;
    }

    std::printf("\n%s\n", ok ? "PHASE 4 / CURVALIGN GRADCHECK PASSED" : "PHASE 4 / CURVALIGN GRADCHECK FAILED");
    return ok ? 0 : 1;
}
