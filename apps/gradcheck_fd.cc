// Permanent finite-difference gradient checker for the hand-rolled adjoint
// chain. The torch gradient oracle is gone (see documentation/plans/
// remove-autodiff.md); central finite differences of the hand forward loss
// are the independent reference.
//
// State: the canonical gradcheck state used throughout the remove-autodiff
// plan — banana pair,
// init(true) -> resample(0.2) -> init(false) — with the FULL production
// configuration (w_h = 1, AIAP/AreaPreserving 0.5/0.5, w_c = 0.1).
//
// Check: compute_gradients_hand vs central differences (h = 1e-6) of the hand
// forward loss w.r.t. the bary coords of sampled FacePoint path-network
// vertices. Gate: 1e-5 rel (the remove-autodiff plan's validation table),
// abs floor 1e-8.
//
// Smoothness guard: the loss landscape contains sliver-triangle regions that
// are locally nonsmooth (diagnosed in plan Phase 5c: fd(1e-5) vs fd(1e-4) disagree
// in SIGN at some rows), where no finite difference is meaningful. Each sample
// is cross-checked with a second step size (h2 = 1e-5); samples where the two
// FD estimates disagree by more than 1e-3 RELATIVE TO THE FD MAGNITUDE
// (abs floor 1e-9 — central-difference cancellation noise at h = 1e-6 is
// ~eps*|L|/h ~ 1e-10) are reported and excluded from the gate (count printed —
// expected to be a small fraction).
//
// All raw values are dumped to gradcheck_fd.txt at %.17g.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>

#include <LayoutOpt/Adjoint/HandGradients.hh>
#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/TriangleStrip.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/Embedding.hh>
#include <LayoutOpt/Init.hh>
#include <LayoutOpt/OptimizationOptions.hh>
#include <LayoutOpt/Resample.hh>
#include <LayoutOpt/ScalarFields.hh>

using namespace LayoutOpt;
namespace fs = std::filesystem;

namespace
{

// one hand forward loss at the CURRENT path-network state
double forward_loss(std::vector<TriangleStrip> const& _strips,
                    TargetMeshData const& _tmd,
                    LayoutData const& _ld,
                    PathNetworkData const& _pnd,
                    OverlayMeshData const& _omd,
                    OptimizationOptions const& _opts)
{
    LeafCtx leaf;
    leaf_collect(_strips, _tmd, _ld, _pnd, leaf);
    Eigen::MatrixX2d const bary = collect_bary(_pnd);
    HandLossCtx ctx;
    return hand_loss_forward(leaf, bary, _strips, _tmd, _ld, _pnd, _omd, _opts, ctx);
}

struct Sample
{
    int pn_idx = -1;
    int coord = 0;
    double hand = 0.0;
    double fd = 0.0;     // central, h
    double fd2 = 0.0;    // central, h2 (smoothness cross-check)
    bool smooth = true;  // fd vs fd2 agree
    double worst = 0.0;  // |fd - hand| / max(atol, rtol*|hand|)
};

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

    // the canonical gradcheck state (matches the retired torch-oracle harnesses)
    compute_layout_embedding_init(tmd, ld, pnd, omd, true);
    resample_layout(tmd, ld, pnd, omd, 0.2);
    std::vector<TriangleStrip> strips;
    compute_layout_embedding_init(tmd, ld, pnd, omd, false, &strips);

    // production configuration (optimize.cc)
    OptimizationOptions opts;
    opts.w_curvature_alignment_loss = 0.1;
    opts.w_harmonic_distorion_loss = 1.0;
    opts.harmonic_options.w_AIAP_SingValDecomp = 0.5;
    opts.harmonic_options.w_AreaPreserving_SingValDecomp = 0.5;

    double hand_loss = 0.0;
    auto hand_grads = compute_gradients_hand(strips, tmd, ld, pnd, omd, opts, &hand_loss);

    // sample FacePoint leaves with a fixed stride (deterministic)
    std::vector<VH> face_vhs;
    for (auto pn_vh : pnd.mesh_->vertices())
        if (pnd.sp_on_target_.value()[pn_vh].type == SurfacePointType::FacePoint)
            face_vhs.push_back(pn_vh);
    int const N_SAMPLES = 48;
    int const stride = std::max(1, (int)face_vhs.size() / N_SAMPLES);

    double const h = 1e-6;
    double const h2 = 1e-5;
    double const atol = 1e-8;
    double const rtol = 1e-5;
    double const smooth_rtol = 1e-3;

    std::vector<Sample> samples;
    auto& sp_attr = pnd.sp_on_target_.value();
    for (size_t k = 0; k < face_vhs.size(); k += (size_t)stride)
    {
        auto pn_vh = face_vhs[k];
        SurfacePoint& sp = sp_attr[pn_vh];
        vec2d const saved = sp.bary_params;

        for (int c = 0; c < 2; ++c)
        {
            auto eval_at = [&](double _delta)
            {
                vec2d b = saved;
                b[c] += _delta;
                sp.bary_params = b;
                double const L = forward_loss(strips, tmd, ld, pnd, omd, opts);
                sp.bary_params = saved;
                return L;
            };

            Sample s;
            s.pn_idx = pn_vh.idx.value;
            s.coord = c;
            s.hand = hand_grads[pn_vh][c];
            s.fd = (eval_at(h) - eval_at(-h)) / (2.0 * h);
            s.fd2 = (eval_at(h2) - eval_at(-h2)) / (2.0 * h2);
            double const fd_mag = std::max(std::abs(s.fd), std::abs(s.fd2));
            s.smooth = std::abs(s.fd - s.fd2) <= std::max(smooth_rtol * fd_mag, 1e-9);
            s.worst = std::abs(s.fd - s.hand) / std::max(atol, rtol * std::abs(s.hand));
            samples.push_back(s);
        }
    }

    double worst = 0.0, worst_abs = 0.0;
    long n_smooth = 0, n_skipped = 0;
    for (auto const& s : samples)
    {
        if (!s.smooth)
        {
            ++n_skipped;
            continue;
        }
        ++n_smooth;
        worst = std::max(worst, s.worst);
        worst_abs = std::max(worst_abs, std::abs(s.fd - s.hand));
    }

    bool const pass = n_smooth > 0 && worst <= 1.0;

    auto report = [&](std::ostream& f)
    {
        char buf[512];
        std::snprintf(buf, sizeof buf, "hand loss %.17g\n", hand_loss);
        f << buf;
        std::snprintf(buf, sizeof buf, "FacePoint leaves %zu, sampled %zu (stride %d), coords checked %zu\n", face_vhs.size(),
                      samples.size() / 2, stride, samples.size());
        f << buf;
        for (auto const& s : samples)
        {
            std::snprintf(buf, sizeof buf, "  pn %5d c%d  hand %.17g  fd %.17g  fd2 %.17g  %s  worst %.6g\n", s.pn_idx, s.coord, s.hand,
                          s.fd, s.fd2, s.smooth ? "smooth   " : "NONSMOOTH", s.worst);
            f << buf;
        }
        std::snprintf(buf, sizeof buf, "smooth %ld  skipped(nonsmooth) %ld\n", n_smooth, n_skipped);
        f << buf;
        std::snprintf(buf, sizeof buf, "worst %.17g (gate <= 1, rel %.3g abs floor %.3g)  worst_abs %.17g\n", worst, rtol, atol, worst_abs);
        f << buf;
        std::snprintf(buf, sizeof buf, "%s\n", pass ? "PASS" : "FAIL");
        f << buf;
    };

    {
        std::ofstream f("gradcheck_fd.txt");
        report(f);
    }
    report(std::cout);

    std::printf("\n%s\n", pass ? "FD GRADCHECK PASSED" : "FD GRADCHECK FAILED");
    return pass ? 0 : 1;
}
