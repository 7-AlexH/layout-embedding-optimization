// Phase 4 / S1 validation (remove-autodiff plan): the hand-rolled leaf stage
// (Adjoint/LeafStage.{hh,cc}) + the full hand gradient driver
// (Adjoint/HandGradients.{hh,cc}) must reproduce production compute_gradients
// end to end: bary leaves -> S1 -> S3 -> S4/S5/S6/S7 + curvature alignment ->
// total loss -> d_bary.
//
// Oracle: the PRODUCTION path itself. Per config the differentiable overlay
// is rebuilt exactly as each optimize iteration does
// (compute_differentiable_intersections_for_overlay_stable + sync_tg_and_torch
// — bitwise idempotent here, reported), leaf grads are zeroed, the loss is
// assembled exactly like eval() (w_h * harmonic_distortion_loss +
// w_c * principal_curvature_alignment_loss), and compute_gradients runs
// backward and collects sp.bary_coords.grad() per FacePoint. Config A is the
// FULL production configuration (w_h=1, w_c=0.1, AIAP/AreaPreserving 0.5/0.5)
// whose total loss is the known Phase 3 baseline 1.2734862918788761 (banana
// pair, post init+resample+init) — the replication self-check.
//
// Hand: leaf_collect / collect_bary once; per config hand_loss_forward +
// hand_loss_backward, plus the packaged compute_gradients_hand (must agree
// with the direct calls bitwise). The rewritten overlay rows (S1 node rows +
// S3 interior rows) must equal production pos_mat_ bitwise, and the 2D strip
// endpoints must equal sp.get_pos(strip.heh_pos_2d) bitwise.
//
// Finite differences (plan section 5: the only oracle NOT derived from
// torch): central differences h=1e-6 of the hand forward loss w.r.t. sampled
// leaf components (top-|grad| + an even stride), gated at
// max(1e-8 abs, 1e-5 rel) per the plan's 1e-5-rel row.
//
// Tolerances vs torch: 1e-9 abs / 1e-7 rel (plan section 5; the plan's
// end-of-Phase-4 "1e-9 abs" leaf gate is the reported d_bary max_abs).
// In-process FP constants are not trustworthy on Windows (torch_cpu.dll
// clobbers callee-saved xmm14/15), so tolerances are loaded from volatiles
// right before use and all raw maxima are dumped to gradcheck_phase4_s1.txt.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>
#include <polymesh/properties.hh>

#include <LayoutOpt/Adjoint/HandGradients.hh>
#include <LayoutOpt/Adjoint/LeafStage.hh>
#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/TriangleStrip.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/DifferentiableIntersection.hh>
#include <LayoutOpt/Embedding.hh>
#include <LayoutOpt/EmbeddingOverlay.hh>
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

struct FdFail
{
    int pn = -1, comp = -1;
    double hand = 0.0, fd = 0.0;
};

struct ConfigResult
{
    std::string name;
    double ref_loss = 0.0;  // production torch loss (eval() assembly)
    double hand_loss = 0.0; // hand forward loss
    double rebuild_max_abs = 0.0;  // pos_mat_ before vs after the graph rebuild (bitwise expected)
    double hand_pos_max_abs = 0.0; // hand rewritten rows vs production pos_mat_ (bitwise expected)
    double seg_max_abs = 0.0;      // hand seg_a/b vs torch sp.get_pos(heh_pos_2d) (bitwise expected)
    double wrapper_grad_max_abs = 0.0; // compute_gradients_hand vs direct d_bary (bitwise expected)
    double wrapper_loss_abs = 0.0;     // its loss_out vs hand_loss (bitwise expected)
    long oracle_undefined_grads = 0;   // FacePoint leaves whose torch .grad() was undefined (0 expected)
    Cmp c_loss, c_bary;
    size_t fd_n = 0;
    double fd_worst_abs = 0.0;   // max |hand - fd|
    double fd_worst_gated = 0.0; // max |hand - fd| / max(fd_atol, fd_rtol * |fd|); pass iff <= 1
    std::vector<FdFail> fd_fails;
    bool pass = false;
};

ConfigResult run_config(std::string const& _name,
                        OptimizationOptions const& _opts,
                        std::vector<TriangleStrip> const& _strips,
                        TargetMeshData& _tmd,
                        LayoutData& _ld,
                        PathNetworkData& _pnd,
                        OverlayMeshData& _omd,
                        LeafCtx const& _leaf,
                        Eigen::MatrixX2d const& _bary,
                        std::vector<std::pair<int, int>> const& _comps)
{
    ConfigResult res;
    res.name = _name;

    //=====================================================================
    // rebuild the differentiable overlay graph, exactly like each production
    // iteration (and the inline init below) does; bitwise idempotent here
    // since neither the sps nor the strips changed
    //=====================================================================
    Eigen::MatrixX3d const pos_before = _omd.pos_mat_.value();
    compute_differentiable_intersections_for_overlay_stable(_strips, _tmd, _ld, _pnd, _omd);
    sync_tg_and_torch(_omd);
    res.rebuild_max_abs = (_omd.pos_mat_.value() - pos_before).cwiseAbs().maxCoeff();

    // zero the leaf grads (configs share the same leaf tensors)
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto& sp = _pnd.sp_on_target_.value()[pn_vh];
        if (sp.bary_coords.defined() && sp.bary_coords.requires_grad() && sp.bary_coords.grad().defined())
            sp.bary_coords.mutable_grad().reset();
    }

    //=====================================================================
    // production reference: eval()'s loss assembly + compute_gradients
    //=====================================================================
    EvalInfo info;
    torch::Tensor loss = torch::zeros({}, torch::dtype(torch::kFloat64));
    if (_opts.w_harmonic_distorion_loss > 0)
        loss = loss + _opts.w_harmonic_distorion_loss * harmonic_distortion_loss(_tmd, _ld, _pnd, _omd, _opts.harmonic_options, info);
    if (_opts.w_curvature_alignment_loss > 0)
        loss = loss + _opts.w_curvature_alignment_loss * principal_curvature_alignment_loss(_tmd, _ld, _pnd, _omd, info);
    res.ref_loss = loss.item<double>();

    auto oracle_attr = compute_gradients(loss, _pnd);

    // extract to plain buffers (torch traffic for the oracle ends here).
    // Only FacePoint rows are read: production fills the attribute only for
    // those (everything else keeps the float32-zeros default = zero grad,
    // matching the zero-initialized g_bary).
    Eigen::MatrixX2d g_bary = Eigen::MatrixX2d::Zero(_bary.rows(), 2);
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        if (_pnd.sp_on_target_.value()[pn_vh].type != SurfacePointType::FacePoint)
            continue;
        torch::Tensor const& g = oracle_attr[pn_vh];
        if (!g.defined())
        {
            // production stores sp.bary_coords.grad() verbatim; undefined here
            // would mean a FacePoint leaf the loss never reached
            if (!_pnd.map_to_layout_vertices_[pn_vh].is_invalid())
                ++res.oracle_undefined_grads;
            continue;
        }
        auto acc = g.accessor<double, 1>();
        g_bary(pn_vh.idx.value, 0) = acc[0];
        g_bary(pn_vh.idx.value, 1) = acc[1];
    }

    //=====================================================================
    // hand forward + self-checks
    //=====================================================================
    HandLossCtx ctx;
    res.hand_loss = hand_loss_forward(_leaf, _bary, _strips, _tmd, _ld, _pnd, _omd, _opts, ctx);

    res.hand_pos_max_abs = (ctx.pos - _omd.pos_mat_.value()).cwiseAbs().maxCoeff();
    for (auto l_eh : _ld.mesh_->edges())
    {
        int const le = l_eh.idx.value;
        TriangleStrip const& strip = _strips[le];
        auto sp_from = _pnd.sp_on_target_.value()[strip.pn_vhs.front()];
        auto sp_to = _pnd.sp_on_target_.value()[strip.pn_vhs.back()];
        auto A = sp_from.get_pos(strip.heh_pos_2d);
        auto B = sp_to.get_pos(strip.heh_pos_2d);
        auto a_acc = A.accessor<double, 1>();
        auto b_acc = B.accessor<double, 1>();
        for (int c = 0; c < 2; ++c)
        {
            res.seg_max_abs = std::max(res.seg_max_abs, std::abs(ctx.seg_a(le, c) - a_acc[c]));
            res.seg_max_abs = std::max(res.seg_max_abs, std::abs(ctx.seg_b(le, c) - b_acc[c]));
        }
    }

    //=====================================================================
    // hand backward + comparison
    //=====================================================================
    Eigen::MatrixX2d d_bary = Eigen::MatrixX2d::Zero(_bary.rows(), 2);
    hand_loss_backward(ctx, _leaf, _tmd, _opts, d_bary);

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

    add(res.c_loss, res.hand_loss, res.ref_loss);
    for (Eigen::Index r = 0; r < d_bary.rows(); ++r)
        for (int c = 0; c < 2; ++c)
            add(res.c_bary, d_bary(r, c), g_bary(r, c));

    //=====================================================================
    // packaged compute_gradients_hand must agree with the direct calls
    //=====================================================================
    {
        double wloss = 0.0;
        auto wattr = compute_gradients_hand(_strips, _tmd, _ld, _pnd, _omd, _opts, &wloss);
        res.wrapper_loss_abs = std::abs(wloss - res.hand_loss);
        for (auto pn_vh : _pnd.mesh_->vertices())
        {
            if (_pnd.sp_on_target_.value()[pn_vh].type != SurfacePointType::FacePoint)
                continue;
            vec2d const& w = wattr[pn_vh];
            res.wrapper_grad_max_abs = std::max(res.wrapper_grad_max_abs, std::abs(w.x() - d_bary(pn_vh.idx.value, 0)));
            res.wrapper_grad_max_abs = std::max(res.wrapper_grad_max_abs, std::abs(w.y() - d_bary(pn_vh.idx.value, 1)));
        }
    }

    //=====================================================================
    // central finite differences of the hand forward (torch-independent
    // arbiter): top-|oracle-grad| components + an even stride
    //=====================================================================
    static volatile double h_v = 1e-6;
    static volatile double fd_atol_v = 1e-8;
    static volatile double fd_rtol_v = 1e-5;
    double const h = h_v;
    double const fd_atol = fd_atol_v;
    double const fd_rtol = fd_rtol_v;

    std::vector<char> pick(_comps.size(), 0);
    {
        std::vector<size_t> order(_comps.size());
        std::iota(order.begin(), order.end(), (size_t)0);
        std::sort(order.begin(), order.end(),
                  [&](size_t x, size_t y)
                  {
                      double const gx = std::abs(g_bary(_comps[x].first, _comps[x].second));
                      double const gy = std::abs(g_bary(_comps[y].first, _comps[y].second));
                      return gx > gy;
                  });
        for (size_t i = 0; i < order.size() && i < 10; ++i)
            pick[order[i]] = 1;
        size_t const stride = std::max<size_t>(1, _comps.size() / 50);
        for (size_t i = 0; i < _comps.size(); i += stride)
            pick[i] = 1;
    }

    for (size_t i = 0; i < _comps.size(); ++i)
    {
        if (!pick[i])
            continue;
        int const pn = _comps[i].first;
        int const c = _comps[i].second;

        Eigen::MatrixX2d bp = _bary;
        HandLossCtx fd_ctx;
        bp(pn, c) = _bary(pn, c) + h;
        double const lp = hand_loss_forward(_leaf, bp, _strips, _tmd, _ld, _pnd, _omd, _opts, fd_ctx);
        bp(pn, c) = _bary(pn, c) - h;
        double const lm = hand_loss_forward(_leaf, bp, _strips, _tmd, _ld, _pnd, _omd, _opts, fd_ctx);
        double const fd = (lp - lm) / (2.0 * h);

        double const d = std::abs(d_bary(pn, c) - fd);
        double const gated = d / std::max(fd_atol, fd_rtol * std::abs(fd));
        res.fd_worst_abs = std::max(res.fd_worst_abs, d);
        res.fd_worst_gated = std::max(res.fd_worst_gated, gated);
        ++res.fd_n;
        if (gated > one && res.fd_fails.size() < 8)
            res.fd_fails.push_back({pn, c, d_bary(pn, c), fd});
    }

    res.pass = res.oracle_undefined_grads == 0 && res.c_loss.worst <= one && res.c_bary.worst <= one
               && res.fd_worst_gated <= one;
    return res;
}

void report(std::ostream& _f, ConfigResult const& _r)
{
    char buf[256];
    std::snprintf(buf, sizeof buf, "config %s\n", _r.name.c_str());
    _f << buf;
    std::snprintf(buf, sizeof buf, "  ref_loss    %.17g\n", _r.ref_loss);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  hand_loss   %.17g\n", _r.hand_loss);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  graph rebuild max_abs %.17g (vs pre-rebuild pos_mat_, bitwise expected)\n", _r.rebuild_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  hand pos rewrite max_abs %.17g (vs production pos_mat_, bitwise expected)\n", _r.hand_pos_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  hand seg A/B max_abs %.17g (vs torch get_pos, bitwise expected)\n", _r.seg_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  oracle undefined FacePoint grads %ld (0 expected)\n", _r.oracle_undefined_grads);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  loss        max_abs %.17g  worst %.17g  (n=%zu)\n", _r.c_loss.max_abs, _r.c_loss.worst, _r.c_loss.n);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  d_bary      max_abs %.17g  worst %.17g  (n=%zu)\n", _r.c_bary.max_abs, _r.c_bary.worst, _r.c_bary.n);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  wrapper     grad max_abs %.17g  loss abs %.17g (bitwise expected)\n", _r.wrapper_grad_max_abs, _r.wrapper_loss_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  fd          worst_abs %.17g  worst_gated %.17g  (n=%zu, h=1e-6, gate max(1e-8, 1e-5*|fd|))\n",
                  _r.fd_worst_abs, _r.fd_worst_gated, _r.fd_n);
    _f << buf;
    for (auto const& f : _r.fd_fails)
    {
        std::snprintf(buf, sizeof buf, "    FD FAIL pn=%d c=%d hand=%.17g fd=%.17g\n", f.pn, f.comp, f.hand, f.fd);
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

    // same embedding state as optimize.cc reaches at its first eval; the final
    // compute_layout_embedding_init(false) is replicated INLINE from its public
    // constituents because the TriangleStrips are local to the production init
    // (Embedding.cc, anonymous namespace) and S1/S3 need them. They cannot be
    // rebuilt afterwards (compute_2D_embeddig_per_strip mutates the path
    // network), so this MUST stay the literal production call sequence — the
    // bitwise-known reference loss (config A = the Phase 3 baseline
    // 1.2734862918788761) doubles as the replication self-check.
    compute_layout_embedding_init(tmd, ld, pnd, omd, true);
    resample_layout(tmd, ld, pnd, omd, 0.2);

    init_path_network_surface_points(tmd, ld, pnd, false);
    compute_path_network_exact(tmd, ld, pnd);
    std::vector<TriangleStrip> strips;
    compute_2D_embeddig_per_strip(tmd, ld, pnd, strips);
    insert_path_network(tmd, pnd, omd);
    compute_path_boundary_mask(pnd, omd);
    compute_mapping_layout_to_overlay(ld, pnd);
    compute_mapping_overlay_to_layout(omd, ld);
    compute_differentiable_intersections_for_overlay_stable(strips, tmd, ld, pnd, omd);
    sync_tg_and_torch(omd);

    // leaf structure + bary values are config-independent
    LeafCtx leaf;
    leaf_collect(strips, tmd, ld, pnd, leaf);
    Eigen::MatrixX2d const bary = collect_bary(pnd);

    size_t n_face = 0, n_edge = 0, n_vertex = 0;
    std::vector<std::pair<int, int>> comps; // FD candidates: FacePoint leaf components
    for (auto const& n : leaf.nodes)
    {
        if (n.type == SurfacePointType::FacePoint)
        {
            ++n_face;
            comps.push_back({n.pn_v, 0});
            comps.push_back({n.pn_v, 1});
        }
        else if (n.type == SurfacePointType::EdgePoint)
            ++n_edge;
        else
            ++n_vertex;
    }

    // config A: the FULL production configuration (optimize.cc)
    OptimizationOptions opt_prod;
    opt_prod.w_harmonic_distorion_loss = 1.0;
    opt_prod.w_curvature_alignment_loss = 0.1;
    opt_prod.harmonic_options.w_AIAP_SingValDecomp = 0.5;
    opt_prod.harmonic_options.w_AreaPreserving_SingValDecomp = 0.5;

    // config B: every distortion branch active + a different curvature weight
    OptimizationOptions opt_all;
    opt_all.w_harmonic_distorion_loss = 1.0;
    opt_all.w_curvature_alignment_loss = 0.7;
    opt_all.harmonic_options.w_SDE_DirectComputation = 0.3;
    opt_all.harmonic_options.w_SDE_SingValDecomp = 0.2;
    opt_all.harmonic_options.w_DE_SingValDecomp = 0.15;
    opt_all.harmonic_options.w_AIAP_SingValDecomp = 0.25;
    opt_all.harmonic_options.w_I_DevFrom1_SingValDecomp = 0.1;
    opt_all.harmonic_options.w_AreaPreserving_SingValDecomp = 0.2;

    std::vector<ConfigResult> results;
    results.push_back(run_config("production(w_h=1,AIAP+AP,w_curv=0.1)", opt_prod, strips, tmd, ld, pnd, omd, leaf, bary, comps));
    results.push_back(run_config("all-branches(w_curv=0.7)", opt_all, strips, tmd, ld, pnd, omd, leaf, bary, comps));

    //=== dump raw values (torch-free file write -> trustworthy on Windows) ===
    {
        std::ofstream f("gradcheck_phase4_s1.txt");
        char buf[128];
        std::snprintf(buf, sizeof buf, "leaves: %zu FacePoint, %zu EdgePoint, %zu VertexPoint (of %zu mapped)\n",
                      n_face, n_edge, n_vertex, leaf.nodes.size());
        f << buf;
        for (auto const& r : results)
            report(f, r);
    }

    std::printf("leaves: %zu FacePoint, %zu EdgePoint, %zu VertexPoint (of %zu mapped)\n",
                n_face, n_edge, n_vertex, leaf.nodes.size());
    bool ok = true;
    for (auto const& r : results)
    {
        report(std::cout, r);
        ok = ok && r.pass;
    }

    std::printf("\n%s\n", ok ? "PHASE 4 / S1 GRADCHECK PASSED" : "PHASE 4 / S1 GRADCHECK FAILED");
    return ok ? 0 : 1;
}
