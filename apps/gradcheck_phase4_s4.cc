// Phase 4 / S4 validation (remove-autodiff plan): the hand-rolled boundary-UV
// stage (Adjoint/BoundaryUvStage.{hh,cc}) must reproduce the torch path's
// forward values and stage-input adjoints.
//
// Stage boundary (plan section 4/S4): overlay 3D positions -> per-layout-edge
// arc lengths -> per-pn-halfedge t -> per-pn-halfedge boundary UVs. Validated
// COMPOSED with the already-validated S5 (cotans), S6 (harmonic solve) and S7
// (distortion) stages. The ONLY leaf is the overlay position tensor: the
// boundary UVs (leaves in the S5/S6 gradchecks) are now DERIVED from it, so
// d_overlay_pos is the total gradient of the loss w.r.t. overlay positions
// (S4 uv path + S5 cotan path + S7 direct-3D path combined). The non-leaf
// edge-length / t / uv tensors are retain_grad()'d so the intermediate
// adjoints stay directly comparable.
//
// Oracle: the production prepare functions themselves, run with the leaf
// swapped into _omd.torch_pos_ (torch_compute_embedded_layout_edge_lengths ->
// torch_compute_t_for_pn_halfedge -> torch_compute_pn_uvs), then the
// production patch loop (torch_prepare_param -> torch_harmonic_param ->
// torch_face_distortion) with the gathered boundary UVs kept ATTACHED to the
// graph. The oracle loss must equal the production harmonic_distortion_loss
// BITWISE (the leaf is a bitwise clone of the production positions).
//
// Hand: edge_lengths_forward -> pn_t_forward -> pn_uvs_forward (S4) +
// cotans_forward (S5) -> prepare_param/harmonic_param_forward (S6) ->
// patch_distortion_forward (S7) -> loss. Backward runs the stages in reverse;
// d_lengths receives contributions from both pn_uvs_backward and
// pn_t_backward, so edge_lengths_backward runs last.
//
// Tolerances: 1e-9 abs / 1e-7 rel (plan section 5). In-process FP constants
// are not trustworthy on Windows (torch_cpu.dll clobbers callee-saved
// xmm14/15), so tolerances are loaded from volatiles right before use and all
// raw maxima are dumped to gradcheck_phase4_s4.txt for an external check.

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

#include <LayoutOpt/Adjoint/BoundaryUvStage.hh>
#include <LayoutOpt/Adjoint/CotanStage.hh>
#include <LayoutOpt/Adjoint/DistortionStage.hh>
#include <LayoutOpt/Adjoint/HarmonicStage.hh>
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
    double ref_loss = 0.0;
    double oracle_loss = 0.0;
    double hand_loss = 0.0;
    bool oracle_bitwise = false;
    long prepare_mismatches = 0;      // STRUCTURAL prepare_param vs torch_prepare_param (faces, classification)
    double bnd_gather_max_abs = 0.0;  // hand boundary-UV gather vs torch (~ulp: the hand uvs are Eigen-computed)
    double lengths_fwd_max_abs = 0.0; // hand edge_lengths_forward vs torch
    double t_fwd_max_abs = 0.0;       // hand pn_t_forward vs torch
    double uvs_fwd_max_abs = 0.0;     // hand pn_uvs_forward vs torch
    double cotans_fwd_max_abs = 0.0;  // hand cotans_forward vs torch_cotans
    double inner_solve_max_abs = 0.0; // hand solve vs oracle sparse solve
    Cmp c_loss, c_uvs, c_t, c_lengths, c_cotans, c_pos;
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
    // production reference
    //=====================================================================
    EvalInfo info;
    torch::Tensor ref_loss_t = harmonic_distortion_loss(_tmd, _ld, _pnd, _omd, _hopt, info);
    res.ref_loss = ref_loss_t.item<double>();

    //=====================================================================
    // stage input: the overlay position leaf. The production S4 functions
    // are re-run with the leaf swapped into _omd.torch_pos_ so the oracle's
    // lengths/t/uvs derive from it; the stacked tensors are retain_grad()'d
    // for the intermediate adjoint comparisons.
    //=====================================================================
    torch::Tensor orig_pos = _omd.torch_pos_.value();
    torch::Tensor pos3_leaf = orig_pos.detach().clone().requires_grad_(true);
    Eigen::MatrixX3d const& pos_e = _omd.pos_mat_.value();

    _omd.torch_pos_.emplace(pos3_leaf);
    torch_compute_embedded_layout_edge_lengths(_ld, _pnd, _omd);
    torch_compute_t_for_pn_halfedge(_ld, _pnd, _omd);
    torch_compute_pn_uvs(_ld, _pnd, _hopt);
    _omd.torch_pos_.emplace(orig_pos); // restore; the graph holds its own refs

    torch::Tensor lengths_t = _ld.torch_embedded_edge_length_.value();
    torch::Tensor t_t = _pnd.torch_t_.value();
    torch::Tensor uvs_t = _pnd.torch_uvs_.value();
    lengths_t.retain_grad();
    t_t.retain_grad();
    uvs_t.retain_grad();

    auto cotans = torch_cotans(pos3_leaf, *_omd.mesh_.get());
    cotans.retain_grad();
    int64_t const n_edges = cotans.size(0);
    int64_t const n_layout_edges = lengths_t.size(0);
    int64_t const n_pn_heh = t_t.size(0);

    //=====================================================================
    // hand forward of the S4 + S5 stages
    //=====================================================================
    EdgeLengthCtx el_ctx;
    Eigen::VectorXd lengths_e;
    edge_lengths_forward(pos_e, _ld, _pnd, el_ctx, lengths_e);

    TCtx t_ctx;
    Eigen::VectorXd t_e;
    pn_t_forward(pos_e, lengths_e, _ld, _pnd, t_ctx, t_e);

    UvCtx uv_ctx;
    Eigen::MatrixX2d uvs_e;
    pn_uvs_forward(lengths_e, t_e, _ld, _pnd, _hopt.fixed_parameter_domain, uv_ctx, uvs_e);

    CotanCtx cot_ctx;
    Eigen::VectorXd cotans_e;
    cotans_forward(pos_e, *_omd.mesh_.get(), cot_ctx, cotans_e);

    // forward closeness, hand vs torch
    {
        auto acc = lengths_t.accessor<double, 1>();
        for (int64_t i = 0; i < n_layout_edges; ++i)
            res.lengths_fwd_max_abs = std::max(res.lengths_fwd_max_abs, std::abs(lengths_e(i) - acc[i]));
    }
    {
        auto acc = t_t.accessor<double, 1>();
        for (int64_t i = 0; i < n_pn_heh; ++i)
            res.t_fwd_max_abs = std::max(res.t_fwd_max_abs, std::abs(t_e(i) - acc[i]));
    }
    {
        auto acc = uvs_t.accessor<double, 2>();
        for (int64_t i = 0; i < n_pn_heh; ++i)
            for (int c = 0; c < 2; ++c)
                res.uvs_fwd_max_abs = std::max(res.uvs_fwd_max_abs, std::abs(uvs_e(i, c) - acc[i][c]));
    }
    {
        auto acc = cotans.accessor<double, 1>();
        for (int64_t i = 0; i < n_edges; ++i)
            res.cotans_fwd_max_abs = std::max(res.cotans_fwd_max_abs, std::abs(cotans_e(i) - acc[i]));
    }

    //=====================================================================
    // per-overlay-heh uv arrays. The torch gather stays ATTACHED to the leaf
    // graph (unlike S5/S6, where boundary UVs were detached into leaves).
    // pn_of_oheh inverts the one-to-one gather for the backward scatter.
    //=====================================================================
    auto heh_count = _omd.mesh_->halfedges().size();
    auto uvs = torch::zeros({(int64_t)heh_count, 2}, torch::dtype(torch::kFloat64));
    Eigen::MatrixX2d o_uvs_e = Eigen::MatrixX2d::Zero((Eigen::Index)heh_count, 2);
    std::vector<int> pn_of_oheh(heh_count, -1);
    for (auto pn_heh : _pnd.mesh_->halfedges())
    {
        auto o_vh_from = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_from()]);
        auto o_vh_to = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_to()]);
        auto o_heh = pm::halfedge_from_to(o_vh_from, o_vh_to);
        uvs[o_heh.idx.value] = uvs_t[pn_heh.idx.value];
        o_uvs_e.row(o_heh.idx.value) = uvs_e.row(pn_heh.idx.value);
        pn_of_oheh[(size_t)o_heh.idx.value] = pn_heh.idx.value;
    }

    int const n_patches = (int)_ld.mesh_->faces().size();
    std::vector<torch::Tensor> SDE_per_patch;
    SDE_per_patch.reserve(n_patches);
    std::vector<PatchHarmonicCtx> hand_hctx;
    hand_hctx.reserve(n_patches);
    std::vector<PatchDistortionCtx> hand_sctx;
    hand_sctx.reserve(n_patches);
    std::vector<std::vector<vec2d>> hand_bnd(n_patches);
    std::vector<std::vector<int>> bnd_src(n_patches);

    for (int patch_i = 0; patch_i < n_patches; ++patch_i)
    {
        //--- torch oracle side -------------------------------------------
        std::vector<VH> inner_vhs_t, bnd_vhs_t;
        std::vector<torch::Tensor> bnd_uvs_t;
        std::vector<FH> fhs_t;
        pm::vertex_attribute<MappingIndex> map_t = _omd.mesh_->vertices().make_attribute<MappingIndex>({-1, true});
        torch_prepare_param(patch_i, _pnd, _omd, uvs, map_t, inner_vhs_t, bnd_vhs_t, bnd_uvs_t, fhs_t);

        auto inner_t = torch_harmonic_param(map_t, _omd.patch_boundary_mask_.value(), inner_vhs_t, bnd_uvs_t, cotans, /*sparse*/ true);

        std::vector<torch::Tensor> SDE_list;
        SDE_list.reserve(fhs_t.size());
        for (auto o_fh : fhs_t)
        {
            auto hehA = o_fh.any_halfedge();
            auto hehB = hehA.next();
            auto hehC = hehB.next();

            auto i = map_t[hehA.vertex_from()];
            auto j = map_t[hehB.vertex_from()];
            auto k = map_t[hehC.vertex_from()];

            torch::Tensor posA_2D = !i.on_boundary ? inner_t[i.idx] : bnd_uvs_t[i.idx];
            torch::Tensor posB_2D = !j.on_boundary ? inner_t[j.idx] : bnd_uvs_t[j.idx];
            torch::Tensor posC_2D = !k.on_boundary ? inner_t[k.idx] : bnd_uvs_t[k.idx];
            torch::Tensor posA_3D = pos3_leaf[hehA.vertex_from().idx.value];
            torch::Tensor posB_3D = pos3_leaf[hehB.vertex_from().idx.value];
            torch::Tensor posC_3D = pos3_leaf[hehC.vertex_from().idx.value];

            auto SDE_info = torch_face_distortion(posA_2D, posB_2D, posC_2D, posA_3D, posB_3D, posC_3D, _hopt);
            SDE_list.push_back(SDE_info.distortion_val);
        }
        SDE_per_patch.push_back(torch::stack(SDE_list).sum());

        //--- hand side ----------------------------------------------------
        std::vector<VH> inner_vhs_h, bnd_vhs_h;
        std::vector<vec2d>& bnd_uvs_h = hand_bnd[patch_i];
        std::vector<FH> fhs_h;
        pm::vertex_attribute<MappingIndex> map_h = _omd.mesh_->vertices().make_attribute<MappingIndex>({-1, true});
        prepare_param(patch_i, _pnd, _omd, o_uvs_e, map_h, inner_vhs_h, bnd_vhs_h, bnd_uvs_h, fhs_h, &bnd_src[patch_i]);

        // port equality: same faces, same classification. UV values are NOT
        // memcmp'd (unlike S5/S6, where the input uvs were production-bitwise
        // by construction): the hand uvs come from the hand S4 chain, whose
        // Eigen segment norms differ from torch's norm kernel by <= 1 ulp on
        // some segments. The value diff is reported as bnd_gather_max_abs and
        // covered by the tolerance-gated adjoint comparisons.
        if (fhs_h.size() != fhs_t.size() || inner_vhs_h.size() != inner_vhs_t.size() || bnd_vhs_h.size() != bnd_vhs_t.size())
            ++res.prepare_mismatches;
        else
        {
            for (size_t k = 0; k < fhs_t.size(); ++k)
                if (fhs_h[k] != fhs_t[k])
                    ++res.prepare_mismatches;
            for (size_t k = 0; k < inner_vhs_t.size(); ++k)
                if (inner_vhs_h[k] != inner_vhs_t[k])
                    ++res.prepare_mismatches;
            for (size_t k = 0; k < bnd_vhs_t.size(); ++k)
                if (bnd_vhs_h[k] != bnd_vhs_t[k])
                    ++res.prepare_mismatches;
            for (size_t k = 0; k < bnd_uvs_t.size(); ++k)
            {
                double const u = bnd_uvs_t[k][0].item<double>();
                double const v = bnd_uvs_t[k][1].item<double>();
                res.bnd_gather_max_abs = std::max(res.bnd_gather_max_abs, std::abs(bnd_uvs_h[k].x() - u));
                res.bnd_gather_max_abs = std::max(res.bnd_gather_max_abs, std::abs(bnd_uvs_h[k].y() - v));
            }
        }

        PatchHarmonicCtx hctx = harmonic_param_forward(map_h, inner_vhs_h, bnd_uvs_h, cotans_e);

        {
            auto acc = inner_t.accessor<double, 2>();
            for (int64_t r = 0; r < inner_t.size(0); ++r)
                for (int c = 0; c < 2; ++c)
                    res.inner_solve_max_abs = std::max(res.inner_solve_max_abs, std::abs(hctx.inner_uvs(r, c) - acc[r][c]));
        }

        hand_sctx.push_back(patch_distortion_forward(fhs_h, map_h, hctx.inner_uvs, bnd_uvs_h, pos_e, _hopt));
        hand_hctx.push_back(std::move(hctx));
    }

    //=====================================================================
    // oracle loss (must equal production BITWISE) + backward
    //=====================================================================
    torch::Tensor oracle_loss_t = torch::stack(SDE_per_patch).sum();
    res.oracle_loss = oracle_loss_t.item<double>();
    res.oracle_bitwise = std::memcmp(&res.oracle_loss, &res.ref_loss, sizeof(double)) == 0;
    oracle_loss_t.backward();

    //=====================================================================
    // extract the oracle grads into plain buffers (torch traffic ends here)
    //=====================================================================
    Eigen::VectorXd g_lengths = Eigen::VectorXd::Zero(n_layout_edges);
    {
        torch::Tensor g = lengths_t.grad();
        if (g.defined())
        {
            auto acc = g.accessor<double, 1>();
            for (int64_t i = 0; i < n_layout_edges; ++i)
                g_lengths(i) = acc[i];
        }
    }
    Eigen::VectorXd g_t = Eigen::VectorXd::Zero(n_pn_heh);
    {
        torch::Tensor g = t_t.grad();
        if (g.defined())
        {
            auto acc = g.accessor<double, 1>();
            for (int64_t i = 0; i < n_pn_heh; ++i)
                g_t(i) = acc[i];
        }
    }
    Eigen::MatrixX2d g_uvs = Eigen::MatrixX2d::Zero(n_pn_heh, 2);
    {
        torch::Tensor g = uvs_t.grad();
        if (g.defined())
        {
            auto acc = g.accessor<double, 2>();
            for (int64_t i = 0; i < n_pn_heh; ++i)
                for (int c = 0; c < 2; ++c)
                    g_uvs(i, c) = acc[i][c];
        }
    }
    Eigen::VectorXd g_cotans = Eigen::VectorXd::Zero(n_edges);
    {
        torch::Tensor g = cotans.grad();
        if (g.defined())
        {
            auto acc = g.accessor<double, 1>();
            for (int64_t i = 0; i < n_edges; ++i)
                g_cotans(i) = acc[i];
        }
    }
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
    // hand loss + backward (pure double/Eigen; no torch from here on)
    //=====================================================================
    res.hand_loss = distortion_loss_forward(hand_sctx, false);

    std::vector<double> d_sde, d_pa;
    distortion_loss_backward(hand_sctx, false, 1.0, d_sde, d_pa);

    Eigen::VectorXd d_cotans = Eigen::VectorXd::Zero(n_edges);
    Eigen::MatrixX3d d_pos = Eigen::MatrixX3d::Zero(pos_e.rows(), 3);
    Eigen::MatrixX2d d_uvs_pn = Eigen::MatrixX2d::Zero(n_pn_heh, 2);
    Eigen::VectorXd d_t = Eigen::VectorXd::Zero(n_pn_heh);
    Eigen::VectorXd d_lengths = Eigen::VectorXd::Zero(n_layout_edges);

    for (int p = 0; p < n_patches; ++p)
    {
        Eigen::MatrixX2d d_inner = Eigen::MatrixX2d::Zero(hand_hctx[p].n_inner, 2);
        std::vector<vec2d> d_bnd(hand_bnd[p].size(), vec2d::Zero());
        // S7 backward: seeds d_inner and the direct boundary/overlay paths
        patch_distortion_backward(hand_sctx[p], d_sde[p], d_pa[p], _hopt, d_inner, d_bnd, d_pos);
        // S6 backward: routes d_inner through the solve into cotans + boundary UVs
        harmonic_param_backward(hand_hctx[p], hand_bnd[p], cotans_e, d_inner, d_cotans, d_bnd);
        // scatter the patch boundary-UV adjoints back to pn-halfedge uv rows
        for (size_t b = 0; b < d_bnd.size(); ++b)
        {
            int const pn = pn_of_oheh[(size_t)bnd_src[p][b]];
            if (pn < 0)
            {
                ++res.prepare_mismatches;
                continue;
            }
            d_uvs_pn(pn, 0) += d_bnd[b].x();
            d_uvs_pn(pn, 1) += d_bnd[b].y();
        }
    }
    // S4 backward: uvs -> (t, lengths) -> positions. d_lengths receives
    // contributions from both pn_uvs_backward and pn_t_backward, so
    // edge_lengths_backward must run last.
    pn_uvs_backward(uv_ctx, t_e, d_uvs_pn, d_lengths, d_t);
    pn_t_backward(t_ctx, pos_e, d_t, d_pos, d_lengths);
    edge_lengths_backward(el_ctx, pos_e, d_lengths, d_pos);
    // S5 backward: fold the accumulated d_cotans into d_overlay_pos
    cotans_backward(cot_ctx, pos_e, d_cotans, d_pos);

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
    for (int64_t i = 0; i < n_pn_heh; ++i)
        for (int c = 0; c < 2; ++c)
            add(res.c_uvs, d_uvs_pn(i, c), g_uvs(i, c));
    for (int64_t i = 0; i < n_pn_heh; ++i)
        add(res.c_t, d_t(i), g_t(i));
    for (int64_t i = 0; i < n_layout_edges; ++i)
        add(res.c_lengths, d_lengths(i), g_lengths(i));
    for (int64_t i = 0; i < n_edges; ++i)
        add(res.c_cotans, d_cotans(i), g_cotans(i));
    for (Eigen::Index r = 0; r < g_pos.rows(); ++r)
        for (int c = 0; c < 3; ++c)
            add(res.c_pos, d_pos(r, c), g_pos(r, c));

    res.pass = res.oracle_bitwise && res.prepare_mismatches == 0
               && res.c_loss.worst <= one && res.c_uvs.worst <= one && res.c_t.worst <= one
               && res.c_lengths.worst <= one && res.c_cotans.worst <= one && res.c_pos.worst <= one;
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
    std::snprintf(buf, sizeof buf, "  prepare_param structural mismatches %ld\n", _r.prepare_mismatches);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  boundary-uv gather max_abs %.17g (vs torch, ~ulp expected)\n", _r.bnd_gather_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  edge_lengths_forward max_abs %.17g (vs torch)\n", _r.lengths_fwd_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  pn_t_forward max_abs %.17g (vs torch)\n", _r.t_fwd_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  pn_uvs_forward max_abs %.17g (vs torch)\n", _r.uvs_fwd_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  cotans_forward max_abs %.17g (vs torch_cotans)\n", _r.cotans_fwd_max_abs);
    _f << buf;
    std::snprintf(buf, sizeof buf, "  inner solve max_abs %.17g (vs oracle sparse solve)\n", _r.inner_solve_max_abs);
    _f << buf;
    Cmp const* cmps[6] = {&_r.c_loss, &_r.c_uvs, &_r.c_t, &_r.c_lengths, &_r.c_cotans, &_r.c_pos};
    char const* names[6] = {"loss", "d_uvs", "d_t", "d_lengths", "d_cotans", "d_overlay_pos"};
    for (int i = 0; i < 6; ++i)
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

    // config A: the production weights
    HarmonicOptions prod;
    prod.w_AIAP_SingValDecomp = 0.5;
    prod.w_AreaPreserving_SingValDecomp = 0.5;

    // config B: every distortion branch active (different d_inner seeds)
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
        std::ofstream f("gradcheck_phase4_s4.txt");
        for (auto const& r : results)
            report(f, r);
    }

    bool ok = true;
    for (auto const& r : results)
    {
        report(std::cout, r);
        ok = ok && r.pass;
    }

    std::printf("\n%s\n", ok ? "PHASE 4 / S4 GRADCHECK PASSED" : "PHASE 4 / S4 GRADCHECK FAILED");
    return ok ? 0 : 1;
}
