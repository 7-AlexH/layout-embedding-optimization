// Permanent end-to-end determinism check (remove-autodiff Phase 6: replaces
// the torch-vs-hand e2e_phase4 harness — there is only one gradient path now,
// so the e2e property worth gating is exact reproducibility).
//
// Two COMPLETE production runs over identical fresh data (banana pair, the
// FULL production configuration from optimize.cc: w_h=1, AIAP/AreaPreserving
// 0.5/0.5, w_c=0.1, vector Adam, production step-size schedule), both driven
// by the production eval()/apply() machinery exactly as optimize.cc drives
// them. The loop differs from optimize.cc only in: no screenshots/viewer/
// write_state (no numeric effect) and a compressed resample schedule
// (i % 6 == 0 -> i = 0, 6 instead of i = 0, 20) so a mid-trajectory resample
// is crossed affordably.
//
// Gates (all BITWISE — the hand adjoint pipeline is sequential plain-double
// arithmetic with no reassociation sources, so two runs over the same input
// must agree exactly):
//   - iter-0 loss vs the recorded Phase-3 reference (rel 1e-12; ulp-level
//     drift allowed since Phase 5b moved detached reads onto Eigen)
//   - per-iteration loss A == B exactly
//   - discrete state fingerprints (vertex-sp conversion counts, pn vertex
//     count, sp type counts, anchor-heh sum) EQ at every iteration
//   - post-apply bary_full() arrays A == B exactly at every iteration
//   - final layout-node positions A == B exactly
//
// A failure here means nondeterminism crept into the pipeline (unordered
// iteration, uninitialized reads, parallel reduction) — exactly the property
// Phase 6d's OpenMP work must preserve.
//
// All raw values go to e2e_determinism.txt at %.17g.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/TriangleStrip.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/Embedding.hh>
#include <LayoutOpt/EmbeddingUtils.hh>
#include <LayoutOpt/FlattenTriangleStrip.hh>
#include <LayoutOpt/Init.hh>
#include <LayoutOpt/ObjectiveFunctions.hh>
#include <LayoutOpt/Optimization.hh>
#include <LayoutOpt/OptimizationOptions.hh>
#include <LayoutOpt/Optimizers.hh>
#include <LayoutOpt/Resample.hh>
#include <LayoutOpt/ScalarFields.hh>
#include <LayoutOpt/Update.hh>

using namespace LayoutOpt;
namespace fs = std::filesystem;

namespace
{

int const N_ITER = 12;
int const RESAMPLE_EVERY = 6; // resamples at i = 0 (init state) and i = 6 (mid-trajectory)

struct IterRec
{
    bool resampled = false;
    double loss = 0.0;
    long conv_count = 0; // vertex-sp conversions during this iteration (resample/init + update)
    // post-apply state fingerprint (the state the NEXT iteration evaluates)
    size_t fp_verts = 0;      // pn vertex count
    long fp_n_vert = 0;       // # VertexPoint sps
    long fp_n_edge = 0;       // # EdgePoint sps
    long fp_n_face = 0;       // # FacePoint sps
    long long fp_heh_sum = 0; // sum of sp anchor halfedge indices
    std::vector<double> fp_bary; // post-apply bary_full() per pn vertex (3 each)
};

struct RunResult
{
    std::vector<IterRec> iters;
    Eigen::MatrixX3d final_node_pos; // per layout-vertex 3D position (by layout idx)
    size_t final_pn_verts = 0;
};

RunResult run_loop(char const* _tag, pm::vertex_attribute<pos3> const& _pos, pm::vertex_attribute<pos3> const& _l_pos)
{
    RunResult res;

    TargetMeshData tmd(_pos);
    tmd.direction_field_data_.emplace(smooth_direction_field(tmd.pos_));
    LayoutData ld(_l_pos);
    PathNetworkData pnd(ld.pos_);
    OverlayMeshData omd(tmd.pos_);
    compute_layout_embedding_init(tmd, ld, pnd, omd, true);

    // production configuration (optimize.cc)
    OptimizationOptions opts;
    opts.w_curvature_alignment_loss = 0.1;
    opts.w_harmonic_distorion_loss = 1.0;
    opts.harmonic_options.w_AIAP_SingValDecomp = 0.5;
    opts.harmonic_options.w_AreaPreserving_SingValDecomp = 0.5;
    opts.harmonic_options.w_I_DevFrom1_SingValDecomp = .0;
    opts.harmonic_options.w_DE_SingValDecomp = .0;
    opts.harmonic_options.w_SDE_SingValDecomp = .0;
    opts.optimizer = Optimizer::Adam;

    // production step constants (optimize.cc)
    double const STEP_SIZE_MAX = 0.0015;
    double const STEP_SIZE_MIN = 0.0005;
    double const STEP_GROWTH = 1.15;
    double const RESAMPLE_SPACING = 0.2;

    OptimizerData od;
    od.init_vetor_adam_param(*pnd.mesh_.get(), STEP_SIZE_MIN);

    std::vector<TriangleStrip> strips; // refreshed by every init/update below

    for (int i = 0; i < N_ITER; ++i)
    {
        IterRec rec;
        long const conv_before = g_vertex_sp_conversion_count;

        if (i % RESAMPLE_EVERY == 0)
        {
            rec.resampled = true;
            od.init_vetor_adam_param(*pnd.mesh_.get(), STEP_SIZE_MIN);
            resample_layout(tmd, ld, pnd, omd, RESAMPLE_SPACING);
            compute_layout_embedding_init(tmd, ld, pnd, omd, false, &strips);
        }

        auto eval_info = eval(tmd, ld, pnd, omd, strips, opts, i, od);
        rec.loss = eval_info.loss;
        res.iters.push_back(rec);

        // apply() (Optimization.cc) inlined so the rebuilt strips are captured
        collapse_pn_to_layout(pnd);
        do_step(tmd, pnd, eval_info.pn_sp_update_dirs, &od);
        compute_layout_embedding_update(tmd, ld, pnd, omd, &strips);

        od.step_size = tg::min(od.step_size * STEP_GROWTH, double(STEP_SIZE_MAX));

        res.iters.back().conv_count = g_vertex_sp_conversion_count - conv_before;

        // post-apply fingerprint of the path network's discrete state
        {
            IterRec& r = res.iters.back();
            r.fp_verts = pnd.mesh_->vertices().size();
            r.fp_bary.reserve(r.fp_verts * 3);
            for (auto pn_vh : pnd.mesh_->vertices())
            {
                auto const& sp = pnd.sp_on_target_.value()[pn_vh];
                r.fp_heh_sum += (long long)sp.heh_idx.value;
                if (sp.type == SurfacePointType::VertexPoint)
                    ++r.fp_n_vert;
                else if (sp.type == SurfacePointType::EdgePoint)
                    ++r.fp_n_edge;
                else
                    ++r.fp_n_face;
                vec3d const b = sp.bary_full();
                r.fp_bary.push_back(b.x());
                r.fp_bary.push_back(b.y());
                r.fp_bary.push_back(b.z());
            }
        }

        std::printf("  [%s] iter %2d%s loss %.17g  conv %ld  fp v%zu V%ld E%ld F%ld heh%lld\n", _tag, i,
                    rec.resampled ? " (resample)" : "           ", rec.loss, res.iters.back().conv_count, res.iters.back().fp_verts,
                    res.iters.back().fp_n_vert, res.iters.back().fp_n_edge, res.iters.back().fp_n_face, res.iters.back().fp_heh_sum);
        std::fflush(stdout);
    }

    // final embedding: 3D positions of the layout nodes
    res.final_node_pos = Eigen::MatrixX3d::Zero((Eigen::Index)ld.mesh_->vertices().size(), 3);
    for (auto pn_vh : pnd.mesh_->vertices())
    {
        auto l_idx = pnd.map_to_layout_vertices_[pn_vh];
        if (l_idx.is_invalid())
            continue;
        res.final_node_pos.row(l_idx.value) = pnd.sp_on_target_.value()[pn_vh].get_pos(tmd.pos_mat_, *tmd.mesh_.get()).transpose();
    }
    res.final_pn_verts = pnd.mesh_->vertices().size();

    return res;
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

    std::printf("run A\n");
    auto A = run_loop("A", pos, l_pos);
    std::printf("run B (identical configuration — must be bitwise equal)\n");
    auto B = run_loop("B", pos, l_pos);

    double const ref0 = 1.2734862918788761; // Phase 3 baseline at the i=0 state
    double const ref0_rtol = 1e-12;         // Phase 5b: detached reads on Eigen -> ulp-level shifts allowed

    double const ref0_abs = std::abs(A.iters[0].loss - ref0);

    // A vs B: everything must match exactly
    bool loss_all_eq = true;
    std::vector<bool> fp_eq(A.iters.size());
    std::vector<double> bary_d(A.iters.size(), -1.0);
    bool fp_all_eq = true;
    double bary_d_max = 0.0;
    for (size_t i = 0; i < A.iters.size(); ++i)
    {
        auto const& a = A.iters[i];
        auto const& b = B.iters[i];
        loss_all_eq = loss_all_eq && a.loss == b.loss;
        fp_eq[i] = a.conv_count == b.conv_count && a.fp_verts == b.fp_verts && a.fp_n_vert == b.fp_n_vert && a.fp_n_edge == b.fp_n_edge
                   && a.fp_n_face == b.fp_n_face && a.fp_heh_sum == b.fp_heh_sum;
        fp_all_eq = fp_all_eq && fp_eq[i];
        if (fp_eq[i] && a.fp_bary.size() == b.fp_bary.size())
        {
            bary_d[i] = 0.0;
            for (size_t k = 0; k < a.fp_bary.size(); ++k)
                bary_d[i] = std::max(bary_d[i], std::abs(a.fp_bary[k] - b.fp_bary[k]));
            bary_d_max = std::max(bary_d_max, bary_d[i]);
        }
        else
        {
            bary_d_max = std::numeric_limits<double>::infinity();
        }
    }
    double const final_pos_max_abs = A.final_node_pos.rows() == B.final_node_pos.rows()
                                         ? (A.final_node_pos - B.final_node_pos).cwiseAbs().maxCoeff()
                                         : std::numeric_limits<double>::infinity();

    bool const pass = ref0_abs <= ref0_rtol * std::abs(ref0) && loss_all_eq && fp_all_eq && bary_d_max == 0.0 && final_pos_max_abs == 0.0;

    auto report = [&](std::ostream& f)
    {
        char buf[512];
        std::snprintf(buf, sizeof buf, "iter-0 loss %.17g  |ref %.17g|  abs %.17g (gate rel %.3g)\n", A.iters[0].loss, ref0, ref0_abs, ref0_rtol);
        f << buf;
        f << "iter  rs  loss_A                 loss_B                 eq  fp  bary_d  conv A/B\n";
        for (size_t i = 0; i < A.iters.size(); ++i)
        {
            auto const& a = A.iters[i];
            auto const& b = B.iters[i];
            std::snprintf(buf, sizeof buf, "%4zu  %s  %.17g  %.17g  %s  %s  %.6g  %ld/%ld (A v%zu V%ld E%ld F%ld heh%lld)\n", i,
                          a.resampled ? "* " : "  ", a.loss, b.loss, a.loss == b.loss ? "EQ  " : "DIFF", fp_eq[i] ? "EQ  " : "DIFF", bary_d[i],
                          a.conv_count, b.conv_count, a.fp_verts, a.fp_n_vert, a.fp_n_edge, a.fp_n_face, a.fp_heh_sum);
            f << buf;
        }
        std::snprintf(buf, sizeof buf, "per-iteration loss: %s (gate: bitwise EQ)\n", loss_all_eq ? "EQ" : "DIFF");
        f << buf;
        std::snprintf(buf, sizeof buf, "discrete fingerprints: %s (gate: EQ at every iteration)\n", fp_all_eq ? "EQ" : "DIFF");
        f << buf;
        std::snprintf(buf, sizeof buf, "post-apply bary max |A-B| %.17g (gate: 0)\n", bary_d_max);
        f << buf;
        std::snprintf(buf, sizeof buf, "final layout-node pos max |A-B| %.17g (gate: 0)\n", final_pos_max_abs);
        f << buf;
        std::snprintf(buf, sizeof buf, "final pn vertex count A %zu  B %zu\n", A.final_pn_verts, B.final_pn_verts);
        f << buf;
        std::snprintf(buf, sizeof buf, "%s\n", pass ? "PASS" : "FAIL");
        f << buf;
    };

    {
        std::ofstream f("e2e_determinism.txt");
        report(f);
    }
    report(std::cout);

    std::printf("\n%s\n", pass ? "DETERMINISM E2E PASSED" : "DETERMINISM E2E FAILED");
    return pass ? 0 : 1;
}
