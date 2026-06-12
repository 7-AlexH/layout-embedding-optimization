#include "HandGradients.hh"

#include <cassert>

#include <omp.h>

#include <polymesh/Mesh.hh>
#include <polymesh/properties.hh>

namespace LayoutOpt
{

double hand_loss_forward(LeafCtx const& _leaf,
                         Eigen::MatrixX2d const& _bary,
                         std::vector<TriangleStrip> const& _strips,
                         TargetMeshData const& _tmd,
                         LayoutData const& _ld,
                         PathNetworkData const& _pnd,
                         OverlayMeshData const& _omd,
                         OptimizationOptions const& _opts,
                         HandLossCtx& _ctx)
{
    assert(_omd.pos_mat_.has_value());

    _ctx = HandLossCtx{};

    int const n_l_edges = (int)_ld.mesh_->edges().size();
    _ctx.pos = _omd.pos_mat_.value();
    _ctx.seg_a = Eigen::MatrixX2d::Zero(n_l_edges, 2);
    _ctx.seg_b = Eigen::MatrixX2d::Zero(n_l_edges, 2);

    // S1: bary -> layout-node overlay rows + 2D strip endpoints
    leaf_forward(_leaf, _bary, _tmd.pos_mat_, _ctx.pos, _ctx.seg_a, _ctx.seg_b);
    // S3: A/B -> overwritten interior overlay rows
    intersections_forward(_strips, _ctx.seg_a, _ctx.seg_b, _tmd, _ld, _pnd, _ctx.ictx, _ctx.pos);

    if (_opts.w_harmonic_distorion_loss > 0)
    {
        // S4
        edge_lengths_forward(_ctx.pos, _ld, _pnd, _ctx.el_ctx, _ctx.lengths);
        pn_t_forward(_ctx.pos, _ctx.lengths, _ld, _pnd, _ctx.t_ctx, _ctx.t);
        pn_uvs_forward(_ctx.lengths, _ctx.t, _ld, _pnd, _opts.harmonic_options.fixed_parameter_domain, _ctx.uv_ctx, _ctx.uvs);
        // S5
        cotans_forward(_ctx.pos, *_omd.mesh_.get(), _ctx.cot_ctx, _ctx.cotans);

        // per-overlay-halfedge UV gather (from-vertex convention, same as the
        // `uvs` tensor in harmonic_distortion_loss)
        auto const heh_count = _omd.mesh_->halfedges().size();
        Eigen::MatrixX2d o_uvs = Eigen::MatrixX2d::Zero((Eigen::Index)heh_count, 2);
        _ctx.pn_of_oheh.assign(heh_count, -1);
        for (auto pn_heh : _pnd.mesh_->halfedges())
        {
            auto o_vh_from = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_from()]);
            auto o_vh_to = _omd.mesh_->handle_of(_pnd.map_to_overlay_vertices_.value()[pn_heh.vertex_to()]);
            auto o_heh = pm::halfedge_from_to(o_vh_from, o_vh_to);
            o_uvs.row(o_heh.idx.value) = _ctx.uvs.row(pn_heh.idx.value);
            _ctx.pn_of_oheh[(size_t)o_heh.idx.value] = pn_heh.idx.value;
        }

        // S6 + S7 per patch. Patches are independent (each reads shared const
        // data and writes only its own slot), so the loop runs under OpenMP;
        // the loss reduction (distortion_loss_forward) stays sequential in
        // patch-index order, so the result is bitwise-deterministic for any
        // thread count/schedule (e2e_determinism gates this).
        int const n_patches = (int)_ld.mesh_->faces().size();
        _ctx.hctx.resize(n_patches);
        _ctx.sctx.resize(n_patches);
        _ctx.bnd_uvs.resize(n_patches);
        _ctx.bnd_src.resize(n_patches);
        _ctx.patch_fhs.resize(n_patches);

        // polymesh attribute registration is NOT thread-safe, so the scratch
        // maps are created (and later destroyed) sequentially, one per thread;
        // each iteration resets exactly the entries prepare_param wrote (the
        // inner/boundary lists), restoring the {-1, true} fresh state.
        int const n_threads = omp_get_max_threads();
        std::vector<pm::vertex_attribute<MappingIndex>> maps;
        maps.reserve((size_t)n_threads);
        for (int t = 0; t < n_threads; ++t)
            maps.push_back(_omd.mesh_->vertices().make_attribute<MappingIndex>({-1, true}));

#pragma omp parallel for schedule(dynamic)
        for (int patch_i = 0; patch_i < n_patches; ++patch_i)
        {
            pm::vertex_attribute<MappingIndex>& map = maps[(size_t)omp_get_thread_num()];
            std::vector<VH> inner_vhs, bnd_vhs;
            std::vector<vec2d>& bnd = _ctx.bnd_uvs[patch_i];
            std::vector<int>& src = _ctx.bnd_src[patch_i];
            std::vector<FH>& fhs = _ctx.patch_fhs[patch_i];
            prepare_param(patch_i, _pnd, _omd, o_uvs, map, inner_vhs, bnd_vhs, bnd, fhs, &src);

            PatchHarmonicCtx h = harmonic_param_forward(map, inner_vhs, bnd, _ctx.cotans);
            _ctx.sctx[patch_i] = patch_distortion_forward(fhs, map, h.inner_uvs, bnd, _ctx.pos, _opts.harmonic_options);
            _ctx.hctx[patch_i] = std::move(h);

            for (auto vh : inner_vhs)
                map[vh] = {-1, true};
            for (auto vh : bnd_vhs)
                map[vh] = {-1, true};
        }
        _ctx.harmonic = distortion_loss_forward(_ctx.sctx, /*normalized*/ false);
    }

    if (_opts.w_curvature_alignment_loss > 0)
        _ctx.curvature = curvature_alignment_forward(_ctx.pos, _tmd, _pnd, _omd, _ctx.curv_ctx);

    // same assembly order as eval(): zeros + w_h * harmonic + w_c * curvature
    _ctx.loss = 0.0;
    if (_opts.w_harmonic_distorion_loss > 0)
        _ctx.loss = _ctx.loss + _opts.w_harmonic_distorion_loss * _ctx.harmonic;
    if (_opts.w_curvature_alignment_loss > 0)
        _ctx.loss = _ctx.loss + _opts.w_curvature_alignment_loss * _ctx.curvature;
    return _ctx.loss;
}

void hand_loss_backward(HandLossCtx const& _ctx,
                        LeafCtx const& _leaf,
                        TargetMeshData const& _tmd,
                        OptimizationOptions const& _opts,
                        Eigen::MatrixX2d& _d_bary)
{
    Eigen::MatrixX3d d_pos = Eigen::MatrixX3d::Zero(_ctx.pos.rows(), 3);
    Eigen::MatrixX2d d_seg_a = Eigen::MatrixX2d::Zero(_ctx.seg_a.rows(), 2);
    Eigen::MatrixX2d d_seg_b = Eigen::MatrixX2d::Zero(_ctx.seg_b.rows(), 2);

    if (_opts.w_harmonic_distorion_loss > 0)
    {
        // S7 loss assembly, seeded with the harmonic weight
        std::vector<double> d_sde, d_pa;
        distortion_loss_backward(_ctx.sctx, /*normalized*/ false, _opts.w_harmonic_distorion_loss, d_sde, d_pa);

        Eigen::VectorXd d_cotans = Eigen::VectorXd::Zero(_ctx.cotans.size());
        Eigen::MatrixX2d d_uvs_pn = Eigen::MatrixX2d::Zero(_ctx.uvs.rows(), 2);
        Eigen::VectorXd d_t = Eigen::VectorXd::Zero(_ctx.t.size());
        Eigen::VectorXd d_lengths = Eigen::VectorXd::Zero(_ctx.lengths.size());

        // S7 -> S6 per patch, boundary adjoints scattered back to pn-uv rows
        for (size_t p = 0; p < _ctx.sctx.size(); ++p)
        {
            Eigen::MatrixX2d d_inner = Eigen::MatrixX2d::Zero(_ctx.hctx[p].n_inner, 2);
            std::vector<vec2d> d_bnd(_ctx.bnd_uvs[p].size(), vec2d::Zero());
            patch_distortion_backward(_ctx.sctx[p], d_sde[p], d_pa[p], _opts.harmonic_options, d_inner, d_bnd, d_pos);
            harmonic_param_backward(_ctx.hctx[p], _ctx.bnd_uvs[p], _ctx.cotans, d_inner, d_cotans, d_bnd);
            for (size_t b = 0; b < d_bnd.size(); ++b)
            {
                int const pn = _ctx.pn_of_oheh[(size_t)_ctx.bnd_src[p][b]];
                assert(pn >= 0);
                d_uvs_pn(pn, 0) += d_bnd[b].x();
                d_uvs_pn(pn, 1) += d_bnd[b].y();
            }
        }

        // S4: uvs -> (t, lengths) -> positions; edge_lengths last (both
        // pn_uvs_backward and pn_t_backward feed d_lengths)
        pn_uvs_backward(_ctx.uv_ctx, _ctx.t, d_uvs_pn, d_lengths, d_t);
        pn_t_backward(_ctx.t_ctx, _ctx.pos, d_t, d_pos, d_lengths);
        edge_lengths_backward(_ctx.el_ctx, _ctx.pos, d_lengths, d_pos);
        // S5
        cotans_backward(_ctx.cot_ctx, _ctx.pos, d_cotans, d_pos);
    }

    // independent curvature branch, seeded with its weight
    if (_opts.w_curvature_alignment_loss > 0)
        curvature_alignment_backward(_ctx.curv_ctx, _ctx.pos, _opts.w_curvature_alignment_loss, d_pos);

    // d_pos is now the TOTAL loss gradient w.r.t. the post-write overlay
    // positions: route the interior written rows to the strip endpoints (S3),
    // then endpoints + node rows to the leaves (S1)
    intersections_backward(_ctx.ictx, d_pos, _tmd, d_seg_a, d_seg_b);
    leaf_backward(_leaf, _tmd.pos_mat_, d_pos, d_seg_a, d_seg_b, _d_bary);
}

pm::vertex_attribute<vec2d> compute_gradients_hand(std::vector<TriangleStrip> const& _strips,
                                                   TargetMeshData const& _tmd,
                                                   LayoutData const& _ld,
                                                   PathNetworkData const& _pnd,
                                                   OverlayMeshData const& _omd,
                                                   OptimizationOptions const& _opts,
                                                   double* _loss_out)
{
    LeafCtx leaf;
    leaf_collect(_strips, _tmd, _ld, _pnd, leaf);
    Eigen::MatrixX2d const bary = collect_bary(_pnd);

    HandLossCtx ctx;
    hand_loss_forward(leaf, bary, _strips, _tmd, _ld, _pnd, _omd, _opts, ctx);
    if (_loss_out)
        *_loss_out = ctx.loss;

    Eigen::MatrixX2d d_bary = Eigen::MatrixX2d::Zero(bary.rows(), 2);
    hand_loss_backward(ctx, leaf, _tmd, _opts, d_bary);

    // production collection convention (compute_gradients): zero default,
    // FacePoint rows filled; gradient reaching any other sp type is discarded
    auto out = _pnd.mesh_->vertices().make_attribute<vec2d>(vec2d::Zero());
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        SurfacePoint const& sp = _pnd.sp_on_target_.value()[pn_vh];
        if (sp.type == SurfacePointType::FacePoint)
            out[pn_vh] = vec2d(d_bary(pn_vh.idx.value, 0), d_bary(pn_vh.idx.value, 1));
    }
    return out;
}

} // namespace LayoutOpt
