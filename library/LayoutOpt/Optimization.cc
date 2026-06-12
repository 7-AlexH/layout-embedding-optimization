#include "LayoutOpt/Optimization.hh"
#include "LayoutOpt/Adjoint/HandGradients.hh"
#include "LayoutOpt/Embedding.hh"
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/ObjectiveFunctions.hh"
#include "LayoutOpt/Optimizers.hh"
#include "LayoutOpt/Update.hh"
#include "LayoutOpt/Visualization/Viewing.hh"


#include "Utils/Timer.hh"

namespace LayoutOpt
{

EvalInfo eval(TargetMeshData& _tmd,
              LayoutData& _ld,
              PathNetworkData& _pnd,
              OverlayMeshData& _omd,
              std::vector<TriangleStrip> const& _strips,
              OptimizationOptions const& _opts,
              int i,
              OptimizerData& _od)
{
    // DEBUG_OUT("iteration " << i << ":")

    EvalInfo eval_info;

    // The hand-rolled adjoint chain: leaf_collect -> hand_loss_forward ->
    // hand_loss_backward (see Adjoint/HandGradients.hh). Loss assembly inside
    // hand_loss_forward is w_h * harmonic + w_c * curvature, the same order
    // as the original torch loss.
    timers.start(TimerCollection::EvalObjective);
    LeafCtx leaf;
    leaf_collect(_strips, _tmd, _ld, _pnd, leaf);
    Eigen::MatrixX2d const bary = collect_bary(_pnd);

    HandLossCtx ctx;
    hand_loss_forward(leaf, bary, _strips, _tmd, _ld, _pnd, _omd, _opts, ctx);
    timers.stop(TimerCollection::EvalObjective);

    eval_info.loss = ctx.loss;

    // visualization fields (parity with the old torch loss): per-face
    // distortion and per-corner patch UVs
    if (_opts.w_harmonic_distorion_loss > 0)
    {
        eval_info.o_distortion_harmonic = _omd.mesh_->faces().make_attribute<double>(0.0);
        eval_info.o_uvs = _omd.mesh_->halfedges().make_attribute<pos2>();
        for (size_t p = 0; p < ctx.patch_fhs.size(); ++p)
        {
            for (size_t k = 0; k < ctx.patch_fhs[p].size(); ++k)
            {
                auto o_fh = ctx.patch_fhs[p][k];
                FaceDistortionCtx const& fc = ctx.sctx[p].faces[k];

                auto hehA = o_fh.any_halfedge();
                HEH const hehs[3] = {hehA, hehA.next(), hehA.next().next()};
                for (int c = 0; c < 3; ++c)
                {
                    MappingIndex const& mi = fc.uv[c];
                    vec2d const v = mi.on_boundary ? ctx.bnd_uvs[p][mi.idx] : vec2d(ctx.hctx[p].inner_uvs.row(mi.idx).transpose());
                    eval_info.o_uvs[hehs[c]] = pos2(v.x(), v.y());
                }
                eval_info.o_distortion_harmonic[o_fh] = fc.distortion;
            }
        }
    }

    // DEBUG_OUT("loss: " << eval_info.loss)

    if (eval_info.loss < std::numeric_limits<double>::max() || tg::is_nan(eval_info.loss))
    {
        eval_info.success = true;
    }
    else // early out if loss is too large or inf
    {
        eval_info.success = false;
        return eval_info;
    }

    timers.start(TimerCollection::Backpropagation);
    Eigen::MatrixX2d d_bary = Eigen::MatrixX2d::Zero(bary.rows(), 2);
    hand_loss_backward(ctx, leaf, _tmd, _opts, d_bary);
    timers.stop(TimerCollection::Backpropagation);

    // gradient collection convention (preserved from the original torch path):
    // zero default, only FacePoint surface points carry a gradient
    auto grads = _pnd.mesh_->vertices().make_attribute<vec2d>(vec2d::Zero());
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        if (_pnd.sp_on_target_.value()[pn_vh].type == SurfacePointType::FacePoint)
            grads[pn_vh] = vec2d(d_bary(pn_vh.idx.value, 0), d_bary(pn_vh.idx.value, 1));
    }
    // preprocess_gradients(_tmd, _pnd, grads);

    switch (_opts.optimizer)
    {
    case LayoutOpt::Optimizer::GradientDescent:
    {
        DEBUG_OUT("gradient descent!");
        eval_info.pn_sp_update_dirs = gradient_descent(_pnd, grads, _od);
        break;
    }
    case LayoutOpt::Optimizer::Adam:
    {
        // DEBUG_OUT("adam");
        eval_info.pn_sp_update_dirs = vector_adam_updates(_tmd, _pnd, grads, _od);
        break;
    }
    }

    return eval_info;
}

void apply(TargetMeshData const& _tmd,
           LayoutData& _ld,
           PathNetworkData& _pnd,
           OverlayMeshData& _omd,
           OptimizerData& _od,
           EvalInfo const& _eval_info,
           std::vector<TriangleStrip>* _strips_out)
{
    timers.start(TimerCollection::Update);
    // DEBUG_OUT("collapse")
    collapse_pn_to_layout(_pnd);

    // DEBUG_OUT("trace")
    auto traces = do_step(_tmd, _pnd, _eval_info.pn_sp_update_dirs, &_od);
    timers.stop(TimerCollection::Update);

    timers.start(TimerCollection::Embedding);
    // DEBUG_OUT("embed")
    compute_layout_embedding_update(_tmd, _ld, _pnd, _omd, _strips_out);
    timers.stop(TimerCollection::Embedding);
}


}
