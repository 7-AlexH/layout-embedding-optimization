#include "LayoutOpt/Optimization.hh"
#include "LayoutOpt/Embedding.hh"
#include "LayoutOpt/EmbeddingUtils.hh"
#include "LayoutOpt/ObjectiveFunctions.hh"
#include "LayoutOpt/Optimizers.hh"
#include "LayoutOpt/Update.hh"
#include "LayoutOpt/Visualization/Viewing.hh"


#include "Utils/Timer.hh"

namespace LayoutOpt
{

EvalInfo eval(TargetMeshData& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, OptimizationOptions const& _opts, int i, OptimizerData& _od)
{
    // DEBUG_OUT("iteration " << i << ":")

    EvalInfo eval_info;

    timers.start(TimerCollection::EvalObjective);

    torch::Tensor loss = torch::zeros({}, torch::dtype(torch::kFloat64));
    if (_opts.w_harmonic_distorion_loss > 0)
    {
        auto loss_harmonic_distortion = _opts.w_harmonic_distorion_loss * harmonic_distortion_loss(_tmd, _ld, _pnd, _omd, _opts.harmonic_options, eval_info);
        loss = loss + loss_harmonic_distortion;

        // DEBUG_OUT("harmonic_distortion_loss: " << loss_harmonic_distortion.item<double>())
    }

    if (_opts.w_curvature_alignment_loss > 0)
    {
        auto loss_curvature_alignment = _opts.w_curvature_alignment_loss * principal_curvature_alignment_loss(_tmd, _ld, _pnd, _omd, eval_info);
        loss = loss + loss_curvature_alignment;
        // DEBUG_OUT("curvature_alignment_loss: " << loss_curvature_alignment.item<double>())
    }

    // if (_opts.w_repel_loss > 0)
    // {
    //     auto loss_repel = repel_loss_v2(_ld, _pnd, _omd);
    //     loss = loss + loss_repel;
    //     DEBUG_OUT("repel_loss: " << loss_repel.item<double>())
    // }

    // if (_opts.w_variance_loss > 0)
    // {
    //     auto loss_variance = _opts.w_variance_loss * variance_loss(_tmd, _ld, _omd, eval_info);
    //     loss = loss + loss_variance;
    //     DEBUG_OUT("variance_loss: " << loss_variance.item<double>())
    // }
    // if (_opts.w_yamabe_loss > 0)
    // {
    //     auto loss_distortion = _opts.w_yamabe_loss * distorion_loss(_tmd, _ld, _omd, eval_info);
    //     loss = loss + loss_distortion;
    //     DEBUG_OUT("distortion_loss: " << loss_distortion.item<double>())
    // }
    // if (_opts.w_distortion_loss_extrinsic > 0)
    // {
    //     auto loss_distortion_extr = _opts.w_distortion_loss_extrinsic * distorion_loss_extrinsic(_tmd, _ld, _omd, eval_info);
    //     loss = loss + loss_distortion_extr;
    //     DEBUG_OUT("distortion_extr_loss: " << loss_distortion_extr.item<double>())
    // }
    // if (_opts.w_total_length_loss > 0)
    // {
    //     auto loss_total_length = _opts.w_total_length_loss * total_length_loss(_pnd, _omd);
    //     loss = loss + loss_total_length;
    //     DEBUG_OUT("length_loss: " << loss_total_length.item<double>())
    // }
    // if (_opts.w_inner_angle_loss > 0)
    // {
    //     auto loss_inner_angle = inner_angle_loss(_pnd, _omd);
    //     loss = loss + loss_inner_angle;

    //     DEBUG_OUT("inner_angle_loss: " << loss_inner_angle.item<double>())
    // }

    timers.stop(TimerCollection::EvalObjective);

    // DEBUG_OUT("loss: " << loss.item<double>())

    eval_info.loss = loss.item<double>();

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
    eval_info.pn_sp_grads = compute_gradients(loss, _pnd);
    timers.stop(TimerCollection::Backpropagation);
    // preprocess_gradients(_tmd, _pnd, eval_info.pn_sp_grads);

    switch (_opts.optimizer)
    {
    case LayoutOpt::Optimizer::GradientDescent:
    {
        DEBUG_OUT("gradient descent!");
        eval_info.pn_sp_update_dirs = gradient_descent(_pnd, eval_info.pn_sp_grads, _od);
        break;
    }
    case LayoutOpt::Optimizer::Adam:
    {
        // DEBUG_OUT("adam");
        eval_info.pn_sp_update_dirs = vector_adam_updates(_tmd, _pnd, eval_info.pn_sp_grads, _od);
        break;
    }
    }

    return eval_info;
}

void apply(TargetMeshData const& _tmd, LayoutData& _ld, PathNetworkData& _pnd, OverlayMeshData& _omd, OptimizerData& _od, EvalInfo const& _eval_info)
{
    timers.start(TimerCollection::Update);
    // DEBUG_OUT("collapse")
    collapse_pn_to_layout(_pnd);

    // DEBUG_OUT("trace")
    auto traces = do_step(_tmd, _pnd, _eval_info.pn_sp_update_dirs, &_od);
    timers.stop(TimerCollection::Update);

    timers.start(TimerCollection::Embedding);
    // DEBUG_OUT("embed")
    compute_layout_embedding_update(_tmd, _ld, _pnd, _omd);
    timers.stop(TimerCollection::Embedding);
}


}
