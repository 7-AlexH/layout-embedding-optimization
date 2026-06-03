#include "OptimizationOptions.hh"

namespace LayoutOpt
{

std::string OptimizationOptions::to_string() const
{
    std::string label = "";

    switch (optimizer)
    {
    case Optimizer::Adam:
    {
        label += "adam_";
        break;
    }
    default:
    {
        label += "gradient_descent_";
        break;
    }
    }

    if (w_harmonic_distorion_loss > 0)
    {
        label += "_w_harmonic_distorion_loss_";
        if (harmonic_options.w_SDE_SingValDecomp > 0)
            label += "w_SDE_SVD_" + std::to_string(harmonic_options.w_SDE_SingValDecomp); //SVD = singular value decompostion
        if (harmonic_options.w_SDE_DirectComputation > 0)
            label += "w_SDE_cf_" + std::to_string(harmonic_options.w_SDE_DirectComputation); //cf = closed form
        if (harmonic_options.w_DE_SingValDecomp)
             label += "w_DE_" + std::to_string(harmonic_options.w_DE_SingValDecomp);
        if (harmonic_options.w_AIAP_SingValDecomp)
             label += "w_AIAP_" + std::to_string(harmonic_options.w_AIAP_SingValDecomp);
        if (harmonic_options.w_I_DevFrom1_SingValDecomp)
             label += "w_I_DevFrom1_"+ std::to_string(harmonic_options.w_I_DevFrom1_SingValDecomp);
        if (harmonic_options.w_AreaPreserving_SingValDecomp)
             label += "w_AreaPres_"+ std::to_string(harmonic_options.w_AreaPreserving_SingValDecomp);

        if (harmonic_options.normalized == true)
            label += "normalized_";
        if (harmonic_options.fixed_parameter_domain == true)
            label += "fixed_square_domain_";

        label += "_" + std::to_string(w_harmonic_distorion_loss);
    }

    if (w_curvature_alignment_loss > 0)
        label += "_w_curvature_alignment_loss_" + std::to_string(w_curvature_alignment_loss);
    if (w_repel_loss > 0)
        label += "_w_repel_loss_" + std::to_string(w_repel_loss);
    if (w_variance_loss > 0)
        label += "_w_variance_loss_" + std::to_string(w_variance_loss);
    if (w_yamabe_loss > 0)
        label += "_w_yamabe_loss_" + std::to_string(w_yamabe_loss);
    if (w_distortion_loss_extrinsic > 0)
        label += "_w_distortion_loss_extrinsic_" + std::to_string(w_distortion_loss_extrinsic);
    if (w_total_length_loss > 0)
        label += "_w_total_length_loss_" + std::to_string(w_total_length_loss);
    if (w_inner_angle_loss > 0)
        label += "_w_inner_angle_loss_" + std::to_string(w_inner_angle_loss);
    return label;
}
}
