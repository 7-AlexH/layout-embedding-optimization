#pragma once
#include <string>
namespace LayoutOpt
{

struct HarmonicOptions
{
    double w_SDE_SingValDecomp = 0.0;
    double w_SDE_DirectComputation = 0.0;
    double w_DE_SingValDecomp = 0.0;
    double w_AIAP_SingValDecomp = 0.0; // as isometric as possible https://dl.acm.org/doi/pdf/10.1145/2766947
    double w_I_DevFrom1_SingValDecomp = 0.0; //0.1
    double w_AreaPreserving_SingValDecomp = 0.0; //1.0

    //should not be used
    const bool normalized = false; // is the result normalized per patch param area
    // parameter domain
    const bool fixed_parameter_domain = false;
};

enum class Optimizer
{
    GradientDescent,
    Adam
};

struct OptimizationOptions
{
    // optimization weights and options
    HarmonicOptions harmonic_options;
    double w_harmonic_distorion_loss = 0.0;

    double w_yamabe_loss = 0.0; //VSC

    // regularization for alignment
    double w_curvature_alignment_loss = 0.0;

    // should not be used
    // regularizer ensuring that the surface points are equally distanced
    double const w_repel_loss = 0.0;
    double const w_variance_loss = 0.0;
    double const w_distortion_loss_extrinsic = 0.0;
    double const w_total_length_loss = 0.0;
    double const w_inner_angle_loss = 0.0;

    Optimizer optimizer = Optimizer::Adam;

    std::string to_string() const;
};


} // namespace LayoutOpt
