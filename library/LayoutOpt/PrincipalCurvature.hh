#pragma once
#include <LayoutOpt/DataStructures/Types.hh>
#include <polymesh/attributes.hh>

#include "LayoutOpt/TorchUtils.hh"

namespace LayoutOpt
{
struct PrincipalCurvature
{
    // real principal curvatures
    double kappa1;
    double kappa2;
    vec3 dir1; //ambient space
    vec3 dir2; //ambient space

    double k_min() const;
    double k_max() const;

    double mean_curvature() const;

    vec3 get_max_dir() const;
    vec3 get_min_dir() const;
    double get_filter_value() const; // scale invariant directional alignment of surface parameterization, Campen et al. 2016
};

pm::vertex_attribute<PrincipalCurvature> principal_curvature(pm::vertex_attribute<pos3> const& _pos);



struct DirectionFieldData
{
    double confidence; // magnitude is proportional to the squared difference of the 1st and 2nd principal curvatures (κ1−κ2)2
    torch::Tensor basis; //tensor shape [2, 3], ie two 3D vectors
    torch::Tensor dir; //smooth direction field --> a 2D vector raised to the 4th power (4-rosy field), norm = 1
};



//using geometry central
/// @short the magnitude is proportional to the squared difference of the 1st and 2nd principal curvatures (κ1−κ2)2
///  (so for instance, if a surface is flat and κ1≈κ2, the magnitude of the field will be near 0).
pm::face_attribute<DirectionFieldData> smooth_direction_field(pm::vertex_attribute<pos3> const& _pos);



} // namespace LayoutOpt
