#pragma once
#include "LayoutOpt/PrincipalCurvature.hh"
namespace LayoutOpt
{

// pushes curvature onto faces, stores the per point mean curvature per face
pm::face_attribute<double> point_wise_mean_curvature(pm::vertex_attribute<PrincipalCurvature> const& _pc, pm::vertex_attribute<pos3> const& _pos);
// following Sharp and Crane 2018 (Variational Surface Cutting)
pm::face_attribute<double> point_wise_gauss_curvature(pm::vertex_attribute<pos3> const& _pos);
pm::face_attribute<double> normal_based_cluster(pm::vertex_attribute<pos3> const& _pos);




} // namespace LayoutOpt
