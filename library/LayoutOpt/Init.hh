#pragma once
#include "LayoutOpt/DataStructures/Types.hh"
namespace LayoutOpt
{

/**
 * @brief Normalizes target and layout geometry (ie., centering, scaling to unit area).
 * @param _t_pos Target positions (in-place).
 * @param _l_pos Layout positions (in-place).
 */
void preprocess_target_and_layout(pm::vertex_attribute<pos3>& _t_pos, pm::vertex_attribute<pos3>& _l_pos);

/**
 * @brief Adds normal-direction noise proportional to average edge length.
 * @param _pos Input positions.
 * @param _amount Relative magnitude (0.01 = 1%).
 * @return Perturbed positions.
 */
pm::vertex_attribute<pos3> add_noise(const pm::vertex_attribute<pos3> &_pos, double _amount = 0.01);

//==============================================================================
//==================================== aux =====================================
//==============================================================================
/**
 * @brief Scales all vertices uniformly.
 * @param _pos Positions (in-place).
 * @param _scale Scaling factor.
 */
void rescale_surface(pm::vertex_attribute<pos3>& _pos, double _scale);

/**
 * @brief Rescales surface to unit area.
 * @param _pos Positions (in-place).
 * @return Applied scale factor.
 */
double rescale_to_unit_area(pm::vertex_attribute<pos3>& _pos);

/**
 * @brief Translates all vertices.
 * @param _pos Positions (in-place).
 * @param _translation_vec Translation vector.
 */
void translate_surface(pm::vertex_attribute<pos3>& _pos, vec3 _translation_vec);

/**
 * @brief Centers surface at origin.
 * @param _pos Positions (in-place).
 * @return Applied translation.
 */
vec3 translate_surface_to_center(pm::vertex_attribute<pos3>& _pos);

//==============================================================================
//================================= validity ===================================
//==============================================================================

/**
 * @brief Checks for degenerate (near-zero area) faces.
 * @param _pos Positions.
 * @param _eps Area threshold.
 * @return True if any degenerate face exists.
 */
bool contains_degenerate_faces(pm::vertex_attribute<pos3> const& _pos, double _eps = MEDIUM_EPS);


} // namespace LayoutOpt
