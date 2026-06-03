#pragma once

#include <torch/serialize/input-archive.h>
#include "LayoutOpt/DataStructures/Types.hh"
namespace LayoutOpt
{
// access
torch::Tensor get_row(torch::Tensor const& _tensor, size_t _idx);

// conversions
// types
pos3 torch_to_pos3(torch::Tensor const& _pos);
pos2 torch_to_pos2(torch::Tensor const& _pos);

// attributes
template <typename tag, typename AttrT>
at::Tensor attr_to_torch(pm::primitive_attribute<tag, AttrT> const& _attr);

// utils

// utils
/// @return tensor of shape [#F] (1D tensor with #F elements) where #F is the number of faces in the mesh
at::Tensor torch_face_areas(at::Tensor const& _torch_pos, polymesh::Mesh const& _mesh);

/// @return tensor of shape [#E] (1D tensor with #E elements) where #E is the number of edges in the mesh
at::Tensor torch_cotans(at::Tensor const& _torch_pos, polymesh::Mesh const& _mesh);

/**
 * @brief, computes 2D embedding of vertices for a given faces, where _3D_origin becomes the origin and the point _3D_x_axis will be lying on the x-axis
 * @param assumes that _3D_origin and _3D_x_axis are in the plane of the face
 * @param positive tells if the vector should be aligned along the positive or negative direction
 */
torch::Tensor torch_compute_2D_face_embedding(HEH _heh, torch::Tensor const& _3D_origin, torch::Tensor const& _3D_x_axis, bool _positive, torch::Tensor const& _3D_pos);


/**
 * @brief Computes a 2D embedding for the vertices of a given face.
 *
 * This function calculates a 2D embedding for the vertices of the face defined by `_heh`.
 * The vertex referenced by `_heh.next().vertex_to()` is set as the origin of the embedding.
 *
 * @param _heh A half-edge handle representing the face.
 * @param _3D_pos A tensor containing the 3D positions of the vertices.
 * @return A torch::Tensor containing the 2D coordinates of the face vertices.
 */
torch::Tensor torch_compute_2D_face_embedding(HEH _heh, torch::Tensor const& _3D_pos);


torch::Tensor torch_compute_bary_cords_2D(torch::Tensor const& _point, HEH const& _heh, pm::vertex_attribute<torch::Tensor> _pos);
torch::Tensor torch_compute_bary_cords_2D(torch::Tensor const& _point, HEH const& _heh, torch::Tensor const& _pos);

torch::Tensor torch_face_normal(pm::face_handle const& _fh, torch::Tensor const& _pos);

torch::Tensor torch_compute_intersection_parameter(torch::Tensor const& _line_a, torch::Tensor const& _line_b);

at::Tensor torch_singular_values(at::Tensor const& _mat);

} // namespace LayoutOpt
