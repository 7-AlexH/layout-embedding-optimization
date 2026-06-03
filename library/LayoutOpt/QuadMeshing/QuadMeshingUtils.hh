#pragma once
#include <polymesh/Mesh.hh>
#include "LayoutOpt/DataStructures/Types.hh"

namespace LayoutOpt
{
pm::face_attribute<double> max_inner_angle(pm::vertex_attribute<pos3> _qm_pos);
pm::face_attribute<double> scaled_jacobian(pm::vertex_attribute<pos3> _qm_pos);

}
