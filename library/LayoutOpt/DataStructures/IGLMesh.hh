#pragma once
#include "Types.hh"

namespace LayoutOpt{
struct IGLMesh
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

IGLMesh to_igl_mesh(const pm::vertex_attribute<pos3>& _pos);
}//namespace LayoutOpt
