#pragma once

#include <polymesh/attributes.hh>
#include "LayoutOpt/DataStructures/Types.hh"

namespace LayoutOpt
{

struct EvalInfo
{
    // check if gradients are available
    bool success = false;

    // loss
    double loss;

    // update directions in bary coords, consumed by the detached do_step
    pm::vertex_attribute<vec2d> pn_sp_update_dirs;

    // for harmonic loss
    pm::face_attribute<double> o_distortion_harmonic;
    pm::halfedge_attribute<pos2> o_uvs;
};

// maps an overlay vertex into the linear system of a per-patch harmonic
// parameterization (inner index or boundary index)
struct MappingIndex
{
    int idx = -1;
    bool on_boundary = false;
};

} // namespace LayoutOpt
