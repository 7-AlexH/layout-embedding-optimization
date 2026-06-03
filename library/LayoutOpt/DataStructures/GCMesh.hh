#pragma once
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include <geometrycentral/utilities/vector3.h>
#include "Types.hh"

namespace LayoutOpt
{

using namespace geometrycentral::surface;

struct GCMesh
{
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> positionGeometry;
};

GCMesh to_gc_mesh(pm::vertex_attribute<pos3> const& _pos);

pos3 to_tg(geometrycentral::Vector3 const& _vec);

} // namespace LayoutOpt
