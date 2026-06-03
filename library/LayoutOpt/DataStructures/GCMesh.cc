#include "GCMesh.hh"
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>
#include "IGLMesh.hh"
#include "geometrycentral/surface/simple_polygon_mesh.h"

namespace LayoutOpt
{
GCMesh to_gc_mesh(pm::vertex_attribute<pos3> const& _pos)
{
    IGLMesh igl_mesh = to_igl_mesh(_pos);
    auto mesh = std::make_unique<ManifoldSurfaceMesh>(igl_mesh.F);
    auto pos = std::make_unique<VertexPositionGeometry>(*mesh, igl_mesh.V);
    return {std::move(mesh), std::move(pos)};
}

pos3 to_tg(geometrycentral::Vector3 const& _vec) { return pos3(_vec.x, _vec.y, _vec.z); }

} // namespace LayoutOpt
