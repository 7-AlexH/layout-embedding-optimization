#pragma once
#include <Eigen/Core>
#include <polymesh/attributes.hh>
#include "LayoutOpt/DataStructures/SurfacePoint.hh"
#include "LayoutOpt/DataStructures/Types.hh"
#include "typed-geometry/types/mat.hh"

namespace LayoutOpt
{

// for initialization
//  project the vertices of the layout onto a face of the target mesh
pm::vertex_attribute<pos3> project_from_to(pm::vertex_attribute<pos3> const& _from_pos, pm::vertex_attribute<pos3> const& _to_pos);

// projection
// find element where it is closest to
std::pair<pos3, FH> project_to_faces(pos3 _p, pm::vertex_attribute<pos3> const& _t_pos);
std::pair<pos3, EH> project_to_edges(pos3 _p, pm::vertex_attribute<pos3> const& _t_pos);
std::pair<pos3, EH> project_to_edges(pos3 _p, pm::vertex_attribute<pos3> const& _t_pos, std::vector<EH> const& _candidat_edges);


double angle_sum(HEH heh_start, HEH heh_end, pm::vertex_attribute<pos3> const& _pos);


struct BarycentricCoordinates
{
    double alpha;
    double beta;
    double gamma() const { return 1.0 - alpha - beta; }
};

BarycentricCoordinates compute_barycentric_coordinates(pos3 _pos, HEH _heh, pm::vertex_attribute<pos3> _positions);

/// @short Clamps barycentric coordinates to _eps to avoid negatives or near-zero values.
BarycentricCoordinates stabilize(BarycentricCoordinates const& _bc, double _eps);

/// @short assumes that a face is shared, if not, the result is random
FH shared_face(SurfacePoint const& _spA, SurfacePoint const& _spB, polymesh::Mesh const& _m);

/// @short assumes that a face is shared, if not, the result is random
/// only for vertex-surfacePoints
HEH shared_halfedge(SurfacePoint const& _sp_from, SurfacePoint const& _sp_to, polymesh::Mesh const& _m);

/// @brief Checks if a corner of a 3D planar polygon is convex.
///
/// Given three consecutive vertices (a, b, c) of a polygon and the face normal,
/// this function determines whether the corner at vertex b forms a convex angle
/// relative to the face's orientation. Works for planar polygons in 3D.
///
/// @param a The previous vertex of the corner.
/// @param b The current vertex at the corner to test.
/// @param c The next vertex of the corner.
/// @param face_normal The normalized normal vector of the polygon's face.
/// @return true if the corner at b is convex; false otherwise.
bool is_convex_corner_3d(pos3 const& a, pos3 const& b, pos3 const& c, vec3 const& face_normal);
std::pair<bool, float> corner_convexity_and_angle_3d(pos3 const& a, pos3 const& b, pos3 const& c, vec3 const& face_normal);

} // namespace LayoutOpt
