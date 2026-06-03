#include "ScalarFields.hh"
#include <polymesh/properties.hh>
#include "LayoutOpt/DataStructures/Types.hh"
#include "LayoutOpt/Visualization/Colors.hh"

#include <glow-extras/viewer/canvas.hh>
#include <glow-extras/viewer/view.hh>

namespace LayoutOpt
{

pm::face_attribute<double> point_wise_mean_curvature(pm::vertex_attribute<PrincipalCurvature> const& _pc, pm::vertex_attribute<pos3> const& _pos)
{
    pm::face_attribute<double> mean_curvature(_pc.mesh());
    auto voronoi_areas = pm::vertex_voronoi_areas(_pos);

    for (auto fh : mean_curvature.mesh().faces())
    {
        double mc_fh = 0.0;
        for (auto vh : fh.vertices())
        {
            mc_fh += (1.0 / 3.0) * _pc[vh].mean_curvature() / (voronoi_areas[vh] + EPS);
        }
        mean_curvature[fh] = mc_fh;
    }
    return mean_curvature;
}

pm::face_attribute<double> point_wise_gauss_curvature(pm::vertex_attribute<pos3> const& _pos)
{
    pm::halfedge_attribute<tg::angle64> triangle_corner_angles(_pos.mesh()); // the value is always associated with the from vertex of the heh
    for (auto heh : _pos.mesh().halfedges())
    {
        // i is from vertex of heh
        auto heh_ij = heh;
        auto heh_ik = heh.prev().opposite();

        auto dir_ij = tg::normalize_safe(_pos[heh_ij.vertex_to()] - _pos[heh_ij.vertex_from()]);
        auto dir_ik = tg::normalize_safe(_pos[heh_ik.vertex_to()] - _pos[heh_ik.vertex_from()]);
        triangle_corner_angles[heh] = tg::angle_between(dir_ij, dir_ik);
    }
    pm::vertex_attribute<tg::angle64> angle_sum_per_vertex(_pos.mesh());
    for (auto vh : _pos.mesh().vertices())
    {
        tg::angle64 angle_sum = tg::angle64::from_degree(0.0);
        for (auto heh : vh.outgoing_halfedges())
        {
            angle_sum += triangle_corner_angles[heh];
        }
        angle_sum_per_vertex[vh] = angle_sum;
    }
    pm::halfedge_attribute<double> augmented_triangle_angles(_pos.mesh()); // the value is always associated with the from vertex of the heh
    for (auto heh : _pos.mesh().halfedges())
    {
        auto i = heh.vertex_from();
        augmented_triangle_angles[heh]
            = (tg::angle64::from_degree(360.0).radians() * triangle_corner_angles[heh].radians()) / angle_sum_per_vertex[i].radians();
    }
    pm::face_attribute<double> point_wise_gauss_curvature_per_face(_pos.mesh());
    pm::face_attribute<double> face_areas = pm::triangle_areas(_pos);
    for (auto fh : _pos.mesh().faces())
    {
        double angle_deviation = tg::angle64::from_degree(180.0).radians();
        for (auto heh : fh.halfedges())
        {
            angle_deviation = angle_deviation - augmented_triangle_angles[heh];
        }
        point_wise_gauss_curvature_per_face[fh] = angle_deviation / face_areas[fh];
    }
    return point_wise_gauss_curvature_per_face;
}

pm::face_attribute<double> normal_based_cluster(pm::vertex_attribute<pos3> const& _pos)
{
    pm::face_attribute<double> normal_cluster(_pos.mesh());
    std::vector<vec3> normals = {vec3::unit_x, -vec3::unit_x, vec3::unit_y, -vec3::unit_y, vec3::unit_z, -vec3::unit_z};
    auto face_normals = pm::face_normals(_pos);
    for (auto fh : _pos.mesh().faces())
    {
        auto fh_normal = face_normals[fh];
        std::vector<double> dots;

        double idx = 0;
        double max_dot = 0;
        for (auto normal : normals)
        {
            auto dot_res = tg::dot(fh_normal, normal);
            if (dot_res > max_dot)
            {
                normal_cluster[fh] = idx;
                max_dot = dot_res;
            }
            idx += 1.0;
        }
    }
    return normal_cluster;
}

} // namespace LayoutOpt
