#include "Init.hh"
#include "LayoutOpt/DataStructures/Types.hh"
#include "Utils.hh"
#include "polymesh/properties.hh"

namespace LayoutOpt
{

void preprocess_target_and_layout(pm::vertex_attribute<pos3>& _t_pos, pm::vertex_attribute<pos3>& _l_pos)
{
    auto translation_vector = translate_surface_to_center(_t_pos);
    translate_surface(_l_pos, translation_vector);

    auto scale = rescale_to_unit_area(_t_pos);
    rescale_surface(_l_pos, scale);

    _l_pos = project_from_to(_l_pos, _t_pos);
}

void rescale_surface(pm::vertex_attribute<pos3>& _pos, double _scale)
{
    _pos.apply([_scale](auto& _p) { _p = _p * _scale; });
}

double rescale_to_unit_area(pm::vertex_attribute<pos3>& _pos)
{
    double area = pm::triangle_areas(_pos).sum();
    if (area <= 0.0)
        return 1.0; // avoid invalid scale

    double scale = 1.0 / std::sqrt(area);
    rescale_surface(_pos, scale);

    return scale;
}

void translate_surface(pm::vertex_attribute<pos3>& _pos, vec3 _translation_vec)
{
    _pos.apply([&_translation_vec](auto& _p) { _p += _translation_vec; });
}

vec3 translate_surface_to_center(pm::vertex_attribute<pos3>& _pos)
{
    auto voronoi_areas = pm::vertex_voronoi_areas(_pos);
    pos3 centroid = pos3::zero;
    double total_area = 0.0;

    for (auto vh : _pos.mesh().vertices())
    {
        double a = voronoi_areas[vh];
        centroid += _pos[vh] * a;
        total_area += a;
    }

    if (total_area > 0.0)
    {
        centroid /= total_area;
        auto translation_vector = pos3::zero - centroid;
        translate_surface(_pos, translation_vector);
        return translation_vector;
    }
    return vec3::zero;
}

bool contains_degenerate_faces(pm::vertex_attribute<pos3> const& _pos, double _eps)
{
    for (auto fh : _pos.mesh().faces())
    {
        auto area = pm::face_area(fh, _pos);
        if (area < _eps)
            return true;
    }
    return false;
}

pm::vertex_attribute<pos3> add_noise(const pm::vertex_attribute<pos3>& _pos, double _amount)
{
    auto edge_lengths = _pos.mesh().edges().map([&](EH eh){return pm::edge_length(eh, _pos);});
    auto vertex_normals = pm::vertex_normals_by_area(_pos);

    tg::rng rng;

    auto noise_pos = _pos.mesh().vertices().map([&](VH vh){return _pos[vh] + tg::uniform(rng, 0.0, 1.0) * _amount * vertex_normals[vh];});
    return noise_pos;
}


} // namespace LayoutOpt
