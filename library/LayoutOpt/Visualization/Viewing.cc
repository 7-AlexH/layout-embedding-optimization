#include "Viewing.hh"
#include "LayoutOpt/ObjectiveFunctions.hh"

#include "LayoutOpt/Visualization/ColorGenerator.hh"
#include "LayoutOpt/Visualization/ColorsMaps.hh"
#include "glow-extras/viewer/canvas.hh"

#include <complex.h>
#include "LayoutOpt/DataStructures/GCMesh.hh"
#include "geometrycentral/surface/direction_fields.h"

namespace LayoutOpt
{

void view_mesh(pm::vertex_attribute<pos3> const& _pos, ViewingOptions const& _vo)
{
    auto v = gv::view();
    if (_vo.view_faces)
    {
        gv::view(_pos, _vo.face_color);
    }

    if (_vo.view_vertices)
    {
        gv::view(gv::points(_pos).point_size_px(_vo.point_size), _vo.vertex_color);
    }

    if (_vo.view_edges)
    {
        gv::view(gv::lines(_pos).line_width_px(_vo.line_width), _vo.edge_color);
    }
}

void view_overlay(LayoutData const& _ld, OverlayMeshData const& _omd)
{
    auto c = gv::canvas();
    double vertex_size = 15;
    double line_size = 5;


    ColorGenerator cg(0, 25);
    auto layout_colors = cg.generate_next_colors(_ld.mesh_->faces().size());

    for (auto vh : _ld.mesh_->vertices())
    {
        if (vh.adjacent_vertices().size() != 2)
        {
            c.add_point(_ld.pos_[vh], MAGENTA).size(vertex_size);
        }
        else
        {
            c.add_point(_ld.pos_[vh], MAGENTA).size(0.9 * vertex_size);
        }
    }

    if (_omd.map_to_layout_faces_.has_value())
    {
        auto const& map_to_layout_faces = _omd.map_to_layout_faces_.value();
        for (auto fh : _omd.mesh_->faces())
        {
            auto color = map_to_layout_faces[fh].is_valid() ? layout_colors[map_to_layout_faces[fh].value] : BLACK;
            c.add_face(fh, _omd.pos_, color);
        }
    }
    if (_omd.patch_boundary_mask_.has_value())
    {
        auto const& patch_boundary_mask = _omd.patch_boundary_mask_.value();
        for (auto eh : _omd.mesh_->edges())
        {
            if (patch_boundary_mask[eh])
            {
                c.add_line(eh, _omd.pos_, MAGENTA).size(line_size);
            }
        }
    }
}

void view_overlay(LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo const& _eval_info)
{
    auto c = gv::canvas();
    double vertex_size = 20;
    double line_size = 5;

    double cap_min = 0.0;
    double cap_max = 0.001;
    auto colors = apply_colormap(_eval_info.o_distortion_harmonic, ColormapNames::plasma, &cap_min, &cap_max);

    for (auto vh : _ld.mesh_->vertices())
    {
        if (vh.adjacent_vertices().size() != 2)
        {
            c.add_point(_ld.pos_[vh], MAGENTA).size(vertex_size);
        }
        else
        {
            c.add_point(_ld.pos_[vh], MAGENTA).size(0.9 * vertex_size);
        }
    }

    if (_omd.map_to_layout_faces_.has_value())
    {
        for (auto fh : _omd.mesh_->faces())
        {
            c.add_face(fh, _omd.pos_, colors[fh]);
        }
    }
    if (_omd.patch_boundary_mask_.has_value())
    {
        auto const& patch_boundary_mask = _omd.patch_boundary_mask_.value();
        for (auto eh : _omd.mesh_->edges())
        {
            if (patch_boundary_mask[eh])
            {
                c.add_line(eh, _omd.pos_, MAGENTA).size(line_size);
            }
        }
    }
}


void view_layout(LayoutData const& _ld)
{
    auto c = gv::canvas();
    double vertex_size = 15;
    double line_size = 5;

    ColorGenerator cg(0, 25);
    auto layout_colors = cg.generate_next_colors(_ld.mesh_->faces().size());

    for (auto vh : _ld.mesh_->vertices())
    {
        c.add_point(_ld.pos_[vh], MAGENTA).size(vertex_size);
    }

    for (auto eh : _ld.mesh_->edges())
    {
        c.add_line(eh, _ld.pos_, MAGENTA).size(line_size);
    }


    for (auto fh : _ld.mesh_->faces())
    {        
        auto color = layout_colors[fh.idx.value];
        c.add_face(fh, _ld.pos_, color);
    }
}

void view_param(LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo const& _eval_info, FH _l_fh)
{
    double cap_min = 0.0;
    double cap_max = 0.001;
    auto colors = apply_colormap(_eval_info.o_distortion_harmonic, ColormapNames::plasma, &cap_min, &cap_max);

    auto c = gv::canvas();
    for (auto fh : _omd.mesh_->faces())
    {
        if (_omd.map_to_layout_faces_.value()[fh].value != _l_fh.idx.value)
            continue;
        auto hehs = fh.halfedges().to_array<3>();

        c.add_face(tg::pos3(_eval_info.o_uvs[hehs[0]]), tg::pos3(_eval_info.o_uvs[hehs[1]]), tg::pos3(_eval_info.o_uvs[hehs[2]]), colors[fh]);
        c.add_line(tg::pos3(_eval_info.o_uvs[hehs[0]]), tg::pos3(_eval_info.o_uvs[hehs[1]]), BLUE).size(0.1);
        c.add_line(tg::pos3(_eval_info.o_uvs[hehs[1]]), tg::pos3(_eval_info.o_uvs[hehs[2]]), BLUE).size(0.1);
        c.add_line(tg::pos3(_eval_info.o_uvs[hehs[2]]), tg::pos3(_eval_info.o_uvs[hehs[0]]), BLUE).size(0.1);
    }
}

void view_frame_field(pm::vertex_attribute<pos3> const& _pos)
{
    // 2. convert to other mesh representation
    GCMesh gcmesh = to_gc_mesh(_pos);

    // 3. compute smooth field and principalCurvatureDirections
    gcmesh.positionGeometry->requireFacePrincipalCurvatureDirections();
    gcmesh.positionGeometry->requireFaceTangentBasis();
    auto aligned_direction_field = computeCurvatureAlignedFaceDirectionField(*gcmesh.positionGeometry, 4);


    //===============================================================================

    auto index = computeVertexIndex(*gcmesh.positionGeometry, aligned_direction_field, 4);
    auto dfd = smooth_direction_field(_pos);

    {
        auto c = gv::canvas();
        c.add_faces(_pos, BLUE_25);

        for (auto vh : _pos.mesh().vertices())
        {
            if (index[vh.idx.value] != 0)
            {
                c.add_point(_pos[vh], RED).size(25);
            }
        }


        for (auto f : _pos.mesh().faces())
        {
            auto df = dfd[f];
            auto confidence = df.confidence;

            auto const& basis = df.basis_eigen;
            auto const& dir = df.dir_eigen;

            auto dir_complex = std::complex<double>(dir.x(), dir.y());

            auto vec = std::pow(dir_complex, 1. / 4);
            auto vec90 = std::complex<double>{0, 1} * std::pow(dir_complex, 1. / 4);

            auto basisX = vec3(basis(0, 0), basis(0, 1), basis(0, 2));
            auto basisY = vec3(basis(1, 0), basis(1, 1), basis(1, 2));

            auto tg_vec = vec2(vec.real(), vec.imag());
            auto tg_vec90 = vec2(vec90.real(), vec90.imag());

            auto vec_ambient = tg_vec.x * basisX + tg_vec.y * basisY;
            auto vec_ambient90 = tg_vec90.x * basisX + tg_vec90.y * basisY;

            double alpha = 0.02;
            c.add_line(pm::triangle_centroid(f, _pos), alpha * vec_ambient, MAGENTA).size(2);
            c.add_line(pm::triangle_centroid(f, _pos), alpha * -vec_ambient, PURPLE).size(2);
            c.add_line(pm::triangle_centroid(f, _pos), alpha * vec_ambient90, GREEN).size(2);
            c.add_line(pm::triangle_centroid(f, _pos), alpha * -vec_ambient90, BLUE).size(2);
        }
    }
}


} // namespace LayoutOpt
