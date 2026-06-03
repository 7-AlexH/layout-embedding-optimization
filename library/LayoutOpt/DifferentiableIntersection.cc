#include "DifferentiableIntersection.hh"
#include "LayoutOpt/FlattenTriangleStrip.hh"
#include "LayoutOpt/TorchUtils.hh"
#include "LayoutOpt/Visualization/ColorGenerator.hh"
#include "LayoutOpt/Visualization/Colors.hh"


#include <glow-extras/viewer/canvas.hh>

namespace LayoutOpt
{

void compute_differentiable_intersections_for_overlay_stable(
    std::vector<TriangleStrip> const& _strips, TargetMeshData const& _tmd, LayoutData const& _ld, PathNetworkData const& _pnd, OverlayMeshData& _omd)
{
    // DEBUG_OUT("compute_differentiable_intersections")

    assert(_ld.map_to_overlay_vertices_.has_value());
    assert(_pnd.map_to_overlay_vertices_.has_value());

    // DEBUG_OUT("intersections")
    _omd.torch_pos_.emplace(attr_to_torch(_omd.pos_));
    // compute differentiable om positions
    _omd.torch_pos_.emplace(attr_to_torch(_omd.pos_));
    compute_differentiable_surface_points(_tmd, _pnd, _omd);
    for (auto _l_eh : _ld.mesh_->edges())
    {
        TriangleStrip const& _strip = _strips[_l_eh.idx.value];
        compute_differentiable_intersection_for_overlay(_strip, _tmd, _pnd, _omd);
    }
}

void compute_differentiable_surface_points(TargetMeshData const& _tmd, PathNetworkData const& _pnd, OverlayMeshData& _omd)
{
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        // 1. set the corresponding variable in path network to requires_grad(true)
        if (_pnd.map_to_layout_vertices_[pn_vh].is_invalid())
            continue;
        auto& sp = _pnd.sp_on_target_.value()[pn_vh];
        sp.bary_coords.set_requires_grad(true);

        // 2. compute the correct overlay positions based on those variables
        auto o_v_idx = _pnd.map_to_overlay_vertices_.value()[pn_vh];
        auto new_pos = sp.get_pos(_tmd.torch_pos_, *_tmd.mesh_.get());
        _omd.torch_pos_.value()[o_v_idx.value] = new_pos;
    }
}

void compute_differentiable_intersection_for_overlay(TriangleStrip const& _strip, TargetMeshData const& _tmd, PathNetworkData const& _pnd, OverlayMeshData& _omd)
{
    assert(_pnd.sp_on_target_.has_value());
    auto sp_from = _pnd.sp_on_target_.value()[_strip.pn_vhs.front()];
    auto sp_to = _pnd.sp_on_target_.value()[_strip.pn_vhs.back()];

    auto A = sp_from.get_pos(_strip.heh_pos_2d);
    auto B = sp_to.get_pos(_strip.heh_pos_2d);

    auto layout_seg = torch::stack({A, B});

    for (size_t i = 1; i < _strip.pn_vhs.size() - 1; ++i)
    {
        auto intersect_pn_vh = _strip.pn_vhs[i];
        auto intersect_hh = _tmd.mesh_->handle_of(_pnd.sp_on_target_.value()[intersect_pn_vh].heh_idx);

        auto from_2D = _strip.heh_pos_2d[intersect_hh];
        auto to_2D = _strip.heh_pos_2d[intersect_hh.opposite()];

        auto from_3D = _tmd.torch_pos_[intersect_hh.vertex_from().idx.value];
        auto to_3D = _tmd.torch_pos_[intersect_hh.vertex_to().idx.value];

        auto intersect_seg = torch::stack({to_2D, from_2D});
        auto params = torch_compute_intersection_parameter(intersect_seg, layout_seg);

        // DEBUG
        // if ((params[0].item<double>() < 0) || (params[0].item<double>() > 1.0))
        // {
        //     DEBUG_VAR(params[0].item<double>())
        //     DEBUG_VAR(params[1].item<double>())
        //     // DEBUG
        //     auto c = gv::canvas();
        //     auto cg = ColorGenerator();
        //     auto tg_A_2D = tg::pos3(torch_to_pos2(A));
        //     auto tg_B_2D = tg::pos3(torch_to_pos2(B));


        //     auto tg_from_2D = tg::pos3(torch_to_pos2(from_2D));
        //     auto tg_to_2D = tg::pos3(torch_to_pos2(to_2D));

        //     c.add_line(tg_A_2D, tg_B_2D, MAGENTA);

        //     c.add_line(tg_from_2D, tg_to_2D, GREEN);

        //     for (auto fh : _strip.target_fhs)
        //     {
        //         std::vector<pos3> positions;
        //         for (auto heh : fh.halfedges())
        //         {
        //             auto pos = pos3(torch_to_pos2(_strip.heh_pos_2d[heh]));
        //             positions.push_back(pos);
        //         }
        //         auto color = cg.generate_next_color();
        //         c.add_face(positions[0], positions[1], positions[2], color);
        //     }
        // }
        // END
        // assert(params[0].item<double>() >= 0 && params[0].item<double>() <= 1.0);

        auto intersect_pos = from_3D * params[1] + to_3D * params[0];
        auto intersect_o_vh = _pnd.map_to_overlay_vertices_.value()[intersect_pn_vh];

        _omd.torch_pos_.value()[intersect_o_vh.value] = intersect_pos;
    }
}

} // namespace LayoutOpt
