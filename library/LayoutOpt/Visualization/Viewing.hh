#pragma once
#include <LayoutOpt/DataStructures/Types.hh>
#include "Colors.hh"
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/ObjectiveFunctions.hh"

#include <glow-extras/glfw/GlfwContext.hh>
#include <glow-extras/viewer/view.hh>

namespace LayoutOpt
{

inline auto default_style()
{
    // Implemented inline so we can use 'auto' because the returned type is an implementation detail.
    return gv::config(gv::dark_ui, gv::no_grid, gv::no_outline, gv::shadow_strength(0.5f), gv::background_color(tg::color3::white),
                      gv::ssao_power(1.5f), gv::shadow_screen_fadeout_distance(50.f));
}
inline auto default_style_no_shadow()
{
    // Implemented inline so we can use 'auto' because the returned type is an implementation detail.
    return gv::config(gv::dark_ui, gv::no_grid, gv::no_outline, gv::background_color(tg::color3::white), gv::ssao_power(1.0f),
                      gv::shadow_screen_fadeout_distance(50.f), gv::no_shadow, gv::no_backfacing_shadow);
}

struct ViewingOptions
{
    bool view_vertices = true;
    bool view_edges = true;
    bool view_faces = true;

    float line_width = 1.0;
    float point_size = 8.0;

    tg::color4 vertex_color = BLUE;
    tg::color4 edge_color = BLUE;
    tg::color4 face_color = BLUE_75;
};

constexpr ViewingOptions vo_target_opaque = {false, true, true, 2.0, 10.0, BLUE, BLUE_75, BLUE_50};
constexpr ViewingOptions vo_target_qm_opaque = {false, true, true, 4.0, 10.0, GREEN, GREEN, GREEN_50};
constexpr ViewingOptions vo_target = {false, true, true, 0.5, 10.0, BLUE, BLUE_75, T3_BLUE_75};
constexpr ViewingOptions vo_layout = {true, true, true, 3.0, 10.0, MAGENTA, MAGENTA_75, MAGENTA_50};

void view_mesh(pm::vertex_attribute<pos3> const& _pos, ViewingOptions const& _vo = ViewingOptions());
void view_frame_field(pm::vertex_attribute<pos3> const& _pos);

void view_layout(LayoutData const& _ld);
void view_overlay(LayoutData const& _ld, OverlayMeshData const& _omd);
void view_overlay(LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo const& _eval_info);
void view_param(LayoutData const& _ld, OverlayMeshData const& _omd, EvalInfo const& _eval_info, FH _l_fh);


} // namespace LayoutOpt
