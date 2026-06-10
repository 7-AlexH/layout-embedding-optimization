#include <filesystem>

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/Embedding.hh>
#include <LayoutOpt/EmbeddingUtils.hh>
#include <LayoutOpt/Visualization/ColorsMaps.hh>
#include <LayoutOpt/Visualization/Viewing.hh>
#include <glow-extras/viewer/canvas.hh>

#include "LayoutOpt/Init.hh"
#include "LayoutOpt/Resample.hh"

using namespace LayoutOpt;
namespace fs = std::filesystem;

int main()
{
    glow::glfw::GlfwContext ctx;
    auto style = default_style();

    //===================================
    // Paths
    //===================================

    fs::path const base_path = fs::path(DATA_PATH) / "Bunny";
    fs::path const mesh_path = base_path / "bunny.obj";
    fs::path const layout_path = base_path / "bunny_layout.obj";

    auto const cam_pos = gv::camera_transform(tg::pos3(-0.301326f, 0.335422f, 0.994961f), tg::pos3(-0.186834f, 0.243625f, 0.683726f));

    // other example
    //  fs::path base_path = fs::path(DATA_PATH) / "Cube/";
    //  fs::path mesh_path = base_path / "cube.obj";
    //  fs::path layout_path = base_path / "cube_layout.obj";
    //  auto const cam_pos = gv::camera_transform(tg::pos3(-1.024356f, 0.432752f, 0.493283f), tg::pos3(-0.820939f, 0.329959f, 0.383541f));

    // !note, if a state is not loading - check if the target and the layout have the same relative size and position!
    // bc the initalization is computed by nearest neighbours

    //===================================
    // Load and preprocess meshes
    //===================================

    pm::Mesh m;
    pm::vertex_attribute<pos3> pos(m);
    pm::load(mesh_path.string().c_str(), m, pos);

    pm::Mesh l;
    pm::vertex_attribute<pos3> l_pos(l);
    pm::load(layout_path.string().c_str(), l, l_pos);

    preprocess_target_and_layout(pos, l_pos);

    //===================================
    // Initialize embedding
    //===================================

    TargetMeshData tmd(pos);
    LayoutData ld(l_pos);
    PathNetworkData pnd(ld.pos_);
    OverlayMeshData omd(tmd.pos_);

    compute_layout_embedding_init(tmd, ld, pnd, omd, true);

    //===================================
    // Visualization
    //===================================

    auto g = gv::grid();
    {
        auto v = gv::view();
        v.configure(cam_pos);
        view_overlay(ld, omd);
    }
    {
        resample_layout(tmd, ld, pnd, omd, 1);
        auto style = default_style_no_shadow();
        auto v = gv::view();
        v.configure(cam_pos, gv::no_shadow);
        view_layout(ld);
    }

    return 0;
}