#include <filesystem>

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>

#include <LayoutOpt/Visualization/Viewing.hh>
#include <glow-extras/viewer/canvas.hh>

using namespace LayoutOpt;
namespace fs = std::filesystem;

int main()
{
    glow::glfw::GlfwContext ctx;
    auto style = default_style();
    constexpr bool open_viewer = true;
    GLOW_VIEWER_CONFIG(glow::viewer::camera_transform(tg::pos3(-0.248235f, 0.568524f, 1.544200f), tg::pos3(0.039091f, 0.068563f, -0.040690f)));
    //===================================
    // Paths
    //===================================
    fs::path const base_path = fs::path(DATA_PATH) / "Bunny";
    fs::path const mesh_path = base_path / "qm_theirs.obj";

    fs::path const layout_path = base_path / "bunny_layout.obj";

    //===================================
    // Load mesh
    //===================================
    pm::Mesh m;
    pm::vertex_attribute<pos3> pos(m);
    pm::load(mesh_path.string().c_str(), m, pos);

    //===================================
    // Detect singular vertices (valence != 4 in quad mesh)
    //===================================
    auto is_singular = m.vertices().make_attribute(false);
    is_singular.compute([](VH vh) { return vh.adjacent_vertices().size() != 4; });

    //===================================
    // Trace separatrices from each singular vertex
    //===================================
    auto is_separatrix = m.edges().make_attribute(false);
    for (auto vh : m.vertices())
    {
        if (!is_singular[vh])
            continue;

        for (auto heh : vh.outgoing_halfedges())
        {
            auto cur = heh;
            while (true)
            {
                is_separatrix[cur] = true;
                if (is_singular[cur.vertex_to()])
                    break;
                cur = cur.next().opposite().next();
            }
        }
    }

    // A patch-start vertex has all edges on separatrices
    auto is_patch_start = m.vertices().make_attribute(false);
    is_patch_start.compute([&](VH vh) { return vh.edges().all([&](EH eh) { return is_separatrix[eh]; }); });

    //===================================
    // Build layout mesh
    //===================================
    pm::Mesh layout;
    pm::vertex_attribute<pos3> layout_pos(layout);
    auto to_layout_vh = m.vertices().make_attribute<VH>();

    // Add a layout vertex for every singular mesh vertex
    for (auto vh : m.vertices())
    {
        if (!is_singular[vh])
            continue;
        auto l_vh = layout.vertices().add();
        to_layout_vh[vh] = l_vh;
        layout_pos[l_vh] = pos[vh];
    }

    // Add a layout vertex for every separatrix endpoint not yet added
    for (auto heh : m.halfedges())
    {
        if (!is_separatrix[heh])
            continue;
        auto target = heh.vertex_to();
        if (to_layout_vh[target].is_valid())
            continue;
        auto l_vh = layout.vertices().add();
        to_layout_vh[target] = l_vh;
        layout_pos[l_vh] = pos[target];
    }

    //===================================
    // Trace patch boundaries and add faces to the layout mesh
    //===================================
    auto visited = m.faces().make_attribute(false);
    for (auto vh : m.vertices())
    {
        if (!is_patch_start[vh])
            continue;

        for (auto heh : vh.outgoing_halfedges())
        {
            if (visited[heh.face()])
                continue;

            std::vector<VH> face_vhs;
            auto cur = heh;
            do
            {
                visited[cur.face()] = true;
                face_vhs.push_back(to_layout_vh[cur.vertex_from()]);
                cur = is_separatrix[cur.next()] ? cur.next() : cur.next().opposite().next();
            } while (cur != heh);

            assert(layout.faces().can_add(face_vhs));
            layout.faces().add(face_vhs);
        }
    }

    //===================================
    // Output
    //===================================
    if (open_viewer)
        view_mesh(layout_pos, vo_layout);

    pm::save(layout_path.string().c_str(), layout_pos);
}