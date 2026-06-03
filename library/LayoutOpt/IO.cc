#include "IO.hh"
#include <fstream>
#include "LayoutOpt/TorchUtils.hh"
#include "LayoutOpt/Utils/Debug.hh"
#include "LayoutOpt/Visualization/ColorsMaps.hh"
#include "polymesh/formats.hh"

#include <glow-extras/viewer/canvas.hh>

namespace LayoutOpt
{

void write_info(std::filesystem::path const& _path, double _loss)
{
    std::ofstream file(_path, std::ios::app); // open in append mode
    if (file)
    {
        file << _loss << '\n';
    }
}

void write_state(std::filesystem::path const& _path, TargetMeshData const& _tmd, LayoutData const& _ld)
{
    DEBUG_OUT("===================write_state=====================");
    namespace fs = std::filesystem;
    auto t_path = _path / "state_t.obj";
    pm::save(t_path.c_str(), _tmd.pos_);

    auto l_path = _path / "state_l.obj";
    pm::save(l_path.c_str(), _ld.pos_);
}

void write_pathnetwork(std::filesystem::path const& _path, TargetMeshData const& _tmd, PathNetworkData const& _pnd)
{
    auto path = _path / "state_pn.obj";

    auto pos = _pnd.mesh_->vertices().make_attribute<pos3>();
    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        pos[pn_vh] = torch_to_pos3(_pnd.sp_on_target_.value()[pn_vh].get_pos(_tmd.torch_pos_, *_tmd.mesh_.get()));
        pm::save(path.c_str(), pos);
    }
}

void write_overlay_and_mask(std::filesystem::path const& _path, OverlayMeshData const& _omd)
{
    if (!_omd.patch_boundary_mask_)
    {
        throw std::runtime_error("patch_boundary_mask_ is not set");
    }
    if (!_omd.map_to_layout_faces_)
    {
        throw std::runtime_error("map_to_layout_faces_ is not set");
    }
    // 1. Write boundary.txt
    {
        std::ofstream ofs(_path / "boundary.txt");
        if (!ofs)
            throw std::runtime_error("Cannot open boundary.txt for writing");
        for (auto e : _omd.mesh_->edges())
        {
            ofs << ((_omd.patch_boundary_mask_).value()[e] ? 1 : 0) << "\n";
        }
    }
    // 2. Write label.txt (map_to_layout_faces_)
    {
        std::ofstream ofs(_path / "label.txt");
        if (!ofs)
            throw std::runtime_error("Cannot open label.txt for writing");
        for (auto f : _omd.mesh_->faces())
        {
            ofs << (_omd.map_to_layout_faces_).value()[f].value << "\n";
        }
    }

    // 3. write mesh
    namespace fs = std::filesystem;
    auto t_path = _path / "state_o.obj";
    pm::save(t_path.c_str(), _omd.pos_);
}

void write_target_labels_per_vertex(std::filesystem::path const& _path, TargetMeshData const& _tmd, OverlayMeshData const& _omd)
{
    auto target_label = _tmd.mesh_->vertices().make_attribute(-1);
    for (auto o_vh : _omd.mesh_->vertices())
    {
        auto pn_vh = _omd.map_to_pn_vertices_.value()[o_vh];
        if (pn_vh.is_valid())
            continue; // it is not a target vertex

        auto t_vh = _tmd.mesh_->handle_of(o_vh);
        target_label[t_vh] = _omd.map_to_layout_faces_.value()[o_vh.any_face()].value;
    }

    {
        std::ofstream ofs(_path / "t_vertex_label.txt");
        if (!ofs)
            throw std::runtime_error("Cannot open label.txt for writing");
        for (auto t_vh : _tmd.mesh_->vertices())
        {
            ofs << target_label[t_vh] << "\n";
        }
    }

    // 3. write mesh
    namespace fs = std::filesystem;
    auto t_path = _path / "state_t.obj";
    pm::save(t_path.c_str(), _tmd.pos_);


    std::ifstream ifs(_path / "t_vertex_label.txt");
    pm::vertex_attribute<int> labels = _tmd.mesh_->vertices().make_attribute(-1);

    int value;
    for (auto t_vh : _tmd.mesh_->vertices())
    {
        if (!(ifs >> value))
        {
            break;
            throw std::runtime_error("Not enough entries in t_label.txt");
        }
        else
        {
            labels[t_vh] = value;
        }
    }

    auto colors = apply_colormap(labels);

    {
        auto v = gv::view(_tmd.pos_, colors);
    }
}

void write_overlay_labels_per_face(std::filesystem::path const& _path, OverlayMeshData const& _omd)
{
    {
        std::ofstream ofs(_path / "l_flabel.txt");
        if (!ofs)
            throw std::runtime_error("Cannot open label.txt for writing");
        for (auto t_fh : _omd.mesh_->faces())
        {
            ofs << _omd.map_to_layout_faces_.value()[t_fh].value << "\n";
        }
    }

    // 3. write mesh
    namespace fs = std::filesystem;
    auto t_path = _path / "state_o.obj";
    pm::save(t_path.c_str(), _omd.pos_);


    std::ifstream ifs(_path / "l_flabel.txt");
    pm::face_attribute<int> labels = _omd.mesh_->faces().make_attribute(-1);

    int value;
    for (auto t_fh : _omd.mesh_->faces())
    {
        if (!(ifs >> value))
        {
            break;
            throw std::runtime_error("Not enough entries in t_label.txt");
        }
        else
        {
            labels[t_fh] = value;
        }
    }

    auto colors = apply_colormap(labels);

    {
        auto v = gv::view(_omd.pos_, colors);
    }
}


void write_uvs(std::filesystem::path const& _path, pm::vertex_attribute<pos3> _pos, pm::halfedge_attribute<pos2> _uvs)
{
    std::ofstream out(_path);
    if (!out)
        return;

    auto const& m = _uvs.mesh();
    auto obj_idx = m.vertices().make_attribute<int>();
    int next_v = 1;
    for (auto v : m.vertices())
    {
        auto p = _pos[v];
        out << "v " << p.x << " " << p.y << " " << p.z << "\n";
        obj_idx[v] = next_v++;
    }

    // 2. write UVs per halfedge and store indices
    auto uv_idx = m.halfedges().make_attribute<int>(); // create if not exists
    int next_uv = 1;
    for (auto h : m.halfedges())
    {
        auto uv = _uvs[h];
        out << "vt " << uv.x << " " << uv.y << "\n";
        uv_idx[h] = next_uv++;
    }
    // 3. write faces
    for (auto f : m.faces())
    {
        out << "f";
        for (auto h : f.halfedges())
        {
            int v_id = obj_idx[h.vertex_from()]; // vertex index from step 1
            int uv_id = uv_idx[h];   // UV index from step 2
            out << " " << v_id << "/" << uv_id;
        }
            out << "\n";
    }
}
} // namespace LayoutOpt
