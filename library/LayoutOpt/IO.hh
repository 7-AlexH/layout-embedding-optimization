#pragma once
#include <filesystem>
#include <polymesh/attributes.hh>
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"

namespace LayoutOpt
{
void write_info(std::filesystem::path const& _path, double _loss);

void write_state(std::filesystem::path const& _path, TargetMeshData const& _tmd, LayoutData const& _ld);

void write_pathnetwork(std::filesystem::path const& _path, TargetMeshData const& _tmd, PathNetworkData const& _pnd);

void write_overlay_and_mask(std::filesystem::path const& _path, OverlayMeshData const& _omd);

void write_target_labels_per_vertex(std::filesystem::path const& _path, const TargetMeshData& _tmd, const OverlayMeshData& _omd);
void write_overlay_labels_per_face(std::filesystem::path const& _path, const OverlayMeshData& _omd);

void write_uvs(std::filesystem::path const& _path, pm::vertex_attribute<pos3> _pos, pm::halfedge_attribute<pos2> _uvs);

} // namespace LayoutOpt
