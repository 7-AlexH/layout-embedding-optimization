#pragma once
#include "LayoutOpt/DataStructures/LayoutEmbedding.hh"
#include "LayoutOpt/QuadMeshing/LoopSubdivision.hh"
#include "LayoutOpt/Resample.hh"

namespace LayoutOpt
{

//=============================================================================================
//=================================adapted from born et al.====================================
//=============================================================================================

using VertexParam = pm::vertex_attribute<tg::dpos2>;
using HalfedgeParam = pm::halfedge_attribute<tg::dpos2>;

/// Takes an embedded quad layout and a valid number of subdivisions
/// per edge. Returns an integer-grid map.
HalfedgeParam parametrize_patches(
    OverlayMeshData const& _omd,
    LoopSubdivData const& _lsd);

enum class LaplaceWeights
{
    Uniform,
    MeanValue,
};

/// Compute harmonic field using mean-value weights.
bool harmonic(
    pm::vertex_attribute<pos3>& _pos,
    pm::vertex_attribute<bool>& _constrained,
    Eigen::MatrixXd& _constraint_values,
    Eigen::MatrixXd& _res,
    LaplaceWeights const _weights,
    bool const _fallback_iterative = false);

/// Compute harmonic field using mean-value weights.
bool harmonic_parametrization(
    pm::vertex_attribute<pos3>& _pos,
    pm::vertex_attribute<bool>& _constrained,
    VertexParam const& _constraint_values,
    VertexParam& _res,
    LaplaceWeights const _weights,
    bool const _fallback_iterative = false);

/// Takes an integer-grid map and extracts a quad mesh.
pm::vertex_attribute<pos3> extract_quad_mesh(
    OverlayMeshData const& _omd,
    LoopSubdivData const& _lsd,
    HalfedgeParam const& _param,
    pm::Mesh& _q,
    pm::face_attribute<pm::face_index>& _q_matching_layout_face);


}
