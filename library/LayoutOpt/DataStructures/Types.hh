#pragma once

#include <LayoutOpt/Utils/Assert.hh>
#include <LayoutOpt/Utils/Debug.hh>

#include <clean-core/vector.hh>
#include <polymesh/Mesh.hh>
#include <typed-geometry/tg.hh>

#include <Eigen/Eigen>

#include <polymesh/std/io.hh>

namespace LayoutOpt
{

using VH = pm::vertex_handle;
using EH = pm::edge_handle;
using HEH = pm::halfedge_handle;
using FH = pm::face_handle;

// polymesh
using pos3 = tg::dpos3;
using pos2 = tg::dpos2;

using vec3 = tg::dvec3;
using vec2 = tg::dvec2;

// clean core
template <class T>
using vector = cc::vector<T>;

// Eigen
using vec3d = Eigen::Vector3d;
using vec2d = Eigen::Vector2d;
using vecXd = Eigen::VectorXd;

template <class T>
using vec3T = Eigen::Vector3<T>;
template <class T>
using vec2T = Eigen::Vector2<T>;

using mat2d = Eigen::Matrix2d;


//====================================
//          constants
//====================================
constexpr double SMALL_EPS = 1e-12;
constexpr double EPS = 1e-9;
constexpr double MEDIUM_EPS = 1e-6;
constexpr double LARGE_EPS = 1e-4;
constexpr double LARGER_EPS = 1e-2;

} // namespace LayoutOpt
