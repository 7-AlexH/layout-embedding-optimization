#include "QuadMeshing.hh"
#include <cassert>
#include <glow-extras/viewer/canvas.hh>
#include "ExactPredicates.h"
#include "LayoutOpt/Visualization/Colors.hh"
#include "polymesh/properties.hh"

namespace LayoutOpt
{
namespace
{

std::pair<double, double> compute_bary(pos2 const& _p, pos2 const& _a, pos2 const& _b, pos2 const& _c)
{
    auto const va = _a - _c;
    auto const vb = _b - _c;
    auto const vp = _p - _c;

    double const d00 = tg::dot(va, va);
    double const d01 = tg::dot(va, vb);
    double const d02 = tg::dot(va, vp);
    double const d11 = tg::dot(vb, vb);
    double const d12 = tg::dot(vb, vp);

    double const denom = d00 * d11 - d01 * d01;

    double const alpha = (d02 * d11 - d01 * d12) / denom;
    double const beta = (d00 * d12 - d01 * d02) / denom;

    return std::make_pair(alpha, beta);
}

double const* ptr(tg::dpos2 const& _p) { return &_p.x; }

bool in_triangle_inclusive(pos2 const& _p, pos2 _a, pos2 _b, pos2 _c, double const _scale = 1.0)
{
    if (_scale != 1.0)
    {
        auto const cog = tg::average(std::vector<pos2>{_a, _b, _c});
        auto const M = tg::translation(cog) * tg::scaling(_scale, _scale) * tg::translation(-cog);
        _a = M * _a;
        _b = M * _b;
        _c = M * _c;
    }

    return orient2d(ptr(_p), ptr(_a), ptr(_b)) >= 0 && orient2d(ptr(_p), ptr(_b), ptr(_c)) >= 0 && orient2d(ptr(_p), ptr(_c), ptr(_a)) >= 0;
}

pos3 point_on_surface(tg::dpos2 const& _p, std::vector<pm::face_handle> const& _t_patch, pm::vertex_attribute<pos3> const& _pos, HalfedgeParam const& _param)
{
    // To fix numerical issues at the patch boundary,
    // try the lookup a few times while slowly growing each individual triangle.
    int const n_attempts = 3;
    double scale = 1.0;
    double const eps = 1e-6;
    for (int i = 0; i < n_attempts; ++i)
    {
        for (auto t_f : _t_patch)
        {
            assert(t_f.halfedges().size() == 3);
            auto const ha = t_f.halfedges().first(); // pointing to vertex a
            auto const hb = ha.next();               // pointing to vertex b
            auto const hc = hb.next();               // pointing to vertex c

            if (in_triangle_inclusive(_p, _param[ha], _param[hb], _param[hc], scale))
            {
                auto [alpha, beta] = compute_bary(_p, _param[ha], _param[hb], _param[hc]);

                // if(alpha > 1.01 || alpha < - 0.01)
                // {
                //     auto c = gv::canvas();

                //     for(auto fh : _t_patch)
                //     {
                //         for(auto heh : fh.halfedges())
                //         {
                //             c.add_line(pos3(_param[heh]), pos3(_param[heh.next()]), BLUE).size(1.0);
                //         }
                //     }

                //     c.add_face(pos3(_param[ha]), pos3(_param[hb]), pos3(_param[hc]), BLUE_50);
                //     c.add_point(pos3(_p), RED);

                //     DEBUG_VAR(alpha)
                //     DEBUG_VAR(beta)
                // }
                // if(beta > 1.01 || beta < - 0.01)
                // {
                //     DEBUG_VAR(alpha)
                //     DEBUG_VAR(beta)
                // }

                if (alpha > 1.0 + EPS || alpha < 0.0 - EPS || beta > 1.0 + EPS || beta < 0.0 - EPS) //!std::isfinite(alpha) || !std::isfinite(beta))
                {
                    alpha = 1.0 / 3.0;
                    beta = 1.0 / 3.0;
                    //std::cout << "Computing barycentric coordinates failed due to degenerate triangle." << std::endl;
                }
                return alpha * _pos[ha.vertex_to()] + beta * _pos[hb.vertex_to()] + (1.0 - alpha - beta) * _pos[hc.vertex_to()];
            }
        }
        scale *= 1.0 + eps;
    }
    std::runtime_error("Triangle lookup failed");
    return pos3();
}

/// Angle at to-vertex between given and next halfedge
auto calc_sector_angle(pm::vertex_attribute<pos3>& _pos, pm::halfedge_handle _h)
{
    auto const v1 = _pos[_h.next().vertex_to()] - _pos[_h.vertex_to()];
    auto const v2 = _pos[_h.vertex_from()] - _pos[_h.vertex_to()];
    return tg::angle_between(v1, v2);
}

double mean_value_weight(pm::vertex_attribute<pos3>& _pos, pm::halfedge_handle _h)
{
    if (_h.edge().is_boundary())
        return 0.0;

    auto const angle_l = calc_sector_angle(_pos, _h.prev());
    auto const angle_r = calc_sector_angle(_pos, _h.opposite());
    auto const edge_length = pm::edge_length(_h, _pos);
    double w_ij = (tan(angle_l.radians() / 2.0) + tan(angle_r.radians() / 2.0)) / edge_length;

    if (w_ij <= 0.0)
        w_ij = 1e-5;

    return w_ij;
}

tg::ipos2 snap(tg::dpos2 const& uv)
{
    auto const res = tg::ipos2(lround(uv.x), lround(uv.y));

    // LE_ASSERT_EPS(res.x, uv.x, 1e-6);
    // LE_ASSERT_EPS(res.y, uv.y, 1e-6);

    return res;
}

int count_subdiv(LoopSubdivData const& _lsd, pm::halfedge_handle const& arc_heh, HalfedgeParam const& _param)
{
    auto const path = _lsd.map_arc_to_overlay_[arc_heh];
    auto const h_first = pm::halfedge_from_to(path[0], path[1]);
    auto const h_last = pm::halfedge_from_to(path[path.size() - 2], path[path.size() - 1]);
    assert(h_first.is_valid());
    assert(h_last.is_valid());

    auto const uv_from = snap(_param[h_first.prev()]);
    auto const uv_to = snap(_param[h_last]);
    if (uv_from.x == uv_to.x)
        return abs(uv_to.y - uv_from.y) - 1;
    else if (uv_from.y == uv_to.y)
        return abs(uv_to.x - uv_from.x) - 1;
    else
        throw std::logic_error("count_subdiv: arc is neither axis-aligned in u nor v");
}


void extract_patch(OverlayMeshData const& _omd,
                   pm::face_handle& _l_f,
                   pm::Mesh& _patch,
                   pm::vertex_attribute<pos3>& _patch_pos,
                   pm::vertex_attribute<pm::vertex_handle>& _v_overlay_to_patch,
                   pm::halfedge_attribute<pm::halfedge_handle>& _h_patch_to_overlay)
{
    // Init result
    _patch.clear();
    _patch_pos = _patch.vertices().make_attribute<pos3>();

    // Index maps
    _v_overlay_to_patch = _omd.mesh_->vertices().make_attribute<pm::vertex_handle>();
    _h_patch_to_overlay = _patch.halfedges().make_attribute<pm::halfedge_handle>();

    // Create region mesh
    for (auto o_fh : _omd.mesh_->faces())
    {
        if (_omd.map_to_layout_faces_.value()[o_fh].value != _l_f.idx.value)
            continue;

        // Add vertices to result mesh
        for (auto o_v : o_fh.vertices())
        {
            if (_v_overlay_to_patch[o_v].is_invalid())
            {
                auto r_v = _patch.vertices().add();
                _patch_pos[r_v] = _omd.pos_[o_v];
                _v_overlay_to_patch[o_v] = r_v;
            }
        }
        // Add face to result mesh
        _patch.faces().add(o_fh.vertices().to_vector([&](auto o_v) { return _v_overlay_to_patch[o_v]; }));

        // Fill halfedge index map
        for (auto o_h : o_fh.halfedges())
        {
            auto const r_v_from = _v_overlay_to_patch[o_h.vertex_from()];
            auto const r_v_to = _v_overlay_to_patch[o_h.vertex_to()];
            auto const r_h = pm::halfedge_from_to(r_v_from, r_v_to);
            assert(r_h.is_valid());

            _h_patch_to_overlay[r_h] = o_h;
            _h_patch_to_overlay[r_h.opposite()] = o_h.opposite();
        }
    }
}
}

HalfedgeParam parametrize_patches(OverlayMeshData const& _omd, LoopSubdivData const& _lsd)
{
    auto param = _omd.mesh_->halfedges().make_attribute<tg::dpos2>();

    // Ensure that _l_subdivisions are loop-wise consistent
    for (auto arc_h : _lsd.arc_mesh_->halfedges())
    {
        auto arc_e = arc_h.edge();
        auto arc_e_opp = arc_h.next().next().edge();
        assert(_lsd.arc_subdivs_[arc_e] == _lsd.arc_subdivs_[arc_e_opp]);
    }

    for (auto arc_f : _lsd.arc_mesh_->faces())
    {
        assert(arc_f.vertices().size() == 4);

        // Extract patch mesh
        pm::Mesh p_m;
        pm::vertex_attribute<pos3> p_pos;
        pm::vertex_attribute<pm::vertex_handle> v_overlay_to_patch;
        pm::halfedge_attribute<pm::halfedge_handle> h_patch_to_overlay;
        extract_patch(_omd, arc_f, p_m, p_pos, v_overlay_to_patch, h_patch_to_overlay);

        // Constrain patch boundary to rectangle
        auto p_constrained = p_m.vertices().make_attribute<bool>(false);
        auto p_constraint_value = p_m.vertices().make_attribute<tg::dpos2>();

        double const width = _lsd.arc_subdivs_[arc_f.halfedges().first().edge()] + 1.0;
        double const height = _lsd.arc_subdivs_[arc_f.halfedges().last().edge()] + 1.0;
        std::vector<tg::dpos2> const corners = {{0.0, 0.0}, {width, 0.0}, {width, height}, {0.0, height}};


        auto cd = gv::canvas_data();
        cd.add_faces(p_pos, GREEN_50);
        cd.add_lines(p_pos, GREEN);

        int corner_idx = 0;
        for (auto arc_heh : arc_f.halfedges())
        {
            double const length_total = _lsd.arc_lengths_(arc_heh);
            double length_acc = 0.0;

            auto const& o_path_vertices = _lsd.map_arc_to_overlay_[arc_heh];

            for (int i = 0; i < o_path_vertices.size() - 1; ++i)
            {
                auto const t_vi = o_path_vertices[i];
                auto const t_vj = o_path_vertices[i + 1];
                double const lambda_i = length_acc / length_total;
                length_acc += tg::length(_omd.pos_[t_vi] - _omd.pos_[t_vj]);

                auto const p_vi = v_overlay_to_patch[t_vi];

                cd.add_point(p_pos[p_vi], RED).size(10);

                p_constrained[p_vi] = true;
                p_constraint_value[p_vi] = (1.0 - lambda_i) * corners[corner_idx] + lambda_i * corners[(corner_idx + 1) % 4];
            }
            ++corner_idx;
            // {
            //     auto c = gv::canvas();
            //     c.add_data(cd);
            // }
        }


        // Compute Tutte embedding
        // Try a few times with successively more uniform weights
        VertexParam p_param;
        if (!harmonic_parametrization(p_pos, p_constrained, p_constraint_value, p_param, LaplaceWeights::MeanValue, false))
        {
            if (!harmonic_parametrization(p_pos, p_constrained, p_constraint_value, p_param, LaplaceWeights::Uniform, true))
            {
                std::runtime_error("Harmonic parametrization failed.");
            }
        }

        for (auto v : p_m.vertices())
        {
            assert(std::isfinite(p_param[v].x));
            assert(std::isfinite(p_param[v].y));
        }

        // Transfer parametrization to target mesh
        for (auto p_h : p_m.halfedges())
        {
            if (!p_h.is_boundary())
                param[h_patch_to_overlay[p_h]] = p_param[p_h.vertex_to()];
        }

        // {
        //     auto c = gv::canvas();
        //     c.add_faces(p_param, BLUE_25);
        //     c.add_lines(p_param, BLUE);
        // }
    }
    return param;
}


bool harmonic(pm::vertex_attribute<pos3>& _pos,
              pm::vertex_attribute<bool>& _constrained,
              Eigen::MatrixXd& _constraint_values,
              Eigen::MatrixXd& _res,
              LaplaceWeights const _weights,
              bool const _fallback_iterative)
{
    assert(_pos.mesh().is_compact());


    int const n = _pos.mesh().vertices().size();
    int const d = _constraint_values.cols();
    assert(_constraint_values.rows() == n);

    // Set up Laplace matrix and rhs
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n, d);
    std::vector<Eigen::Triplet<double>> triplets;
    for (auto v : _pos.mesh().vertices())
    {
        int const i = v.idx.value;

        if (_constrained[v])
        {
            triplets.push_back(Eigen::Triplet<double>(i, i, 1.0));
            rhs.row(i) = _constraint_values.row(i);
        }
        else
        {
            assert(!v.is_boundary());

            for (auto h : v.outgoing_halfedges())
            {
                int const j = h.vertex_to().idx.value;
                double w_ij;
                if (_weights == LaplaceWeights::Uniform)
                    w_ij = 1.0;
                else if (_weights == LaplaceWeights::MeanValue)
                    w_ij = mean_value_weight(_pos, h);
                else
                    std::runtime_error("");

                triplets.push_back(Eigen::Triplet<double>(i, j, w_ij));
                triplets.push_back(Eigen::Triplet<double>(i, i, -w_ij));
            }
        }
    }

    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(L);
    if (solver.info() == Eigen::Success)
    {
        _res = solver.solve(rhs);
        if (solver.info() == Eigen::Success)
            return true;
    }

    std::cout << "LU solve failed" << std::endl;

    if (_fallback_iterative)
    {
        std::cout << "Falling back to iterative solver" << std::endl;
    }

    return false;
}

bool harmonic_parametrization(pm::vertex_attribute<pos3>& _pos,
                              pm::vertex_attribute<bool>& _constrained,
                              VertexParam const& _constraint_values,
                              VertexParam& _res,
                              LaplaceWeights const _weights,
                              bool const _fallback_iterative)
{
    int const n = _pos.mesh().vertices().size();
    int const d = 2;

    // Convert constraints
    Eigen::MatrixXd constraint_values = Eigen::MatrixXd::Zero(n, d);
    for (auto v : _pos.mesh().vertices())
        constraint_values.row(v.idx.value) = Eigen::Vector2d(_constraint_values[v].x, _constraint_values[v].y);

    // Compute
    Eigen::MatrixXd res_mat;
    if (!harmonic(_pos, _constrained, constraint_values, res_mat, _weights, _fallback_iterative))
        return false;

    // Convert result
    _res = _pos.mesh().vertices().make_attribute<tg::dpos2>();
    for (auto v : _pos.mesh().vertices())
        _res[v] = tg::dpos2(res_mat(v.idx.value, 0), res_mat(v.idx.value, 1));

    return true;
}

pm::vertex_attribute<pos3> extract_quad_mesh(OverlayMeshData const& _omd,
                                             LoopSubdivData const& _lsd,
                                             HalfedgeParam const& _param,
                                             polymesh::Mesh& _q,
                                             pm::face_attribute<polymesh::face_index>& _q_matching_layout_face)
{
    exactinit();

    _q.clear();
    auto q_pos = _q.vertices().make_attribute<pos3>();
    _q_matching_layout_face = _q.faces().make_attribute<pm::face_index>();

    // Per layout vertex, cache vertex index.
    // Per layout halfedge, cache list of vertex indices. First and last stay unused.
    auto vv_cache = _lsd.arc_mesh_->vertices().make_attribute<pm::vertex_handle>();
    auto hv_cache = _lsd.arc_mesh_->halfedges().make_attribute<std::vector<pm::vertex_handle>>();

    auto cd = gv::canvas_data();
    cd.add_faces(_omd.pos_, GREEN_50);

    for (auto arc_fh : _lsd.arc_mesh_->faces())
    {
        // Determine patch dimensions
        auto const l_h_u = arc_fh.any_halfedge(); // u direction
        auto const l_h_v = l_h_u.next();

        int const n_u = count_subdiv(_lsd, l_h_u, _param) + 2;
        int const n_v = count_subdiv(_lsd, l_h_v, _param) + 2;

        // Per layout face, cache grid of vertex indices.
        // First halfedge defines u direction.
        std::vector<std::vector<pm::vertex_handle>> fv_cache(n_u, std::vector<pm::vertex_handle>(n_v));

        // Enumerate patch vertices
        for (int u = 0; u < n_u; ++u)
        {
            for (int v = 0; v < n_v; ++v)
            {
                // Look-up vertex in cache
                pm::vertex_handle q_v;
                if ((u == 0 || u == n_u - 1) && (v == 0 || v == n_v - 1))
                {
                    // Patch vertex
                    pm::vertex_handle l_v;
                    if (u == 0 && v == 0)
                        l_v = l_h_u.vertex_from();
                    else if (u == n_u - 1 && v == 0)
                        l_v = l_h_u.vertex_to();
                    else if (u == n_u - 1 && v == n_v - 1)
                        l_v = l_h_v.vertex_to();
                    else if (u == 0 && v == n_v - 1)
                        l_v = l_h_v.next().vertex_to();
                    else
                        std::logic_error("");

                    // Cache lookup
                    q_v = vv_cache[l_v];
                    if (q_v.is_invalid())
                    {
                        q_v = _q.vertices().add();
                        vv_cache[l_v] = q_v;
                    }
                }
                else if (u == 0 || u == n_u - 1 || v == 0 || v == n_v - 1)
                {
                    // Patch boundary interior vertex
                    pm::halfedge_handle l_h;
                    int idx = -1;
                    int n = -1;
                    if (v == 0)
                    {
                        l_h = l_h_u;
                        idx = u;
                        n = n_u;
                    }
                    else if (u == n_u - 1)
                    {
                        l_h = l_h_v;
                        idx = v;
                        n = n_v;
                    }
                    else if (v == n_v - 1)
                    {
                        l_h = l_h_v.next();
                        idx = n_u - 1 - u;
                        n = n_u;
                    }
                    else if (u == 0)
                    {
                        l_h = l_h_u.prev();
                        idx = n_v - 1 - v;
                        n = n_v;
                    }
                    else
                        std::logic_error("");

                    assert(idx >= 1);
                    assert(idx < n - 1);

                    // Init halfedge cache vectors
                    if (hv_cache[l_h].empty())
                        hv_cache[l_h] = std::vector<pm::vertex_handle>(n);
                    auto const l_h_opp = l_h.opposite();
                    if (hv_cache[l_h_opp].empty())
                        hv_cache[l_h_opp] = std::vector<pm::vertex_handle>(n);

                    assert(hv_cache[l_h].size() == n);
                    assert(hv_cache[l_h_opp].size() == n);

                    // Cache lookup
                    assert(hv_cache[l_h][idx] == hv_cache[l_h_opp][n - 1 - idx]);
                    q_v = hv_cache[l_h][idx];
                    if (q_v.is_invalid())
                    {
                        q_v = _q.vertices().add();
                        hv_cache[l_h][idx] = q_v;
                        hv_cache[l_h_opp][n - 1 - idx] = q_v;
                    }
                }
                else
                {
                    // Patch interior vertex
                    q_v = _q.vertices().add();
                }

                // Get patch target triangles
                auto const o_patch
                    = _omd.mesh_->faces().filter([&](FH o_fh) { return _omd.map_to_layout_faces_.value()[o_fh].value == arc_fh.idx.value;}).to_vector();
                //  LE_ASSERT(!t_patch.empty());

                // Compute position
                auto const p_param = tg::dpos2((double)u, (double)v);

                if(u == 0 || u == n_u - 1 || v == 0 || v == n_v - 1) // special handeling for boundary
                {
                    auto find_pos_along_arc = [&](const std::vector<VH>& points_on_overlay, HEH arc_heh,  int u_or_v)
                    {
                        const auto arc_length = _lsd.arc_lengths_[arc_heh];
                        auto target_length = (double(u_or_v) / double(_lsd.arc_subdivs_[arc_heh]+ 1)) * arc_length;

                        // DEBUG_VAR(u_or_v)
                        // DEBUG_VAR(_lsd.arc_subdivs_[arc_heh]+ 1)
                        // DEBUG_VAR(arc_length)
                        // DEBUG_VAR(target_length)

                        auto traveled_length = 0.;
                        for(int i = 0; i < points_on_overlay.size() - 1; ++i)
                        {
                            auto edge_length = tg::length(_omd.pos_[points_on_overlay[i + 1]]  - _omd.pos_[points_on_overlay[i]]);
                            traveled_length += edge_length;
                            if(traveled_length > target_length)
                            {
                                auto extra = traveled_length - target_length;
                                double alpha =  1.0 - extra/edge_length;
                                q_pos[q_v] = tg::lerp( _omd.pos_[points_on_overlay[i]],  _omd.pos_[points_on_overlay[i + 1]], alpha);
                                return;
                            }
                        }
                    };
                    HEH  arc_heh;
                    if(v == 0)
                    {
                        arc_heh = l_h_u;
                        const auto& points_on_overlay = _lsd.map_arc_to_overlay_[arc_heh];
                        if(u == 0)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.front()];

                        }
                        else if(u == n_u - 1)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.back()];
                        }
                        else
                        {
                            find_pos_along_arc(points_on_overlay, arc_heh, u);
                        }
                    }
                    if(u == n_u - 1)
                    {
                        arc_heh = l_h_u.next();
                        const auto& points_on_overlay = _lsd.map_arc_to_overlay_[arc_heh];
                        if(v == 0)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.front()];

                        }
                        else if(v == n_v - 1)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.back()];
                        }
                        else
                        {
                            find_pos_along_arc(points_on_overlay, arc_heh, v);
                        }
                    }
                    if(v == n_v - 1)
                    {
                        arc_heh = l_h_u.next().next().opposite(); //to account for direction change
                        const auto& points_on_overlay = _lsd.map_arc_to_overlay_[arc_heh];
                        if(u == 0)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.front()];

                        }
                        else if(u == n_u - 1)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.back()];
                        }
                        else
                        {
                            find_pos_along_arc(points_on_overlay, arc_heh, u);
                        }
                    }
                    if(u == 0)
                    {
                        arc_heh = l_h_u.next().next().next().opposite(); // to account for direction change
                        const auto& points_on_overlay = _lsd.map_arc_to_overlay_[arc_heh];
                        if(v == 0)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.front()];

                        }
                        else if(v == n_v - 1)
                        {
                            q_pos[q_v] = _omd.pos_[points_on_overlay.back()];
                        }
                        else
                        {
                            find_pos_along_arc(points_on_overlay, arc_heh, v);
                        }
                    }
                }
                else // innver vertex
                {
                    q_pos[q_v] = point_on_surface(p_param, o_patch, _omd.pos_, _param);
                }

                cd.add_point(pos3(q_pos[q_v]), RED).size(10);

                // Add vertex to cache
                fv_cache[u][v] = q_v;

                // Add face
                if (u >= 1 && v >= 1)
                {
                    auto const q_f = _q.faces().add(fv_cache[u - 1][v - 1], fv_cache[u][v - 1], fv_cache[u][v], fv_cache[u - 1][v]);

                    _q_matching_layout_face[q_f] = arc_fh.idx;
                }
            }
        }
    }

    // Assert no boundaries
    for (auto e : _q.edges())
        assert(!e.is_boundary());

    for (auto v : _q.vertices())
    {
        assert(std::isfinite(q_pos[v].x));
        assert(std::isfinite(q_pos[v].y));
        assert(std::isfinite(q_pos[v].z));
    }

    return q_pos;
}
}
