#include "Update.hh"

#include <cassert>
#include <vector>

#include "LayoutOpt/DataStructures/SurfacePoint.hh"
#include "LayoutOpt/GeomUtils.hh"

namespace LayoutOpt
{

pm::vertex_attribute<std::vector<SurfacePoint>> do_step(TargetMeshData const& _tmd, PathNetworkData& _pnd, pm::vertex_attribute<vec2d> const& _dirs, OptimizerData* _od)
{
    pm::vertex_attribute<std::vector<SurfacePoint>> traces(*_pnd.mesh_.get());

    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto const& cur_sp = _pnd.sp_on_target_.value()[pn_vh];

        auto heh_canonical_frame = _tmd.mesh_->handle_of(cur_sp.heh_idx);
        auto fh_canonical_frame = heh_canonical_frame.face();
        std::vector<VH> vhs_canonical_fame = {heh_canonical_frame.vertex_from(), heh_canonical_frame.vertex_to(), heh_canonical_frame.next().vertex_to()}; // A, B, C with C being the origin

        // step in bary coords (do_step runs post-collapse: every sp is a FacePoint)
        auto sp_new = cur_sp.copy();
        {
            vec3d const bary = cur_sp.bary_full();
            vec2d const& dir = _dirs[pn_vh];
            double const a_new = bary[0] + dir.x();
            double const b_new = bary[1] + dir.y();
            sp_new.bary_params = vec2d(a_new, b_new);
        }

        if (sp_new.is_inside_element())
        {
            _pnd.sp_on_target_.value()[pn_vh] = sp_new;
            continue;
        }

        // if the new sp is not inside the face, we have to unfold the strip to find the new face
        // build the first triangle in 2D
        Eigen::MatrixX2d target_pos_2D = Eigen::MatrixX2d::Zero((Eigen::Index)_tmd.mesh_->vertices().size(), 2);
        auto const face_2D_coords = compute_2D_face_embedding(heh_canonical_frame, _tmd.pos_mat_);
        for (size_t i = 0; i < vhs_canonical_fame.size(); ++i)
        {
            target_pos_2D.row(vhs_canonical_fame[i].idx.value) = face_2D_coords.row(i);
        }

        vec2d const from_point_2D = cur_sp.get_pos(target_pos_2D, *_tmd.mesh_.get());
        vec2d const to_point_2D = sp_new.get_pos(target_pos_2D, *_tmd.mesh_.get());

        auto iter_heh = HEH::invalid;
        // find first intersecting halfedge
        for (auto heh : fh_canonical_frame.halfedges())
        {
            vec2d const candidat_from = target_pos_2D.row(heh.vertex_from().idx.value).transpose();
            vec2d const candidat_to = target_pos_2D.row(heh.vertex_to().idx.value).transpose();
            auto const params_for_candidat_seg = compute_intersection_parameter(candidat_from, candidat_to, from_point_2D, to_point_2D);
            auto const params_for_update_seg = compute_intersection_parameter(from_point_2D, to_point_2D, candidat_from, candidat_to);

            if (!params_for_candidat_seg.has_value())
                continue;

            if (params_for_candidat_seg->minCoeff() >= 0.0 && params_for_candidat_seg->maxCoeff() <= 1.0)
            {
                if (params_for_update_seg->minCoeff() >= 0.0 - EPS && params_for_update_seg->maxCoeff() <= 1.0 + EPS)
                {
                    iter_heh = heh;
                    traces[pn_vh].push_back(SurfacePoint(vec2d((*params_for_candidat_seg)[0], 0.0), iter_heh, SurfacePointType::EdgePoint));
                    break;
                }
            }
        }
        assert(iter_heh.is_valid());

        FH fh_final;
        while (iter_heh.is_valid())
        {
            // compute 2D embedding of this fh
            //  compute height in 3D
            auto hh = iter_heh;
            auto hh_opp = hh.opposite();

            vec3d const fn_b = face_normal(hh_opp.face(), _tmd.pos_mat_);

            vec3d const posA = _tmd.pos_mat_.row(hh_opp.vertex_from().idx.value).transpose();
            vec3d const posB = _tmd.pos_mat_.row(hh_opp.vertex_to().idx.value).transpose();

            vec3d const hh_opp_vec_norm = normalized_eps(vec3d(posB - posA));
            vec3d const height_vec_norm = normalized_eps(fn_b.cross(hh_opp_vec_norm));

            vec3d const hh_opp_next_vec = _tmd.pos_mat_.row(hh_opp.next().vertex_to().idx.value).transpose() - _tmd.pos_mat_.row(hh_opp.next().vertex_from().idx.value).transpose();
            double const height_vec_norm_for_real = hh_opp_next_vec.dot(height_vec_norm);
            vec3d const height_vec = height_vec_norm_for_real * height_vec_norm;

            vec3d const proj_to_base = vec3d(_tmd.pos_mat_.row(hh_opp.next().vertex_to().idx.value).transpose()) - height_vec;
            double const vec_to_proj_length = (proj_to_base - posA).dot(hh_opp_vec_norm);

            // reconstruct in 2D
            vec2d const posA_2D = target_pos_2D.row(hh_opp.vertex_from().idx.value).transpose();
            vec2d const posB_2D = target_pos_2D.row(hh_opp.vertex_to().idx.value).transpose();

            vec2d const hh_opp_vec_norm_2D = normalized_eps(vec2d(posB_2D - posA_2D));

            vec2d const height_vec_norm_2D(-hh_opp_vec_norm_2D.y(), hh_opp_vec_norm_2D.x());
            vec2d const height_point = posA_2D + vec_to_proj_length * hh_opp_vec_norm_2D + height_vec_norm_for_real * height_vec_norm_2D;

            // // "opposite" vertex of hh
            auto opp_vh = hh.opposite().next().vertex_to();

            target_pos_2D.row(opp_vh.idx.value) = height_point.transpose();

            fh_final = iter_heh.opposite_face();
            auto test_hehs = {iter_heh.opposite().next(), iter_heh.opposite().next().next()};
            iter_heh = HEH::invalid;

            for (auto heh : test_hehs)
            {
                vec2d const candidat_from = target_pos_2D.row(heh.vertex_from().idx.value).transpose();
                vec2d const candidat_to = target_pos_2D.row(heh.vertex_to().idx.value).transpose();
                auto const params_for_candidat_seg = compute_intersection_parameter(candidat_from, candidat_to, from_point_2D, to_point_2D);
                auto const params_for_update_seg = compute_intersection_parameter(from_point_2D, to_point_2D, candidat_from, candidat_to);

                if (!params_for_candidat_seg.has_value())
                    continue;

                if (params_for_candidat_seg->minCoeff() >= 0.0 && params_for_candidat_seg->maxCoeff() <= 1.0)
                {
                    if (params_for_update_seg->minCoeff() >= 0.0 && params_for_update_seg->maxCoeff() <= 1.0)
                    {
                        iter_heh = heh;
                        traces[pn_vh].push_back(SurfacePoint(vec2d((*params_for_candidat_seg)[0], 0.0), iter_heh, SurfacePointType::EdgePoint));
                        break;
                    }
                }
            }
        }
        assert(fh_final.is_valid());
        auto heh_final = fh_final.any_halfedge();
        std::vector<VH> vhs_final = {heh_final.vertex_from(), heh_final.vertex_to(), heh_final.next().vertex_to()};

        if (_od != nullptr)
        {
            // parallel-transport Adam's first moment: interpret it as bary coords in
            // the canonical frame, walk the resulting vector into the final frame and
            // re-express it in the final face's bary coords.
            vec2d const& m = _od->m[pn_vh];
            vec2d const canonical_A = target_pos_2D.row(vhs_canonical_fame[0].idx.value).transpose();
            vec2d const canonical_B = target_pos_2D.row(vhs_canonical_fame[1].idx.value).transpose();
            vec2d const canonical_C = target_pos_2D.row(vhs_canonical_fame[2].idx.value).transpose();
            vec2d const adam_vec = m.x() * canonical_A + m.y() * canonical_B + (1.0 - m.x() - m.y()) * canonical_C;

            vec2d const final_origin = target_pos_2D.row(vhs_final[2].idx.value).transpose();

            // in new frame
            vec2d const adam_vec_to_in_new_frame = final_origin + adam_vec;

            vec3d const bc_in_new_frame = compute_bary_coords_2D(adam_vec_to_in_new_frame, heh_final, target_pos_2D);
            _od->m[pn_vh] = vec2d(bc_in_new_frame[0], bc_in_new_frame[1]);
        }

        vec3d const bc = compute_bary_coords_2D(to_point_2D, heh_final, target_pos_2D);
        sp_new = SurfacePoint(vec2d(bc[0], bc[1]), heh_final, SurfacePointType::FacePoint);
        _pnd.sp_on_target_.value()[pn_vh] = sp_new;
    }
    return traces;
}


} // namespace LayoutOpt
