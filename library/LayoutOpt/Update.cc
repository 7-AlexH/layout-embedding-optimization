#include "Update.hh"
#include <torch/nn/functional/normalization.h>
#include "LayoutOpt/DataStructures/SurfacePoint.hh"
#include "LayoutOpt/TorchUtils.hh"


#include <glow-extras/viewer/canvas.hh>
#include "LayoutOpt/Visualization/ColorGenerator.hh"
#include "LayoutOpt/Visualization/Colors.hh"

namespace LayoutOpt
{

pm::vertex_attribute<std::vector<SurfacePoint>> do_step(TargetMeshData const& _tmd, PathNetworkData& _pnd, pm::vertex_attribute<at::Tensor> const& _dirs, OptimizerData* _od)
{
    pm::vertex_attribute<std::vector<SurfacePoint>> traces(*_pnd.mesh_.get());

    for (auto pn_vh : _pnd.mesh_->vertices())
    {
        auto const& cur_sp = _pnd.sp_on_target_.value()[pn_vh];

        auto heh_canonical_frame = _tmd.mesh_->handle_of(cur_sp.heh_idx);
        auto fh_canonical_frame = heh_canonical_frame.face();
        std::vector<VH> vhs_canonical_fame = {heh_canonical_frame.vertex_from(), heh_canonical_frame.vertex_to(), heh_canonical_frame.next().vertex_to()}; // A, B, C with C being the origin

        auto sp_new = _pnd.sp_on_target_.value()[pn_vh].copy();
        sp_new.bary_coords += _dirs[pn_vh];

        if (sp_new.is_inside_element())
        {
            _pnd.sp_on_target_.value()[pn_vh] = sp_new;
            continue;
        }

        // if the new sp is not inside the face, we have to unfold the strip to find the new face
        // build the first triangle in 2D
        auto target_pos_2D = torch::zeros({_tmd.mesh_->vertices().size(), 2}, torch::dtype(torch::kFloat64));
        auto face_2D_coords = torch_compute_2D_face_embedding(heh_canonical_frame, _tmd.torch_pos_);
        for (size_t i = 0; i < vhs_canonical_fame.size(); ++i)
        {
            auto vh = vhs_canonical_fame[i];

            target_pos_2D[vh.idx.value][0] = face_2D_coords[i][0];
            target_pos_2D[vh.idx.value][1] = face_2D_coords[i][1];
        }

        auto from_point_2D = cur_sp.get_pos(target_pos_2D, *_tmd.mesh_.get());
        auto to_point_2D = sp_new.get_pos(target_pos_2D, *_tmd.mesh_.get());

        auto update_seg = torch::stack({from_point_2D, to_point_2D});

        auto iter_heh = HEH::invalid;
        // find first intersecting halfedge
        for (auto heh : fh_canonical_frame.halfedges())
        {
            auto candidat_seg = torch::stack({target_pos_2D[heh.vertex_from().idx.value], target_pos_2D[heh.vertex_to().idx.value]});
            auto params_for_candidat_seg = torch_compute_intersection_parameter(candidat_seg, update_seg);
            auto params_for_update_seg = torch_compute_intersection_parameter(update_seg, candidat_seg);

            if (!params_for_candidat_seg.defined())
                continue;

            if (params_for_candidat_seg.min().item<double>() >= 0.0 && params_for_candidat_seg.max().item<double>() <= 1.0)
            {
                if (params_for_update_seg.min().item<double>() >= 0.0 - EPS && params_for_update_seg.max().item<double>() <= 1.0 + EPS)
                {
                    iter_heh = heh;
                    traces[pn_vh].push_back(SurfacePoint(torch::tensor({params_for_candidat_seg[0].item<double>()}), iter_heh, SurfacePointType::EdgePoint));
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

            auto fn_a = torch_face_normal(hh.face(), _tmd.torch_pos_);
            auto fn_b = torch_face_normal(hh_opp.face(), _tmd.torch_pos_);

            auto posA = _tmd.torch_pos_[hh_opp.vertex_from().idx.value];
            auto posB = _tmd.torch_pos_[hh_opp.vertex_to().idx.value];

            auto hh_opp_vec_norm = posB - posA;
            hh_opp_vec_norm = torch::nn::functional::normalize(hh_opp_vec_norm, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
            auto height_vec_norm = torch::linalg_cross(fn_b, hh_opp_vec_norm);
            height_vec_norm = torch::nn::functional::normalize(height_vec_norm, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

            auto hh_opp_next_vec = _tmd.torch_pos_[hh_opp.next().vertex_to().idx.value] - _tmd.torch_pos_[hh_opp.next().vertex_from().idx.value];
            auto height_vec_norm_for_real = torch::dot(hh_opp_next_vec, height_vec_norm);
            auto height_vec = torch::dot(hh_opp_next_vec, height_vec_norm) * height_vec_norm;

            auto proj_to_base = _tmd.torch_pos_[hh_opp.next().vertex_to().idx.value] - height_vec;
            auto vec_to_proj_length = torch::dot(proj_to_base - posA, hh_opp_vec_norm);
            auto vec_to_proj = torch::dot(proj_to_base - posA, hh_opp_vec_norm) * hh_opp_vec_norm;

            // reconstruct in 2D
            auto posA_2D = target_pos_2D[hh_opp.vertex_from().idx.value];
            auto posB_2D = target_pos_2D[hh_opp.vertex_to().idx.value];

            auto hh_opp_vec_norm_2D = posB_2D - posA_2D;
            hh_opp_vec_norm_2D = torch::nn::functional::normalize(hh_opp_vec_norm_2D, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

            auto height_vec_norm_2D = torch::stack({-hh_opp_vec_norm_2D[1], hh_opp_vec_norm_2D[0]});
            auto height_point = posA_2D + vec_to_proj_length * hh_opp_vec_norm_2D + height_vec_norm_for_real * height_vec_norm_2D;

            // // "opposite" vertex of hh
            auto opp_vh = hh.opposite().next().vertex_to();

            target_pos_2D[opp_vh.idx.value][0] = height_point[0];
            target_pos_2D[opp_vh.idx.value][1] = height_point[1];

            fh_final = iter_heh.opposite_face();
            auto test_hehs = {iter_heh.opposite().next(), iter_heh.opposite().next().next()};
            iter_heh = HEH::invalid;

            for (auto heh : test_hehs)
            {
                auto candidat_seg = torch::stack({target_pos_2D[heh.vertex_from().idx.value], target_pos_2D[heh.vertex_to().idx.value]});
                auto params_for_candidat_seg = torch_compute_intersection_parameter(candidat_seg, update_seg);
                auto params_for_update_seg = torch_compute_intersection_parameter(update_seg, candidat_seg);

                if (!params_for_candidat_seg.defined())
                    continue;

                if (params_for_candidat_seg.min().item<double>() >= 0.0 && params_for_candidat_seg.max().item<double>() <= 1.0)
                {
                    if (params_for_update_seg.min().item<double>() >= 0.0 && params_for_update_seg.max().item<double>() <= 1.0)
                    {
                        iter_heh = heh;
                        traces[pn_vh].push_back(SurfacePoint(torch::tensor({params_for_candidat_seg[0].item<double>()}), iter_heh, SurfacePointType::EdgePoint));
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
            // auto c = gv::canvas();

            // auto tg_canonical_pos0 = pos3(torch_to_pos2(target_pos_2D[vhs_canonical_fame[0].idx.value]));
            // auto tg_canonical_pos1 = pos3(torch_to_pos2(target_pos_2D[vhs_canonical_fame[1].idx.value]));
            // auto tg_canonical_origin = pos3(torch_to_pos2(target_pos_2D[vhs_canonical_fame[2].idx.value]));

            // c.add_point(pos3::zero, BLACK_75).size(8);
            // c.add_point(tg_canonical_origin, BLACK).size(10);
            // c.add_face(tg_canonical_pos0, tg_canonical_pos1, tg_canonical_origin, BLUE_50);

            // adam vec in canonical frame
            auto sp_adam_vec_canonical_frame = _pnd.sp_on_target_.value()[pn_vh].copy();
            sp_adam_vec_canonical_frame.bary_coords = _od->m[pn_vh];
            auto adam_vec = sp_adam_vec_canonical_frame.get_pos(target_pos_2D, *_tmd.mesh_.get());
            // vec3 tg_adam_vec = pos3(torch_to_pos2(adam_vec)) - tg_canonical_origin;

            // c.add_line(tg_canonical_origin, tg_adam_vec, BLUE);

            // auto tg_final_pos0 = pos3(torch_to_pos2(target_pos_2D[vhs_final[0].idx.value]));
            // auto tg_final_pos1 = pos3(torch_to_pos2(target_pos_2D[vhs_final[1].idx.value]));
            auto final_origin = target_pos_2D[vhs_final[2].idx.value];
            // auto tg_final_origin = pos3(torch_to_pos2(final_origin));

            // c.add_point(tg_final_origin, BLACK).size(10);
            // c.add_face(tg_final_pos0, tg_final_pos1, tg_final_origin, MAGENTA_50);

            // in new frame
            auto adam_vec_to_in_new_frame = final_origin + adam_vec;
            // c.add_line(tg_final_origin, pos3(torch_to_pos2(adam_vec_to_in_new_frame)), MAGENTA_75);

            auto bc_in_new_frame = torch_compute_bary_cords_2D(adam_vec_to_in_new_frame, heh_final, target_pos_2D);
            _od->m[pn_vh] = torch::stack({bc_in_new_frame[0], bc_in_new_frame[1]});

            // debug
            // auto sp_debug = SurfacePoint(torch::stack({bc_in_new_frame[0], bc_in_new_frame[1]}), heh_final, SurfacePointType::FacePoint);
            // auto tg_debug = pos3(torch_to_pos2(sp_debug.get_pos(target_pos_2D, *_tmd.mesh_.get())));

            // c.add_point(tg_debug, MAGENTA).size(15);


            // auto sp_origin = SurfacePoint(torch::zeros({2}, torch::dtype(torch::kFloat64)), heh_final, SurfacePointType::FacePoint);
            // auto tg_origin = pos3(torch_to_pos2(sp_origin.get_pos(target_pos_2D, *_tmd.mesh_.get())));

            // c.add_line(tg_origin, tg_debug, MAGENTA).size(10);
        }

        auto bc = torch_compute_bary_cords_2D(to_point_2D, heh_final, target_pos_2D);
        sp_new = SurfacePoint(torch::stack({bc[0], bc[1]}), heh_final, SurfacePointType::FacePoint);
        _pnd.sp_on_target_.value()[pn_vh] = sp_new;
    }
    return traces;
}


} // namespace LayoutOpt
