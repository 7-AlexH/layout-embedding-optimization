#include "PrincipalCurvature.hh"

#include <igl/principal_curvature.h>
#include "LayoutOpt/DataStructures/GCMesh.hh"
#include "LayoutOpt/DataStructures/IGLMesh.hh"

#include <glow-extras/viewer/canvas.hh>
#include "LayoutOpt/Visualization/Colors.hh"

#include <geometrycentral/surface/direction_fields.h>

namespace LayoutOpt
{

pm::vertex_attribute<PrincipalCurvature> principal_curvature(pm::vertex_attribute<pos3> const& _pos)
{
    IGLMesh igl_mesh = to_igl_mesh(_pos);
    // Compute curvature directions via quadric fitting
    Eigen::MatrixXd PD1, PD2;
    Eigen::VectorXd PV1, PV2;
    igl::principal_curvature(igl_mesh.V, igl_mesh.F, PD1, PD2, PV1, PV2);

    pm::vertex_attribute<PrincipalCurvature> pcs(_pos.mesh());
    pcs.compute(
        [&](VH _vh)
        {
            PrincipalCurvature pc = {PV1(_vh.idx.value), PV2(_vh.idx.value), vec3(PD1.row(_vh.idx.value)), vec3(PD2.row(_vh.idx.value))};
            return pc;
        });
    return pcs;
}

double PrincipalCurvature::k_min() const
{
    if (tg::abs(kappa1) < tg::abs(kappa2))
        return kappa1;
    else
        return kappa2;
}

double PrincipalCurvature::k_max() const
{
    if (tg::abs(kappa1) > tg::abs(kappa2))
        return kappa1;
    else
        return kappa2;
}

double PrincipalCurvature::mean_curvature() const { return 0.5 * (kappa1 + kappa2); }

vec3 PrincipalCurvature::get_max_dir() const
{
    if (tg::abs(kappa1) > tg::abs(kappa2))
    {
        return dir1;
    }
    else
    {
        return dir2;
    }
}

vec3 PrincipalCurvature::get_min_dir() const
{
    if (tg::abs(kappa1) < tg::abs(kappa2))
    {
        return dir1;
    }
    else
    {
        return dir2;
    }
}

double PrincipalCurvature::get_filter_value() const
{
    return tg::abs(k_max() - k_min()) * tg::pow((k_max() * k_max() + k_min() * k_min()), -0.5);
    // return tg::abs(k_max() / k_min());
    // return tg::pow2(k_max() * k_max() - k_min() * k_min());
}

pm::face_attribute<DirectionFieldData> smooth_direction_field(pm::vertex_attribute<pos3> const& _pos)
{
    // 1. define return
    auto dir_field_data = _pos.mesh().faces().make_attribute<DirectionFieldData>();

    // 2. convert to other mesh representation
    GCMesh gcmesh = to_gc_mesh(_pos);

    // 3. compute smooth field and principalCurvatureDirections
    gcmesh.positionGeometry->requireFacePrincipalCurvatureDirections();
    gcmesh.positionGeometry->requireFaceTangentBasis();
    auto aligned_direction_field = computeCurvatureAlignedFaceDirectionField(*gcmesh.positionGeometry, 4);

    for (auto f : gcmesh.mesh->faces())
    {
        // 3.1 confidence
        auto fh = _pos.mesh().faces()[f.getIndex()];
        dir_field_data[fh].confidence = gcmesh.positionGeometry->facePrincipalCurvatureDirections[f].norm();

        // 3.2 basis
        auto basis = gcmesh.positionGeometry->faceTangentBasis[f];
        auto basisX = basis[0];
        auto basisY = basis[1];

        auto t_basisX = torch::tensor({basisX.x, basisX.y, basisX.z}, torch::dtype(torch::kFloat64));
        auto t_basisY = torch::tensor({basisY.x, basisY.y, basisY.z}, torch::dtype(torch::kFloat64));

        dir_field_data[fh].basis = torch::stack({t_basisX, t_basisY});

        // 3.3 dir
        dir_field_data[fh].dir = torch::tensor({aligned_direction_field[f].x, aligned_direction_field[f].y}, torch::dtype(torch::kFloat64));
    }
    return dir_field_data;
}

} // namespace LayoutOpt
