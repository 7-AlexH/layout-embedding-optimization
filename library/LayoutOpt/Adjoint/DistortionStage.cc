#include "DistortionStage.hh"

#include <cassert>
#include <cmath>

namespace LayoutOpt
{

FaceDistortionCtx face_distortion_forward(vec2d const& _a2, vec2d const& _b2, vec2d const& _c2,
                                          vec3d const& _a3, vec3d const& _b3, vec3d const& _c3,
                                          HarmonicOptions const& _opt)
{
    FaceDistortionCtx c;

    // --- triangle rebuild (mirrors torch_face_distortion) ---
    c.ab3 = _b3 - _a3;
    c.ac3 = _c3 - _a3;
    c.ab2 = _b2 - _a2;
    c.ac2 = _c2 - _a2;

    c.l_ab3 = c.ab3.norm();
    c.u_ab3 = c.ab3 / c.l_ab3;
    c.l_ab2 = c.ab2.norm();
    c.u_ab2 = c.ab2 / c.l_ab2;
    c.l_ac3 = c.ac3.norm();
    c.u_ac3 = c.ac3 / c.l_ac3;
    c.l_ac2 = c.ac2.norm();
    c.u_ac2 = c.ac2 / c.l_ac2;

    vec3d const axis = c.u_ab3.cross(c.u_ac3);
    double const axis_norm = axis.norm();
    c.unit_axis = axis / axis_norm;
    c.num3 = axis.dot(c.unit_axis); // torch: dot(cross(uAB3, uAC3), unit_axis)
    c.den3 = c.u_ab3.dot(c.u_ac3);
    c.angle3 = std::atan2(c.num3, c.den3);

    c.num2 = c.u_ab2.x() * c.u_ac2.y() - c.u_ab2.y() * c.u_ac2.x();
    c.den2 = c.u_ab2.x() * c.u_ac2.x() + c.u_ab2.y() * c.u_ac2.y();
    c.angle2 = std::atan2(c.num2, c.den2);

    // A at the origin, B along +x, C polar
    c.ref_B = vec2d(c.l_ab3, 0.0);
    c.param_B = vec2d(c.l_ab2, 0.0);
    c.ref_C = vec2d(c.l_ac3 * std::cos(c.angle3), c.l_ac3 * std::sin(c.angle3));
    c.param_C = vec2d(c.l_ac2 * std::cos(c.angle2), c.l_ac2 * std::sin(c.angle2));

    // --- compute_distortion (A at origin: ref_AB = ref_B, ref_AC = ref_C, ...) ---
    c.ref_area = 0.5 * (c.ref_B.x() * c.ref_C.y() - c.ref_B.y() * c.ref_C.x());
    c.param_area = 0.5 * (c.param_B.x() * c.param_C.y() - c.param_B.y() * c.param_C.x());

    // early-outs, same order as torch (NaN areas fall through the first
    // comparison and are caught by the self-inequality check)
    if (c.ref_area < EPS || c.param_area < EPS)
    {
        c.skipped = true;
        c.distortion = 0.0;
        c.param_area = 0.0;
        return c;
    }
    if (c.param_area != c.param_area || c.ref_area != c.ref_area)
    {
        c.skipped = true;
        c.distortion = 0.0;
        c.param_area = 0.0;
        return c;
    }
    assert(c.ref_area >= 0 && c.param_area >= 0); // TORCH_CHECK parity (unreachable: areas >= EPS here)

    // J = M_param * M_ref^-1, columns are the edge vectors
    mat2d M;
    M << c.ref_B.x(), c.ref_C.x(), //
        c.ref_B.y(), c.ref_C.y();
    c.P << c.param_B.x(), c.param_C.x(), //
        c.param_B.y(), c.param_C.y();
    c.K = M.inverse();
    mat2d const J = c.P * c.K;

    // closed-form 2x2 singular values (torch_singular_values)
    c.e = (J(0, 0) + J(1, 1)) * 0.5;
    c.f = (J(0, 0) - J(1, 1)) * 0.5;
    c.g = (J(1, 0) + J(0, 1)) * 0.5;
    c.h = (J(1, 0) - J(0, 1)) * 0.5;
    c.q = std::sqrt(c.e * c.e + c.h * c.h);
    c.r = std::sqrt(c.f * c.f + c.g * c.g);
    c.s0 = c.q + c.r;
    c.s1 = c.q - c.r;

    double dist = 0.0;
    if (_opt.w_SDE_DirectComputation > 0)
    {
        // literal port incl. the (ra*ra/pa)*pa association of the torch code
        double const factor1 = 1.0 + (c.ref_area * c.ref_area / c.param_area * c.param_area);

        double const norm_ref_AB = c.ref_B.norm();
        double const norm_ref_AC = c.ref_C.norm();
        double const norm_param_AB = c.param_B.norm();
        double const norm_param_AC = c.param_C.norm();

        double const factor2 = (norm_param_AC * norm_param_AC * norm_ref_AB * norm_ref_AB + norm_param_AB * norm_param_AB * norm_ref_AC * norm_ref_AC)
                               / (4.0 * c.ref_area);
        double const factor3 = (c.param_C.dot(c.param_B) * c.ref_C.dot(c.ref_B)) / (2.0 * c.ref_area);

        dist += _opt.w_SDE_DirectComputation * (factor1 * (factor2 - factor3));
    }
    if (_opt.w_SDE_SingValDecomp > 0)
    {
        double const val = c.ref_area * (c.s0 * c.s0 + c.s1 * c.s1)
                           + c.param_area * (1.0 / (c.s0 * c.s0 + EPS) + 1.0 / (c.s1 * c.s1 + EPS));
        dist += _opt.w_SDE_SingValDecomp * val;
    }
    if (_opt.w_DE_SingValDecomp > 0)
    {
        dist += _opt.w_DE_SingValDecomp * (c.ref_area * (c.s0 * c.s0 + c.s1 * c.s1));
    }
    if (_opt.w_AIAP_SingValDecomp > 0)
    {
        // same value comparison as torch (subgradient: only the selected
        // singular values receive gradient)
        double s_min, s_max;
        if (c.s0 < c.s1)
        {
            s_min = c.s0;
            s_max = c.s1;
        }
        else
        {
            s_min = c.s1;
            s_max = c.s0;
        }
        dist += _opt.w_AIAP_SingValDecomp * (c.ref_area * (s_max * s_max + 1.0 / (s_min * s_min)));
    }
    if (_opt.w_I_DevFrom1_SingValDecomp > 0)
    {
        dist += _opt.w_I_DevFrom1_SingValDecomp * (c.ref_area * ((c.s0 - 1.0) * (c.s0 - 1.0) + (c.s1 - 1.0) * (c.s1 - 1.0)));
    }
    if (_opt.w_AreaPreserving_SingValDecomp > 0)
    {
        double const m = 1.0 - c.s0 * c.s1;
        dist += _opt.w_AreaPreserving_SingValDecomp * (c.ref_area * (m * m));
    }
    c.distortion = dist;
    return c;
}

FaceDistortionGrad face_distortion_backward(FaceDistortionCtx const& c,
                                            double _d_distortion,
                                            double _d_param_area,
                                            HarmonicOptions const& _opt)
{
    FaceDistortionGrad gr; // members zero-initialized
    if (c.skipped)
        return gr; // torch returns fresh disconnected zeros: no gradient flows

    // adjoint accumulators
    double s0b = 0.0, s1b = 0.0;
    double rab = 0.0;              // ref_area adjoint
    double pab = _d_param_area;    // param_area adjoint (the returned output)
    vec2d refBb = vec2d::Zero();   // rebuilt-corner adjoints
    vec2d refCb = vec2d::Zero();
    vec2d parBb = vec2d::Zero();
    vec2d parCb = vec2d::Zero();

    // --- distortion branches (reverse of the forward accumulation) ---
    if (_opt.w_SDE_DirectComputation > 0)
    {
        double const ra = c.ref_area, pa = c.param_area;
        double const factor1 = 1.0 + (ra * ra / pa * pa);
        double const nrAB = c.ref_B.norm(), nrAC = c.ref_C.norm();
        double const npAB = c.param_B.norm(), npAC = c.param_C.norm();
        double const S = npAC * npAC * nrAB * nrAB + npAB * npAB * nrAC * nrAC;
        double const factor2 = S / (4.0 * ra);
        double const dotP = c.param_C.dot(c.param_B);
        double const dotR = c.ref_C.dot(c.ref_B);
        double const D = dotP * dotR;
        double const factor3 = D / (2.0 * ra);
        double const G = factor2 - factor3;

        double const f1b = _d_distortion * _opt.w_SDE_DirectComputation * G;
        double const Gb = _d_distortion * _opt.w_SDE_DirectComputation * factor1;
        double const f2b = Gb;
        double const f3b = -Gb;

        // factor1 = 1 + ((ra*ra)/pa)*pa — literal chain (the pa partial is
        // algebraically zero; keep the exact form for clarity)
        rab += f1b * (2.0 * ra / pa * pa);
        pab += f1b * (ra * ra / pa - (ra * ra / (pa * pa)) * pa);

        // factor2 = S / (4*ra)
        double const Sb = f2b / (4.0 * ra);
        rab += f2b * (-S / (4.0 * ra * ra));
        double const nrABb = Sb * 2.0 * nrAB * npAC * npAC;
        double const nrACb = Sb * 2.0 * nrAC * npAB * npAB;
        double const npABb = Sb * 2.0 * npAB * nrAC * nrAC;
        double const npACb = Sb * 2.0 * npAC * nrAB * nrAB;

        // factor3 = (dotP * dotR) / (2*ra)
        double const Db = f3b / (2.0 * ra);
        rab += f3b * (-D / (2.0 * ra * ra));
        double const dotPb = Db * dotR;
        double const dotRb = Db * dotP;

        refBb += nrABb * (c.ref_B / nrAB) + dotRb * c.ref_C;
        refCb += nrACb * (c.ref_C / nrAC) + dotRb * c.ref_B;
        parBb += npABb * (c.param_B / npAB) + dotPb * c.param_C;
        parCb += npACb * (c.param_C / npAC) + dotPb * c.param_B;
    }
    if (_opt.w_SDE_SingValDecomp > 0)
    {
        double const w = _opt.w_SDE_SingValDecomp * _d_distortion;
        double const i0 = 1.0 / (c.s0 * c.s0 + EPS);
        double const i1 = 1.0 / (c.s1 * c.s1 + EPS);
        rab += w * (c.s0 * c.s0 + c.s1 * c.s1);
        pab += w * (i0 + i1);
        s0b += w * (c.ref_area * 2.0 * c.s0 - c.param_area * 2.0 * c.s0 * i0 * i0);
        s1b += w * (c.ref_area * 2.0 * c.s1 - c.param_area * 2.0 * c.s1 * i1 * i1);
    }
    if (_opt.w_DE_SingValDecomp > 0)
    {
        double const w = _opt.w_DE_SingValDecomp * _d_distortion;
        rab += w * (c.s0 * c.s0 + c.s1 * c.s1);
        s0b += w * c.ref_area * 2.0 * c.s0;
        s1b += w * c.ref_area * 2.0 * c.s1;
    }
    if (_opt.w_AIAP_SingValDecomp > 0)
    {
        double const w = _opt.w_AIAP_SingValDecomp * _d_distortion;
        bool const sel = c.s0 < c.s1; // same comparison as forward/torch
        double const s_min = sel ? c.s0 : c.s1;
        double const s_max = sel ? c.s1 : c.s0;
        rab += w * (s_max * s_max + 1.0 / (s_min * s_min));
        double const smaxb = w * c.ref_area * 2.0 * s_max;
        double const sminb = w * c.ref_area * (-2.0 / (s_min * s_min * s_min));
        if (sel)
        {
            s0b += sminb;
            s1b += smaxb;
        }
        else
        {
            s1b += sminb;
            s0b += smaxb;
        }
    }
    if (_opt.w_I_DevFrom1_SingValDecomp > 0)
    {
        double const w = _opt.w_I_DevFrom1_SingValDecomp * _d_distortion;
        rab += w * ((c.s0 - 1.0) * (c.s0 - 1.0) + (c.s1 - 1.0) * (c.s1 - 1.0));
        s0b += w * c.ref_area * 2.0 * (c.s0 - 1.0);
        s1b += w * c.ref_area * 2.0 * (c.s1 - 1.0);
    }
    if (_opt.w_AreaPreserving_SingValDecomp > 0)
    {
        double const w = _opt.w_AreaPreserving_SingValDecomp * _d_distortion;
        double const m = 1.0 - c.s0 * c.s1;
        rab += w * (m * m);
        s0b += w * c.ref_area * 2.0 * m * (-c.s1);
        s1b += w * c.ref_area * 2.0 * m * (-c.s0);
    }

    // --- singular-value adjoint: s0 = q + r, s1 = q - r ---
    double const qb = s0b + s1b;
    double const rb = s0b - s1b;
    // q = sqrt(e^2 + h^2), r = sqrt(f^2 + g^2). r == 0 (exactly conformal
    // face) gives NaN here — torch's sqrt backward NaNs identically.
    double const eb = qb * (c.e / c.q);
    double const hb = qb * (c.h / c.q);
    double const fb = rb * (c.f / c.r);
    double const gb = rb * (c.g / c.r);

    mat2d Jb;
    Jb(0, 0) = 0.5 * (eb + fb);
    Jb(1, 1) = 0.5 * (eb - fb);
    Jb(1, 0) = 0.5 * (gb + hb);
    Jb(0, 1) = 0.5 * (gb - hb);

    // J = P * K with K = M^-1:  Pb = Jb K^T,  Kb = P^T Jb,  Mb = -K^T Kb K^T
    mat2d const Pb = Jb * c.K.transpose();
    mat2d const Kb = c.P.transpose() * Jb;
    mat2d const Mb = -(c.K.transpose() * Kb * c.K.transpose());

    parBb += Pb.col(0);
    parCb += Pb.col(1);
    refBb += Mb.col(0);
    refCb += Mb.col(1);

    // --- areas: ref_area = 0.5*(refB.x*refC.y - refB.y*refC.x) ---
    refBb.x() += 0.5 * rab * c.ref_C.y();
    refBb.y() -= 0.5 * rab * c.ref_C.x();
    refCb.x() -= 0.5 * rab * c.ref_B.y();
    refCb.y() += 0.5 * rab * c.ref_B.x();
    parBb.x() += 0.5 * pab * c.param_C.y();
    parBb.y() -= 0.5 * pab * c.param_C.x();
    parCb.x() -= 0.5 * pab * c.param_B.y();
    parCb.y() += 0.5 * pab * c.param_B.x();

    // --- triangle rebuild: refB = (l_ab3, 0), refC = l_ac3*(cos a3, sin a3) ---
    double const cos3 = std::cos(c.angle3), sin3 = std::sin(c.angle3);
    double const cos2 = std::cos(c.angle2), sin2 = std::sin(c.angle2);
    double const l_ab3b = refBb.x(); // refB.y is the constant 0
    double const l_ac3b = refCb.x() * cos3 + refCb.y() * sin3;
    double const angle3b = c.l_ac3 * (-sin3 * refCb.x() + cos3 * refCb.y());
    double const l_ab2b = parBb.x();
    double const l_ac2b = parCb.x() * cos2 + parCb.y() * sin2;
    double const angle2b = c.l_ac2 * (-sin2 * parCb.x() + cos2 * parCb.y());

    // --- atan2(num, den): d/dnum = den/(den^2+num^2), d/dden = -num/(...) ---
    double const denom3 = c.den3 * c.den3 + c.num3 * c.num3;
    double const num3b = angle3b * c.den3 / denom3;
    double const den3b = -angle3b * c.num3 / denom3;
    double const denom2 = c.den2 * c.den2 + c.num2 * c.num2;
    double const num2b = angle2b * c.den2 / denom2;
    double const den2b = -angle2b * c.num2 / denom2;

    // num3 = dot(cross(u_ab3, u_ac3), unit_axis) with unit_axis =
    // cross(..)/|cross(..)|: the path through unit_axis cancels exactly, the
    // net adjoint into the cross product is num3b * unit_axis.
    // Cross adjoint: cb of c = u x v gives ub = v x cb, vb = cb x u.
    vec3d const crossb = num3b * c.unit_axis;
    vec3d const u_ab3b = c.u_ac3.cross(crossb) + den3b * c.u_ac3;
    vec3d const u_ac3b = crossb.cross(c.u_ab3) + den3b * c.u_ab3;

    // num2 = uab2.x*uac2.y - uab2.y*uac2.x ; den2 = dot(uab2, uac2)
    vec2d u_ab2b, u_ac2b;
    u_ab2b.x() = num2b * c.u_ac2.y() + den2b * c.u_ac2.x();
    u_ab2b.y() = -num2b * c.u_ac2.x() + den2b * c.u_ac2.y();
    u_ac2b.x() = -num2b * c.u_ab2.y() + den2b * c.u_ab2.x();
    u_ac2b.y() = num2b * c.u_ab2.x() + den2b * c.u_ab2.y();

    // --- norm/unit pairs: u = w/l, l = |w|  =>
    //     l_total = l_direct - dot(ub, w)/l^2 ;  wb = ub/l + l_total * u ---
    vec3d const ab3b = u_ab3b / c.l_ab3 + (l_ab3b - u_ab3b.dot(c.ab3) / (c.l_ab3 * c.l_ab3)) * c.u_ab3;
    vec3d const ac3b = u_ac3b / c.l_ac3 + (l_ac3b - u_ac3b.dot(c.ac3) / (c.l_ac3 * c.l_ac3)) * c.u_ac3;
    vec2d const ab2b = u_ab2b / c.l_ab2 + (l_ab2b - u_ab2b.dot(c.ab2) / (c.l_ab2 * c.l_ab2)) * c.u_ab2;
    vec2d const ac2b = u_ac2b / c.l_ac2 + (l_ac2b - u_ac2b.dot(c.ac2) / (c.l_ac2 * c.l_ac2)) * c.u_ac2;

    // --- edge vectors to corners: ab = b - a, ac = c - a ---
    gr.d_b3 = ab3b;
    gr.d_c3 = ac3b;
    gr.d_a3 = -(ab3b + ac3b);
    gr.d_b2 = ab2b;
    gr.d_c2 = ac2b;
    gr.d_a2 = -(ab2b + ac2b);
    return gr;
}

PatchDistortionCtx patch_distortion_forward(std::vector<FH> const& _patch_fhs,
                                            pm::vertex_attribute<MappingIndex> const& _map_to_vec,
                                            Eigen::MatrixX2d const& _inner_uvs,
                                            std::vector<vec2d> const& _boundary_uvs,
                                            Eigen::MatrixX3d const& _overlay_pos,
                                            HarmonicOptions const& _opt)
{
    PatchDistortionCtx p;
    p.faces.reserve(_patch_fhs.size());

    auto uv_of = [&](MappingIndex const& mi) -> vec2d
    { return mi.on_boundary ? _boundary_uvs[mi.idx] : vec2d(_inner_uvs.row(mi.idx).transpose()); };

    for (auto o_fh : _patch_fhs)
    {
        auto const hehA = o_fh.any_halfedge();
        auto const hehB = hehA.next();
        auto const hehC = hehB.next();
        auto const vA = hehA.vertex_from();
        auto const vB = hehB.vertex_from();
        auto const vC = hehC.vertex_from();
        MappingIndex const iA = _map_to_vec[vA];
        MappingIndex const iB = _map_to_vec[vB];
        MappingIndex const iC = _map_to_vec[vC];

        FaceDistortionCtx fc = face_distortion_forward(uv_of(iA), uv_of(iB), uv_of(iC),
                                                       vec3d(_overlay_pos.row(vA.idx.value).transpose()),
                                                       vec3d(_overlay_pos.row(vB.idx.value).transpose()),
                                                       vec3d(_overlay_pos.row(vC.idx.value).transpose()), _opt);
        fc.o_vh[0] = vA.idx.value;
        fc.o_vh[1] = vB.idx.value;
        fc.o_vh[2] = vC.idx.value;
        fc.uv[0] = iA;
        fc.uv[1] = iB;
        fc.uv[2] = iC;

        p.sde += fc.distortion;
        p.param_area += fc.param_area;
        p.faces.push_back(fc);
    }
    return p;
}

void patch_distortion_backward(PatchDistortionCtx const& _ctx,
                               double _d_sde,
                               double _d_param_area,
                               HarmonicOptions const& _opt,
                               Eigen::MatrixX2d& _d_inner_uvs,
                               std::vector<vec2d>& _d_boundary_uvs,
                               Eigen::MatrixX3d& _d_overlay_pos)
{
    for (auto const& fc : _ctx.faces)
    {
        FaceDistortionGrad const g = face_distortion_backward(fc, _d_sde, _d_param_area, _opt);
        vec2d const* d2[3] = {&g.d_a2, &g.d_b2, &g.d_c2};
        vec3d const* d3[3] = {&g.d_a3, &g.d_b3, &g.d_c3};
        for (int k = 0; k < 3; ++k)
        {
            if (fc.uv[k].on_boundary)
                _d_boundary_uvs[fc.uv[k].idx] += *d2[k];
            else
                _d_inner_uvs.row(fc.uv[k].idx) += d2[k]->transpose();
            _d_overlay_pos.row(fc.o_vh[k]) += d3[k]->transpose();
        }
    }
}

double distortion_loss_forward(std::vector<PatchDistortionCtx> const& _patches, bool _normalized)
{
    double loss = 0.0;
    for (auto const& p : _patches)
        loss += _normalized ? p.sde / p.param_area : p.sde;
    return loss;
}

void distortion_loss_backward(std::vector<PatchDistortionCtx> const& _patches,
                              bool _normalized,
                              double _d_loss,
                              std::vector<double>& _d_sde,
                              std::vector<double>& _d_param_area)
{
    _d_sde.assign(_patches.size(), 0.0);
    _d_param_area.assign(_patches.size(), 0.0);
    for (size_t i = 0; i < _patches.size(); ++i)
    {
        if (_normalized)
        {
            double const pa = _patches[i].param_area;
            _d_sde[i] = _d_loss / pa;
            _d_param_area[i] = -_d_loss * _patches[i].sde / (pa * pa);
        }
        else
        {
            _d_sde[i] = _d_loss;
        }
    }
}

} // namespace LayoutOpt
