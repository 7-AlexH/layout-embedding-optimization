#include "TorchUtils.hh"

#include <ATen/core/TensorBody.h>
#include <torch/nn.h>
#include <cassert>
#include <glow-extras/viewer/canvas.hh>
#include "LayoutOpt/Utils/Timer.hh"
#include "LayoutOpt/Visualization/Colors.hh"

namespace LayoutOpt
{
namespace
{

at::Tensor torch_compute_bary_cords_2D(at::Tensor const& _point, torch::Tensor p0, torch::Tensor p1, torch::Tensor p2)
{
    torch::Tensor T = torch::stack({p0, p1, p2}, 0);
    T = torch::cat({T, torch::ones({3, 1})}, 1); // Shape: [3, 3], append ones for homogeneous coordinates

    // Construct the position vector
    torch::Tensor P = torch::cat({_point, torch::tensor({1.0})}); // Shape: [3]

    // Solve for barycentric coordinates
    return torch::linalg_solve(T.t(), P); // Shape: [3], containing (lambda0, lambda1, lambda2)
}

}

at::Tensor get_row(at::Tensor const& _tensor, size_t _idx)
{
    assert(_tensor.dim() == 2 && "Tensor should have 2 dimensions.");
    assert(_tensor.size(0) > 0 && _tensor.size(1) > 0 && "Tensor should be non-empty (n x m).");
    assert(_idx < _tensor.size(0) && "Index is out of range for rows.");

    return _tensor.index({static_cast<int64_t>(_idx)});
}

//==========================================================================
template <typename tag, typename AttrT>
at::Tensor attr_to_torch(pm::primitive_attribute<tag, AttrT> const& _attr)
{
    std::vector<torch::Tensor> torch_list;
    auto size = _attr.size();
    auto data = _attr.data();

    for (auto i = 0; i < size; ++i)
    {
        auto const& value = data[i];

        if constexpr (std::is_same_v<AttrT, pos3>)
        {
            torch_list.push_back(torch::tensor({value.x, value.y, value.z}, torch::dtype(torch::kFloat64)));
        }
        else if constexpr (std::is_same_v<AttrT, pos2>)
        {
            torch_list.push_back(torch::tensor({value.x, value.y}, torch::dtype(torch::kFloat64)));
        }
        else
        {
            torch_list.push_back(torch::tensor({value}, torch::dtype(torch::kFloat64)));
        }
    }
    return torch::stack(torch_list);
}
// instantiation
template at::Tensor attr_to_torch<pm::vertex_tag, pos3>(pm::primitive_attribute<pm::vertex_tag, pos3> const& _attr);
template at::Tensor attr_to_torch<pm::halfedge_tag, pos2>(pm::primitive_attribute<pm::halfedge_tag, pos2> const& _attr);
template at::Tensor attr_to_torch<pm::vertex_tag, double>(pm::primitive_attribute<pm::vertex_tag, double> const& _attr);
template at::Tensor attr_to_torch<pm::face_tag, double>(pm::primitive_attribute<pm::face_tag, double> const& _attr);


at::Tensor torch_face_areas(torch::Tensor const& _torch_pos, polymesh::Mesh const& _mesh)
{
    // Vectorized computations for efficiency
    int64_t F = _mesh.faces().size();

    // Preallocate vertex index arrays
    std::vector<int64_t> v0_idx(F), v1_idx(F), v2_idx(F);

    int64_t f = 0;
    for (auto fh : _mesh.faces())
    {
        auto [vh0, vh1, vh2] = fh.vertices().to_array<3>();
        v0_idx[f] = vh0.idx.value;
        v1_idx[f] = vh1.idx.value;
        v2_idx[f] = vh2.idx.value;
        f++;
    }

    auto device = _torch_pos.device();

    torch::Tensor v0_idx_tensor = torch::tensor(v0_idx, torch::kInt64).to(device);
    torch::Tensor v1_idx_tensor = torch::tensor(v1_idx, torch::kInt64).to(device);
    torch::Tensor v2_idx_tensor = torch::tensor(v2_idx, torch::kInt64).to(device);

    // Gather positions for all faces
    auto pos0 = _torch_pos.index_select(0, v0_idx_tensor); // [F,3]
    auto pos1 = _torch_pos.index_select(0, v1_idx_tensor); // [F,3]
    auto pos2 = _torch_pos.index_select(0, v2_idx_tensor); // [F,3]

    // Triangle edge vectors
    auto vec0 = pos1 - pos0; // [F,3]
    auto vec1 = pos2 - pos0; // [F,3]

    // Cross product and norm
    auto cross = torch::cross(vec0, vec1, 1); // [F,3]
    auto norm = cross.norm(2, 1);             // [F]

    // Triangle area = 0.5 * |cross|
    auto area = 0.5 * norm; // [F]
    return area;
}

at::Tensor torch_cotans(at::Tensor const& _torch_pos, polymesh::Mesh const& _mesh)
{
    // Vectorized computations for efficiency
    int64_t E = _mesh.edges().size();
    std::vector<int64_t> i_idx(E), j_idx(E), a_idx(E), b_idx(E);

    // Gather vertex indices for all edges
    int64_t e = 0;
    for (auto eh : _mesh.edges())
    {
        auto h0 = eh.halfedgeA();
        auto h1 = eh.halfedgeB();

        i_idx[e] = h0.vertex_to().idx.value;
        j_idx[e] = h1.vertex_to().idx.value;
        a_idx[e] = h0.next().vertex_to().idx.value;
        b_idx[e] = h1.next().vertex_to().idx.value;
        e++;
    }

    // Convert to tensors on same device as _torch_pos
    auto device = _torch_pos.device();
    torch::Tensor i_idx_tensor = torch::tensor(i_idx, torch::kInt64).to(device);
    torch::Tensor j_idx_tensor = torch::tensor(j_idx, torch::kInt64).to(device);
    torch::Tensor a_idx_tensor = torch::tensor(a_idx, torch::kInt64).to(device);
    torch::Tensor b_idx_tensor = torch::tensor(b_idx, torch::kInt64).to(device);

    // Gather positions
    auto pi = _torch_pos.index_select(0, i_idx_tensor); // [E,3]
    auto pj = _torch_pos.index_select(0, j_idx_tensor); // [E,3]
    auto pa = _torch_pos.index_select(0, a_idx_tensor); // [E,3]
    auto pb = _torch_pos.index_select(0, b_idx_tensor); // [E,3]

    // Edge vectors
    auto e_ia = pi - pa; // [E,3]
    auto e_ja = pj - pa; // [E,3]
    auto e_ib = pi - pb; // [E,3]
    auto e_jb = pj - pb; // [E,3]

    // Cross norms
    auto denom_a = torch::cross(e_ia, e_ja, 1).norm(2, 1); // [E]
    auto denom_b = torch::cross(e_ib, e_jb, 1).norm(2, 1); // [E]

    // Dot products
    auto dot_a = (e_ia * e_ja).sum(1); // [E]
    auto dot_b = (e_ib * e_jb).sum(1); // [E]

    // Cotangents
    auto cot_a = dot_a / (denom_a + 1e-10);
    auto cot_b = dot_b / (denom_b + 1e-10);

    // Final per-edge cotangent
    return cot_a + cot_b; // [E]
}


//==========================================================================
pos3 torch_to_pos3(at::Tensor const& _pos)
{
    assert(_pos.dim() == 1 && _pos.size(0) == 3 && "Tensor must have shape [3]");
    return pos3(_pos[0].item<double>(), _pos[1].item<double>(), _pos[2].item<double>());
}

pos2 torch_to_pos2(at::Tensor const& _pos)
{
    if (!(_pos.dim() == 1 && _pos.size(0) == 2))
    {
        DEBUG_VAR(_pos.dim())
        DEBUG_VAR(_pos.size(0))
    }

    assert(_pos.dim() == 1 && _pos.size(0) == 2 && "Tensor must have shape [2]");
    return pos2(_pos[0].item<double>(), _pos[1].item<double>());
}

torch::Tensor torch_compute_2D_face_embedding(HEH _heh, torch::Tensor const& _3D_origin, torch::Tensor const& _3D_x_axis, bool _positive, torch::Tensor const& _3D_pos)
{
    auto const& A = _3D_pos[_heh.vertex_from().idx.value];
    auto const& B = _3D_pos[_heh.vertex_to().idx.value];
    auto const& C = _3D_pos[_heh.next().vertex_to().idx.value];

    auto const& O = _3D_origin; // renaming
    auto const& X = _3D_x_axis; // renaming

    // face normal
    auto AC = C - A;
    auto AB = B - A;
    auto normal = torch::nn::functional::normalize(torch::linalg_cross(AB, AC), torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

    // computing all lengths and angles in 3D
    // vectors
    auto OA = A - O;
    auto OB = B - O;
    auto OC = C - O;
    auto OX = X - O;
    // lengths
    auto length_OA = torch::norm(OA);
    auto length_OB = torch::norm(OB);
    auto length_OC = torch::norm(OC);
    // normalized
    auto OA_normalized = torch::nn::functional::normalize(OA, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
    auto OB_normalized = torch::nn::functional::normalize(OB, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
    auto OC_normalized = torch::nn::functional::normalize(OC, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
    auto OX_normalized = torch::nn::functional::normalize(OX, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
    // angles
    auto angle_XOA = torch::atan2(torch::dot(torch::linalg_cross(OX_normalized, OA_normalized), normal), torch::dot(OA_normalized, OX_normalized));
    auto angle_XOB = torch::atan2(torch::dot(torch::linalg_cross(OX_normalized, OB_normalized), normal), torch::dot(OB_normalized, OX_normalized));
    auto angle_XOC = torch::atan2(torch::dot(torch::linalg_cross(OX_normalized, OC_normalized), normal), torch::dot(OC_normalized, OX_normalized));

    // correct for alignment along neg x direction
    if (!_positive)
    {
        angle_XOA += M_PI;
        angle_XOB += M_PI;
        angle_XOC += M_PI;
    }

    auto OA_2D = length_OA * torch::stack({torch::cos(angle_XOA), torch::sin(angle_XOA)});
    auto OB_2D = length_OB * torch::stack({torch::cos(angle_XOB), torch::sin(angle_XOB)});
    auto OC_2D = length_OC * torch::stack({torch::cos(angle_XOC), torch::sin(angle_XOC)});

    // VIEW
    if (false)
    {
        auto posA = torch_to_pos3(A);
        auto posB = torch_to_pos3(B);
        auto posC = torch_to_pos3(C);
        auto posO = torch_to_pos3(O);
        auto posX = torch_to_pos3(X);

        auto posA_2D = pos3(torch_to_pos2(OA_2D));
        auto posB_2D = pos3(torch_to_pos2(OB_2D));
        auto posC_2D = pos3(torch_to_pos2(OC_2D));

        auto g = gv::grid();
        {
            auto v = gv::view();
            auto cam = gv::CameraController::create();
            v.configure(cam);
            auto c = gv::canvas();
            c.add_point(posA, BLUE);
            c.add_point(posB, BLUE);
            c.add_point(posC, BLUE);
            c.add_point(posO, RED);
            c.add_point(posX, GREEN);
        }
        {
            auto c = gv::canvas();
            c.add_line(tg::pos3::zero, 10 * tg::vec3::unit_x).color(BLACK);
            c.add_point(pos3(posA_2D));
            c.add_point(pos3(posB_2D));
            c.add_point(pos3(posC_2D));
        }
    }

    return torch::stack({OA_2D, OB_2D, OC_2D});
}


torch::Tensor torch_compute_2D_face_embedding(HEH _heh, at::Tensor const& _3D_pos)
{
    auto const& p0 = _3D_pos[_heh.vertex_from().idx.value];
    auto const& p1 = _3D_pos[_heh.vertex_to().idx.value];
    auto const& p2 = _3D_pos[_heh.next().vertex_to().idx.value]; // this will be the new origin

    auto e0 = p0 - p2;
    auto e1 = p1 - p2;

    auto length_e0 = torch::norm(e0);
    auto length_e1 = torch::norm(e1);

    auto e0_normalized = torch::nn::functional::normalize(e0, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
    auto e1_normalized = torch::nn::functional::normalize(e1, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

    auto angle = torch::atan2(torch::norm(torch::linalg_cross(e0_normalized, e1_normalized)), torch::dot(e0_normalized, e1_normalized));

    auto e0_2D = length_e0 * torch::tensor({1.0, 0.0}); // x-axis
    auto e1_2D = length_e1 * torch::stack({torch::cos(angle), torch::sin(angle)});

    return torch::stack({e0_2D, e1_2D, torch::zeros({2})});
}

at::Tensor torch_compute_bary_cords_2D(at::Tensor const& _point, const HEH& _heh, pm::vertex_attribute<at::Tensor> _pos)
{
    std::vector<VH> vhs = {_heh.vertex_from(), _heh.vertex_to(), _heh.next().vertex_to()};
    auto const& p0 = _pos[vhs[0]];
    auto const& p1 = _pos[vhs[1]];
    auto const& p2 = _pos[vhs[2]];

    return torch_compute_bary_cords_2D(_point, p0, p1, p2);
}


at::Tensor torch_compute_bary_cords_2D(at::Tensor const& _point, const HEH& _heh, at::Tensor const& _pos)
{
    std::vector<VH> vhs = {_heh.vertex_from(), _heh.vertex_to(), _heh.next().vertex_to()};
    auto const& p0 = _pos[vhs[0].idx.value];
    auto const& p1 = _pos[vhs[1].idx.value];
    auto const& p2 = _pos[vhs[2].idx.value];

    return torch_compute_bary_cords_2D(_point, p0, p1, p2);
}

at::Tensor torch_face_normal(polymesh::face_handle const& _fh, at::Tensor const& _pos)
{
    auto [vh0, vh1, vh2] = _fh.vertices().to_array<3>();
    auto const& p0 = _pos[vh0.idx.value];
    auto const& p1 = _pos[vh1.idx.value];
    auto const& p2 = _pos[vh2.idx.value];
    auto f_n = torch::nn::functional::normalize(torch::linalg_cross(p1 - p0, p2 - p0), torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
    return f_n;
}

torch::Tensor torch_compute_intersection_parameter(torch::Tensor const& _line_a, torch::Tensor const& _line_b)
{
    // Extract endpoints: each line is a 2x2 tensor with each row a 2D coordinate.
    auto const& a0 = _line_a[0];
    auto const& a1 = _line_a[1];
    auto const& b0 = _line_b[0];
    auto const& b1 = _line_b[1];

    // Compute direction vectors.
    auto a = a1 - a0;
    auto b = b1 - b0;

    // Compute the cross product (scalar in 2D)
    auto cross = a[0] * b[1] - a[1] * b[0];
    double cross_val = cross.item<double>();
    if (std::abs(cross_val) < 1e-8)
    {
        DEBUG_OUT("Lines are parallel or nearly parallel.")
        return at::Tensor{};
    }

    // Compute parameter t such that intersection = a0 + t*(a1 - a0)
    auto b0_a0 = b0 - a0;
    auto t_numer = b0_a0[0] * b[1] - b0_a0[1] * b[0];
    auto t = t_numer / cross;

    // Barycentric coordinates relative to line_a are (1-t, t)
    return torch::stack({1.0 - t, t}, 0);
}

at::Tensor torch_singular_values(at::Tensor const& _mat)
{
    // Ensure the input is a 2x2 matrix
    assert(_mat.dim() == 2 && _mat.size(0) == 2 && _mat.size(1) == 2 && "Input must be a 2x2 matrix.");

    auto e = (_mat[0][0] + _mat[1][1]) * 0.5;
    auto f = (_mat[0][0] - _mat[1][1]) * 0.5;
    auto g = (_mat[1][0] + _mat[0][1]) * 0.5;
    auto h = (_mat[1][0] - _mat[0][1]) * 0.5;
    auto q = sqrt(e * e + h * h);
    auto r = sqrt(f * f + g * g);

    auto sing_val_0 = q + r;
    auto sing_val_1 = q - r;

    return torch::stack({sing_val_0, sing_val_1});
}


} // namespace LayoutOpt
