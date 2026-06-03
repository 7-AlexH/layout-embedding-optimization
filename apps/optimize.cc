#include <filesystem>
#include <fstream>

#include <glow-extras/viewer/canvas.hh>
#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>
#include <torch/cuda.h>

#include <LayoutOpt/DataStructures/LayoutEmbedding.hh>
#include <LayoutOpt/DataStructures/Types.hh>
#include <LayoutOpt/Embedding.hh>
#include <LayoutOpt/EmbeddingUtils.hh>
#include <LayoutOpt/Init.hh>
#include <LayoutOpt/IO.hh>
#include <LayoutOpt/Optimization.hh>
#include <LayoutOpt/OptimizationOptions.hh>
#include <LayoutOpt/Optimizers.hh>
#include <LayoutOpt/Resample.hh>
#include <LayoutOpt/ScalarFields.hh>
#include <LayoutOpt/Visualization/ColorsMaps.hh>
#include <LayoutOpt/Visualization/Viewing.hh>

using namespace LayoutOpt;
namespace fs = std::filesystem;

int main()
{
    glow::glfw::GlfwContext ctx;
    auto style = default_style();

    //===================================
    // Options
    //===================================

    OptimizationOptions opts;
    opts.w_curvature_alignment_loss = 0.1;
    opts.w_harmonic_distorion_loss = 1.0;
    opts.harmonic_options.w_AIAP_SingValDecomp = 0.5;
    opts.harmonic_options.w_AreaPreserving_SingValDecomp = 0.5;

    // experiments, turned off since they are not proven to be useful
    opts.harmonic_options.w_I_DevFrom1_SingValDecomp = .0;
    opts.harmonic_options.w_DE_SingValDecomp = .0;
    opts.harmonic_options.w_SDE_SingValDecomp = .0;

    opts.optimizer = Optimizer::Adam;

    bool const enable_screenshots = true;
    auto const screenshot_size = tg::ivec2(2560, 1440);
    int const screenshot_samples = 64;
    auto const cam_pos = gv::camera_transform(tg::pos3(0.801545f, 0.222864f, -1.134743f), tg::pos3(0.000195f, 0.051322f, 0.009690f));

    //===================================
    // Paths
    //===================================

    fs::path base_path = fs::path(DATA_PATH) / "Spot/";
    fs::path path = base_path / "spot.obj";
    fs::path l_path = base_path / "spot_layout.obj";
    
    fs::path folder_name_images = base_path / "images";
    fs::create_directories(folder_name_images);

    //===================================
    // Load and preprocess meshes
    //===================================

    pm::Mesh m;
    pm::vertex_attribute<pos3> pos(m);
    pm::load(path.c_str(), m, pos);

    pm::Mesh l;
    pm::vertex_attribute<pos3> l_pos(l);
    pm::load(l_path.c_str(), l, l_pos);

    preprocess_target_and_layout(pos, l_pos);
    assert(!contains_degenerate_faces(pos));

    //===================================
    // Initialize embedding
    //===================================

    TargetMeshData tmd(pos);
    tmd.direction_field_data_.emplace(smooth_direction_field(tmd.pos_));
    LayoutData ld(l_pos);
    PathNetworkData pnd(ld.pos_);
    OverlayMeshData omd(tmd.pos_);
    compute_layout_embedding_init(tmd, ld, pnd, omd, true);
    
    bool view_init = true;
    if(view_init)
    {
        auto v = gv::view();
        v.configure(cam_pos);
        view_overlay(ld, omd);
    }

    //===================================
    // Optimize
    //===================================

    auto format_frame_number = [](int _frame_number) -> std::string
    {
        assert(_frame_number >= 0);
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << _frame_number;
        return oss.str();
    };

    constexpr double STEP_SIZE_MAX = 0.0015;
    constexpr double STEP_SIZE_MIN = 0.0005;

    OptimizerData od;
    od.init_vetor_adam_param(*pnd.mesh_.get(), STEP_SIZE_MIN);

    int ITER_UNTIL_REMESH = 5;
    constexpr int MAX_ITER = 100;
    for (int i = 0; i < MAX_ITER; ++i)
    {
        DEBUG_OUT("start iteration i: " << i)
        write_state(fs::path(DATA_PATH), tmd, ld);

        if (i % ITER_UNTIL_REMESH == 0)
        {
            DEBUG_OUT("reset vector adam")
            od.init_vetor_adam_param(*pnd.mesh_.get(), STEP_SIZE_MIN);
            DEBUG_OUT("resample")
            resample_layout(tmd, ld, pnd, omd, 0.2);
            compute_layout_embedding_init(tmd, ld, pnd, omd, false);
            ITER_UNTIL_REMESH += 15;
        }

        DEBUG_OUT("eval")
        auto eval_info = eval(tmd, ld, pnd, omd, opts, i, od);

        if (enable_screenshots)
        {
            auto screenshot_path = folder_name_images / (format_frame_number(i) + ".png");
            auto cfg_screenshot = gv::config(gv::headless_screenshot(screenshot_size, screenshot_samples, screenshot_path.string(), GL_RGBA8));
            auto v = gv::view();
            v.configure(cam_pos);
            view_overlay(ld, omd);
        }

        DEBUG_OUT("apply")
        apply(tmd, ld, pnd, omd, od, eval_info);
        od.step_size = tg::min(od.step_size * 1.15, STEP_SIZE_MAX); // warmup
    }
}
