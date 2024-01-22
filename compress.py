# %%
import gc
import json
import os
import time
import uuid
from argparse import ArgumentParser, Namespace
from os import path
from shutil import copyfile
from typing import Dict, Tuple

import torch
from tqdm import tqdm

# %%
from arguments import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from compression.vq import CompressionSettings, compress_gaussians
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from finetune import finetune
from utils.image_utils import psnr
from utils.loss_utils import ssim


def unique_output_folder():
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    return os.path.join("./output_vq/", unique_str[0:10])


def calc_importance(
    gaussians: GaussianModel, scene, pipeline_params
) -> Tuple[torch.Tensor, torch.Tensor]:
    scaling = gaussians.scaling_qa(
        gaussians.scaling_activation(gaussians._scaling.detach())
    )
    cov3d = gaussians.covariance_activation(
        scaling, 1.0, gaussians.get_rotation.detach(), True
    ).requires_grad_(True)
    scaling_factor = gaussians.scaling_factor_activation(
        gaussians.scaling_factor_qa(gaussians._scaling_factor.detach())
    )

    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None
    num_pixels = 0
    for camera in tqdm(scene.getTrainCameras(), desc="Calculating sensitivity"):
        cov3d_scaled = cov3d * scaling_factor.square()
        rendering = render(
            camera,
            gaussians,
            pipeline_params,
            background,
            clamp_color=False,
            cov3d=cov3d_scaled,
        )["render"]
        loss = rendering.sum()
        loss.backward()
        num_pixels += rendering.shape[1]*rendering.shape[2]

    importance = torch.cat(
        [gaussians._features_dc.grad, gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels
    cov_grad = cov3d.grad/num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    return importance.detach(), cov_grad.detach()


def render_and_eval(
    gaussians: GaussianModel,
    scene: Scene,
    model_params: ModelParams,
    pipeline_params: PipelineParams,
) -> Dict[str, float]:
    with torch.no_grad():
        ssims = []
        psnrs = []
        lpipss = []

        views = scene.getTestCameras()

        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for view in tqdm(views, desc="Rendering progress"):
            rendering = render(view, gaussians, pipeline_params, background)[
                "render"
            ].unsqueeze(0)
            gt = view.original_image[0:3, :, :].unsqueeze(0)

            ssims.append(ssim(rendering, gt))
            psnrs.append(psnr(rendering, gt))
            lpipss.append(lpips(rendering, gt, net_type="vgg"))
            gc.collect()
            torch.cuda.empty_cache()

        return {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }


def run_vq(
    model_params: ModelParams,
    optim_params: OptimizationParams,
    pipeline_params: PipelineParams,
    comp_params: CompressionParams,
):
    gaussians = GaussianModel(
        model_params.sh_degree, quantization=not optim_params.not_quantization_aware
    )
    scene = Scene(
        model_params, gaussians, load_iteration=comp_params.load_iteration, shuffle=True
    )

    if comp_params.start_checkpoint:
        (checkpoint_params, first_iter) = torch.load(comp_params.start_checkpoint)
        gaussians.restore(checkpoint_params, optim_params)


    timings ={}

    # %%

    start_time = time.time()
    color_importance, gaussian_sensitivity = calc_importance(
        gaussians, scene, pipeline_params
    )
    end_time = time.time()
    timings["sensitivity_calculation"] = end_time-start_time
    # %%
    print("vq compression..")
    with torch.no_grad():
        start_time = time.time()
        color_importance_n = color_importance.amax(-1)

        gaussian_importance_n = gaussian_sensitivity.amax(-1)

        torch.cuda.empty_cache()

        color_compression_settings = CompressionSettings(
            codebook_size=comp_params.color_codebook_size,
            importance_prune=comp_params.color_importance_prune,
            importance_include=comp_params.color_importance_include,
            steps=int(comp_params.color_cluster_iterations),
            decay=comp_params.color_decay,
            batch_size=comp_params.color_batch_size,
        )

        gaussian_compression_settings = CompressionSettings(
            codebook_size=comp_params.gaussian_codebook_size,
            importance_prune=None,
            importance_include=comp_params.gaussian_importance_include,
            steps=int(comp_params.gaussian_cluster_iterations),
            decay=comp_params.gaussian_decay,
            batch_size=comp_params.gaussian_batch_size,
        )

        compress_gaussians(
            gaussians,
            color_importance_n,
            gaussian_importance_n,
            color_compression_settings if not comp_params.not_compress_color else None,
            gaussian_compression_settings
            if not comp_params.not_compress_gaussians
            else None,
            comp_params.color_compress_non_dir,
            prune_threshold=comp_params.prune_threshold,
        )
        end_time = time.time()
        timings["clustering"]=end_time-start_time

    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(comp_params.output_vq, exist_ok=True)

    copyfile(
        path.join(model_params.model_path, "cfg_args"),
        path.join(comp_params.output_vq, "cfg_args"),
    )
    model_params.model_path = comp_params.output_vq

    with open(
        os.path.join(comp_params.output_vq, "cfg_args_comp"), "w"
    ) as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(comp_params))))

    iteration = scene.loaded_iter + comp_params.finetune_iterations
    if comp_params.finetune_iterations > 0:

        start_time = time.time()
        finetune(
            scene,
            model_params,
            optim_params,
            comp_params,
            pipeline_params,
            testing_iterations=[
                -1
            ],
            debug_from=-1,
        )
        end_time = time.time()
        timings["finetune"]=end_time-start_time

        # %%
    out_file = path.join(
        comp_params.output_vq,
        f"point_cloud/iteration_{iteration}/point_cloud.npz",
    )
    start_time = time.time()
    gaussians.save_npz(out_file, sort_morton=not comp_params.not_sort_morton)
    end_time = time.time()
    timings["encode"]=end_time-start_time
    timings["total"]=sum(timings.values())
    with open(f"{comp_params.output_vq}/times.json","w") as f:
        json.dump(timings,f)
    file_size = os.path.getsize(out_file) / 1024**2
    print(f"saved vq finetuned model to {out_file}")

    # eval model
    print("evaluating...")
    metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)
    metrics["size"] = file_size
    print(metrics)
    with open(f"{comp_params.output_vq}/results.json","w") as f:
        json.dump({f"ours_{iteration}":metrics},f,indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(description="Compression script parameters")
    model = ModelParams(parser, sentinel=True)
    model.data_device = "cuda"
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    comp = CompressionParams(parser)
    args = get_combined_args(parser)

    if args.output_vq is None:
        args.output_vq = unique_output_folder()

    model_params = model.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pipeline.extract(args)
    comp_params = comp.extract(args)

    run_vq(model_params, optim_params, pipeline_params, comp_params)
