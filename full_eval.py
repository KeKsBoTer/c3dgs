import os
from argparse import ArgumentParser

mipnerf360_scenes = ["flowers", "garden", "stump", "treehill","room", "counter", "kitchen", "bonsai","bicycle"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = [ "playroom","drjohnson"]
nerf_synthetic=["chair",  "drums","ficus", "hotdog", "lego","materials", "mic", "ship"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--model_dir", default="./models")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)
all_scenes.extend(nerf_synthetic)


def run_experiment(model_dir:str,
            scene:str,
            output_dir:str,
            color_cb_size=2**12,
            gaussian_cb_size=2**12,
            color_importance_include=0.6*1e-6,
            gaussian_importance_include=0.3*1e-5):
    cmd = [
        "--model_path", f"{model_dir}/{scene}",
        "--load_iteration",str(30000),
        "--finetune_iterations", "5000",
        "--output_vq", f"{output_dir}/{scene}",
        "--data_device","cuda",
        "--gaussian_codebook_size", str(gaussian_cb_size),
        "--color_codebook_size", str(color_cb_size),
        "--color_importance_include",str(color_importance_include), 
        "--gaussian_importance_include",str(gaussian_importance_include),
    ]
    os.system(
        "python run_vq_ours.py "+ " ".join(cmd)
    )

for scene in all_scenes:
    run_experiment(args.model_dir,scene,args.output_path)