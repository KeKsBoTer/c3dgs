<div align="center">

# Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis


<font size="4">
Simon Niedermayr &emsp; Josef Stumpfegger  &emsp; Rüdiger Westermann
</font>
<br>

<font size="4">
 Technical University of Munich 
</font>

<a href="https://keksboter.github.io/c3dgs/">Webpage</a> | <a href="https://arxiv.org/abs/2401.02436">arXiv</a> 

<img src="docs/static/img/pipeline.svg" alt="Comrpression Pipeline"/>
</div>

## Abstract
Recently, high-fidelity scene reconstruction with an optimized 3D Gaussian splat representation has been introduced for novel view synthesis from sparse image sets. Making such representations suitable for applications like network streaming and rendering on low-power devices requires significantly reduced memory consumption as well as improved rendering efficiency.
We propose a compressed 3D Gaussian splat representation that utilizes sensitivity-aware vector clustering with quantization-aware training to compress directional colors and Gaussian parameters. The learned codebooks have low bitrates and achieve a compression rate of up to **31x** on real-world scenes with only minimal degradation of visual quality. We demonstrate that the compressed splat representation can be efficiently rendered with hardware rasterization on lightweight GPUs at up to **4x** higher framerates than reported via an optimized GPU compute pipeline. Extensive experiments across multiple datasets demonstrate the robustness and rendering speed of the proposed approach. 

## Citation
If you find our work useful, please cite:
```
@misc{niedermayr2023compressed,
    title={Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis}, 
    author={Simon Niedermayr and Josef Stumpfegger and Rüdiger Westermann},
    year={2023},
    eprint={2401.02436},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Installation

### Requirements

- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions
- CUDA SDK 12 for PyTorch extensions
- C++ Compiler and CUDA SDK must be compatible


Please refer to the [original 3D Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting) for more details about requirements.

### Cloning the Repository
```
git clone https://github.com/KeKsBoTer/c3dgs
```

### Setup 

Our default, provided install method is based on Conda package and environment management:

```
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate c3dgs
```

### Running 

To compress a scene reconstructed by 3D Gaussian Splatting, simply use:

```
python compress.py --model_path <model_folder> --data_device "cuda" --output_vq <output_folder>
```

Note: After the compression is complete the script will compute the metrics (PSNR, SSIM, LPIPS) for the test images.

### Evaluation

For a more detailed evaluation and rendering run:

```bash
python render.py -m <path to compressed model> # Generate renderings
python metrics.py -m <path to compressed model> # Compute error metrics on renderings
```

## Interactive Viewers

Our renderer supports direct rendering of the compressed files on many platforms with improved rendering speed.
[It can be found here](https://github.com/KeKsBoTer/web-splat).


Alternatively, you can convert the compressed `.npz` scene files back to `.ply` files and open them with the [SIBR Viewer](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers):

```bash
python npz2ply.py <npz_file> [--ply_file <PLY_FILE>]
```
