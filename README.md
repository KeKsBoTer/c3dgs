# Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis

[Arxiv](https://arxiv.org/abs/2401.02436) | [Project Page](https://keksboter.github.io/c3dgs)

## Abstract
Recently, high-fidelity scene reconstruction with an optimized 3D Gaussian splat representation has been introduced for novel view synthesis from sparse image sets. Making such representations suitable for applications like network streaming and rendering on low-power devices requires significantly reduced memory consumption as well as improved rendering efficiency.
We propose a compressed 3D Gaussian splat representation that utilizes sensitivity-aware vector clustering with quantization-aware training to compress directional colors and Gaussian parameters. The learned codebooks have low bitrates and achieve a compression rate of up to **31x** on real-world scenes with only minimal degradation of visual quality. We demonstrate that the compressed splat representation can be efficiently rendered with hardware rasterization on lightweight GPUs at up to **4x** higher framerates than reported via an optimized GPU compute pipeline. Extensive experiments across multiple datasets demonstrate the robustness and rendering speed of the proposed approach. 

**Code coming soon**

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
