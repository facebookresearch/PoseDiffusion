# PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

![Teaser](https://raw.githubusercontent.com/posediffusion/posediffusion.github.io/main/resources/teaser.gif)

<p dir="auto">[<a href="https://arxiv.org/pdf/2306.15667.pdf" rel="nofollow">Paper</a>]
[<a href="https://posediffusion.github.io/" rel="nofollow">Project Page</a>]</p>


## Quick Start
We provide a simple installation script, which assumes Python 3.9 and CUDA 11.6 now.

```.bash
source install.sh
```

The model ckpt is available in [dropbox](https://www.dropbox.com/s/tqzrv9i0umdv17d/co3d_model_Apr16.pth?dl=0).

An example usage is as below:

```.bash
python demo.py image_folder="samples/apple" ckpt="/PATH/TO/DOWNLOADED/CKPT"
```

Using a Quadro GP100 GPU, the inference time for a 20-frame sequence wo GGS is around 0.8 second, with GGS is around 80 seconds (including the time of 20-seconds matching extration).


## Changelog

### Co3D Model V1 (2023-04-18)
- Switched to encoder-only transformer 
- Adopted a different method for time embedding and pose embedding


## Acknowledgement

Thanks for the great implementation of [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [guided-diffusion
](https://github.com/openai/guided-diffusion), and [hloc](https://github.com/cvg/Hierarchical-Localization).



## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.


## TODO

- [ ] A General Dataset Class
- [ ] Evaluation Pipeline
- [ ] A Script to Train NeRF
