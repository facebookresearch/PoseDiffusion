# PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

![Teaser](https://raw.githubusercontent.com/posediffusion/posediffusion.github.io/main/resources/teaser.gif)

<p dir="auto">[<a href="https://arxiv.org/pdf/2306.15667.pdf" rel="nofollow">Paper</a>]
[<a href="https://posediffusion.github.io/" rel="nofollow">Project Page</a>]</p>

## Installation
We provide a simple installation script which by default installs a conda env of Python 3.9, PyTorch 1.13 and CUDA 11.6.

```.bash
source install.sh
```

## Quick Start

Download the model checkpoint trained on Co3D from [Dropbox](https://www.dropbox.com/s/tqzrv9i0umdv17d/co3d_model_Apr16.pth?dl=0). The predicted camera poses and focal lengths are defined in [NDC coordinate](https://pytorch3d.org/docs/cameras).

Here's an example of how to use it:

```.bash
python demo.py image_folder="samples/apple" ckpt="/PATH/TO/DOWNLOADED/CKPT"
```

Feel free to test with your own data by specifying a different `image_folder`. 

Using a Quadro GP100 GPU, the inference time for a 20-frame sequence without GGS is approximately 0.8 seconds, and with GGS it’s around 80 seconds (including 20 seconds for matching extraction).

You can choose to enable or disable GGS in `./cfgs/default.yaml`.

We use [Visdom](https://github.com/fossasia/visdom) by default for visualization. Please ensure that your Visdom settings are correctly configured to visualize the results accurately; however, Visdom is not necessary for running the model.

## Training

Start by following the instructions [here](https://github.com/amyxlase/relpose-plus-plus#pre-processing-co3d) to preprocess the annotations of Co3D V2 dataset. This will significantly reduce data processing time during training.

Next, specify the paths `CO3D_DIR` and `CO3D_ANNOTATION_DIR` in `./cfgs/default_train.yaml`.

Now, you can start training with:

```bash
python train.py
```

All configurations are specified inside `./cfgs/default_train.yaml`.

For multi-GPU training, launch the training script using [accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch), e.g., training on 8 GPUs (processes) in 1 node (machines):

```bash
accelerate launch train.py --num_processes=8 --multi_gpu --num_machines=1
```

Please notice that we use Visdom to record logs.

## Changelog

### Co3D Model V1 (2023-04-18)
- Switched to encoder-only transformer 
- Adopted a different method for time embedding and pose embedding

## Acknowledgement

Thanks for the great implementation of [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [guided-diffusion](https://github.com/openai/guided-diffusion), [hloc](https://github.com/cvg/Hierarchical-Localization), [relpose](https://github.com/jasonyzhang/relpose).


## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

