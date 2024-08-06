# PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

![Teaser](https://raw.githubusercontent.com/posediffusion/posediffusion.github.io/main/resources/teaser.gif)

<p dir="auto">[<a href="https://arxiv.org/pdf/2306.15667.pdf" rel="nofollow">Paper</a>]
[<a href="https://posediffusion.github.io/" rel="nofollow">Project Page</a>]</p>


**Updates:**

- [Aug 6, 2024] To access the raw data for Figures 5, 7, and 10, please download our submission from arXiv. You can do this by clicking the [Download source] button at https://arxiv.org/format/2306.15667. The submission includes the following files: plots_co3dv2.tex, plots_re10k.tex, and plots_co3dv2_suppl.tex. These files use data from csvs.tex and csvs_suppl.tex, which are also included in the LaTeX submission source, to generate the figures.

- [Apr 24, 2024] You may also have an interest in [VGGSfM](https://github.com/facebookresearch/vggsfm), where a model similar to PoseDiffusion is used as the camera predictor. It also supports to optimize camera parameters through bundle adjustment.  

- [Apr 24, 2024] Updated the checkpoint for RealEstate10K dataset.



## Installation
We provide a simple installation script that, by default, sets up a conda environment with Python 3.9, PyTorch 1.13, and CUDA 11.6.

```.bash
source install.sh
```

## Quick Start

### 1. Download Checkpoint

You can download the model checkpoint trained on [Co3D](https://drive.google.com/file/d/14Waj4RCrdezx_jG9Am9WlvLfvhyuCQ3k/view?usp=sharing) or [RealEstate10K](https://drive.google.com/file/d/18kyF8XLKsTGtKEFb0NAF2tceS2eoIV-a/view?usp=sharing). Please note that the checkpoint for RealEstate10K was re-trained on an image size of 336 (you need to change the `image_size` in the coorresponding config). The predicted camera poses and focal lengths are defined in [NDC coordinate](https://pytorch3d.org/docs/cameras).





### 2. Run the Demo

```.bash
python demo.py image_folder="samples/apple" ckpt="/PATH/TO/DOWNLOADED/CKPT"
```

You can experiment with your own data by specifying a different `image_folder`.


On a Quadro GP100 GPU, the inference time for a 20-frame sequence is approximately 0.8 seconds without GGS and around 80 seconds with GGS (including 20 seconds for matching extraction).

You can choose to enable or disable GGS (or other settings) in `./cfgs/default.yaml`.

We use [Visdom](https://github.com/fossasia/visdom) by default for visualization. Ensure your Visdom settings are correctly configured to visualize the results accurately. However, Visdom is not necessary for running the model.

## Training

### 1. Preprocess Annotations

Start by following the instructions [here](https://github.com/amyxlase/relpose-plus-plus#pre-processing-co3d) to preprocess the annotations of the Co3D V2 dataset. This will significantly reduce data processing time during training.

### 2. Specify Paths

Next, specify the paths for `CO3D_DIR` and `CO3D_ANNOTATION_DIR` in `./cfgs/default_train.yaml`. `CO3D_DIR` should be set to the path where your downloaded Co3D dataset is located, while `CO3D_ANNOTATION_DIR` should point to the location of the annotation files generated after completing the preprocessing in step 1.

### 3. Start Training

- For 1-GPU Training:
  ```bash
  python train.py
  ```

- For multi-GPU training, launch the training script using [accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch), e.g., training on 8 GPUs (processes) in 1 node (machines):
  ```bash
  accelerate launch --num_processes=8 --multi_gpu --num_machines=1 train.py 
  ```
  
All configurations are specified inside `./cfgs/default_train.yaml`. Please notice that we use Visdom to record logs. 

For each iteration, the training should take around 1~3 seconds depending on difference devices. You can check it by looking at the `sec/it` of the log. The whole training should take around 2-3 days on 8 A100 GPUs.

**NOTE**: In some clusters we found the publicly released training code can be super slow when using multiple GPUs. This looks because that `accelerate` does not work well under some settings and hence the data loading is very slow (if not hangs out). The simplest solution is to remove `accelerate (accelerator)` from the code, and use pytorch's own distributed trainer or [pytorch-lighting](https://github.com/Lightning-AI/pytorch-lightning) to launch the training. This problem does not affect single GPU training. Please submit an issue if you observed a higher number or report your case [here](https://github.com/facebookresearch/PoseDiffusion/issues/33) (this should be related to the data loading of `accelerate`, a simple solution is to use pytorch's own distributed training). 


## Testing

### 1. Specify Paths

Please specify the paths `CO3D_DIR`, `CO3D_ANNOTATION_DIR`, and `resume_ckpt` in `./cfgs/default_test.yaml`. The flag `resume_ckpt` refers to your downloaded model checkpoint.

### 2. Run Testing

```bash
python test.py
```

You can check different testing settings by adjusting `num_frames`, `GGS.enable`, and others in `./cfgs/default_test.yaml`.


## Acknowledgement

Thanks for the great implementation of [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [guided-diffusion](https://github.com/openai/guided-diffusion), [hloc](https://github.com/cvg/Hierarchical-Localization), [relpose](https://github.com/jasonyzhang/relpose).


## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.
