# PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

## Please be aware that the latest code updates are now being pushed to the 'dev' branch.

Installation assumes Python 3.9 and CUDA 11.6
```.bash
bash install.sh
```

The ckpt is available in [dropbox](https://www.dropbox.com/s/tqzrv9i0umdv17d/co3d_model_Apr16.pth?dl=0).

Example usage:

```.bash
python demo.py image_folder="samples/apple" ckpt="co3d_model.pth"
```

By a Quadro GP100 GPU on FAIR cluster, the inference time for a 20-frame sequence wo GGS is around 0.8 second, with GGS is around 80 seconds (including the time of 20-seconds matching extration).

Our current implementation of GGS is slightly different from mentioned in the submission.

## Changelog

### Co3D Model V1 (2023-04-18)
- Switched to encoder-only transformer 
- Adopted a different method for time embedding and pose embedding


## Acknowledgement

Thanks for the great implementation of [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), [guided-diffusion
](https://github.com/openai/guided-diffusion), and [hloc](https://github.com/cvg/Hierarchical-Localization).


## TODO

- [x] Config System
- [x] Model Architecture
- [x] Installation
- [x] Model Weights
- [x] Verification
- [x] Match Extraction  
- [x] Verify Match Extraction  
- [x] GGS
- [x] Case Verification of GGS
- [ ] A General Dataset Class
- [x] Update Model Weights to V1
- [x] Installation Script
- [ ] Evaluation Pipeline
- [ ] Statistical Verification of GGS


##

- [ ] BARF
- [ ] GGS coordinate from cropped to uncropped
- [ ] Visualization tool
- [ ] PyTorch 2.0
- [ ] Training Pipeline
- [ ] Retraining
- [ ] Large-scale training
- [ ] Simplified Installation
- [ ] General cross dataset testing



<!-- 
## Wait a Second
- [ ] Non Rigid
- [ ] Unsupervised
- [ ] Large Model (Ongoing)
 -->






