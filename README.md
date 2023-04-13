# PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

Please refer to [pixar_replay](https://github.com/fairinternal/pixar_replay) for the installation instruction.

The ckpt is available in [dropbox](https://www.dropbox.com/s/unsgup5yu2pmusk/co3d_model0.pth?dl=0).

Example usage:

```.bash
python test.py image_folder="samples/apple" ckpt="co3d_model0.pth"
```

By a Quadro GP100 GPU on FAIR cluster, the inference time for a 20-frame sequence wo GGS is around 0.8 second, with GGS is around 80 seconds (including the time of 20-seconds matching extration).

Our current implementation of GGS is slightly different from mentioned in the submission.

## Acknowledgement

Thanks for the great implementation of [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) and [hloc](https://github.com/cvg/Hierarchical-Localization).


## TODO

- [x] Config System
- [x] Model Architecture
- [x] Installation
- [x] Model Weights
- [x] Verification
- [x] Match Extraction  
- [x] Verify Match Extraction  
- [x] GGS
- [ ] Verify GGS
- [ ] BARF
- [ ] Update Model Weights to V1
- [ ] GGS coordinate from cropped to uncropped
- [ ] Visualization tool
- [ ] PyTorch 2.0
- [ ] Training Pipeline
- [ ] Retraining
- [ ] Large-scale training
- [ ] Simplified Installation
- [ ] General cross dataset testing




## Wait a Second
- [ ] Non Rigid
- [ ] Unsupervised
- [ ] Large Model (Ongoing)










