# PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment

Please refer to [pixar_replay](https://github.com/fairinternal/pixar_replay) for the installation instruction.

The ckpt is available in [dropbox](https://www.dropbox.com/s/unsgup5yu2pmusk/co3d_model0.pth?dl=0) and TBD.

Example usage:

```.bash
python test.py TEST.image_folder="samples/apple" TEST.ckpt="co3d_model0.pth"
```

By a Quadro GP100 GPU on FAIR cluster, the inference time for a 20-frame sequence is around 0.8 second.

## TODO

- [x] Config System
- [x] Model Architecture
- [x] Installation
- [x] Model Weights
- [x] Verification
- [ ] Visualization tool
- [ ] PyTorch 2.0
- [ ] Training Pipeline
- [ ] Retraining
- [ ] Large-scale training
- [ ] Simplified Installation
- [ ] General cross dataset testing
- [ ] Match Extraction  
- [ ] Geometry Guided Sampling




## Wait a Second
- [ ] Non Rigid
- [ ] Unsupervised
- [ ] Large Model










