<div align="center">

<h1>Controllable Human-centric Keyframe Interpolation with Generative Prior</h1>

<div>
    <span class="author-block">
        <a href="https://gseancdat.github.io/" target="_blank">Zujin Guo</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://wusize.github.io/" target="_blank">Size Wu</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://caizhongang.com/" target="_blank">Zhongang Cai</a><sup>2</sup>,</span>
    <span class="author-block">
        <a href="https://weivision.github.io/" target="_blank">Wei Li</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://www.mmlab-ntu.com/person/ccloy/" target="_blank">Chen Change Loy</a><sup>1</sup>
</div>
<div class="is-size-5 publication-authors">
<!-- <span class="author-block"><a href="https://www.mmlab-ntu.com/index.html" target="_blank">S-Lab, Nanyang Technological University</a></span> -->
<span class="author-block"><sup>1</sup>S-Lab, Nanyang Technological University,</span>
<span class="author-block"><sup>2</sup>SenseTime Research</span>
</div>

<div>
    <strong>NeurIPS 2025</strong>
</div>

<div>
    <h4 align="center">
        <a href="https://gseancdat.github.io/projects/PoseFuse3D_KI" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
         <a href="https://arxiv.org/abs/2506.03119" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2506.03119-b31b1b.svg">
        </a>
        <img src="https://komarev.com/ghpvc/?username=GSeanCDAT-PoseFuse3D-KI&label=Visitors&color=3977dd">
    </h4>
</div>


<img src="assets/teaser_web.png" width="100%"/>

We introduce PoseFuse3D Keyframe Interpolator (<strong>PoseFuse3D-KI</strong>), a novel framework that integrates <strong>3D human guidance signals</strong> into the diffusion process for Controllable Human-centric Keyframe Interpolation (CHKI).

:open_book: For more visual results of PoseFuse3D-KI, go checkout our <a href="https://gseancdat.github.io/projects/PoseFuse3D_KI" target="_blank">project page</a>.

---
</div>

## 🚩 News
* **2026.03.31**: 📦 Dataset and smplx linear interpolator are released!
* **2025.12.31**: 🚀 Inference code and model checkpoints are released!
* **2025.09.17**: 🎉 **PoseFuse3D-KI** has been accepted by NeurIPS 2025!


## ⚙️ Installation

```bash
# Create a conda environment
conda create -n pf3d python=3.10 -y
conda activate pf3d

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install dependencies
pip install -r requirements.txt
```


## 📂 Preparation
1.  **SMPL-X Models**: Download the [SMPL-X](https://smpl-x.is.tue.mpg.de/) body models and place them in `./data/smplx_files/smplx`.
2.  **Checkpoints**: Download our model [checkpoints](https://entuedu-my.sharepoint.com/:u:/g/personal/zujin001_e_ntu_edu_sg/IQDGLSo4pRuxQp2TnwMG2CrkATQxOGkAvAkiTPA-Aze6UDA?e=7ClDjp) and place them in the `./ckpt` directory.
3.  **Dataset**: See [data.md](data.md) for dataset download links, data format details, and citation guidelines.

The expected file structure should be as follows: 
```text
PoseFuse3D-KI/
├── ckpt/ 
│   ├── pf3d/
│   │   ├── control_model.safetensors
│   │   └── pytorch_lora_weights.safetensors
│   └── pf3d_flf2v/
│       ├── control_model.safetensors
│       └── pytorch_lora_weights.safetensors
├── data/
│   └── smplx_files/
│       └── smplx/ 
│           ├── MANO_SMPLX_vertex_ids.pkl
│           ├── SMPL-X__FLAME_vertex_ids.npy
│           ├── SMPLX_NEUTRAL.pkl
│           ├── SMPLX_to_J14.pkl
│           ├── SMPLX_NEUTRAL.npz
│           ├── SMPLX_MALE.npz
│           └── SMPLX_FEMALE.npz
```


## 🚀 Demo
We provide two interpolation examples in the `./examples` directory. The structure of an example is as follows:

```text
PoseFuse3D-KI/
└── examples/
    └── example1/
        ├── frames/
        │   ├── frame_0000.png      # First keyframe
        │   └── frame_0024.png      # Last keyframe
        ├── pose/
        │   ├── frame_0000.png      # 2D pose guidance
        │   └── ...
        └── smplx_ann/              # 3D SMPL-X annotations
            ├── img_cs/
            ├── meta/
            └── smplx/
```

To run the interpolation demo, use the following command:
```bash
# Standard interpolation
sh demo.sh

# Interpolation with FLF2V 
sh demo_flf2v.sh
```
The results will be saved in `./examples/example1/output`.


## 📝 Citation
If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{guo2025controllable,
    title={Controllable Human-centric Keyframe Interpolation with Generative Prior},
    author={Guo, Zujin and Wu, Size and Cai, Zhongang and Li, Wei and Loy, Chen Change},
    booktitle={Advances in Neural Information Processing Systems},
    year={2025}
}
```


