# Training Data

## Download

The training data can be downloaded from the following link:

[Download Training Data](https://entuedu-my.sharepoint.com/:f:/g/personal/zujin001_e_ntu_edu_sg/IgD-qC1ny8eAQJPm0nGMRJbKAfo4sEBjwvIlU9qHE1W0KKY?e=9Yc4da)

The download contains three files:
- `sportsslomo_fullbody.tar.gz` 
- `new_pexels_ready.tar.gz`
- `split.zip`

## Data Format

Each sub-dataset in the download follows the same directory structure:

```text
<dataset_root>/
├── frames/                # Video frames (PNG)
│   └── <clip_id>/
│       ├── frame_0000.png
│       ├── frame_0004.png
│       └── ...
├── bbox/                  # Bounding box annotations (JSON)
│   └── <clip_id>/
│       ├── frame_0000.json
│       └── ...
├── seg/                   # Segmentation masks (compressed NumPy, .npy.gz)
│   └── <clip_id>/
│       ├── frame_0000.npy.gz
│       └── ...
├── poses_133/             # 2D pose keypoint annotations (JSON)
│   └── <clip_id>/
│       ├── frame_0000.json
│       └── ...
└── smplx_ann/             # 3D SMPL-X body model annotations
    └── <clip_id>/
        ├── img_cs/        # Rendered 3D pose visualizations (PNG)
        ├── meta/          # Per-human metadata (JSON)
        └── smplx/         # SMPL-X body parameters (NPZ)
```


## Citation

If you find this work useful, please cite the following works:

```bibtex
@inproceedings{guo2025controllable,
    title={Controllable Human-centric Keyframe Interpolation with Generative Prior},
    author={Guo, Zujin and Wu, Size and Cai, Zhongang and Li, Wei and Loy, Chen Change},
    booktitle={Advances in Neural Information Processing Systems},
    year={2025}
}
```

The SportsSloMo dataset is from:

```bibtex
@inproceedings{chen2024sportsslomo,
    title={SportsSloMo: A New Benchmark and Baselines for Human-centric Video Frame Interpolation},
    author={Jiaben Chen and Huaizu Jiang},
    year={2024},
    booktitle={CVPR}
}
```
