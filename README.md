# 🏡 🏯 UnderOneFacade 🏢 🏡

**Building on [ZAHA](https://github.com/OloOcki/zaha), UnderOneFacade is — to date — the largest benchmark for facade semantic segmentation of point clouds.**

[[Dataset]](#) [[Paper]](#) [[Benchmark]](#) [[More]](#)

<!-- 把你的方法图/teaser图放进 img/ 文件夹，文件名换成你自己的 -->
![teaser](images/teaser.png)

## 🌟 Highlights

- **2.7 billion** annotated points — the largest cross-country 3D facade benchmark to date
- Spans **three continents** (UK, Germany, Singapore), combining Victorian, Haussmann, and Southeast Asian colonial/modern architectural styles
- Centimeter-accurate geometry from **multi-sensor acquisition** (TLS + MLS: Leica RTC360, Leica BLK360, MODISSA platform)
- Adopts and extends the **LoFG (Level of Facade Generalization)** taxonomy across countries — 15 classes at LoFG3, 5 at LoFG2
- Benchmarks **6 representative architectures** (PointNet++, KPConv, DGCNN, PointTransformer v1, PointTransformer v3, OctFormer)
- Reveals strong **cross-continental domain shift**: several models degrade by more than 30 F1 points between European and Asian facades
- ❗The link above provides the dataset at original point cloud density; for region-specific subsets (Singapore / Germany / UK), see [here](#)

## 📹 Preview
[There will be a video here]

## 📝 Abstract

Globally consistent semantic digital twins require centimeter-accurate and geographically transferable 3D facade segmentation. However, progress in facade parsing is limited by the lack of large-scale, standardized benchmarks for evaluating cross-domain generalization. Existing datasets are geographically narrow, semantically inconsistent, or insufficiently precise. We introduce UnderOneFacade, the largest cross-country and cross-continent 3D facade benchmark to date, comprising centimeter-accurate point clouds with hierarchical, harmonized, and architecturally grounded semantic labels totaling 2.7 billion annotated points. Through a systematic evaluation of representative point-, graph- and transformer-based architectures, we show that current methods struggle to recognize fine-grained architectural elements and degrade significantly across geographic domains, with the best models achieving only up to 33 IoU on the fine-grained LoFG3 benchmark. By combining geometric precision with standardized semantics at unprecedented scale, UnderOneFacade establishes a rigorous benchmark for developing robust and transferable 3D segmentation models. The dataset, evaluation scripts, and pretrained models will be released upon publication.

## 🎓 Publication

Please find our paper accepted at ECCV 2026:

---

## Supported models

| Family | Methods |
|--------|---------|
| Point-based | PointNet, PointNet++, KPConv |
| Graph-based | DGCNN |
| Transformer-based | OctFormer, Point Transformer v1 (PTv1), Point Transformer v3 (PTv3) |

## Repository layout

```
UnderOneFacade/
├── config.yaml              # Edit data/output/weights paths here
├── train.py                 # Train on source country/countries
├── finetune.py              # Fine-tune pretrained weights on target
├── test.py                  # Zero-shot evaluation
├── datasets/                # Unified facade dataloaders
├── models/                  # Model engines + pointops CUDA extension
├── utils/                   # Data I/O, metrics, config helpers
│   ├── load_data.py         # Read .npy / .asc point clouds
│   ├── misc.py              # Config, paths, checkpoint helpers
│   └── metrics.py           # Segmentation metrics
├── run_experiments.sh       # Batch training sweeps
├── install.sh               # Build PTv1 pointops extension
└── asset/                   # Paper PDF
```

## Setup

### 1. Environment

```bash
conda create -n underonefacade python=3.10 -y
conda activate underonefacade
pip install -r requirements.txt
```

Adjust `spconv-cu118` in `requirements.txt` to match your CUDA version. For `torch-scatter` / `torch-sparse` / `torch-cluster`, you may need the PyG wheel index for your torch/CUDA build.

### 2. Build Point Transformer v1 ops (required for PTv1)

```bash
bash install.sh
# or manually:
# cd models/pointops_lib && pip install -e .
```

### 3. Dataset

Download the UnderOneFacade dataset and arrange it as:

```
/path/to/UnderOneFacade_data/
├── UnderOneFacade_Nottingham/data/LentonRd_SS_0.05/
│   ├── training/
│   ├── validation/
│   └── test/
├── UnderOneFacade_Singapore/
│   ├── train/
│   ├── validation/
│   └── test/
└── UnderOneFacade_ZAHA/
    ├── training_npy_0.05/
    ├── validation_npy_0.05/
    └── test_npy_0.05/
```

Point format per file: `(N, 8)` with columns `[X, Y, Z, R, G, B, Intensity, Label]`.
Labels are 1-indexed in the files; the code applies `label_offset: 1`.

Supported file formats:
- `.npy` — Singapore, ZAHA
- `.asc` — Nottingham (header row skipped)

To load a file directly:

```python
from utils.load_data import read_point_cloud

points = read_point_cloud("/path/to/facade.npy")  # (N, 8) float32
```

`load_file` is an alias for `read_point_cloud`. Datasets use the same reader via `datasets/facade_dataset.py`.

### 4. Config

Edit `data.root`, `output.root`, and `weights.root` in `config.yaml`.

## Training

Train on one or more source countries:

```bash
python train.py \
  --model dgcnn \
  --lofg lofg3 \
  --features xyz \
  --train_countries nottingham \
  --val_countries nottingham \
  --config config.yaml
```

Arguments:

- `--model`: `pointnet`, `pointnet2`, `dgcnn`, `kpconv`, `kpconv_full`, `ptv1`, `ptv3`, `octformer`
- `--lofg`: `lofg2` (5 classes) or `lofg3` (15 classes)
- `--features`: `xyz` or `rgbi` (xyz + rgb + intensity)
- `--train_countries` / `--val_countries`: comma-separated `nottingham`, `singapore`, `zaha`

Checkpoints and metrics are written under `output.root/train/<run_name>/`.

## Zero-shot test

```bash
python test.py \
  --model dgcnn \
  --lofg lofg3 \
  --features xyz \
  --weights auto \
  --source_countries nottingham \
  --test_countries singapore \
  --config config.yaml
```

With `--weights auto`, weights are resolved from `weights.root` using the naming convention:
`{weights_root}/{source_key}/{model}_{lofg}_{features}.pth`.

## Fine-tuning

```bash
python finetune.py \
  --model dgcnn \
  --lofg lofg3 \
  --features xyz \
  --pretrained auto \
  --source_countries nottingham \
  --finetune_countries singapore \
  --config config.yaml
```

## Batch experiments

One script covers the paper cross-domain training sweeps:

```bash
chmod +x run_experiments.sh

./run_experiments.sh sin_uk          # Singapore -> Nottingham (all baselines)
./run_experiments.sh uk_sin          # Nottingham -> Singapore
./run_experiments.sh sin_uk cuda     # same, force CUDA

# Resume or retrain in existing output folders:
./run_experiments.sh resume auto outputs/train/<run_name>
./run_experiments.sh retrain cuda outputs/train/<run_name>
```

Each sweep runs all baseline models (`octformer`, `ptv3`, `kpconv`, `dgcnn`, `pointnet2`, `ptv1`) for both `lofg2` and `lofg3` with `rgbi` features.

## Metrics

The code reports OA, mean Precision/Recall/F1, and per-class IoU/F1. Results are saved as `test_results.txt` in each run directory.

## Citation

```bibtex
@inproceedings{anonymous2026underoefacade,
  title={UnderOneFacade: Worldwide Facade Semantic Segmentation Benchmark Dataset},
  author={Anonymous},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## 🤝 Acknowledgments


