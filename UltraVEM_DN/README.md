# UltraVEM — Denoising

Denoising for 3D volume electron microscopy (vEM), built on the nnU-Net framework.

Training needs **clean volumes only**. The corruption is synthesised on the fly from a
mixed Poisson–Gaussian noise model, so no paired noisy/clean acquisition of the same
specimen is required — which for vEM is usually impossible to obtain anyway.

The trainer and the inference script live in the companion `nnUNet` repository:

| Component | Location |
|---|---|
| Trainer | `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_DN.py` |
| Inference | `nnunetv2/inference/predict_dn.py` |

This repository holds the SLURM entry points, the shared environment, and a demo
notebook.

---

## How it works

Per training step, starting from a clean volume that nnU-Net has already z-score
normalized:

```
clean ──add Poisson+Gaussian noise──▶ noisy ──Gaussian filter──▶ baseline
                                                                    │
                                              network(baseline) ──▶ residual
                                                                    │
                                            output = baseline + residual
                                                     compared against clean
```

The network predicts a **residual** on top of a Gaussian-filtered baseline rather than
the denoised volume directly. It therefore starts from a reasonable denoising instead of
from nothing, and can only fall below the baseline by actively learning to.

Noise is applied in raw `[0,1]` intensity space — where the physical model lives — and
converted back into the normalized space the network operates in, so the training and
evaluation noise distributions are identical by construction.

Validation reports PSNR and SSIM for the output alongside PSNR for the noisy input and
the Gaussian baseline, so the gain attributable to the network is visible directly rather
than inferred.

---

## Configuration

Nothing about the task is hard-coded in the trainer — there is no config file and no
built-in default. Every hyperparameter below is **required**: training refuses to start
until you set each one explicitly, so a run can never silently pick up a value you did
not choose.

| Variable | Meaning |
|---|---|
| `nnUNet_DN_GAUSSIAN_SIGMA` | Read-noise standard deviation, in `[0,1]` intensity units |
| `nnUNet_DN_POISSON_SCALE` | Photon-count scale for the shot-noise term (lower = stronger) |
| `nnUNet_DN_BASELINE_SIGMA` | Standard deviation, in voxels, of the Gaussian pre-filter |
| `nnUNet_DN_EPOCHS` | Number of training epochs |
| `nnUNet_DN_LR` | Initial learning rate |
| `nnUNet_DN_SSIM_WEIGHT` | Weight of the SSIM term in the loss (`0` disables it) |
| `nnUNet_DN_GRAD_CLIP_NORM` | Gradient-norm clipping threshold |
| `nnUNet_DN_SSIM_WINDOW_SIZE` | Side length, in voxels, of the SSIM pooling window |

Set them in [`scripts/env.sh`](scripts/env.sh), or inline per submission:

```bash
nnUNet_DN_GAUSSIAN_SIGMA=<sigma> nnUNet_DN_POISSON_SCALE=<scale> \
nnUNet_DN_BASELINE_SIGMA=<sigma> nnUNet_DN_EPOCHS=<n> nnUNet_DN_LR=<lr> \
nnUNet_DN_SSIM_WEIGHT=<weight> nnUNet_DN_GRAD_CLIP_NORM=<norm> \
nnUNet_DN_SSIM_WINDOW_SIZE=<size> \
    sbatch --account=<your-account> scripts/train_denoise.slurm
```

Because one trainer class covers every setting, the chosen noise parameters and epoch
count are appended to the nnU-Net output folder name. Two runs differing only in those
settings therefore land in separate directories instead of silently overwriting each
other.

---

## Data layout

Standard nnU-Net raw format, with clean volumes as the images:

```
$nnUNet_raw/<Dataset>/
    imagesTr/<case>_0000.nii.gz    clean volumes (the reconstruction target)
    labelsTr/<case>.nii.gz         dummy all-ones masks
    imagesTs/<case>_0000.nii.gz    volumes to denoise at inference time
    dataset.json
```

`labelsTr/` exists only because nnU-Net requires one label file per case; the denoising
trainer never reads its contents. Preprocess as usual:

```bash
nnUNetv2_plan_and_preprocess -d <DATASET> --verify_dataset_integrity
```

---

## Usage

Edit [`scripts/env.sh`](scripts/env.sh) once for your cluster, then submit from the
repository root.

### Train

With every hyperparameter filled in in [`scripts/env.sh`](scripts/env.sh):

```bash
DATASET=<id> sbatch --account=<your-account> scripts/train_denoise.slurm
```

Or pass them all inline (see [Configuration](#configuration) above):

```bash
DATASET=<id> nnUNet_DN_GAUSSIAN_SIGMA=<sigma> nnUNet_DN_POISSON_SCALE=<scale> \
nnUNet_DN_BASELINE_SIGMA=<sigma> nnUNet_DN_EPOCHS=<n> nnUNet_DN_LR=<lr> \
nnUNet_DN_SSIM_WEIGHT=<weight> nnUNet_DN_GRAD_CLIP_NORM=<norm> \
nnUNet_DN_SSIM_WINDOW_SIZE=<size> \
    sbatch --account=<your-account> scripts/train_denoise.slurm
```

To warm-start from pre-trained encoder weights — for example the UltraVEM MAE checkpoint
from the pre-training repository — set `PRETRAINED_WEIGHTS` to the checkpoint path.
Leave it unset to train from scratch.

### Predict

```bash
INPUT_FOLDER=<dir> OUTPUT_FOLDER=<dir> MODEL_FOLDER=<dir> \
    sbatch --account=<your-account> scripts/predict_denoise.slurm
```

The Gaussian baseline sigma and residual scaling are read from the checkpoint, so
inference reproduces the trained model without further flags. `predict_dn.py` warns if
you override the sigma, because the network's residual is only valid on the baseline it
was trained against.

Both scripts can also be run directly without SLURM; see `--help` on
`nnunetv2/inference/predict_dn.py`.

### Demo

[`ultravem_DN_demo.ipynb`](ultravem_DN_demo.ipynb) walks through checking a pre-trained
encoder against the trainer's network, preparing data, training, running inference, and
visualizing a denoised volume.

---

## Citation

Built on nnU-Net; please cite:

```bibtex
@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021}
}
```
