# UltraVEM — Super-resolution

Anisotropic super-resolution for 3D volume electron microscopy (vEM), built on the
nnU-Net framework.

Training needs **high-resolution volumes only**. The low-resolution input is synthesised
on the fly by subsampling along each axis and interpolating back to the original grid,
reproducing what a coarser acquisition would have recorded — so no paired low/high
resolution acquisition of the same specimen is required.

This matters for vEM specifically: serial-section and FIB-SEM data are typically fine
in-plane and coarse along z, and the axis-wise scale makes that anisotropy explicit
rather than assuming a single isotropic factor.

The trainer and inference script live in the companion `nnUNet` repository:

| Component | Location |
|---|---|
| Trainer | `nnunetv2/training/nnUNetTrainer/nnUNetTrainer_SR.py` |
| Inference | `nnunetv2/inference/predict_sr.py` |

This repository holds the SLURM entry points, the shared environment, and a demo
notebook.

---

## How it works

Per training step, starting from a high-resolution volume that nnU-Net has already
z-score normalized:

```
high-res ──subsample per axis──▶ low-res grid ──interpolate back──▶ network input
                                                                          │
                                                             network(input)
                                                                          │
                                                     restored, compared against high-res
```

The scale is `(sx, sy, sz)` in **image space**: a factor of N on an axis means it was
acquired N times coarser than the target. For example `(1, 1, 5)` restores an
`[8, 8, 40]` nm acquisition to `[8, 8, 8]` nm; `(2, 2, 5)` additionally halves the
in-plane spacing. A factor of 1 leaves that axis untouched.

Note the axis order: image space is `(x, y, z)` while nnU-Net's tensors are `(z, y, x)`.
The trainer reverses the factors internally, so you always specify them in image order.

Validation reports PSNR and SSIM against the high-resolution volume.

---

## Configuration

Nothing about the task is hard-coded in the trainer — there is no config file and no
built-in default. Every hyperparameter below is **required**: training refuses to start
until you set each one explicitly, so a run can never silently pick up a value you did
not choose.

| Variable | Meaning |
|---|---|
| `nnUNet_SR_SCALE` | Per-axis scale `"sx,sy,sz"` in image space (or encode it in the trainer name, below) |
| `nnUNet_SR_EPOCHS` | Number of epochs (or encode it in the trainer name, below) |
| `nnUNet_SR_LR` | Initial learning rate |
| `nnUNet_SR_WEIGHT_DECAY` | Optimizer weight decay |
| `nnUNet_SR_GRAD_CLIP_NORM` | Gradient-norm clipping threshold |
| `nnUNet_SR_FREEZE_EPOCHS` | Epochs to keep a pre-trained encoder frozen (`0` disables) |
| `nnUNet_SR_SSIM_WINDOW_SIZE` | Side length, in voxels, of the SSIM pooling window |
| `nnUNet_SR_WEIGHT_CHARBONNIER` | Weight of the Charbonnier (smooth L1) loss term |
| `nnUNet_SR_WEIGHT_SSIM` | Weight of the SSIM loss term (`0` disables it) |
| `nnUNet_SR_WEIGHT_PERCEPTUAL` | Weight of the VGG perceptual loss term (`0` disables it) |

Set them in [`scripts/env.sh`](scripts/env.sh), or inline per submission:

```bash
nnUNet_SR_SCALE=<sx,sy,sz> nnUNet_SR_EPOCHS=<n> nnUNet_SR_LR=<lr> \
nnUNet_SR_WEIGHT_DECAY=<wd> nnUNet_SR_GRAD_CLIP_NORM=<norm> nnUNet_SR_FREEZE_EPOCHS=<n> \
nnUNet_SR_SSIM_WINDOW_SIZE=<size> nnUNet_SR_WEIGHT_CHARBONNIER=<w> \
nnUNet_SR_WEIGHT_SSIM=<w> nnUNet_SR_WEIGHT_PERCEPTUAL=<w> \
    sbatch --account=<your-account> scripts/train_superres.slurm
```

### Scale in the trainer name

The scale and epoch count may instead be encoded directly in the trainer name, which is
often more convenient when launching a sweep:

```
-tr nnUNetTrainer_SR_x1y1z5          scale (1, 1, 5); epochs still required separately
-tr nnUNetTrainer_SR_x2y2z5_100ep    scale (2, 2, 5), 100 epochs
-tr nnUNetTrainer_SR_x1y1z1_5        scale (1, 1, 1.5)  — underscore is the decimal point
```

These names are resolved on demand, so **any** scale works without editing code. This is
not a hidden default — the numbers come from the name you chose. Precedence is
environment variable > trainer name > error (training refuses to start if neither sets
the scale or the epoch count). Every other hyperparameter above is always required from
the environment regardless of which form you use for scale/epochs.

Starting from pre-trained weights is not part of the trainer name — pass
`-pretrained_weights <checkpoint>` (or set `PRETRAINED_WEIGHTS`) instead.

Because one trainer class covers every setting, the scale and epoch count are appended
to the nnU-Net output folder name, so runs differing only in those settings land in
separate directories instead of silently overwriting each other.

---

## Data layout

Standard nnU-Net raw format, with high-resolution volumes as the images:

```
$nnUNet_raw/<Dataset>/
    imagesTr/<case>_0000.nii.gz    high-resolution volumes (the reconstruction target)
    labelsTr/<case>.nii.gz         dummy all-ones masks
    imagesTs/<case>_0000.nii.gz    low-resolution volumes to restore at inference time
    dataset.json
```

`labelsTr/` exists only because nnU-Net requires one label file per case; the
super-resolution trainer never reads its contents. Preprocess as usual:

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
DATASET=<id> sbatch --account=<your-account> scripts/train_superres.slurm
```

Or pass them all inline (see [Configuration](#configuration) above):

```bash
DATASET=<id> nnUNet_SR_SCALE=<sx,sy,sz> nnUNet_SR_EPOCHS=<n> nnUNet_SR_LR=<lr> \
nnUNet_SR_WEIGHT_DECAY=<wd> nnUNet_SR_GRAD_CLIP_NORM=<norm> nnUNet_SR_FREEZE_EPOCHS=<n> \
nnUNet_SR_SSIM_WINDOW_SIZE=<size> nnUNet_SR_WEIGHT_CHARBONNIER=<w> \
nnUNet_SR_WEIGHT_SSIM=<w> nnUNet_SR_WEIGHT_PERCEPTUAL=<w> \
    sbatch --account=<your-account> scripts/train_superres.slurm
```

To warm-start from pre-trained encoder weights — for example the UltraVEM MAE checkpoint
from the pre-training repository — set `PRETRAINED_WEIGHTS` to the checkpoint path.
`nnUNet_SR_FREEZE_EPOCHS` then holds that encoder fixed for the first N epochs so the
freshly initialised decoder adapts first.

### Predict

```bash
INPUT_FOLDER=<dir> OUTPUT_FOLDER=<dir> MODEL_FOLDER=<dir> \
    sbatch --account=<your-account> scripts/predict_superres.slurm
```

The scale is read from the checkpoint. `predict_sr.py` refuses to guess if the
checkpoint does not record one, rather than silently producing a wrongly-scaled
restoration; pass `--sr_factors <sx> <sy> <sz>` in that case.

### Demo

[`ultravem_SR_demo.ipynb`](ultravem_SR_demo.ipynb) walks through the degradation model and
compares a restored volume against the low-resolution input and the high-resolution
reference.

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
