"""
nnUNet denoising trainer (3D)
=============================

Self-supervised denoising: the training set contains only CLEAN volumes and the
corruption is synthesised on the fly, so no paired noisy/clean acquisition is
needed.

  - imagesTr/  clean volumes (the reconstruction target)
  - labelsTr/  dummy all-ones masks (nnU-Net requires one label file per case)

Data flow per step:
  batch['data']  clean volume, z-score normalized by nnU-Net
    -> add mixed Poisson-Gaussian noise on the fly            -> noisy
    -> Gaussian pre-filter                                    -> baseline
    -> network(baseline) predicts a residual                  -> residual
    -> output = baseline + residual, compared against the clean volume

Learning a residual on top of a Gaussian baseline means the model starts from a
reasonable denoising rather than from nothing, and can only fall below the
baseline by actively learning to.

Configuration
-------------
Every hyperparameter is set by the reader through an environment variable; none
has a value baked into this file, so a run cannot silently pick up settings you
did not choose. All of the following are required:

  nnUNet_DN_GAUSSIAN_SIGMA   read-noise standard deviation, in [0,1] intensity units
  nnUNet_DN_POISSON_SCALE    photon-count scale for the shot-noise term (lower = stronger)
  nnUNet_DN_BASELINE_SIGMA   standard deviation, in voxels, of the Gaussian pre-filter
  nnUNet_DN_EPOCHS           number of training epochs
  nnUNet_DN_LR               initial learning rate
  nnUNet_DN_SSIM_WEIGHT      weight of the SSIM term in the loss (0 disables it)
  nnUNet_DN_GRAD_CLIP_NORM   gradient-norm clipping threshold
  nnUNet_DN_SSIM_WINDOW_SIZE side length, in voxels, of the SSIM pooling window

Example:

    nnUNet_DN_GAUSSIAN_SIGMA=<sigma> nnUNet_DN_POISSON_SCALE=<scale> \\
    nnUNet_DN_BASELINE_SIGMA=<sigma> nnUNet_DN_EPOCHS=<n> nnUNet_DN_LR=<lr> \\
    nnUNet_DN_SSIM_WEIGHT=<weight> nnUNet_DN_GRAD_CLIP_NORM=<norm> \\
    nnUNet_DN_SSIM_WINDOW_SIZE=<size> \\
        nnUNetv2_train <DATASET> <CONFIGURATION> <FOLD> -tr nnUNetTrainer_Denoising

The chosen values are appended to the nnU-Net output folder name, so runs that
differ only in their settings do not overwrite one another's checkpoints.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from time import time as get_time

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context


# ============================================================================
#  Required-parameter helper
# ============================================================================

def _require_env(name: str, cast, hint: str = None):
    """Read a required hyperparameter from the environment.

    Raises rather than falling back to a built-in number: every setting that
    changes what gets trained must be a choice the reader made, not a value
    hidden in this file.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        extra = f" ({hint})" if hint else ""
        raise ValueError(
            f"{name} is not set. This trainer requires every hyperparameter to be "
            f"provided explicitly{extra}. Set it with `export {name}=<value>` (or "
            f"pass it inline) before training; see the module docstring for the "
            f"full list."
        )
    try:
        return cast(raw)
    except ValueError as e:
        raise ValueError(f"Environment variable {name}={raw!r} is not valid: {e}") from e


# ============================================================================
#  Loss functions
# ============================================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.epsilon ** 2))


class SSIMLoss3D(nn.Module):
    """SSIM computed slice-wise over a 3D volume.

    The stabilising constants are derived from the observed data range rather
    than assumed to be [0,1], which matters here because the volumes reaching
    the loss are z-score normalized. The window size is supplied by the caller
    so it cannot drift from the configured value.
    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        if pred.ndim == 5:
            B, C, D, H, W = pred.shape
            if H < self.window_size or W < self.window_size:
                return torch.tensor(0.0, device=pred.device)
            p = pred.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
            t = target.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
        elif pred.ndim == 4:
            if pred.shape[-1] < self.window_size or pred.shape[-2] < self.window_size:
                return torch.tensor(0.0, device=pred.device)
            p, t = pred, target
        else:
            return torch.tensor(0.0, device=pred.device)

        data_range = (t.max() - t.min()).item()
        if data_range < 1e-6:
            return torch.tensor(0.0, device=pred.device)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        pad = self.window_size // 2
        mu_p = F.avg_pool2d(p, self.window_size, stride=1, padding=pad)
        mu_t = F.avg_pool2d(t, self.window_size, stride=1, padding=pad)
        sigma_p2 = F.avg_pool2d(p*p, self.window_size, stride=1, padding=pad) - mu_p**2
        sigma_t2 = F.avg_pool2d(t*t, self.window_size, stride=1, padding=pad) - mu_t**2
        sigma_pt = F.avg_pool2d(p*t, self.window_size, stride=1, padding=pad) - mu_p*mu_t
        ssim = ((2*mu_p*mu_t + C1)*(2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1)*(sigma_p2 + sigma_t2 + C2))
        return 1.0 - ssim.mean()


class DenoisingLoss(nn.Module):
    """Charbonnier reconstruction loss, optionally with an SSIM structural term."""

    def __init__(self, ssim_weight, ssim_window_size):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_weight = ssim_weight
        self.ssim_window_size = ssim_window_size
        self.ssim_loss = SSIMLoss3D(window_size=ssim_window_size) if ssim_weight > 0 else None

    def forward(self, pred, target):
        loss = self.charbonnier(pred, target)
        # SSIM needs at least one full window in-plane; skip it on smaller tiles
        # (deep-supervision heads at coarse scales) rather than returning zeros.
        if self.ssim_loss is not None and min(pred.shape[-2:]) >= self.ssim_window_size:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, target)
        return loss


# ============================================================================
#  Noise generation (on-the-fly, like SR's AnisotropicDegradation)
# ============================================================================

def add_noise_torch(clean_zscore, gaussian_sigma, poisson_scale, data_std=1.0, data_mean=0.0):
    """
    Add mixed Poisson-Gaussian noise to a z-scored volume.

    The corruption is defined in raw [0,1] intensity space, which is where the
    physical noise model lives, so the volume is converted out of and back into
    the z-scored space the network operates in. Using the same parameters at
    training and evaluation time keeps the two distributions identical.

    Args:
        clean_zscore: clean volume, z-score normalized
        gaussian_sigma: read-noise standard deviation, in [0,1] intensity units
        poisson_scale: photon-count scale for the shot-noise term
        data_std, data_mean: normalization statistics used to invert the z-score
    """
    # z-score -> raw [0,1]
    raw = torch.clamp(clean_zscore * data_std + data_mean, 0.0, 1.0)

    # Shot noise: scale to photon counts, sample, scale back.
    lam = torch.clamp(raw * poisson_scale, min=torch.finfo(raw.dtype).tiny)
    poisson_noisy = torch.poisson(lam) / poisson_scale

    # Read noise
    gaussian_noise = torch.randn_like(raw) * gaussian_sigma
    noisy_raw = torch.clamp(poisson_noisy + gaussian_noise, 0.0, 1.0)

    # raw -> z-score
    return (noisy_raw - data_mean) / max(data_std, torch.finfo(raw.dtype).eps)


# ============================================================================
#  Main Denoising Trainer
# ============================================================================

class nnUNetTrainer_Denoising(nnUNetTrainer):
    """
    Denoising trainer.

      - imagesTr/ holds clean volumes; labelsTr/ holds dummy all-ones masks
      - noise is synthesised on the fly in train_step
      - the network predicts a residual on top of a Gaussian-filtered baseline

    Every hyperparameter is required from the environment (see the module
    docstring); none is hard-coded, so a training run cannot silently use
    settings the reader did not choose.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.gaussian_sigma = _require_env(
            "nnUNet_DN_GAUSSIAN_SIGMA", float,
            "read-noise std in [0,1] intensity units")
        self.poisson_scale = _require_env(
            "nnUNet_DN_POISSON_SCALE", float,
            "photon-count scale for the shot-noise term")
        self.baseline_sigma = _require_env(
            "nnUNet_DN_BASELINE_SIGMA", float,
            "Gaussian pre-filter standard deviation, in voxels")
        self.num_epochs = _require_env("nnUNet_DN_EPOCHS", int)
        self.initial_lr = _require_env("nnUNet_DN_LR", float)
        self.ssim_weight = _require_env("nnUNet_DN_SSIM_WEIGHT", float,
                                        "set to 0 to disable the SSIM term")
        self.grad_clip_norm = _require_env("nnUNet_DN_GRAD_CLIP_NORM", float)
        self.ssim_window_size = _require_env("nnUNet_DN_SSIM_WINDOW_SIZE", int)
        self.num_input_channels = len(dataset_json.get('channel_names', {})) or 1

        # Normalization statistics, needed to invert the z-score before applying
        # the noise model in raw intensity space.
        fip = plans.get("foreground_intensity_properties_per_channel",
                        plans.get("foreground_intensity_properties_by_modality", {}))
        first = fip.get("0", {})
        self.data_std = first.get("std", 1.0)
        self.data_mean = first.get("mean", 0.0)

        # One trainer class serves every configuration, so the settings that
        # change the result are folded into the output path. Without this, two
        # runs differing only in noise parameters or epoch count would silently
        # write into the same folder and overwrite each other's checkpoints.
        run_tag = (f"g{self.gaussian_sigma:g}_p{self.poisson_scale:g}"
                   f"_bl{self.baseline_sigma:g}_ep{self.num_epochs}")
        self.output_folder_base = self.output_folder_base + "__" + run_tag
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')
        # The base class already created its log file under the original folder
        # and holds an absolute path to it; move it under the new one, otherwise
        # every run would keep logging into the un-tagged directory.
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, os.path.basename(self.log_file))

        print(f"  [DN] gaussian_sigma={self.gaussian_sigma} poisson_scale={self.poisson_scale} "
              f"baseline_sigma={self.baseline_sigma}\n"
              f"       epochs={self.num_epochs} lr={self.initial_lr} "
              f"ssim_weight={self.ssim_weight}\n"
              f"       data_mean={self.data_mean:.4f} data_std={self.data_std:.4f}\n"
              f"       output_folder={self.output_folder}")

    # --- Loss ---
    def _build_loss(self):
        base_loss = DenoisingLoss(ssim_weight=self.ssim_weight,
                                  ssim_window_size=self.ssim_window_size)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            return DeepSupervisionWrapper(base_loss, weights)
        return base_loss

    # --- Network: 1 channel in → 1 channel out ---
    @staticmethod
    def build_network_architecture(architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision=True):
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels, num_input_channels,
            enable_deep_supervision)

    # --- Optimizer: AdamW, deliberately departing from base nnU-Net's SGD
    # (Nesterov momentum 0.99); AdamW suits this reconstruction task better.
    # The PolyLR schedule itself is the same as base nnU-Net. ---
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.initial_lr,
            weight_decay=self.weight_decay, eps=1e-5)
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    # --- Train step: Residual from Gaussian filter baseline (like SR's residual from trilinear) ---
    def train_step(self, batch):
        """
        Residual denoising from Gaussian filter baseline:

        1. batch['data'] = clean GT (z-score normalized by nnUNet)
        2. Add noise on-the-fly → noisy
        3. Apply Gaussian filter → gauss_filtered (baseline denoising)
        4. Network predicts RESIDUAL: what to add to gauss_filtered
        5. output = gauss_filtered + residual
        6. Loss compares output to clean GT

        This guarantees output >= Gaussian filter baseline, just like SR
        guarantees output >= trilinear interpolation baseline.
        """
        data = batch['data'].to(self.device, non_blocking=True)  # clean GT

        with torch.no_grad():
            noisy = add_noise_torch(data, self.gaussian_sigma, self.poisson_scale,
                                    self.data_std, self.data_mean)
            gauss_filtered = self._gaussian_filter_3d(noisy)

        # Residual target: what the network should learn to add
        if self.enable_deep_supervision:
            target_residual = self._make_ds_residual_targets(data, gauss_filtered)
        else:
            target_residual = data - gauss_filtered

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            residual_pred = self.network(gauss_filtered)
            l = self.loss(residual_pred, target_residual)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    # --- Validation step ---
    def validation_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)  # clean GT

        with torch.no_grad():
            noisy = add_noise_torch(data, self.gaussian_sigma, self.poisson_scale,
                                    self.data_std, self.data_mean)
            gauss_filtered = self._gaussian_filter_3d(noisy)

        if self.enable_deep_supervision:
            target_residual = self._make_ds_residual_targets(data, gauss_filtered)
        else:
            target_residual = data - gauss_filtered

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            residual_pred = self.network(gauss_filtered)
            l = self.loss(residual_pred, target_residual)

        # Reconstruct: gauss_filtered + predicted_residual
        res_pred = residual_pred[0] if self.enable_deep_supervision else residual_pred
        pred = gauss_filtered + res_pred
        gt = data

        data_range = (gt.max() - gt.min()).item()
        data_range = max(data_range, 1e-10)

        # PSNR of final output
        mse = F.mse_loss(pred, gt).item()
        psnr = 10 * np.log10((data_range ** 2) / (mse + 1e-10))

        # PSNR of Gaussian filter baseline
        mse_gauss = F.mse_loss(gauss_filtered, gt).item()
        psnr_gauss = 10 * np.log10((data_range ** 2) / (mse_gauss + 1e-10))

        # PSNR of noisy input
        mse_noisy = F.mse_loss(noisy, gt).item()
        psnr_noisy = 10 * np.log10((data_range ** 2) / (mse_noisy + 1e-10))

        ssim_val = self._compute_ssim_slice(pred, gt, data_range)

        return {
            'loss': l.detach().cpu().numpy(),
            'psnr': psnr,
            'psnr_gauss': psnr_gauss,
            'psnr_noisy': psnr_noisy,
            'mse': mse,
            'ssim': ssim_val,
        }

    def _make_ds_residual_targets(self, clean, gauss_filtered):
        """Create multi-scale residual targets for deep supervision."""
        deep_supervision_scales = self._get_deep_supervision_scales()
        targets = [clean - gauss_filtered]  # full-res residual
        for scale in deep_supervision_scales[1:]:
            clean_ds = F.interpolate(clean, scale_factor=list(scale),
                                     mode='trilinear', align_corners=False)
            gauss_ds = F.interpolate(gauss_filtered, scale_factor=list(scale),
                                     mode='trilinear', align_corners=False)
            targets.append(clean_ds - gauss_ds)
        return targets

    @torch.no_grad()
    def _gaussian_filter_3d(self, x):
        """
        Apply a 3D Gaussian filter on GPU using separable convolution.

        Sigma is in voxel units and comes from `self.baseline_sigma`, so the
        baseline strength is exactly what the reader configured.
        """
        sigma = self.baseline_sigma

        # Kernel wide enough to cover the bulk of the Gaussian, forced odd so it
        # is symmetric about the centre voxel, and never smaller than one voxel
        # on each side.
        ks = max(int(4 * sigma + 0.5) | 1, 3)
        half = ks // 2

        # 1D Gaussian kernel
        coords = torch.arange(ks, dtype=torch.float32, device=x.device) - half
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        B, C, D, H, W = x.shape

        # Separable 3D convolution: apply 1D kernel along each axis
        # Axis Z (dim=2)
        k_z = kernel_1d.view(1, 1, ks, 1, 1).expand(C, 1, ks, 1, 1)
        out = F.conv3d(x, k_z, padding=(half, 0, 0), groups=C)

        # Axis Y (dim=3)
        k_y = kernel_1d.view(1, 1, 1, ks, 1).expand(C, 1, 1, ks, 1)
        out = F.conv3d(out, k_y, padding=(0, half, 0), groups=C)

        # Axis X (dim=4)
        k_x = kernel_1d.view(1, 1, 1, 1, ks).expand(C, 1, 1, 1, ks)
        out = F.conv3d(out, k_x, padding=(0, 0, half), groups=C)

        return out

    @torch.no_grad()
    def _compute_ssim_slice(self, pred, target, data_range):
        try:
            mid = pred.shape[2] // 2
            p = pred[0, 0, mid]
            t = target[0, 0, mid]
            C1 = (0.01 * data_range) ** 2
            C2 = (0.03 * data_range) ** 2
            mu_p, mu_t = p.mean(), t.mean()
            sig_p, sig_t = p.std(), t.std()
            sig_pt = ((p - mu_p) * (t - mu_t)).mean()
            ssim = ((2*mu_p*mu_t + C1)*(2*sig_pt + C2)) / \
                   ((mu_p**2 + mu_t**2 + C1)*(sig_p**2 + sig_t**2 + C2))
            return ssim.item()
        except Exception:
            return 0.0

    # --- on_validation_epoch_end: fully override, same as SR trainer ---
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['loss'])
        psnr_here = np.mean(outputs_collated['psnr'])
        psnr_gauss = np.mean(outputs_collated['psnr_gauss'])
        psnr_noisy = np.mean(outputs_collated['psnr_noisy'])
        ssim_here = np.mean(outputs_collated['ssim'])

        self.logger.log('val_losses', loss_here, self.current_epoch)
        psnr_scaled = psnr_here / 50.0
        self.logger.log('mean_fg_dice', psnr_scaled, self.current_epoch)
        self.logger.log('dice_per_class_or_region', [psnr_here], self.current_epoch)

        gain_over_gauss = psnr_here - psnr_gauss
        self.print_to_log_file(
            f"  DN PSNR: {psnr_here:.2f} dB  |  Gauss PSNR: {psnr_gauss:.2f} dB  |  "
            f"Gain vs Gauss: {gain_over_gauss:+.2f} dB  |  Noisy: {psnr_noisy:.2f} dB  |  "
            f"SSIM: {ssim_here:.4f}"
        )

    # --- on_epoch_end: fully override, same as SR trainer ---
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', get_time(), self.current_epoch)

        self.print_to_log_file('train_loss',
                               np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss',
                               np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        psnr_val = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        self.print_to_log_file(f'PSNR: {psnr_val}')

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"New best EMA PSNR (scaled): {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    # --- perform_actual_validation: skip segmentation validation (like SR) ---
    def perform_actual_validation(self, save_probabilities=False):
        self.print_to_log_file("DN trainer: skipping segmentation-based validation. "
                               "Use nnunetv2/inference/predict_dn.py for full evaluation "
                               "against held-out volumes.")

    # --- save_checkpoint: with DN metadata (like SR saves scale_factors) ---
    def save_checkpoint(self, filename):
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                from torch._dynamo import OptimizedModule
                mod = self.network.module if self.is_ddp else self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod
                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'dn_gaussian_sigma': self.gaussian_sigma,
                    'dn_poisson_scale': self.poisson_scale,
                    'dn_baseline_sigma': self.baseline_sigma,
                    'dn_num_output_channels': self.num_input_channels,
                    'dn_data_std': self.data_std,
                    'dn_data_mean': self.data_mean,
                }
                torch.save(checkpoint, filename)
