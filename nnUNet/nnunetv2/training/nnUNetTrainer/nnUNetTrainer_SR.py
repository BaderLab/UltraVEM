"""
nnUNet super-resolution trainer (3D)
====================================

Repurposes the nnU-Net segmentation pipeline for volumetric super-resolution
with per-axis scaling.

Training needs HIGH-RESOLUTION volumes only. The low-resolution input is
synthesised on the fly by subsampling along each axis and interpolating back to
the original grid, reproducing what a coarser acquisition would have recorded.
No paired low/high-resolution acquisition of the same specimen is required.

Data flow per step:
  batch['data']  high-resolution volume, z-score normalized by nnU-Net
    -> subsample per axis, interpolate back to the grid   -> low-resolution
    -> network(low_resolution)                            -> restored volume
    -> loss against the high-resolution volume

The scale is given as (sx, sy, sz) in image space: a factor of 5 on z means z
was acquired five times coarser than the target, e.g. an [8, 8, 40] nm
acquisition restored to [8, 8, 8] nm.

Data preparation:
    imagesTr/    high-resolution volumes (the reconstruction target)
    labelsTr/    dummy all-ones masks (nnU-Net requires one label file per case)
    imagesTs/    low-resolution volumes to restore at inference time

Configuration
-------------
Every hyperparameter is set by the reader through an environment variable; none
has a value baked into this file, so a run cannot silently pick up settings you
did not choose. All of the following are required:

  nnUNet_SR_SCALE              per-axis scale as "sx,sy,sz" in image space
                                (or encode it in the trainer name, see below)
  nnUNet_SR_EPOCHS              number of training epochs
                                (or encode it in the trainer name, see below)
  nnUNet_SR_LR                  initial learning rate
  nnUNet_SR_WEIGHT_DECAY        optimizer weight decay
  nnUNet_SR_GRAD_CLIP_NORM      gradient-norm clipping threshold
  nnUNet_SR_FREEZE_EPOCHS       epochs to keep a pre-trained encoder frozen (0 disables)
  nnUNet_SR_SSIM_WINDOW_SIZE    side length, in voxels, of the SSIM pooling window
  nnUNet_SR_WEIGHT_CHARBONNIER  weight of the Charbonnier (smooth L1) loss term
  nnUNet_SR_WEIGHT_SSIM         weight of the SSIM loss term (0 disables it)
  nnUNet_SR_WEIGHT_PERCEPTUAL   weight of the VGG perceptual loss term (0 disables it)

The scale and epoch count may instead be encoded in the trainer name:
`-tr nnUNetTrainer_SR_x1y1z5` sets a scale of (1, 1, 5), and an optional
`_<n>ep` suffix sets the epoch count, e.g. `nnUNetTrainer_SR_x1y1z5_100ep`. Any
scale works this way without adding code. Precedence is environment variable >
trainer name > error.

Example:

    nnUNet_SR_LR=<lr> nnUNet_SR_WEIGHT_DECAY=<wd> nnUNet_SR_GRAD_CLIP_NORM=<norm> \\
    nnUNet_SR_FREEZE_EPOCHS=<n> nnUNet_SR_SSIM_WINDOW_SIZE=<size> nnUNet_SR_WEIGHT_CHARBONNIER=<w> \\
    nnUNet_SR_WEIGHT_SSIM=<w> nnUNet_SR_WEIGHT_PERCEPTUAL=<w> \\
        nnUNetv2_train <DATASET> <CONFIGURATION> <FOLD> -tr nnUNetTrainer_SR_x1y1z5_100ep

The chosen values are appended to the nnU-Net output folder name, so runs that
differ only in their settings do not overwrite one another's checkpoints.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List
from time import time as get_time

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs


# ============================================================================
#  Required-parameter helper
# ============================================================================

def _require_env(name: str, cast, hint: str = None, alt: str = None):
    """Read a required hyperparameter from the environment.

    Raises rather than falling back to a built-in number: every setting that
    changes what gets trained must be a choice the reader made, not a value
    hidden in this file. `alt`, if given, is mentioned as another way to set it
    (used for scale/epochs, which can also come from the trainer name).
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        extra = f" ({hint})" if hint else ""
        via_alt = f" or {alt}" if alt else ""
        raise ValueError(
            f"{name} is not set{via_alt}. This trainer requires every hyperparameter to be "
            f"provided explicitly{extra}. Set it with `export {name}=<value>` (or pass it "
            f"inline) before training; see the module docstring for the full list."
        )
    try:
        return cast(raw)
    except ValueError as e:
        raise ValueError(f"Environment variable {name}={raw!r} is not valid: {e}") from e


def parse_scale(value) -> Tuple[float, float, float]:
    """Normalise a scale given as "sx,sy,sz", a sequence, or a single factor."""
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,\s]+", value.strip()) if p]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    try:
        factors = [float(p) for p in parts]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot parse a scale from {value!r}: {e}") from e
    if len(factors) == 1:          # isotropic shorthand
        factors = factors * 3
    if len(factors) != 3:
        raise ValueError(f"Scale needs 3 components (x, y, z), got {value!r}")
    if any(f < 1 for f in factors):
        raise ValueError(f"Scale factors must be >= 1, got {factors}")
    return tuple(factors)


# ============================================================================
#  Loss functions for super resolution
# ============================================================================

class CharbonnierLoss(nn.Module):
    """Smooth L1 loss commonly used in SR. More robust to outliers than MSE."""
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.epsilon ** 2))


class SSIMLoss3D(nn.Module):
    """SSIM loss for 2D/3D volumes using avg pooling.
    Adapts C1/C2 to actual data range (critical for z-score normalized data)."""
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ndim = prediction.ndim - 2
        ws = self.window_size

        # Clamp window size to not exceed any spatial dim
        spatial_shape = prediction.shape[2:]
        effective_ws = min(ws, *spatial_shape)
        if effective_ws < 2:
            return torch.tensor(0.0, device=prediction.device)

        # Adapt C1, C2 to actual data range (NOT hardcoded to 1.0!)
        # nnUNet z-score normalizes: data ~ N(0,1), range roughly [-3, 3] or wider
        data_range = target.detach().max() - target.detach().min()
        data_range = torch.clamp(data_range, min=1e-6)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        if ndim == 2:
            pool_fn = lambda x: F.avg_pool2d(x, effective_ws, stride=1, padding=effective_ws // 2)
        elif ndim == 3:
            pool_fn = lambda x: F.avg_pool3d(x, effective_ws, stride=1, padding=effective_ws // 2)
        else:
            return torch.tensor(0.0, device=prediction.device)

        mu_p = pool_fn(prediction)
        mu_t = pool_fn(target)

        sigma_pp = pool_fn(prediction * prediction) - mu_p * mu_p
        sigma_tt = pool_fn(target * target) - mu_t * mu_t
        sigma_pt = pool_fn(prediction * target) - mu_p * mu_t

        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_pp + sigma_tt + C2))

        return 1.0 - ssim.mean()


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss — standard in SR papers (SRGAN, ESRGAN, SwinIR).
    
    Computes L1 distance between VGG feature maps of prediction and target.
    Works on 3D volumes by extracting 2D slices along the degraded axis.
    
    This provides gradient signal based on high-level feature similarity,
    encouraging the network to produce perceptually sharp results rather
    than just minimizing pixel MSE (which leads to blurriness).
    
    NOTE: Requires torchvision. If not available, gracefully returns 0.
    """
    def __init__(self, n_slices_per_batch: int = 4):
        super().__init__()
        self.n_slices = n_slices_per_batch
        self._vgg = None
        self._available = None
    
    def _init_vgg(self, device):
        """Lazy init VGG features — only loads when first needed."""
        if self._available is not None:
            return self._available
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]  # Up to relu3_3
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self._vgg = vgg.to(device)
            self._available = True
        except Exception:
            try:
                from torchvision.models import vgg16
                vgg = vgg16(pretrained=True).features[:16]
                vgg.eval()
                for p in vgg.parameters():
                    p.requires_grad = False
                self._vgg = vgg.to(device)
                self._available = True
            except Exception:
                self._available = False
        return self._available
    
    def _normalize_for_vgg(self, x):
        """Normalize from z-score space to VGG input range [0,1] then ImageNet normalize."""
        # Map to [0, 1] using min-max per batch
        x_min = x.amin(dim=(-1, -2), keepdim=True)
        x_max = x.amax(dim=(-1, -2), keepdim=True)
        x_01 = (x - x_min) / (x_max - x_min + 1e-8)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x_01 - mean) / std
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._init_vgg(prediction.device):
            return torch.tensor(0.0, device=prediction.device)
        
        ndim = prediction.ndim - 2
        if ndim < 2:
            return torch.tensor(0.0, device=prediction.device)
        
        # For 3D: sample random slices along axis 0 (z-axis, the degraded one)
        if ndim == 3:
            B, C, D, H, W = prediction.shape
            n = min(self.n_slices, D)
            indices = torch.randperm(D, device=prediction.device)[:n]
            pred_slices = prediction[:, :, indices].permute(0, 2, 1, 3, 4).reshape(B * n, C, H, W)
            tgt_slices = target[:, :, indices].permute(0, 2, 1, 3, 4).reshape(B * n, C, H, W)
        else:
            pred_slices = prediction
            tgt_slices = target
        
        # Repeat single channel to 3 channels for VGG
        if pred_slices.shape[1] == 1:
            pred_slices = pred_slices.expand(-1, 3, -1, -1)
            tgt_slices = tgt_slices.expand(-1, 3, -1, -1)
        
        # Normalize
        pred_norm = self._normalize_for_vgg(pred_slices)
        tgt_norm = self._normalize_for_vgg(tgt_slices)
        
        # Extract features
        with torch.no_grad():
            tgt_feat = self._vgg(tgt_norm)
        pred_feat = self._vgg(pred_norm)  # Need gradients for prediction
        
        return F.l1_loss(pred_feat, tgt_feat)


class SRLoss(nn.Module):
    """
    Combined SR loss: Charbonnier (smooth L1) + optional SSIM + optional perceptual.
    This replaces the Dice+CE loss used in segmentation.
    """
    def __init__(self, weight_charbonnier: float, weight_ssim: float,
                 weight_perceptual: float, ssim_window_size: int):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss3D(window_size=ssim_window_size)
        self.perceptual_loss = PerceptualLoss() if weight_perceptual > 0 else None
        self.weight_charbonnier = weight_charbonnier
        self.weight_ssim = weight_ssim
        self.weight_perceptual = weight_perceptual

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.weight_charbonnier * self.charbonnier(prediction, target)
        if self.weight_ssim > 0:
            loss = loss + self.weight_ssim * self.ssim_loss(prediction, target)
        if self.weight_perceptual > 0 and self.perceptual_loss is not None:
            loss = loss + self.weight_perceptual * self.perceptual_loss(prediction, target)
        return loss


# ============================================================================
#  On-the-fly degradation module
# ============================================================================

class AnisotropicDegradation(nn.Module):
    """
    Simulates anisotropic low-resolution acquisition via stride subsampling.

    Takes every N-th voxel along each degraded axis (equivalent to acquiring
    at coarser spacing), then trilinear-upsamples back to the original grid.
    
    This matches the 'stride' degradation in the data preparation script:
    physically exact simulation of what happens when you image at 40nm z
    instead of 8nm z.

    Args:
        scale_factors: (sx, sy, sz) in image space, where a factor of N means that
            axis was acquired N times coarser than the target. Note the axis
            order: image space is (x, y, z) while nnU-Net's tensors are
            (z, y, x), so the factors are reversed internally.
    """
    def __init__(self, scale_factors: Tuple[float, float, float]):
        super().__init__()
        self.scale_factors_xyz = scale_factors
        # nnUNet 3D data shape: (B, C, Z, Y, X) → internal scale = (z, y, x)
        self.scale_factors_internal = (scale_factors[2], scale_factors[1], scale_factors[0])

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        ndim = data.ndim - 2

        if ndim == 3:
            sf = self.scale_factors_internal  # (z, y, x) order
            orig_shape = data.shape[2:]  # (Z, Y, X)

            if all(f <= 1.0 for f in sf):
                return data

            # Stride subsampling: take every N-th voxel along each degraded axis
            # For (5, 1, 1): takes every 5th z-slice, keeps all y and x
            strides = [max(1, int(round(f))) for f in sf]
            lr = data[:, :, ::strides[0], ::strides[1], ::strides[2]]
            
            # Trilinear upsample back to original size
            lr = F.interpolate(lr, size=list(orig_shape), mode='trilinear', align_corners=True)

        elif ndim == 2:
            sf = (self.scale_factors_xyz[1], self.scale_factors_xyz[0])
            orig_shape = data.shape[2:]

            if all(f <= 1.0 for f in sf):
                return data

            strides = [max(1, int(round(f))) for f in sf]
            lr = data[:, :, ::strides[0], ::strides[1]]
            lr = F.interpolate(lr, size=list(orig_shape), mode='bilinear', align_corners=True)
        else:
            lr = data

        return lr


# ============================================================================
#  Main Super Resolution Trainer
# ============================================================================

class nnUNetTrainer_SuperResolution(nnUNetTrainer):
    """
    nnUNet-based Super Resolution Trainer.

    Converts the segmentation pipeline into image-to-image regression for
    volumetric super-resolution with fully flexible per-axis scaling.

    Key modifications from base nnUNetTrainer:
    1. Network: output channels = input channels (image regression, not classification)
    2. Loss: Charbonnier + SSIM (not Dice + CE)
    3. Optimizer: AdamW (not SGD) — better for regression
    4. Input: LR image created on-the-fly from HR ground truth via degradation
    5. Validation: PSNR metric instead of Dice
    6. scale_factors = (sx, sy, sz) in image (x,y,z) space — fully customizable

    The network architecture, patch size, data augmentation, and preprocessing
    all come from the standard nnUNet plans — you only need to change the trainer.
    """

    #: Optional per-axis scale pinned by a generated alias subclass (see
    #: `make_scale_trainer` below); None means "take it from the environment".
    SCALE_FACTORS = None
    #: Optional epoch count pinned by a generated alias subclass.
    NUM_EPOCHS = None

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Precedence: environment variable > trainer-name-derived value > error.
        scale_str = os.environ.get("nnUNet_SR_SCALE")
        if scale_str:
            self.scale_factors = parse_scale(scale_str)
        elif self.SCALE_FACTORS is not None:
            self.scale_factors = self.SCALE_FACTORS
        else:
            raise ValueError(
                "No per-axis scale was provided. Set nnUNet_SR_SCALE=<sx,sy,sz>, or use a "
                "trainer name that encodes it, e.g. -tr nnUNetTrainer_SR_x1y1z5."
            )

        epochs_str = os.environ.get("nnUNet_SR_EPOCHS")
        if epochs_str:
            self.num_epochs = int(epochs_str)
        elif self.NUM_EPOCHS is not None:
            self.num_epochs = self.NUM_EPOCHS
        else:
            raise ValueError(
                "No epoch count was provided. Set nnUNet_SR_EPOCHS=<n>, or use a trainer name "
                "that encodes it, e.g. -tr nnUNetTrainer_SR_x1y1z5_100ep."
            )

        self.initial_lr = _require_env("nnUNet_SR_LR", float)
        self.weight_decay = _require_env("nnUNet_SR_WEIGHT_DECAY", float)
        self.grad_clip_norm = _require_env("nnUNet_SR_GRAD_CLIP_NORM", float)
        self.ssim_window_size = _require_env("nnUNet_SR_SSIM_WINDOW_SIZE", int)

        self.weight_charbonnier = _require_env("nnUNet_SR_WEIGHT_CHARBONNIER", float)
        self.weight_ssim = _require_env("nnUNet_SR_WEIGHT_SSIM", float,
                                        "set to 0 to disable the SSIM term")
        self.weight_perceptual = _require_env("nnUNet_SR_WEIGHT_PERCEPTUAL", float,
                                              "set to 0 to disable the perceptual term")

        # Foreground oversampling is meaningless for a reconstruction task: there
        # is no foreground class to bias sampling towards.
        self.oversample_foreground_percent = 0.0

        # Optionally hold a pre-trained encoder fixed for the first N epochs so
        # the freshly initialised decoder adapts before the encoder is disturbed.
        self.freeze_encoder_epochs = _require_env(
            "nnUNet_SR_FREEZE_EPOCHS", int, "set to 0 to disable freezing")
        self._encoder_frozen = False

        # Degradation module (built in initialize())
        self._degradation = None

        # One trainer class serves every configuration, so the settings that
        # change the result are folded into the output path. Without this, two
        # runs differing only in scale or epoch count would write into the same
        # folder and overwrite each other's checkpoints.
        scale_tag = "x{:g}y{:g}z{:g}".format(*self.scale_factors)
        run_tag = f"scale-{scale_tag}_ep{self.num_epochs}"
        self.output_folder_base = self.output_folder_base + "__" + run_tag
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')
        # The base class already opened a log file under the untagged folder;
        # move it so this run logs where its checkpoints go.
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, os.path.basename(self.log_file))

        print(f"  [SR] scale_factors(x,y,z)={self.scale_factors} epochs={self.num_epochs}\n"
              f"       lr={self.initial_lr} weight_decay={self.weight_decay} "
              f"grad_clip_norm={self.grad_clip_norm}\n"
              f"       loss weights: charbonnier={self.weight_charbonnier} "
              f"ssim={self.weight_ssim} perceptual={self.weight_perceptual}\n"
              f"       freeze_encoder_epochs={self.freeze_encoder_epochs}\n"
              f"       output_folder={self.output_folder}")

    def initialize(self):
        """Build SR-specific network (regression output) and loss."""
        if not self.was_initialized:
            self._set_batch_size_and_oversample()

            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # Build network: output = same channels as input (image regression)
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.num_input_channels,  # <-- KEY CHANGE: regression output
                self.enable_deep_supervision
            ).to(self.device)

            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                from torch import distributed as dist
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # Initialize degradation module
            # Simple stride subsampling: take every Nth voxel along z
            self._degradation = AnisotropicDegradation(self.scale_factors).to(self.device)

            self.was_initialized = True
            
            # Monkey-patch nnUNet's load_pretrained_weights to use our
            # encoder-only loader. This is called by run_training.py AFTER
            # initialize() when -pretrained_weights is specified.
            try:
                import nnunetv2.run.load_pretrained_weights as lpw
                _original_load = lpw.load_pretrained_weights
                trainer_ref = self  # capture reference for closure
                
                def _patched_load(network, fname, verbose=False):
                    """Redirect to our encoder-only loader."""
                    trainer_ref.load_pretrained_encoder(fname)
                
                lpw.load_pretrained_weights = _patched_load
                self.print_to_log_file("Patched load_pretrained_weights for encoder-only SSL loading")
            except Exception as e:
                self.print_to_log_file(f"Could not patch load_pretrained_weights: {e}")
                self.print_to_log_file("If using -pretrained_weights, call load_pretrained_encoder manually")
        else:
            raise RuntimeError("Trainer already initialized.")

    def load_pretrained_encoder(self, pretrained_weights_file: str):
        """
        Properly load encoder-only pretrained weights (e.g., from nnSSL).
        
        Unlike nnUNet's default load_pretrained_weights which tries to load
        the full model (and fails/loads garbage for encoder-only checkpoints),
        this method:
        
        1. Loads ONLY encoder weights from the checkpoint
        2. Handles key mismatches between SSL and nnUNet naming conventions
        3. Leaves decoder weights randomly initialized (fresh for SR task)
        4. Sets up discriminative learning rates (encoder: lr/10, decoder: lr)
        5. Logs exactly what was loaded and what was skipped
        
        Called automatically by nnUNet's training pipeline when -pretrained_weights
        is specified — we monkey-patch load_pretrained_weights to use this instead.
        """
        from torch._dynamo import OptimizedModule
        
        self.print_to_log_file(f"\n{'='*60}")
        self.print_to_log_file(f"Loading nnSSL encoder weights from: {pretrained_weights_file}")
        
        # Load checkpoint
        saved = torch.load(pretrained_weights_file, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'network_weights' in saved:
            pretrained_dict = saved['network_weights']
        elif 'state_dict' in saved:
            pretrained_dict = saved['state_dict']
        elif 'model' in saved:
            pretrained_dict = saved['model']
        elif 'encoder' in saved:
            pretrained_dict = saved['encoder']
        else:
            # Assume the whole file is a state dict
            pretrained_dict = saved
        
        self.print_to_log_file(f"  Checkpoint keys: {len(pretrained_dict)}")
        
        # Get the actual network module
        net = self.network
        if self.is_ddp:
            net = net.module
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
        
        model_dict = net.state_dict()
        self.print_to_log_file(f"  Network keys: {len(model_dict)}")
        
        # Strip common prefixes from pretrained keys (nnSSL may add prefixes)
        cleaned_pretrained = {}
        prefixes_to_strip = ['module.', '_orig_mod.', 'backbone.', 'encoder_model.', 
                             'model.', 'network.']
        for key, val in pretrained_dict.items():
            clean_key = key
            for prefix in prefixes_to_strip:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            cleaned_pretrained[clean_key] = val
        
        # Identify which keys are encoder keys in the current model
        encoder_keys = set()
        decoder_keys = set()
        other_keys = set()
        for k in model_dict.keys():
            if 'encoder' in k or 'stages' in k or 'conv_blocks_context' in k:
                encoder_keys.add(k)
            elif 'decoder' in k or 'transpconv' in k or 'seg_layers' in k or 'conv_blocks_localization' in k:
                decoder_keys.add(k)
            else:
                other_keys.add(k)
        
        # Match pretrained weights to model keys
        # Strategy: try exact match first, then try matching encoder keys
        loaded_keys = []
        skipped_keys = []
        shape_mismatch_keys = []
        
        for model_key in model_dict.keys():
            if model_key in cleaned_pretrained:
                if model_dict[model_key].shape == cleaned_pretrained[model_key].shape:
                    model_dict[model_key] = cleaned_pretrained[model_key]
                    loaded_keys.append(model_key)
                else:
                    shape_mismatch_keys.append(
                        f"{model_key}: model={model_dict[model_key].shape} vs pretrained={cleaned_pretrained[model_key].shape}")
            else:
                skipped_keys.append(model_key)
        
        # Load the matched weights
        net.load_state_dict(model_dict)
        
        # Report
        n_encoder_loaded = sum(1 for k in loaded_keys if k in encoder_keys)
        n_decoder_loaded = sum(1 for k in loaded_keys if k in decoder_keys)
        n_encoder_skipped = sum(1 for k in skipped_keys if k in encoder_keys)
        n_decoder_skipped = sum(1 for k in skipped_keys if k in decoder_keys)
        
        self.print_to_log_file(f"\n  Loaded:  {len(loaded_keys)} params "
                               f"(encoder: {n_encoder_loaded}, decoder: {n_decoder_loaded})")
        self.print_to_log_file(f"  Skipped: {len(skipped_keys)} params "
                               f"(encoder: {n_encoder_skipped}, decoder: {n_decoder_skipped})")
        if shape_mismatch_keys:
            self.print_to_log_file(f"  Shape mismatches: {len(shape_mismatch_keys)}")
            for s in shape_mismatch_keys[:5]:
                self.print_to_log_file(f"    {s}")
        
        # Verify encoder was actually loaded
        if n_encoder_loaded == 0:
            self.print_to_log_file("\n  WARNING: No encoder weights were loaded!")
            self.print_to_log_file("  Pretrained keys (first 10):")
            for k in list(cleaned_pretrained.keys())[:10]:
                self.print_to_log_file(f"    {k}")
            self.print_to_log_file("  Model encoder keys (first 10):")
            for k in list(encoder_keys)[:10]:
                self.print_to_log_file(f"    {k}")
        else:
            self.print_to_log_file(f"\n  Encoder successfully loaded ({n_encoder_loaded} params)")
        
        # Set up discriminative learning rates: encoder lr/10, decoder full lr
        self._setup_discriminative_lr()
        
        self.print_to_log_file(f"{'='*60}\n")
    
    def _setup_discriminative_lr(self):
        """
        Set up discriminative learning rates after loading pretrained encoder.
        Encoder: lr * 0.1 (preserve pretrained features)
        Decoder: lr * 1.0 (learn SR task from scratch)
        """
        from torch._dynamo import OptimizedModule
        
        net = self.network
        if self.is_ddp:
            net = net.module
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
        
        # Collect encoder and decoder parameter ids
        encoder_param_ids = set()
        if hasattr(net, 'encoder'):
            encoder_param_ids = set(id(p) for p in net.encoder.parameters())
        elif hasattr(net, 'stages'):
            encoder_param_ids = set(id(p) for p in net.stages.parameters())
        
        if not encoder_param_ids:
            self.print_to_log_file("  Could not identify encoder for discriminative LR")
            return
        
        current_lr = self.initial_lr
        encoder_lr = current_lr * 0.1
        
        param_groups = [
            {'params': [p for p in self.network.parameters() 
                       if id(p) not in encoder_param_ids and p.requires_grad],
             'lr': current_lr, 'name': 'decoder'},
            {'params': [p for p in self.network.parameters()
                       if id(p) in encoder_param_ids and p.requires_grad],
             'lr': encoder_lr, 'name': 'encoder'},
        ]
        
        # Rebuild optimizer with discriminative LR
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=current_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        # Rebuild LR scheduler
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        self.lr_scheduler = PolyLRScheduler(self.optimizer, current_lr, self.num_epochs)
        
        n_encoder = sum(p.numel() for p in self.network.parameters() if id(p) in encoder_param_ids)
        n_decoder = sum(p.numel() for p in self.network.parameters() if id(p) not in encoder_param_ids)
        self.print_to_log_file(f"  Discriminative LR: encoder={encoder_lr:.2e} ({n_encoder:,} params), "
                               f"decoder={current_lr:.2e} ({n_decoder:,} params)")

    @staticmethod
    def get_training_transforms(
            patch_size: np.ndarray,
            rotation_for_DA: dict,
            deep_supervision_scales: Union[List, Tuple],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ):
        """
        SR-SPECIFIC augmentation: Keep ONLY spatial transforms + mirroring.

        We MUST remove intensity augmentations (noise, blur, brightness, contrast,
        gamma, SimulateLowResolution) because they corrupt the HR ground truth.
        In SR, the target IS the image data itself — any intensity modification
        to batch['data'] before our degradation step means we're training against
        a corrupted target, causing loss to increase.
        """
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
        from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
        from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
        from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Spatial transform: rotation + scaling only (no elastic deform for SR)
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # NO GaussianNoise — corrupts HR target
        # NO GaussianBlur — corrupts HR target
        # NO Brightness/Contrast — corrupts HR target
        # NO SimulateLowResolution — we do our own degradation
        # NO Gamma — corrupts HR target

        # Mirroring is safe (geometric only)
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            from batchgeneratorsv2.transforms.utils.masking import MaskImageTransform
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(RemoveLabelTansform(-1, 0))

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    def _build_loss(self):
        """SR loss with deep supervision."""
        loss = SRLoss(
            weight_charbonnier=self.weight_charbonnier,
            weight_ssim=self.weight_ssim,
            weight_perceptual=self.weight_perceptual,
            ssim_window_size=self.ssim_window_size,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_optimizers(self):
        """AdamW with PolyLR — better for image regression than SGD."""
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def _make_ds_targets(self, hr_data: torch.Tensor) -> list:
        """Create multi-scale deep supervision targets from HR image."""
        deep_supervision_scales = self._get_deep_supervision_scales()
        ndim = hr_data.ndim - 2
        targets = [hr_data]  # full resolution

        for scale in deep_supervision_scales[1:]:
            if ndim == 3:
                t = F.interpolate(hr_data, scale_factor=list(scale), mode='trilinear', align_corners=True)
            else:
                t = F.interpolate(hr_data, scale_factor=list(scale), mode='bilinear', align_corners=True)
            targets.append(t)
        return targets

    def _get_encoder(self):
        """Get the encoder submodule from the nnUNet network."""
        from torch._dynamo import OptimizedModule
        net = self.network
        if self.is_ddp:
            net = net.module
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
        # nnUNet's PlainConvUNet has .encoder attribute
        if hasattr(net, 'encoder'):
            return net.encoder
        # Fallback: try common attribute names
        for attr in ['stages', 'conv_blocks_context', 'down_path']:
            if hasattr(net, attr):
                return getattr(net, attr)
        return None

    def _freeze_encoder(self):
        """Freeze encoder parameters — only decoder trains."""
        encoder = self._get_encoder()
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = False
            self._encoder_frozen = True
            n_frozen = sum(1 for p in encoder.parameters() if not p.requires_grad)
            self.print_to_log_file(f"  Encoder FROZEN ({n_frozen} params). Only decoder will train.")
        else:
            self.print_to_log_file("  WARNING: Could not find encoder submodule — skipping freeze.")

    def _unfreeze_encoder(self):
        """Unfreeze encoder parameters for full fine-tuning."""
        encoder = self._get_encoder()
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = True
            self._encoder_frozen = False
            self.print_to_log_file("  Encoder UNFROZEN. Full network training.")
            # Re-add encoder params to optimizer with lower LR
            # This implements discriminative learning rates:
            # encoder gets lr/10, decoder keeps original lr
            encoder_params = set(id(p) for p in encoder.parameters())
            param_groups = [
                {'params': [p for p in self.network.parameters()
                           if id(p) not in encoder_params and p.requires_grad],
                 'lr': self.optimizer.param_groups[0]['lr']},
                {'params': [p for p in self.network.parameters()
                           if id(p) in encoder_params and p.requires_grad],
                 'lr': self.optimizer.param_groups[0]['lr'] * 0.1},
            ]
            # Rebuild optimizer with discriminative LR
            current_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=current_lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
            self.print_to_log_file(f"  Discriminative LR: decoder={current_lr:.2e}, encoder={current_lr*0.1:.2e}")

    def on_train_epoch_start(self):
        """Handle encoder freeze/unfreeze schedule."""
        super().on_train_epoch_start()
        if self.freeze_encoder_epochs > 0:
            if self.current_epoch == 0:
                self._freeze_encoder()
            elif self.current_epoch == self.freeze_encoder_epochs and self._encoder_frozen:
                self._unfreeze_encoder()

    def train_step(self, batch: dict) -> dict:
        """
        Residual SR training step with skip connection:
        1. HR ground truth from batch['data']
        2. Create LR input via stride subsampling + trilinear upsample
        3. Network predicts RESIDUAL from interpolated input
        4. Reconstructed output = interpolated + residual
        5. Loss compares RECONSTRUCTED OUTPUT to GT (not residual to residual!)
        
        Key insight: SSIM loss must be on (interp + residual) vs GT, not on
        residual vs (GT - interp). SSIM of near-zero residuals is meaningless.
        This ensures both PSNR and SSIM of SR output >= interpolation baseline.
        """
        data = batch['data'].to(self.device, non_blocking=True)  # HR ground truth

        # Create interpolated LR input
        with torch.no_grad():
            lr_interp = self._degradation(data)

        self.optimizer.zero_grad(set_to_none=True)

        from nnunetv2.utilities.helpers import dummy_context
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            residual_pred = self.network(lr_interp)
            
            # Reconstruct: output = interp + residual
            if self.enable_deep_supervision:
                # Deep supervision: add interp at each scale
                ds_scales = self._get_deep_supervision_scales()
                output_list = []
                target_list = [data]
                ndim = data.ndim - 2
                
                # Full resolution
                output_list.append(lr_interp + residual_pred[0])
                
                # Downsampled scales
                for i, scale in enumerate(ds_scales[1:], 1):
                    if ndim == 3:
                        lr_ds = F.interpolate(lr_interp, scale_factor=list(scale), 
                                            mode='trilinear', align_corners=True)
                        gt_ds = F.interpolate(data, scale_factor=list(scale),
                                            mode='trilinear', align_corners=True)
                    else:
                        lr_ds = F.interpolate(lr_interp, scale_factor=list(scale),
                                            mode='bilinear', align_corners=True)
                        gt_ds = F.interpolate(data, scale_factor=list(scale),
                                            mode='bilinear', align_corners=True)
                    output_list.append(lr_ds + residual_pred[i])
                    target_list.append(gt_ds)
                
                l = self.loss(output_list, target_list)
            else:
                output = lr_interp + residual_pred
                l = self.loss(output, data)

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

    def validation_step(self, batch: dict) -> dict:
        """Residual SR validation: loss on reconstructed output vs GT."""
        data = batch['data'].to(self.device, non_blocking=True)

        with torch.no_grad():
            lr_interp = self._degradation(data)

        from nnunetv2.utilities.helpers import dummy_context
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            residual_pred = self.network(lr_interp)
            
            if self.enable_deep_supervision:
                ds_scales = self._get_deep_supervision_scales()
                output_list = [lr_interp + residual_pred[0]]
                target_list = [data]
                ndim = data.ndim - 2
                for i, scale in enumerate(ds_scales[1:], 1):
                    if ndim == 3:
                        lr_ds = F.interpolate(lr_interp, scale_factor=list(scale),
                                            mode='trilinear', align_corners=True)
                        gt_ds = F.interpolate(data, scale_factor=list(scale),
                                            mode='trilinear', align_corners=True)
                    else:
                        lr_ds = F.interpolate(lr_interp, scale_factor=list(scale),
                                            mode='bilinear', align_corners=True)
                        gt_ds = F.interpolate(data, scale_factor=list(scale),
                                            mode='bilinear', align_corners=True)
                    output_list.append(lr_ds + residual_pred[i])
                    target_list.append(gt_ds)
                l = self.loss(output_list, target_list)
                pred_hr = output_list[0]
            else:
                pred_hr = lr_interp + residual_pred
                l = self.loss(pred_hr, data)

        target_hr = data
        data_range = (target_hr.max() - target_hr.min()).item()
        data_range = max(data_range, 1e-10)

        mse = F.mse_loss(pred_hr, target_hr).item()
        psnr = 10 * np.log10((data_range ** 2) / (mse + 1e-10))

        mse_interp = F.mse_loss(lr_interp, target_hr).item()
        psnr_interp = 10 * np.log10((data_range ** 2) / (mse_interp + 1e-10))

        ssim_val = self._compute_ssim_slice(pred_hr, target_hr, data_range)
        frc_ratio = self._compute_frc_resolution_ratio(pred_hr, target_hr)

        return {
            'loss': l.detach().cpu().numpy(),
            'psnr': psnr,
            'psnr_interp': psnr_interp,
            'mse': mse,
            'ssim': ssim_val,
            'frc_ratio': frc_ratio,
        }

    @torch.no_grad()
    def _compute_ssim_slice(self, pred: torch.Tensor, target: torch.Tensor,
                            data_range: float) -> float:
        """Compute SSIM on a middle 2D slice for efficiency during training."""
        # Take middle slice along z (axis 2, the degraded axis)
        mid = pred.shape[2] // 2
        pred_2d = pred[0, 0, mid]  # (H, W)
        tgt_2d = target[0, 0, mid]

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu_p = pred_2d.mean()
        mu_t = tgt_2d.mean()
        sig_p = pred_2d.std()
        sig_t = tgt_2d.std()
        sig_pt = ((pred_2d - mu_p) * (tgt_2d - mu_t)).mean()

        ssim = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sig_p**2 + sig_t**2 + C2))
        return ssim.item()

    @torch.no_grad()
    def _compute_frc_resolution_ratio(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Fourier Ring Correlation (FRC) based resolution ratio.
        
        Computes the fraction of spatial frequencies where FRC > 1/7 threshold
        (half-bit criterion). Higher ratio = more high-frequency detail preserved.
        
        This is computed on a middle 2D slice for efficiency.
        Returns: fraction of frequency rings where FRC > threshold (0 to 1).
        """
        try:
            mid = pred.shape[2] // 2
            pred_2d = pred[0, 0, mid].cpu().float().numpy()
            tgt_2d = target[0, 0, mid].cpu().float().numpy()
            
            H, W = pred_2d.shape
            
            # 2D FFT
            F_pred = np.fft.fft2(pred_2d)
            F_tgt = np.fft.fft2(tgt_2d)
            
            # Frequency grid (ring indices)
            fy = np.fft.fftfreq(H).reshape(-1, 1)
            fx = np.fft.fftfreq(W).reshape(1, -1)
            freq_radius = np.sqrt(fy**2 + fx**2)
            
            # Bin into rings
            max_freq = 0.5  # Nyquist
            n_rings = min(H, W) // 2
            ring_edges = np.linspace(0, max_freq, n_rings + 1)
            
            frc_values = []
            for i in range(n_rings):
                mask = (freq_radius >= ring_edges[i]) & (freq_radius < ring_edges[i+1])
                if mask.sum() == 0:
                    continue
                
                num = np.real(np.sum(F_pred[mask] * np.conj(F_tgt[mask])))
                den = np.sqrt(np.sum(np.abs(F_pred[mask])**2) * np.sum(np.abs(F_tgt[mask])**2))
                
                if den > 1e-10:
                    frc_values.append(num / den)
                else:
                    frc_values.append(0.0)
            
            if len(frc_values) == 0:
                return 0.0
            
            # Half-bit threshold ≈ 1/7 for typical images
            threshold = 1.0 / 7.0
            n_above = sum(1 for v in frc_values if v > threshold)
            return n_above / len(frc_values)
            
        except Exception:
            return 0.0

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Log PSNR (SR and interp baseline), SSIM, and FRC."""
        outputs_collated = collate_outputs(val_outputs)
        loss_here = np.mean(outputs_collated['loss'])
        psnr_here = np.mean(outputs_collated['psnr'])
        psnr_interp = np.mean(outputs_collated['psnr_interp'])
        ssim_here = np.mean(outputs_collated['ssim'])
        frc_here = np.mean(outputs_collated['frc_ratio'])

        if self.is_ddp:
            from torch import distributed as dist
            losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
            psnrs = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(psnrs, outputs_collated['psnr'])
            psnr_here = np.vstack(psnrs).mean()
            ssims = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(ssims, outputs_collated['ssim'])
            ssim_here = np.vstack(ssims).mean()
            frcs = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(frcs, outputs_collated['frc_ratio'])
            frc_here = np.vstack(frcs).mean()

        self.logger.log('val_losses', loss_here, self.current_epoch)
        psnr_scaled = psnr_here / 50.0
        self.logger.log('mean_fg_dice', psnr_scaled, self.current_epoch)
        self.logger.log('dice_per_class_or_region', [psnr_here], self.current_epoch)

        psnr_gain = psnr_here - psnr_interp
        self.print_to_log_file(
            f"  Val loss: {loss_here:.6f}  |  SR PSNR: {psnr_here:.2f} dB  |  "
            f"Interp PSNR: {psnr_interp:.2f} dB  |  Gain: {psnr_gain:+.2f} dB  |  "
            f"SSIM: {ssim_here:.4f}  |  FRC: {frc_here:.4f}"
        )

    def on_epoch_end(self):
        """Best model selection based on PSNR (via EMA dice slot)."""
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

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Override: skip the segmentation-based validation entirely.
        For SR we just rely on the online PSNR metric during training.
        Full SR inference should be done with predict_sr.py separately.
        """
        self.print_to_log_file("SR trainer: skipping segmentation-based validation. "
                               "Use nnunetv2/inference/predict_sr.py for full inference.")

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint with SR-specific metadata (scale_factors)."""
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                from torch._dynamo import OptimizedModule

                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
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
                    # SR-specific metadata
                    'sr_scale_factors': self.scale_factors,
                    'sr_num_output_channels': self.num_input_channels,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')


# ============================================================================
#  Trainer-name shorthand
# ============================================================================
#
# nnU-Net selects a trainer by class name (`-tr <name>`), so a name has to exist
# for every configuration one wants to launch that way. Rather than writing a
# near-identical subclass per scale and schedule -- which is where the
# hard-coded scale factors and epoch counts used to live -- names matching
#
#     nnUNetTrainer_SR_x<sx>y<sy>z<sz>[_<n>ep]
#
# are resolved on demand, so any scale works without adding code:
#
#     -tr nnUNetTrainer_SR_x1y1z5          scale (1, 1, 5); epochs still required separately
#     -tr nnUNetTrainer_SR_x2y2z5_100ep    scale (2, 2, 5), 100 epochs
#
# Equivalently, set both through the environment on the base trainer:
#
#     nnUNet_SR_SCALE=<sx,sy,sz> nnUNet_SR_EPOCHS=<n> \
#         nnUNetv2_train <DATASET> <CONFIG> <FOLD> -tr nnUNetTrainer_SuperResolution
#
# All other hyperparameters (lr, weight decay, loss weights, ...) are always
# required from the environment regardless of which form is used for scale/epochs.
#
# Starting from pre-trained weights is not part of the trainer name: pass
# `-pretrained_weights <checkpoint>` to nnUNetv2_train instead.

_SR_NAME_PATTERN = re.compile(
    r"^nnUNetTrainer_SR_x(?P<sx>\d+(?:_\d+)?)"
    r"y(?P<sy>\d+(?:_\d+)?)"
    r"z(?P<sz>\d+(?:_\d+)?)"
    r"(?:_(?P<epochs>\d+)ep)?$"
)


def _scale_component(text: str) -> float:
    """'5' -> 5.0, '1_5' -> 1.5 (underscore stands in for the decimal point)."""
    return float(text.replace("_", "."))


def make_scale_trainer(name: str):
    """Build a trainer subclass from a `nnUNetTrainer_SR_x..y..z..` name.

    Returns None when the name does not match, which keeps this module from
    claiming names that belong to other trainers during nnU-Net's class search.
    """
    match = _SR_NAME_PATTERN.match(name)
    if match is None:
        return None

    scale = tuple(_scale_component(match.group(axis)) for axis in ("sx", "sy", "sz"))
    epochs = match.group("epochs")
    attrs = {
        "SCALE_FACTORS": scale,
        "NUM_EPOCHS": int(epochs) if epochs is not None else None,
        "__doc__": f"Super-resolution trainer with per-axis scale {scale}"
                   + (f" trained for {epochs} epochs." if epochs else "."),
        "__module__": __name__,
    }
    return type(name, (nnUNetTrainer_SuperResolution,), attrs)


def __getattr__(name: str):
    """Resolve `nnUNetTrainer_SR_x..y..z..` names on first access (PEP 562).

    nnU-Net looks trainers up with hasattr/getattr over every module in the
    trainer package, so synthesising the class here is enough for `-tr` to work.
    Anything not matching the pattern raises AttributeError as usual.
    """
    cls = make_scale_trainer(name)
    if cls is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = cls          # cache so repeated lookups return the same class
    return cls
