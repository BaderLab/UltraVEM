"""
Super-resolution inference
==========================

Applies a trained super-resolution model to a folder of low-resolution volumes.

Volumes are loaded and normalized exactly as during training, run through the
network with sliding-window tiling, and written back out. The per-axis scale is
taken from the checkpoint so inference reproduces the trained model; override it
only to deliberately apply a different scale.

Usage:
    python predict_sr.py \
        --input_folder  /path/to/imagesTs \
        --output_folder /path/to/predictions \
        --model_folder  /path/to/nnUNet_results/<dataset>/<trainer>__<plans>__<config> \
        --fold <FOLD> \
        --checkpoint <CHECKPOINT_FILE>

Run with --help for the full list of options.
"""

import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("ERROR: pip install nibabel"); sys.exit(1)

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


def load_plans_and_normalization(model_folder):
    """Load plans.json and extract normalization parameters."""
    # plans.json is in the parent of fold_X
    plans_path = os.path.join(model_folder, "plans.json")
    if not os.path.exists(plans_path):
        # Try parent directory
        plans_path = os.path.join(os.path.dirname(model_folder), "plans.json")
    
    with open(plans_path) as f:
        plans = json.load(f)
    
    # Get foreground intensity properties for z-score normalization
    fip = plans.get("foreground_intensity_properties_per_channel",
                    plans.get("foreground_intensity_properties_by_modality", {}))
    
    # Get dataset.json
    ds_path = os.path.join(model_folder, "dataset.json")
    if not os.path.exists(ds_path):
        ds_path = os.path.join(os.path.dirname(model_folder), "dataset.json")
    with open(ds_path) as f:
        dataset_json = json.load(f)
    
    return plans, fip, dataset_json


def build_network_from_checkpoint(checkpoint_path, plans, dataset_json, device):
    """Build the SR network from checkpoint metadata and load weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    trainer_name = checkpoint.get("trainer_name", "nnUNetTrainer_SuperResolution")
    sr_scale_factors = checkpoint.get("sr_scale_factors")
    if sr_scale_factors is None:
        raise SystemExit(
            "[ERROR] The checkpoint does not record 'sr_scale_factors', so the scale it was "
            "trained for is unknown.\n"
            "        Guessing would silently produce a wrongly-scaled restoration; pass the "
            "scale explicitly with --sr_factors <sx> <sy> <sz>."
        )
    sr_num_output_channels = checkpoint.get("sr_num_output_channels", 1)
    
    print(f"  Trainer: {trainer_name}")
    print(f"  SR scale factors: {sr_scale_factors}")
    print(f"  Output channels: {sr_num_output_channels}")
    
    # Build network using nnUNet's infrastructure
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    
    plans_manager = PlansManager(plans)
    
    # Get configuration name from checkpoint
    config_name = checkpoint["init_args"]["configuration"]
    configuration_manager = plans_manager.get_configuration(config_name)
    
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    
    # Find trainer class
    import nnunetv2
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name, "nnunetv2.training.nnUNetTrainer"
    )
    if trainer_class is None:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        trainer_class = nnUNetTrainer
    
    # Build with deep supervision OFF for inference
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        sr_num_output_channels,
        enable_deep_supervision=False,
    )
    
    # Load weights — use strict=False to handle deep supervision key mismatches
    weights = checkpoint["network_weights"]
    # Filter out deep supervision keys that don't exist in the inference network
    model_keys = set(network.state_dict().keys())
    filtered_weights = {k: v for k, v in weights.items() if k in model_keys}
    missing = model_keys - set(filtered_weights.keys())
    extra = set(weights.keys()) - model_keys
    if missing:
        print(f"  WARNING: Missing keys in checkpoint: {missing}")
    if extra:
        print(f"  INFO: Extra keys in checkpoint (deep supervision, ignored): {len(extra)} keys")
    
    network.load_state_dict(filtered_weights, strict=False)
    network.to(device)
    network.eval()
    
    print(f"  Network loaded successfully")
    print(f"  Patch size: {configuration_manager.patch_size}")
    
    # Get inference mirroring axes
    mirror_axes = checkpoint.get("inference_allowed_mirroring_axes", None)

    return network, configuration_manager, plans_manager, mirror_axes, sr_scale_factors


def zscore_normalize(data, fip, channel=0):
    """Apply the same z-score normalization nnUNet uses during preprocessing."""
    ch_key = str(channel)
    if ch_key in fip:
        mean_val = fip[ch_key].get("mean", 0.0)
        std_val = fip[ch_key].get("std", 1.0)
        percentile_00_5 = fip[ch_key].get("percentile_00_5", None)
        percentile_99_5 = fip[ch_key].get("percentile_99_5", None)
        
        # Clip to percentiles first (same as nnUNet)
        if percentile_00_5 is not None and percentile_99_5 is not None:
            data = np.clip(data, percentile_00_5, percentile_99_5)
        
        # Z-score normalize
        data = (data - mean_val) / max(std_val, 1e-8)
        return data, mean_val, std_val
    else:
        # Fallback: per-image z-score
        mean_val = float(data.mean())
        std_val = float(max(data.std(), 1e-8))
        data = (data - mean_val) / std_val
        return data, mean_val, std_val


def reverse_zscore(data, mean_val, std_val):
    """Reverse z-score normalization."""
    return data * std_val + mean_val


@torch.inference_mode()
def predict_volume(network, data_tensor, patch_size, device, mirror_axes=None,
                   tile_step_size=0.5, use_gaussian=True):
    """
    Sliding window prediction on a single volume.
    data_tensor: (C, D, H, W) tensor
    Returns: (C, D, H, W) numpy array
    """
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
    
    # Pad to be divisible by patch_size
    # IMPORTANT: use data mean for padding, NOT 0!
    # In z-score space, mean ≈ -1, so padding with 0 creates a ~1.0 discontinuity
    # at borders, which confuses the network and shifts output values.
    pad_value = float(data_tensor.mean())
    data_padded, slicer_revert = pad_nd_image(
        data_tensor, patch_size, "constant", {"value": pad_value}, True, None
    )
    
    spatial_shape = data_padded.shape[1:]  # (D, H, W)
    
    # Compute sliding window positions
    steps = compute_steps_for_sliding_window(spatial_shape, patch_size, tile_step_size)
    
    slicers = []
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                slicers.append(
                    tuple([slice(None), *[slice(si, si + ti) 
                           for si, ti in zip((sx, sy, sz), patch_size)]])
                )
    
    # Gaussian weighting
    if use_gaussian:
        gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1./8, 
                                     value_scaling_factor=10, device=device)
    else:
        gaussian = 1
    
    # Preallocate output
    n_out_channels = data_tensor.shape[0]  # SR: output channels = input channels
    predicted = torch.zeros((n_out_channels, *spatial_shape), dtype=torch.float32, device="cpu")
    n_preds = torch.zeros(spatial_shape, dtype=torch.float32, device="cpu")
    
    data_padded = data_padded.to(device)
    
    for sl in tqdm(slicers, desc="  Sliding window"):
        patch = data_padded[sl][None]  # (1, C, pD, pH, pW)
        
        # Forward pass
        pred = network(patch)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # Take highest resolution if deep supervision
        pred = pred[0]  # Remove batch dim -> (C, pD, pH, pW)
        
        # Test-time augmentation (mirroring)
        if mirror_axes is not None and len(mirror_axes) > 0:
            mirror_axes_shifted = [m + 2 for m in mirror_axes]  # shift for batch dim
            axes_combos = [c for i in range(len(mirror_axes_shifted)) 
                          for c in itertools.combinations(mirror_axes_shifted, i + 1)]
            for axes in axes_combos:
                flipped_input = torch.flip(data_padded[sl][None], list(axes))
                flipped_pred = network(flipped_input)
                if isinstance(flipped_pred, (list, tuple)):
                    flipped_pred = flipped_pred[0]
                pred = pred + torch.flip(flipped_pred[0], [a - 1 for a in axes])  # -1 because no batch dim
            pred = pred / (len(axes_combos) + 1)
        
        pred = pred.cpu().float()
        
        if use_gaussian:
            pred *= gaussian.cpu().float()
        
        predicted[sl] += pred
        n_preds[sl[1:]] += gaussian.cpu().float() if use_gaussian else 1
    
    # Normalize by prediction count
    predicted /= n_preds.clamp(min=1e-8)
    
    # Revert padding
    predicted = predicted[(slice(None), *slicer_revert[1:])]
    
    return predicted.numpy()


def main():
    parser = argparse.ArgumentParser(description="Minimal SR prediction - bypasses nnUNet preprocessing")
    parser.add_argument("--input_folder", "-i", required=True, help="Folder with test NIfTI images (*_0000.nii.gz)")
    parser.add_argument("--output_folder", "-o", required=True, help="Output folder for SR predictions")
    parser.add_argument("--model_folder", "-m", required=True, help="nnUNet model folder (contains fold_X)")
    parser.add_argument("--fold", "-f", type=str, default="0", help="Fold to use")
    parser.add_argument("--checkpoint", "-chk", default="checkpoint_best.pth", help="Checkpoint filename")
    parser.add_argument("--device", "-d", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--tile_step", type=float, default=0.5, help="Sliding window step size")
    parser.add_argument("--no_mirror", action="store_true", help="Disable test-time mirroring")
    parser.add_argument("--z_scale", type=int, default=None,
                       help="[Anisotropic legacy] Z-only upsampling factor. "
                            "If unset, falls back to sr_scale_factors stored in the checkpoint.")
    parser.add_argument("--sr_factors", type=int, nargs=3, default=None,
                       metavar=("X", "Y", "Z"),
                       help="Explicit (x, y, z) upsampling factors. Overrides the checkpoint. "
                            "Use this for 3D isotropic SR if the checkpoint metadata is missing.")
    parser.add_argument("--hr_z_size", type=int, default=None,
                       help="[Anisotropic legacy] Explicit HR z-size. If unset, target is "
                            "LR_z * z_scale_factor.")
    parser.add_argument("--hr_shape", type=int, nargs=3, default=None,
                       metavar=("X", "Y", "Z"),
                       help="Explicit HR (x, y, z) shape. Overrides factor-based target sizes.")
    parser.add_argument("--save_zscore", action="store_true", 
                       help="Also save the prediction in z-score space (for debugging)")
    parser.add_argument("--residual_scale", type=float, default=1.0,
                       help="Scale factor for residual (default: 1.0)")
    parser.add_argument("--adaptive_blend", action="store_true",
                       help="Suppress residual in smooth regions to preserve SSIM")
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device(args.device)
    
    # Load model
    print("=" * 60)
    print("Loading model...")
    checkpoint_path = os.path.join(args.model_folder, f"fold_{args.fold}", args.checkpoint)
    plans, fip, dataset_json = load_plans_and_normalization(args.model_folder)
    
    print(f"  Normalization parameters:")
    for ch, props in fip.items():
        print(f"    Ch {ch}: mean={props.get('mean', '?'):.4f}, std={props.get('std', '?'):.4f}, "
              f"p0.5={props.get('percentile_00_5', '?')}, p99.5={props.get('percentile_99_5', '?')}")
    
    network, config_manager, plans_manager, mirror_axes, sr_scale_factors = build_network_from_checkpoint(
        checkpoint_path, plans, dataset_json, device
    )
    
    if args.no_mirror:
        mirror_axes = None
    
    patch_size = config_manager.patch_size
    print(f"  Patch size: {patch_size}")
    print(f"  Mirror axes: {mirror_axes}")
    
    # Find test files
    input_files = sorted(glob.glob(os.path.join(args.input_folder, "*_0000.nii.gz")))
    if not input_files:
        input_files = sorted(glob.glob(os.path.join(args.input_folder, "*.nii.gz")))
    
    # ---- Resolve effective SR scale factors in (x, y, z) order ----
    # Priority: --sr_factors > --z_scale (z only) > checkpoint sr_scale_factors
    ckpt_sr_xyz = tuple(int(s) for s in sr_scale_factors)
    if args.sr_factors is not None:
        sr_xyz = tuple(int(s) for s in args.sr_factors)
        sr_source = "--sr_factors CLI override"
    elif args.z_scale is not None:
        sr_xyz = (1, 1, int(args.z_scale))
        sr_source = "--z_scale CLI (legacy z-only)"
    else:
        sr_xyz = ckpt_sr_xyz
        sr_source = "checkpoint sr_scale_factors"

    print(f"\nFound {len(input_files)} input files")
    print(f"SR scale factors (x,y,z): {sr_xyz}  [source: {sr_source}]")
    if sr_xyz != ckpt_sr_xyz:
        print(f"  NOTE: differs from checkpoint sr_scale_factors {ckpt_sr_xyz}")
    needs_upsample = any(s > 1 for s in sr_xyz)
    print(f"  Will upsample before inference: {needs_upsample}")
    print("=" * 60)
    
    for fpath in input_files:
        case_name = os.path.basename(fpath).replace("_0000.nii.gz", "").replace(".nii.gz", "")
        print(f"\n--- {case_name} ---")
        
        # Load with nibabel
        nii = nib.load(fpath)
        data_orig = nii.get_fdata(dtype=np.float32)
        affine = nii.affine
        spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float64)
        
        print(f"  Input shape: {data_orig.shape}, spacing: {spacing}")
        print(f"  Input range: [{data_orig.min():.4f}, {data_orig.max():.4f}]")
        
        # Apply z-score normalization (same as nnUNet preprocessing)
        data_norm, mean_val, std_val = zscore_normalize(data_orig.copy(), fip, channel=0)
        print(f"  After z-score: [{data_norm.min():.4f}, {data_norm.max():.4f}] "
              f"(mean={mean_val:.4f}, std={std_val:.4f})")
        
        # Convert axis order: nibabel (X,Y,Z) → SimpleITK (Z,Y,X) → transpose_forward
        transpose_forward = plans.get("transpose_forward", [0, 1, 2])
        transpose_backward = plans.get("transpose_backward", [0, 1, 2])
        
        data_zyx = data_norm.transpose(2, 1, 0)  # (X,Y,Z) → (Z,Y,X)
        data_transposed = data_zyx.transpose(transpose_forward)
        
        print(f"  Internal shape (after axis reorder): {data_transposed.shape}")
        
        data_tensor = torch.from_numpy(data_transposed[None].copy()).float()  # (1, D', H', W')
        
        # Trilinear upsample LR → HR shape, per-axis, matching the training degradation.
        # The trainer does stride(::sr_scale_factors) → trilinear upsample → network.
        # sr_xyz is in (x, y, z) order matching nibabel; map to internal axes:
        #   nibabel (X,Y,Z)  -- transpose(2,1,0) -->  (Z,Y,X)
        #   (Z,Y,X)          -- transpose_forward -->  internal
        # So scale for internal axis i = sr_zyx[transpose_forward[i]].
        if needs_upsample:
            sr_zyx = (sr_xyz[2], sr_xyz[1], sr_xyz[0])
            sr_internal = [sr_zyx[transpose_forward[i]] for i in range(3)]

            current_shape = list(data_tensor.shape[1:])  # (D, H, W) in internal order
            target_shape = [current_shape[i] * sr_internal[i] for i in range(3)]

            # Per-axis HR-shape overrides (legacy --hr_z_size and new --hr_shape).
            # --hr_shape is in (x, y, z); map to internal order then apply.
            hr_xyz_override = list(args.hr_shape) if args.hr_shape is not None else [None, None, None]
            if args.hr_z_size is not None and hr_xyz_override[2] is None:
                hr_xyz_override[2] = args.hr_z_size  # legacy
            if any(o is not None for o in hr_xyz_override):
                hr_zyx_override = [hr_xyz_override[2], hr_xyz_override[1], hr_xyz_override[0]]
                for i in range(3):
                    src_axis = transpose_forward[i]
                    if hr_zyx_override[src_axis] is not None:
                        target_shape[i] = hr_zyx_override[src_axis]

            for i in range(3):
                naive = current_shape[i] * sr_internal[i]
                if naive != target_shape[i]:
                    print(f"  NOTE: axis {i}: LR*scale={naive} ≠ HR target={target_shape[i]} "
                          f"(override applied)")

            print(f"  Upsampling per internal axis: scales={tuple(sr_internal)}")
            print(f"  Full shape: {tuple(current_shape)} → {tuple(target_shape)}")

            data_tensor = F.interpolate(
                data_tensor[None],  # (1, 1, D, H, W)
                size=target_shape,
                mode='trilinear', align_corners=True
            )[0]  # back to (1, D', H', W')
            print(f"  After upsample: {tuple(data_tensor.shape)}")
        
        # Save interpolated input for residual skip connection
        interp_zscore = data_tensor.numpy().copy()  # (1, D, H, W) in z-score space
        
        # Run prediction — network outputs RESIDUAL
        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
            residual_zscore = predict_volume(
                network, data_tensor, patch_size, device,
                mirror_axes=mirror_axes,
                tile_step_size=args.tile_step,
            )
        
        # Add residual to interpolated input (skip connection)
        scaled_residual = residual_zscore * args.residual_scale
        pred_zscore = interp_zscore + scaled_residual
        
        # Optional: adaptive blending — suppress residual where it might hurt SSIM
        # In smooth regions (low local variance), the residual adds noise.
        # In detailed regions (high local variance), the residual adds sharpness.
        if args.adaptive_blend:
            from scipy.ndimage import uniform_filter
            # Compute local variance of the interpolated input
            interp_vol = interp_zscore[0]  # (D, H, W)
            local_mean = uniform_filter(interp_vol, size=7)
            local_var = uniform_filter(interp_vol**2, size=7) - local_mean**2
            local_var = np.maximum(local_var, 0)
            
            # Normalize variance to [0, 1] — high variance = keep residual, low = suppress
            var_max = np.percentile(local_var, 99)
            alpha = np.clip(local_var / max(var_max, 1e-10), 0, 1)
            
            # Blend: in smooth regions (alpha~0), use interp; in detailed (alpha~1), use SR
            pred_zscore = interp_zscore + scaled_residual * alpha[None]
            print(f"  Adaptive blend: alpha mean={alpha.mean():.3f}, "
                  f"fraction>0.5: {(alpha>0.5).mean():.3f}")
        
        print(f"  Residual range: [{residual_zscore.min():.4f}, {residual_zscore.max():.4f}]")
        print(f"  Residual mean:  {residual_zscore.mean():.6f}, std: {residual_zscore.std():.6f}")
        if args.residual_scale != 1.0:
            print(f"  Residual scale: {args.residual_scale}")
        print(f"  Interp range:   [{interp_zscore.min():.4f}, {interp_zscore.max():.4f}]")
        print(f"  Output range:   [{pred_zscore.min():.4f}, {pred_zscore.max():.4f}]")
        
        # Reverse transpose: transpose_backward
        pred_transposed = pred_zscore[0].transpose(transpose_backward)
        # Reverse SimpleITK to nibabel order: (Z,Y,X) → (X,Y,Z)
        pred_xyz = pred_transposed.transpose(2, 1, 0)
        
        # Reverse z-score normalization
        pred_original = reverse_zscore(pred_xyz, mean_val, std_val)
        
        print(f"  Final output shape: {pred_original.shape}")
        print(f"  Final output range: [{pred_original.min():.4f}, {pred_original.max():.4f}]")
        
        # Save with isotropic HR spacing
        min_spacing = spacing.min()
        hr_affine = np.diag([min_spacing, min_spacing, min_spacing, 1.0])
        out_path = os.path.join(args.output_folder, f"{case_name}.nii.gz")
        nib.save(nib.Nifti1Image(pred_original.astype(np.float32), hr_affine), out_path)
        print(f"  Saved: {out_path}")
        
        if args.save_zscore:
            out_path_z = os.path.join(args.output_folder, f"{case_name}_zscore.nii.gz")
            nib.save(nib.Nifti1Image(pred_xyz.astype(np.float32), hr_affine), out_path_z)
            print(f"  Saved (z-score): {out_path_z}")
    
    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()