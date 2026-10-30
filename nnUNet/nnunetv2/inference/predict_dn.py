"""
Denoising inference
===================

Applies a trained denoising model to a folder of volumes.

The model predicts a residual on top of a Gaussian-filtered baseline, mirroring
training exactly:

    baseline  = gaussian_filter(input, sigma=baseline_sigma)
    denoised  = baseline + alpha * network(baseline)

The baseline sigma and the residual scaling default to the values recorded in
the checkpoint, so inference reproduces the trained model without extra flags.

Usage:
    python predict_dn.py \
        --input_folder  /path/to/imagesTs \
        --output_folder /path/to/predictions \
        --model_folder  /path/to/nnUNet_results/<dataset>/<trainer>__<plans>__<config> \
        --fold <FOLD> \
        --checkpoint <CHECKPOINT_FILE>

Run with --help for the full list of options.
"""

import argparse
import glob
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
    plans_path = os.path.join(model_folder, "plans.json")
    if not os.path.exists(plans_path):
        plans_path = os.path.join(os.path.dirname(model_folder), "plans.json")

    with open(plans_path) as f:
        plans = json.load(f)

    fip = plans.get("foreground_intensity_properties_per_channel",
                    plans.get("foreground_intensity_properties_by_modality", {}))

    ds_path = os.path.join(model_folder, "dataset.json")
    if not os.path.exists(ds_path):
        ds_path = os.path.join(os.path.dirname(model_folder), "dataset.json")
    with open(ds_path) as f:
        dataset_json = json.load(f)

    return plans, fip, dataset_json


def build_network_from_checkpoint(checkpoint_path, plans, dataset_json, device):
    """Build the DN network from checkpoint metadata and load weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    trainer_name = checkpoint.get("trainer_name", "nnUNetTrainer_Denoising")
    dn_baseline_sigma = checkpoint.get("dn_baseline_sigma")
    if dn_baseline_sigma is None:
        raise SystemExit(
            "[ERROR] The checkpoint does not record 'dn_baseline_sigma', so the Gaussian "
            "baseline it was trained against is unknown.\n"
            "        Guessing would silently corrupt every output; pass it explicitly with "
            "--baseline_sigma <value>."
        )
    dn_num_output_channels = checkpoint.get("dn_num_output_channels", 1)

    print(f"  Trainer: {trainer_name}")
    print(f"  Baseline sigma (voxels): {dn_baseline_sigma}")
    print(f"  Output channels: {dn_num_output_channels}")

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

    plans_manager = PlansManager(plans)
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

    # Build with deep supervision OFF
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        dn_num_output_channels,
        enable_deep_supervision=False,
    )

    # Load weights
    weights = checkpoint["network_weights"]
    model_keys = set(network.state_dict().keys())
    filtered_weights = {k: v for k, v in weights.items() if k in model_keys}
    extra = set(weights.keys()) - model_keys
    if extra:
        print(f"  INFO: {len(extra)} extra keys (deep supervision, ignored)")

    network.load_state_dict(filtered_weights, strict=False)
    network.to(device)
    network.eval()

    print(f"  Network loaded. Patch size: {configuration_manager.patch_size}")

    mirror_axes = checkpoint.get("inference_allowed_mirroring_axes", None)
    return network, configuration_manager, plans_manager, mirror_axes, dn_baseline_sigma


def zscore_normalize(data, fip, channel=0):
    """Apply the same z-score normalization nnUNet uses during preprocessing."""
    ch_key = str(channel)
    if ch_key in fip:
        mean_val = fip[ch_key].get("mean", 0.0)
        std_val = fip[ch_key].get("std", 1.0)
        p005 = fip[ch_key].get("percentile_00_5", None)
        p995 = fip[ch_key].get("percentile_99_5", None)

        if p005 is not None and p995 is not None:
            data = np.clip(data, p005, p995)

        data = (data - mean_val) / max(std_val, 1e-8)
        return data, mean_val, std_val
    else:
        mean_val = float(data.mean())
        std_val = float(max(data.std(), 1e-8))
        data = (data - mean_val) / std_val
        return data, mean_val, std_val


def reverse_zscore(data, mean_val, std_val):
    return data * std_val + mean_val


def gaussian_filter_3d_raw(data_3d, sigma):
    """Apply Gaussian filter on raw data using PyTorch."""
    return _gaussian_filter_3d_torch(data_3d, sigma)


def gaussian_filter_3d_zscore(data_3d, sigma):
    """Apply Gaussian filter on z-scored data using PyTorch. Matches trainer exactly."""
    return _gaussian_filter_3d_torch(data_3d, sigma)


def _gaussian_filter_3d_torch(data_3d, sigma):
    """Core PyTorch separable Gaussian filter. Works on any domain.

    `sigma` (voxels) must be the same value the trainer used: the network
    predicts a residual on top of this baseline, so a mismatched sigma corrupts
    every output while still looking plausible.
    """
    ks = max(int(4 * sigma + 0.5) | 1, 3)
    half = ks // 2
    coords = torch.arange(ks, dtype=torch.float32) - half
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    x = torch.from_numpy(data_3d[None, None].copy()).float()
    k_z = kernel_1d.view(1, 1, ks, 1, 1)
    out = F.conv3d(x, k_z, padding=(half, 0, 0))
    k_y = kernel_1d.view(1, 1, 1, ks, 1)
    out = F.conv3d(out, k_y, padding=(0, half, 0))
    k_x = kernel_1d.view(1, 1, 1, 1, ks)
    out = F.conv3d(out, k_x, padding=(0, 0, half))

    return out[0, 0].numpy()


@torch.inference_mode()
def predict_volume(network, data_tensor, patch_size, device, mirror_axes=None,
                   tile_step_size=0.5):
    """Sliding window prediction. data_tensor: (C, D, H, W). Returns (C, D, H, W) numpy."""
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window

    data_tensor = data_tensor.to(device)
    C = data_tensor.shape[0]
    assert C == 1, f"Expected 1 channel, got {C}"

    data_padded, slicer_revert = pad_nd_image(data_tensor.cpu().numpy(),
                                               patch_size, 'constant', {'constant_values': 0},
                                               True, None)
    data_padded = torch.from_numpy(data_padded).to(device)
    padded_shape = data_padded.shape[1:]

    steps = compute_steps_for_sliding_window(padded_shape, patch_size, tile_step_size)

    gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1./8, dtype=torch.float16,
                                device=device)

    aggregated = torch.zeros([C] + list(padded_shape), dtype=torch.float32, device=device)
    weight_map = torch.zeros([1] + list(padded_shape), dtype=torch.float32, device=device)

    import itertools
    all_positions = list(itertools.product(*steps))
    n_patches = len(all_positions)

    for idx, pos in enumerate(all_positions):
        slicers = tuple(slice(p, p + ps) for p, ps in zip(pos, patch_size))
        patch = data_padded[(slice(None),) + slicers][None]  # (1, C, pD, pH, pW)

        pred = network(patch)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = pred[0]  # remove batch dim

        aggregated[(slice(None),) + slicers] += pred * gaussian
        weight_map[(slice(None),) + slicers] += gaussian

    aggregated /= weight_map.clamp(min=1e-8)
    result = aggregated.cpu().numpy()

    # Revert padding
    result = result[tuple(slicer_revert)]
    return result


def main():
    parser = argparse.ArgumentParser(description="Denoising inference (residual on a Gaussian baseline)")
    parser.add_argument("--input_folder", "-i", required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--model_folder", "-m", required=True)
    parser.add_argument("--fold", "-f", type=int, default=0)
    parser.add_argument("--checkpoint", "-chk", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile_step", type=float, default=0.5)
    parser.add_argument("--no_mirror", action="store_true")
    parser.add_argument("--baseline_sigma", type=float, default=None,
                        help="Standard deviation, in voxels, of the Gaussian pre-filter applied "
                             "before the network. Defaults to the value recorded in the "
                             "checkpoint; override only to deliberately mismatch training and "
                             "inference.")
    parser.add_argument("--residual_alpha", type=float, default=None,
                        help="Scaling applied to the predicted residual: 0 returns the plain "
                             "Gaussian baseline, 1 applies the residual in full. Defaults to 1 "
                             "(the model as trained); lower it only to trade sharpness for "
                             "stability.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print("=" * 60)
    print("Loading DN model...")
    checkpoint_path = os.path.join(args.model_folder, f"fold_{args.fold}", args.checkpoint)
    plans, fip, dataset_json = load_plans_and_normalization(args.model_folder)

    print(f"  Normalization parameters:")
    for ch, props in fip.items():
        print(f"    Ch {ch}: mean={props.get('mean', '?'):.4f}, std={props.get('std', '?'):.4f}, "
              f"p0.5={props.get('percentile_00_5', '?')}, p99.5={props.get('percentile_99_5', '?')}")

    network, config_manager, plans_manager, mirror_axes, ckpt_baseline_sigma = build_network_from_checkpoint(
        checkpoint_path, plans, dataset_json, device
    )

    # Resolve the settings that must agree with training. Taking these from the
    # checkpoint by default means inference reproduces the trained model unless
    # the caller deliberately asks otherwise.
    baseline_sigma = args.baseline_sigma if args.baseline_sigma is not None else ckpt_baseline_sigma
    if args.baseline_sigma is not None and args.baseline_sigma != ckpt_baseline_sigma:
        print(f"  WARNING: --baseline_sigma {args.baseline_sigma!r} overrides the checkpoint's "
              f"{ckpt_baseline_sigma!r}. The Gaussian baseline will not match the one the "
              f"residual was trained against.")
    residual_alpha = 1.0 if args.residual_alpha is None else args.residual_alpha
    print(f"  Baseline sigma in use: {baseline_sigma}")
    print(f"  Residual alpha: {residual_alpha}")

    if args.no_mirror:
        mirror_axes = None

    patch_size = config_manager.patch_size
    print(f"  Patch size: {patch_size}")
    print(f"  Mirror axes: {mirror_axes}")

    # Find test files
    input_files = sorted(glob.glob(os.path.join(args.input_folder, "*_0000.nii.gz")))
    if not input_files:
        input_files = sorted(glob.glob(os.path.join(args.input_folder, "*.nii.gz")))

    print(f"\nFound {len(input_files)} input files")
    print("=" * 60)

    for fpath in input_files:
        case_name = os.path.basename(fpath).replace("_0000.nii.gz", "").replace(".nii.gz", "")
        print(f"\n--- {case_name} ---")

        # Load with SimpleITK — matches prepare script and evaluate script
        sitk_img = sitk.ReadImage(fpath)
        data_orig = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (Z, Y, X)
        spacing = sitk_img.GetSpacing()  # (x, y, z) spacing
        nii_sitk_origin = sitk_img.GetOrigin()
        nii_sitk_direction = sitk_img.GetDirection()

        print(f"  Input shape (Z,Y,X): {data_orig.shape}, spacing: {spacing}")
        print(f"  Input range: [{data_orig.min():.4f}, {data_orig.max():.4f}]")

        # Z-score normalize (same as nnUNet preprocessing)
        data_norm, mean_val, std_val = zscore_normalize(data_orig.copy(), fip, channel=0)
        print(f"  After z-score: [{data_norm.min():.4f}, {data_norm.max():.4f}] "
              f"(mean={mean_val:.4f}, std={std_val:.4f})")

        # Apply transpose_forward
        transpose_forward = plans.get("transpose_forward", [0, 1, 2])
        transpose_backward = plans.get("transpose_backward", [0, 1, 2])

        data_norm_T = data_norm.transpose(transpose_forward)
        data_raw_T = data_orig.transpose(transpose_forward)

        # Gaussian filter in Z-SCORE space (matches training exactly)
        # Training does: add noise in raw → z-score → PyTorch gaussian filter
        # Prediction:    load noisy raw → z-score → PyTorch gaussian filter
        gauss_zscore_T = gaussian_filter_3d_zscore(data_norm_T, baseline_sigma)
        gauss_tensor = torch.from_numpy(gauss_zscore_T[None].copy()).float()

        # Also compute Gaussian filter in raw space for final output composition
        gauss_raw_T = gaussian_filter_3d_raw(data_raw_T, baseline_sigma)

        print(f"  Internal shape: {data_norm_T.shape}")
        print(f"  Gauss zscore range: [{gauss_zscore_T.min():.4f}, {gauss_zscore_T.max():.4f}]")
        print(f"  Gauss raw range:    [{gauss_raw_T.min():.4f}, {gauss_raw_T.max():.4f}]")

        # Network input = Gaussian filtered z-scored image (same as training)
        # Network output = RESIDUAL in z-score space
        with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
            residual_zscore = predict_volume(
                network, gauss_tensor, patch_size, device,
                mirror_axes=mirror_axes,
                tile_step_size=args.tile_step,
            )

        # Convert residual from z-score to raw: residual_raw = residual_zscore * std
        residual_raw = residual_zscore[0] * std_val

        alpha = residual_alpha
        print(f"  Residual zscore range: [{residual_zscore.min():.4f}, {residual_zscore.max():.4f}]")

        # Final output in RAW space: gauss_raw + alpha * residual_raw
        denoised_raw_T = gauss_raw_T + alpha * residual_raw
        denoised_raw_T = np.clip(denoised_raw_T, 0.0, 1.0)

        # Reverse transpose → (Z, Y, X)
        denoised_original = denoised_raw_T.transpose(transpose_backward)

        print(f"  Final shape (Z,Y,X): {denoised_original.shape}")
        print(f"  Final range: [{denoised_original.min():.4f}, {denoised_original.max():.4f}]")

        # Save with SimpleITK — matches prepare script and evaluate script
        out_path = os.path.join(args.output_folder, f"{case_name}.nii.gz")
        out_img = sitk.GetImageFromArray(denoised_original.astype(np.float32))
        out_img.SetSpacing(tuple(float(s) for s in spacing))
        out_img.SetOrigin(tuple(float(o) for o in nii_sitk_origin))
        out_img.SetDirection(tuple(float(d) for d in nii_sitk_direction))
        sitk.WriteImage(out_img, out_path)
        print(f"  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()