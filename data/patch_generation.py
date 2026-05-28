#!/usr/bin/env python3
import os
import glob
import argparse
from itertools import product

import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from safetensors.torch import save_file

SUFFIXES = [""]
POLS = ["VV", "VH"]


def apply_azimuth_lowpass_numpy(complex_patch: np.ndarray, ratio: float) -> np.ndarray:
    """Applies a continuous top-hat filter in the Doppler domain."""
    spectrum = np.fft.fftshift(np.fft.fft(complex_patch, axis=-1), axes=-1)

    keep_bins = int(spectrum.shape[-1] * ratio)
    start_idx = (spectrum.shape[-1] - keep_bins) // 2

    mask = np.zeros_like(spectrum)
    mask[..., start_idx: start_idx + keep_bins] = 1.0

    truncated = spectrum * mask
    low_res = np.fft.ifft(np.fft.ifftshift(truncated, axes=-1), axis=-1)
    return low_res


def iter_h5_files(root_dir, recursive=True):
    pattern = "**/*.h5" if recursive else "*.h5"
    return sorted(glob.glob(os.path.join(root_dir, pattern), recursive=recursive))


def get_hw_from_h5(h5_path, group):
    with h5py.File(h5_path, "r") as f:
        g = f[group]
        for ref in ["i_VV", "i_VH"]:
            if ref in g:
                return g[ref].shape
        for k in g.keys():
            obj = g[k]
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                return obj.shape
    raise RuntimeError(f"Could not infer (H, W) from {h5_path}")


def get_dataset_pairs(h5_file, group):
    g = h5_file[group]
    pairs = []
    for sfx in SUFFIXES:
        for pol in POLS:
            i_name = f"i_{pol}{sfx}"
            q_name = f"q_{pol}{sfx}"
            if i_name not in g or q_name not in g:
                raise KeyError(f"Missing {group}/{i_name} or {group}/{q_name}")
            pairs.append((i_name, q_name, g[i_name], g[q_name]))
    return pairs


def get_block_coords_from_shape(h, w, block_size):
    h_coords = np.arange(0, h, block_size, dtype=int)
    w_coords = np.arange(0, w, block_size, dtype=int)
    return list(product(h_coords, w_coords))


def read_complex_block(dataset_pairs, h0, w0, bh, bw):
    chans = []
    for _, _, ds_i, ds_q in dataset_pairs:
        i = ds_i[h0:h0 + bh, w0:w0 + bw].astype(np.float32, copy=False)
        q = ds_q[h0:h0 + bh, w0:w0 + bw].astype(np.float32, copy=False)
        chans.append(i)
        chans.append(q)
    # Returns 4 channels: [I_vv, Q_vv, I_vh, Q_vh]
    return np.stack(chans, axis=0).astype(np.float32, copy=False)


def process_one_h5(h5_path, in_dir, out_dir, patch_size, stride, group, zero_thr, ratio_thr, block_size):
    tile_name = os.path.splitext(os.path.basename(h5_path))[0]
    rel_dir = os.path.relpath(os.path.dirname(h5_path), in_dir)
    product_name = rel_dir.split(os.sep)[0]

    product_out_dir = os.path.join(out_dir, product_name)
    os.makedirs(product_out_dir, exist_ok=True)

    try:
        H, W = get_hw_from_h5(h5_path, group)
        block_coords = get_block_coords_from_shape(H, W, block_size)

        n_total = 0
        n_saved = 0
        n_skipped = 0

        with h5py.File(h5_path, "r") as f:
            dataset_pairs = get_dataset_pairs(f, group)

            for bh0, bw0 in block_coords:
                bh = min(block_size, H - bh0)
                bw = min(block_size, W - bw0)

                bh = (bh // patch_size) * patch_size
                bw = (bw // patch_size) * patch_size

                if bh < patch_size or bw < patch_size:
                    continue

                block = read_complex_block(dataset_pairs, bh0, bw0, bh, bw)

                for h1 in range(0, bh - patch_size + 1, stride):
                    for w1 in range(0, bw - patch_size + 1, stride):
                        n_total += 1

                        # x is shape (4, patch_size, patch_size) -> [I_vv, Q_vv, I_vh, Q_vh]
                        x = block[:, h1:h1 + patch_size, w1:w1 + patch_size]

                        if ratio_thr is not None:
                            intensity_vv = x[0] ** 2 + x[1] ** 2
                            no_data_ratio = float((intensity_vv <= zero_thr).sum() / (patch_size * patch_size))
                            if no_data_ratio > ratio_thr:
                                n_skipped += 1
                                continue

                        # Create Complex patches for FFT processing
                        c_vv = x[0] + 1j * x[1]
                        c_vh = x[2] + 1j * x[3]

                        # Build the 20-channel array (5 blocks of 4 channels)
                        out_channels = [x[0], x[1], x[2], x[3]] # Logical Index 0: Full Aperture

                        # The Walk-Back Downsampling Limits
                        for ratio in [0.875, 0.750, 0.625, 0.500]:
                            low_vv = apply_azimuth_lowpass_numpy(c_vv, ratio)
                            low_vh = apply_azimuth_lowpass_numpy(c_vh, ratio)
                            out_channels.extend([
                                low_vv.real.astype(np.float32),
                                low_vv.imag.astype(np.float32),
                                low_vh.real.astype(np.float32),
                                low_vh.imag.astype(np.float32)
                            ])

                        # Stack to (20, 256, 256)
                        x_20 = np.stack(out_channels, axis=0)

                        out_h = bh0 + h1
                        out_w = bw0 + w1
                        out_path = os.path.join(
                            product_out_dir,
                            f"{tile_name}__{out_h}_{out_w}.safetensors"
                        )

                        x_t = torch.from_numpy(x_20.copy())
                        save_file({"x": x_t}, out_path)
                        n_saved += 1

        return (h5_path, n_saved, n_total, n_skipped, None)

    except Exception as e:
        return (h5_path, 0, 0, 0, str(e))


def _auto_workers():
    env = os.environ.get("JOBS", "").strip()
    if env:
        try:
            n = int(env)
        except ValueError:
            n = 0
    else:
        n = 0
    if n <= 0:
        n = os.cpu_count() or 1
    return max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root directory containing .h5 files")
    ap.add_argument("--out_dir", required=True, help="Output directory for .safetensors patches")

    # Defaults updated for physical ASR constraints
    ap.add_argument("--patch_size", type=int, default=256, help="Patch size, default: 256")
    ap.add_argument("--stride", type=int, default=256, help="Overlap stride (256 = Zero Overlap)")

    ap.add_argument("--block_size", type=int, default=1024, help="Block size. Multiple of 256.")
    ap.add_argument("--group", default="bands", help="H5 group name, default: bands")
    ap.add_argument("--no_recursive", action="store_true", help="Do not search recursively for .h5")
    ap.add_argument("--ratio_thr", type=float, default=0.5, help="Skip patch if zero ratio > ratio_thr")
    ap.add_argument("--zero_thr", type=float, default=0.0, help="Consider values <= zero_thr as zero")
    args = ap.parse_args()

    if args.stride is None:
        args.stride = args.patch_size

    if args.block_size < args.patch_size or args.block_size % args.patch_size != 0:
        raise ValueError("Invalid block_size. Must be multiple of patch_size (e.g., 1024).")

    os.makedirs(args.out_dir, exist_ok=True)
    h5_files = iter_h5_files(args.in_dir, recursive=(not args.no_recursive))
    print(f"Found {len(h5_files)} .h5 files under: {args.in_dir}")
    if not h5_files: return

    n_jobs = _auto_workers()
    print(f"Using n_jobs={n_jobs}")

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(process_one_h5)(
            p, args.in_dir, args.out_dir, args.patch_size, args.stride,
            args.group, args.zero_thr, args.ratio_thr, args.block_size,
        )
        for p in h5_files
    )

    ok = [r for r in results if r[4] is None]
    bad = [r for r in results if r[4] is not None]

    print("\nSummary")
    print(f"  Patches saved:            {sum(r[1] for r in ok)}")
    if bad:
        print("\nErrors (first 20):")
        for p, *_rest, err in bad[:20]:
            print(f"  {p}: {err}")


if __name__ == "__main__":
    main()


