import argparse
import csv
import glob
import os

import h5py
import numpy as np
from tqdm import tqdm


def _compute_patch_starts(length, patch_size, stride):
    """Compute start indices that tile an axis with overlap and include the trailing edge."""
    if length < patch_size:
        return []

    starts = list(range(0, length - patch_size + 1, stride))
    last_valid = length - patch_size
    if starts[-1] != last_valid:
        starts.append(last_valid)
    return starts


def crop_h5_dataset_to_overlapping_patches(
    input_root,
    output_root,
    patch_size=128,
    overlap=64,
    overwrite=False,
    manifest_name="patch_manifest.csv",
    ref_band="bands/i_VV",
):
    """Create overlapping H5 patches and a CSV manifest describing them."""
    if overlap >= patch_size:
        raise ValueError("`overlap` must be smaller than `patch_size`.")

    stride = patch_size - overlap
    source_files = sorted(glob.glob(os.path.join(input_root, "**", "*.h5"), recursive=True))

    if not source_files:
        raise FileNotFoundError(f"No .h5 files found in {input_root}")

    os.makedirs(output_root, exist_ok=True)
    manifest_path = os.path.join(output_root, manifest_name)
    if os.path.exists(manifest_path) and not overwrite:
        raise FileExistsError(
            f"Manifest already exists at {manifest_path}. Use overwrite=True to replace it."
        )

    manifest_rows = []
    written_files = 0

    for src_path in tqdm(source_files, desc="Tiling Full Scenes"):
        with h5py.File(src_path, "r") as f_src:
            dataset_paths = []

            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dataset_paths.append(name)

            f_src.visititems(_collect)

            if not dataset_paths:
                continue

            ref_dataset_path = ref_band if ref_band in f_src else dataset_paths[0]
            ref_shape = f_src[ref_dataset_path].shape
            if len(ref_shape) < 2:
                continue

            height, width = ref_shape[-2], ref_shape[-1]
            row_starts = _compute_patch_starts(height, patch_size, stride)
            col_starts = _compute_patch_starts(width, patch_size, stride)

            if not row_starts or not col_starts:
                continue

            rel_dir = os.path.relpath(os.path.dirname(src_path), input_root)
            base_name = os.path.splitext(os.path.basename(src_path))[0]

            for top in row_starts:
                for left in col_starts:
                    patch_name = f"{base_name}_y{top:04d}_x{left:04d}.h5"
                    patch_dir = output_root if rel_dir == "." else os.path.join(output_root, rel_dir)
                    os.makedirs(patch_dir, exist_ok=True)
                    patch_path = os.path.join(patch_dir, patch_name)

                    if os.path.exists(patch_path):
                        if not overwrite:
                            raise FileExistsError(
                                f"Patch already exists at {patch_path}. Use overwrite=True to replace it."
                            )
                        os.remove(patch_path)

                    ref_patch = f_src[ref_dataset_path][top:top + patch_size, left:left + patch_size]
                    null_ratio = float(np.count_nonzero(ref_patch == 0) / ref_patch.size)

                    with h5py.File(patch_path, "w") as f_patch:
                        for dset_path in dataset_paths:
                            src_dset = f_src[dset_path]
                            data = src_dset[...]
                            if data.ndim >= 2:
                                slices = [slice(None)] * data.ndim
                                slices[-2] = slice(top, top + patch_size)
                                slices[-1] = slice(left, left + patch_size)
                                data = data[tuple(slices)]

                            dset = f_patch.create_dataset(dset_path, data=data)
                            for attr_name, attr_val in src_dset.attrs.items():
                                dset.attrs[attr_name] = attr_val

                        for attr_name, attr_val in f_src.attrs.items():
                            f_patch.attrs[attr_name] = attr_val

                        f_patch.attrs["source_file"] = src_path
                        f_patch.attrs["source_product"] = base_name
                        f_patch.attrs["crop_top"] = int(top)
                        f_patch.attrs["crop_left"] = int(left)
                        f_patch.attrs["src_coords"] = [int(top), int(left)]
                        f_patch.attrs["patch_size"] = int(patch_size)
                        f_patch.attrs["overlap"] = int(overlap)
                        f_patch.attrs["null_ratio"] = null_ratio

                    manifest_rows.append({
                        "patch_path": os.path.relpath(patch_path, output_root),
                        "patch_name": patch_name,
                        "source_file": src_path,
                        "source_product": base_name,
                        "ref_band": ref_dataset_path,
                        "y": int(top),
                        "x": int(left),
                        "patch_size": int(patch_size),
                        "overlap": int(overlap),
                        "height": int(patch_size),
                        "width": int(patch_size),
                        "null_ratio": null_ratio,
                    })
                    written_files += 1

    with open(manifest_path, "w", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "patch_path",
                "patch_name",
                "source_file",
                "source_product",
                "ref_band",
                "y",
                "x",
                "patch_size",
                "overlap",
                "height",
                "width",
                "null_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"Done. Saved {written_files} cropped patches of "
        f"{patch_size}x{patch_size} with overlap {overlap} to {output_root}. "
        f"Manifest written to {manifest_path}."
    )


def tile_to_patch(
    input_dir,
    output_dir,
    patch_size=128,
    overlap=64,
    overwrite=False,
    manifest_name="patch_manifest.csv",
    ref_band="bands/i_VV",
):
    """Compatibility wrapper around the overlapping H5 tiling utility."""
    return crop_h5_dataset_to_overlapping_patches(
        input_root=input_dir,
        output_root=output_dir,
        patch_size=patch_size,
        overlap=overlap,
        overwrite=overwrite,
        manifest_name=manifest_name,
        ref_band=ref_band,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilities for generating overlapping H5 patches.")
    parser.add_argument("--crop", action="store_true", help="Create cropped overlapping patches from H5 files.")
    parser.add_argument("--input-root", type=str, help="Root directory containing source .h5 files.")
    parser.add_argument("--output-root", type=str, help="Destination directory for cropped .h5 patches.")
    parser.add_argument("--patch-size", type=int, default=128, help="Square patch size in pixels.")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap with neighbouring patches in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already existing patch files.")
    parser.add_argument("--manifest-name", type=str, default="patch_manifest.csv", help="CSV manifest filename.")
    parser.add_argument("--ref-band", type=str, default="bands/i_VV", help="Reference band for tiling and no-data ratio.")
    args = parser.parse_args()

    if args.crop:
        if not args.input_root or not args.output_root:
            raise ValueError("--input-root and --output-root are required when using --crop.")

        crop_h5_dataset_to_overlapping_patches(
            input_root=args.input_root,
            output_root=args.output_root,
            patch_size=args.patch_size,
            overlap=args.overlap,
            overwrite=args.overwrite,
            manifest_name=args.manifest_name,
            ref_band=args.ref_band,
        )
