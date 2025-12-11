# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Run pylinac PicketFence on all pf#.tif images in a folder,
using numpy arrays and pylinac.core.image.Image to bypass pydicom.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pylinac.core.image import Image as PylinacImage
from pylinac.picketfence import PicketFence

DEFAULT_REGEX = r"^pf\d+\.tif$"


def find_images(root: Path, pattern_regex: str, recursive: bool):
    rx = re.compile(pattern_regex, re.IGNORECASE)
    glob = "**/*" if recursive else "*"
    for p in root.glob(glob):
        if p.is_file() and rx.match(p.name):
            yield p


def load_and_normalize(path: Path) -> np.ndarray:
    """Load TIFF -> grayscale float32 array 0–1."""
    img = Image.open(path)

    if img.mode not in ("L", "I;16"):
        img = ImageOps.grayscale(img)

    arr = np.array(img).astype(np.float32)

    mn = float(arr.min())
    mx = float(arr.max())
    if mx == mn:
        raise ValueError("Image is uniform; cannot analyze.")

    return (arr - mn) / (mx - mn)


def build_pylinac_image(arr: np.ndarray) -> PylinacImage:
    """Construct a pylinac Image object that NEVER touches pydicom."""
    # The Image class accepts raw numpy
    img = PylinacImage(arr)

    # Ensure metadata required by PicketFence exists
    # (pixel spacing defaults to 1 if missing, which is fine)
    if not hasattr(img, "dpmm"):
        # fake dpmm to avoid NaN issues: assume 1 pixel = 1 mm
        img._dpmm = 1.0
    if not hasattr(img, "cax"):
        img.cax = (arr.shape[1] // 2, arr.shape[0] // 2)

    return img


def run_picket_fence(arr: np.ndarray, tol: float, action: float) -> PicketFence:
    """Run PicketFence on a numpy array via pylinac Image wrapper."""
    image_obj = build_pylinac_image(arr)
    pf = PicketFence(image_obj)

    # pylinac version differences: enforce tolerance ordering
    if tol < action:
        tol, action = action, tol

    # many older versions require positional args only
    try:
        pf.analyze(tol, action)
    except TypeError:
        pf.analyze(tol)

    return pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tol", type=float, default=0.7)
    ap.add_argument("--action", type=float, default=0.5)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--pattern", default=DEFAULT_REGEX)
    args = ap.parse_args()

    if args.tol < args.action:
        print("Swapping tol/action to satisfy pylinac.")
        args.tol, args.action = args.action, args.tol

    root = Path(args.folder)
    tifs = list(find_images(root, args.pattern, args.recursive))

    if not tifs:
        print("No matching TIFF files found.")
        sys.exit(1)

    print(f"Found {len(tifs)} image(s).\n")
    failures = 0

    for tif in sorted(tifs):
        print(f"Processing {tif.name}...")
        try:
            arr = load_and_normalize(tif)
            pf = run_picket_fence(arr, args.tol, args.action)

            out_pdf = tif.with_name(f"{tif.stem}_PF.pdf")
            pf.publish_pdf(str(out_pdf))

            print(f"  Saved {out_pdf.name}")
        except Exception as e:
            failures += 1
            print(f"[ERROR] {tif.name}: {e}")

    if failures:
        print(f"\nCompleted with {failures} error(s).")
    else:
        print("\nCompleted without errors.")


if __name__ == "__main__":
    main()
