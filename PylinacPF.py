# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Run pylinac PicketFence on all pf#.tif images in a folder by normalizing them
to numpy arrays and writing temporary DICOM files that pylinac can read.
"""

import argparse
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pylinac.picketfence import PicketFence

DEFAULT_REGEX = r"^pf\d+\.tif$"


def find_images(root: Path, pattern_regex: str, recursive: bool):
    rx = re.compile(pattern_regex, re.IGNORECASE)
    glob = "**/*" if recursive else "*"
    for p in root.glob(glob):
        if p.is_file() and rx.match(p.name):
            yield p


def load_and_normalize(path: Path) -> np.ndarray:
    """Load TIFF -> grayscale float32 array scaled 0-1."""
    img = Image.open(path)

    if img.mode not in ("L", "I;16"):
        img = ImageOps.grayscale(img)

    arr = np.array(img).astype(np.float32)

    mn = float(arr.min())
    mx = float(arr.max())
    if mx == mn:
        raise ValueError("Image is uniform; cannot analyze.")

    return (arr - mn) / (mx - mn)


def write_array_to_dicom(arr: np.ndarray, path: Path) -> Path:
    """Persist the array to a minimal DICOM file for pylinac to read."""
    arr_uint16 = np.clip(arr, 0, 1)
    arr_uint16 = (arr_uint16 * np.iinfo(np.uint16).max).astype(np.uint16)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "RTIMAGE"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S")
    ds.Rows, ds.Columns = arr_uint16.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [1.0, 1.0]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PixelData = arr_uint16.tobytes()

    ds.save_as(path, write_like_original=False)
    return Path(path)


def run_picket_fence(
    arr: np.ndarray,
    tol: float,
    action: float,
    *,
    height_threshold: float,
    required_prominence: float,
    invert: bool,
) -> PicketFence:
    """Run PicketFence on a numpy array by writing a temporary DICOM file."""
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp_path = write_array_to_dicom(arr, Path(tmp.name))

    pf = None
    try:
        pf = PicketFence(tmp_path)

        if tol < action:
            tol, action = action, tol

        pf.analyze(
            tol,
            action,
            height_threshold=height_threshold,
            required_prominence=required_prominence,
            invert=invert,
        )
        return pf
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tol", type=float, default=0.7)
    ap.add_argument("--action", type=float, default=0.5)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--pattern", default=DEFAULT_REGEX)
    ap.add_argument("--height-threshold", type=float, default=0.1)
    ap.add_argument("--prominence", type=float, default=0.05)
    ap.add_argument("--invert", action="store_true")
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
            pf = run_picket_fence(
                arr,
                args.tol,
                args.action,
                height_threshold=args.height_threshold,
                required_prominence=args.prominence,
                invert=args.invert,
            )

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
