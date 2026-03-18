#!/usr/bin/env python3
"""
Voxelize all mzML files in two source directories, preserving provenance in output structure and metadata.

- Source 1: mzMLs produced from conversion
  /home/simonp/FoundationMSMS/data/mzml/massive/MSV000083793
- Source 2: mzMLs downloaded directly
  /home/simonp/FoundationMSMS/data/ccms_peak/massive/MSV000083793/DIA_raw
- Output: Voxel files in
  /home/simonp/FoundationMSMS/data/voxel/massive/MSV000083793/{from_converted_raw,from_downloaded}
  Each voxel .npz will have a .meta.json with provenance info.

Usage:
    python -m foundationmsms.preprocessing.voxelize_with_provenance --msv MSV000083793 --provenance downloaded
"""
from pathlib import Path
import json
from datetime import datetime
import argparse
from foundationmsms.preprocessing.voxelize import mzml_to_voxel_npz


def main():
    parser = argparse.ArgumentParser(description="Voxelize mzML files with provenance tracking.")
    parser.add_argument("--msv", required=True, help="MSV accession, e.g. MSV000083793")
    parser.add_argument("--provenance", required=True, choices=["converted", "downloaded"], help="Which mzML source to use: 'converted' or 'downloaded'")
    parser.add_argument("--mz-bin", type=float, default=1.0)
    parser.add_argument("--mz-parent-bin", type=float, default=1.0)
    parser.add_argument("--rt-bin-sec", type=float, default=1.0)
    args = parser.parse_args()

    msv = args.msv
    provenance = args.provenance

    if provenance == "converted":
        src_dir = Path(f"/home/simonp/FoundationMSMS/data/mzml/massive/{msv}")
        out_voxel_dir = Path(f"/home/simonp/FoundationMSMS/data/voxel/massive/{msv}/from_converted_raw/npz")
        out_meta_dir = Path(f"/home/simonp/FoundationMSMS/data/voxel/massive/{msv}/from_converted_raw/meta")
    else:
        src_dir = Path(f"/home/simonp/FoundationMSMS/data/ccms_peak/massive/{msv}/DIA_raw")
        out_voxel_dir = Path(f"/home/simonp/FoundationMSMS/data/voxel/massive/{msv}/from_downloaded/npz")
        out_meta_dir = Path(f"/home/simonp/FoundationMSMS/data/voxel/massive/{msv}/from_downloaded/meta")

    out_voxel_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    for mzml in src_dir.glob("*.mzML"):
        voxel_path = out_voxel_dir / (mzml.stem + ".npz")
        meta_path = out_meta_dir / (mzml.stem + ".meta.json")
        print(f"Voxelizing {mzml} -> {voxel_path}\nMeta: {meta_path}")
        try:
            mzml_to_voxel_npz(
                mzml, voxel_path,
                mz_bin=args.mz_bin,
                mz_parent_bin=args.mz_parent_bin,
                rt_bin_sec=args.rt_bin_sec
            )
            meta = {
                "source_mzml": str(mzml.resolve()),
                "provenance": provenance,
                "created": datetime.now().isoformat(),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"Failed to voxelize {mzml}: {e}")

if __name__ == "__main__":
    main()
