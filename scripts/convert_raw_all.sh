#!/usr/bin/env bash
set -e

export MSCONVERT_IMAGE="chambm/pwiz-skyline-i-agree-to-the-vendor-licenses"

for src in pride massive; do
  for d in /home/simonp/FoundationMSMS/data/raw/${src}/*; do
    if [[ -d "$d" ]]; then
      dataset=$(basename "$d")
      out_dir="/home/simonp/FoundationMSMS/data/mzml/${src}/${dataset}"
      python -m foundationmsms.preprocessing convert-raw --raw-dir "$d" --mzml-dir "$out_dir"
    fi
  done
done
