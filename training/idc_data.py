"""Pull real CT/MR images + their SEG masks from IDC's open buckets (the SEGRoulette data source).

Each segroulette.json row: c=source series crdc uuid, s=seg series crdc uuid, m=modality (CT/MR/PT),
cb/sb=buckets, col=collection. IDC open buckets key objects under <crdc_series_uuid>/, fetched
anonymously. Returns (volume float32 [z,y,x], labelmap uint8 [z,y,x]) on the SOURCE grid.

NOTE: IDC segs are used only to place realistic fg/bg prompts on real anatomy — NOT as ground truth.
The teacher labels the distillation target; the seg just tells us where a plausible object is.
"""
import os, json, numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import SimpleITK as sitk
import pydicom
import highdicom as hd

_s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))


def _download_series(bucket, crdc_uuid, dest):
    os.makedirs(dest, exist_ok=True)
    paths, tok = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": crdc_uuid + "/"}
        if tok:
            kw["ContinuationToken"] = tok
        r = _s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            base = os.path.basename(o["Key"])
            if not base or o["Key"].endswith("/"):
                continue                       # skip S3 "directory marker" keys
            p = os.path.join(dest, base)
            _s3.download_file(bucket, o["Key"], p)
            paths.append(p)
        if r.get("IsTruncated"):
            tok = r["NextContinuationToken"]
        else:
            break
    return paths


def _hdvol_to_sitk(vol):
    """highdicom Volume (array [d0,d1,d2] + affine voxel->LPS) -> SimpleITK image (LPS)."""
    A = np.asarray(vol.affine, dtype=float)
    M, origin = A[:3, :3], A[:3, 3]
    sp = np.linalg.norm(M, axis=0)                 # spacing along array axes d0,d1,d2
    D = M / sp                                     # unit direction columns for d0,d1,d2
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.array.astype(np.uint8)))
    # sitk array is [z,y,x] == [d0,d1,d2]; physical index order (x,y,z) == (d2,d1,d0)
    img.SetSpacing((float(sp[2]), float(sp[1]), float(sp[0])))
    img.SetDirection(np.stack([D[:, 2], D[:, 1], D[:, 0]], axis=1).flatten().tolist())
    img.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
    return img


def _resample_iso(img, spacing, is_label):
    osp, osz = np.array(img.GetSpacing()), np.array(img.GetSize())
    nsz = np.maximum(1, np.round(osz * osp / spacing)).astype(int).tolist()
    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing((spacing, spacing, spacing))
    rf.SetSize(nsz)
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetOutputDirection(img.GetDirection())
    rf.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return rf.Execute(img)


def load_case(row, workdir, spacing=None):
    """Return (volume[z,y,x] float32, labelmap[z,y,x] uint8), optionally resampled to `spacing` mm iso.

    Source via SimpleITK; SEG via highdicom get_volume() (per-frame position + orientation), then
    resample onto the source grid — correct regardless of the SEG's row/slice orientation."""
    src_dir, seg_dir = os.path.join(workdir, "src"), os.path.join(workdir, "seg")
    _download_series(row["cb"], row["c"], src_dir)
    seg_paths = _download_series(row["sb"], row["s"], seg_dir)

    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(src_dir)
    if not ids:
        raise RuntimeError("no DICOM series in source")
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(src_dir, ids[0]))
    img = reader.Execute()

    seg = hd.seg.segread(seg_paths[0])
    seg_vol = seg.get_volume(combine_segments=True, relabel=False, skip_overlap_checks=True)
    seg_img = _hdvol_to_sitk(seg_vol)
    res = sitk.Resample(seg_img, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    if spacing:
        img = _resample_iso(img, spacing, False)
        res = sitk.Resample(seg_img, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    label = sitk.GetArrayFromImage(res).astype(np.uint8)
    if label.shape != vol.shape:
        raise RuntimeError(f"seg {label.shape} != vol {vol.shape}")
    return vol, label


if __name__ == "__main__":
    import sys, tempfile
    rows = json.load(open(os.path.expanduser("~/segroulette.json"))).get("rows")
    ctmr = [r for r in rows if r.get("m") in ("CT", "MR")]
    r = ctmr[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
    print("case:", r["col"], r["m"], r["sd"], f'{r.get("sz")}MB')
    with tempfile.TemporaryDirectory() as wd:
        vol, lab = load_case(r, wd)
    print("VOL", vol.shape, "range", round(float(vol.min()), 1), round(float(vol.max()), 1))
    print("LABEL", lab.shape, "segments", sorted(int(x) for x in np.unique(lab)),
          "fg_voxels", int((lab > 0).sum()))
