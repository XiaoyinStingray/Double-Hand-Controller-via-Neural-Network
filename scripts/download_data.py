"""Download and extract the IPN Hand / IPN HandS data used by this project.

    python scripts/download_data.py                    # annotations + landmarks + videos
    python scripts/download_data.py --only landmarks
    python scripts/download_data.py --only mp4 --skip-extract
    python scripts/download_data.py --only mp-hands    # optional MediaPipe baseline

Data goes to data/raw/ (gitignored). Checksums are verified when known; a
mismatch is reported but does not abort, because semantic checks are done by
scripts/audit_ipn.py.

Note: the "mp4" archives actually contain .avi files under videos/.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

GDRIVE_IDS = {
    "annotations": "1HybcqZqnNKZGgJJXCMjj6tLFbpsNP_a1",
    "landmarks": "1pbCWxfzGZyexHe89lZtGgnHWfa958XJv",
    # NOTE: the MP Hands file is currently 404 on Google Drive; MP Holistic
    # contains the same hand_landmarks and is used as the MediaPipe baseline.
    "mp-hands": "1Y-3_MUHJ6OYpv-BNof5vYDRR4YNgrGQ",
    "mp-holistic": "1hdtmr4JpK14rt1zMcbN39-oTedn2sBx1",
    "mp4": "1O4Fn_jbAEcKIksXHmQIMcHTyaDw4wt9x",
}

EXPECTED_SHA256 = {
    # NOTE: the README currently lists 4686a2eb... but the served file is this
    # one (content audited and consistent). Kept here to detect future changes.
    "annotations": "520dfbf3d52ecab6535d8841bba55e94d6b512596ed1cff4505d2c1345bcbaec",
    "landmarks": "493bf53638d5b1477e28f05c41884f0cabdcb7b84f20d68f3d5d94e0b46fad4d",
    "mp-hands": "e511bb2b902207b82a296190daadc30a057f3abc16f628579542d2b1274ce598",
    "mp-holistic": "4746db8afa7376ae647c4c61c58cba963aa22fbc3458edbc06fc88775ab972b8",
}

ZIPS = {
    "annotations": (RAW / "IPN_HandS_annotations.zip", RAW / "IPN_HandS_annotations"),
    "landmarks": (RAW / "IPN_HandS_landmarks_GT.zip", RAW / "IPN_HandS_landmarks_GT"),
    "mp-hands": (RAW / "IPN_HandS_landmarks_MPHands.zip", RAW / "IPN_HandS_landmarks_MPHands"),
    "mp-holistic": (RAW / "IPN_HandS_landmarks_MPHolistic.zip",
                    RAW / "IPN_HandS_landmarks_MPHolistic"),
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def find_videos(path: Path):
    return [p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS]



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gdown(file_id: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get] {file_id} -> {out}")
    subprocess.run(
        [sys.executable, "-m", "gdown", file_id, "-O", str(out)],
        check=True,
    )


def gdown_folder(folder_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[get] folder {folder_id} -> {out_dir}")
    subprocess.run(
        [sys.executable, "-m", "gdown", "--folder", folder_id, "-O", str(out_dir)],
        check=True,
    )


def unzip(zip_path: Path, out_dir: Path, force: bool = False) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[skip] {out_dir} already extracted")
        return
    print(f"[unzip] {zip_path.name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def extract_tgz(folder: Path, out_dir: Path, force: bool = False) -> None:
    archives = sorted(folder.rglob("*.tgz"))
    if not archives:
        print(f"[warn] no .tgz found under {folder}")
        return
    existing = find_videos(out_dir)
    if existing and not force:
        print(f"[skip] {len(existing)} video files already extracted")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for tgz in archives:
        print(f"[untar] {tgz.name} ...")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(out_dir)
    videos = find_videos(out_dir)
    print(f"[ok] {len(videos)} video files under {out_dir}")


def get_zip(kind: str, force: bool) -> None:
    zip_path, out_dir = ZIPS[kind]
    if not zip_path.exists() or force:
        gdown(GDRIVE_IDS[kind], zip_path)
    else:
        print(f"[skip] {zip_path.name} exists")
    expected = EXPECTED_SHA256.get(kind)
    if expected:
        actual = sha256(zip_path)
        if actual == expected:
            print(f"[ok] sha256 {actual[:12]}...")
        else:
            print(f"[warn] sha256 mismatch: {actual} != {expected}")
    unzip(zip_path, out_dir, force=force)


def get_mp4(force: bool, skip_extract: bool) -> None:
    folder = RAW / "ipn_mp4"
    if not any(folder.rglob("*.tgz")) or force:
        gdown_folder(GDRIVE_IDS["mp4"], folder)
    else:
        print(f"[skip] tgz files already in {folder}")
    if not skip_extract:
        # The archives contain a top-level videos/ directory, so extract into
        # ipn_mp4/ directly to get ipn_mp4/videos/<video>.avi.
        extract_tgz(folder, folder, force=force)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "annotations", "landmarks",
                                           "mp-hands", "mp-holistic", "mp4"], default="all")
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("--skip-extract", action="store_true",
                        help="download mp4 tgz but do not extract")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    kinds = ["annotations", "landmarks"] if args.only == "all" else [args.only]

    for kind in kinds:
        if kind == "mp4":
            get_mp4(args.force, args.skip_extract)
        else:
            get_zip(kind, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
