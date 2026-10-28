#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_multiple_in_one.py
========================

TUJUAN
------
Dataset Regensburg Pediatric Appendicitis berisi gambar USG di ``US_Pictures/``.
Sebagian gambar sebenarnya adalah *screenshot* yang memuat beberapa snapshot
sekaligus (mis. dua tampilan kiri-kanan, atau grid 2x2). Daftar nama file
tersebut ada di file teks ``multiple_in_one`` (satu nama file per baris).

Skrip ini:
  1. membaca daftar ``multiple_in_one``;
  2. memecah tiap gambar di daftar itu menjadi beberapa file terpisah (panel);
  3. menyalin/mengonversi gambar lain (yang tidak perlu dipecah) ke format
     yang sama, sehingga folder output *lengkap* dan seragam;
  4. membuat ``image_manifest.csv``: satu baris per gambar hasil akhir, dengan
     kolom ``US_Number`` yang bisa langsung di-join ke ``app_data.xlsx``;
  5. membuat ``image_manifest_with_tabular.csv``: manifest di atas yang sudah
     digabung dengan data tabular (join kiri; satu pasien -> banyak gambar,
     sesuai konsep MIL / "bag" of instances).

Folder asli ``US_Pictures/`` TIDAK PERNAH diubah. Semua hasil ditulis ke folder
output baru (default: ``US_Pictures_clean/``).

CARA PAKAI
----------
    # 1) Lihat dulu bagaimana gambar akan dipotong (tidak menulis hasil akhir,
    #    hanya gambar preview dengan garis potong merah di <out>/_preview/):
    python split_multiple_in_one.py --preview

    # 2) Kalau garis potong sudah benar, jalankan proses penuh:
    python split_multiple_in_one.py

    # Opsi lain:
    #   --root  DIR   folder proyek (default: folder tempat skrip ini berada)
    #   --out   DIR   folder output (default: <root>/US_Pictures_clean)
    #   --snap        geser garis potong ke celah gelap terdekat (opsional;
    #                 default: potong tepat di tengah)
    #   --only-split  hanya tulis panel hasil pecahan, jangan salin gambar lain

Dependensi: pillow, numpy, pandas, openpyxl.

KONVENSI NAMA
-------------
Nama asli:   "<subject>.<view> <label>.<ext>"  contoh "465.6 App_M.bmp"
Nama baru :  "S<subject 4 digit>_V<view 2 digit>_P<panel 2 digit>.png"
             contoh "S0465_V06_P01.png", "S0465_V06_P02.png"
             Bila dua file asli berbagi <subject>.<view> yang sama (mis. "92.3
             App.bmp" & "92.3 App_M.bmp"), yang kedua diberi sufiks "_D02".

* ``subject``  = nomor pasien = kolom ``US_Number`` di app_data.xlsx
* ``view``     = nomor tampilan/gambar milik pasien tersebut
* ``panel``    = urutan panel hasil pecahan (kiri->kanan, atas->bawah);
                 P01 untuk gambar yang tidak dipecah.
Label asli (App, RLQ, FF, ... lihat README) disimpan di kolom ``label`` pada
manifest, bukan di nama file, supaya nama file seragam dan mudah di-parse.
Semua gambar disimpan sebagai PNG (lossless), jadi BMP/JPG/PNG jadi seragam.

BAGAIMANA GAMBAR DIPECAH
------------------------
Tiap file di ``LAYOUTS`` diberi tata letak (baris, kolom), dan dipotong TEPAT
DI TENGAH:
  * 2 panel kiri|kanan -> satu garis potong vertikal di tengah lebar (x = W/2)
  * 2 panel atas/bawah -> satu garis potong horizontal di tengah tinggi (y = H/2)
  * 4 panel (2x2)      -> garis potong di tengah lebar DAN tengah tinggi
  * (1, 1)             -> tidak dipecah, hanya dikonversi ke PNG
Opsi ``--snap`` (opsional) menggeser garis ke celah gelap terdekat.
Tidak ada resize/crop lain.

Jika ada file yang hasilnya kurang pas, cukup ubah entri di ``LAYOUTS``
(atau tambahkan baris) lalu jalankan ulang --preview.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- #
# KONFIGURASI
# --------------------------------------------------------------------------- #

# Tata letak (baris, kolom) tiap file di ``multiple_in_one``.
# (1, 2) = dua panel kiri|kanan ; (2, 2) = grid 2x2 ; (2, 1) = atas/bawah ;
# (1, 1) = tidak dipecah (hanya dikonversi).
# Nilai ini hasil pengamatan visual awal -- VERIFIKASI dengan --preview.
# File yang ada di ``multiple_in_one`` tapi tidak ada di sini dianggap (1, 2).
LAYOUTS: Dict[str, Tuple[int, int]] = {
    "9.2 Appendix.bmp": (1, 2),
    "12.2 Appendix.png": (2, 2),
    "14.1 Appendix.png": (2, 2),
    "14.3 Appendix.png": (1, 2),
    "43.1 App L and T.png": (2, 1),
    "64.2 FF retrovesical.bmp": (1, 2),
    "465.6 App_M.bmp": (1, 2),
    "465.7 App_M.bmp": (1, 2),
    "465.8 App_M.bmp": (1, 2),
    "520.3 App_M.bmp": (1, 2),
    "530.2 App.bmp": (1, 2),
    "537.6 App.bmp": (1, 2),
    "545.2 App.bmp": (1, 2),
    "587.3 App.bmp": (1, 2),
    "747.6 App.bmp": (1, 2),
    "833.2 App.bmp": (1, 2),
    "833.3 App.bmp": (1, 2),
    "833.4 App.bmp": (1, 2),
    "833.5 App.bmp": (1, 2),
    # Foto utuh (4032x3024): TIDAK dipotong, hanya dikonversi ke PNG.
    "962.3.jpg": (1, 1),
    "962.4.jpg": (1, 1),
}
DEFAULT_LAYOUT = (1, 2)

# Seberapa jauh (fraksi lebar/tinggi satu panel) garis potong boleh digeser
# ketika mencari celah gelap terdekat.
SNAP_WINDOW = 0.12

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg")

# Pola nama: "<subject>.<view>[ <label>].<ext>"
NAME_RE = re.compile(r"^(?P<subject>\d+)\.(?P<view>\d+)\s*(?P<label>.*)$")


# --------------------------------------------------------------------------- #
# PARSING NAMA FILE
# --------------------------------------------------------------------------- #

def parse_name(filename: str) -> Optional[dict]:
    """Ubah "465.6 App_M.bmp" -> {subject: 465, view: 6, label: "App_M"}.

    Menangani kasus ekstensi ganda ("1.4 Appendix.png.bmp") dan nama tanpa
    label ("962.3.jpg"). Mengembalikan None jika pola tidak cocok.
    """
    stem = filename
    # buang SEMUA ekstensi gambar di ujung (".png.bmp" -> habis keduanya)
    while stem.lower().endswith(IMG_EXTS):
        stem = stem[: stem.rfind(".")]
    m = NAME_RE.match(stem.strip())
    if not m:
        return None
    return {
        "subject": int(m["subject"]),
        "view": int(m["view"]),
        "label": m["label"].strip(),
    }


def make_id(subject: int, view: int, panel: int, dup: int = 1) -> str:
    """Bentuk image_id. ``dup`` > 1 dipakai bila dua file asli punya
    <subject>.<view> yang sama (mis. "92.3 App.bmp" dan "92.3 App_M.bmp");
    file kedua diberi sufiks "_D02" agar tidak menimpa file pertama."""
    base = f"S{subject:04d}_V{view:02d}_P{panel:02d}"
    return base if dup == 1 else f"{base}_D{dup:02d}"


# --------------------------------------------------------------------------- #
# PEMOTONGAN
# --------------------------------------------------------------------------- #

def _snap_cuts(profile: np.ndarray, n: int, window: float) -> List[int]:
    """Cari n-1 posisi potong sepanjang 1 sumbu.

    ``profile`` = rata-rata kecerahan per kolom (atau per baris). Posisi awal
    dibagi rata; tiap posisi digeser ke titik terendah (paling gelap) di dalam
    jendela +/- ``window`` * lebar-panel. window=0 -> potong rata.
    """
    length = len(profile)
    step = length / n
    cuts = []
    for i in range(1, n):
        c = int(round(i * step))
        if window > 0:
            half = max(1, int(step * window))
            lo, hi = max(1, c - half), min(length - 1, c + half)
            # haluskan agar tidak tertipu noise 1 piksel
            seg = np.convolve(profile[lo:hi], np.ones(3) / 3, mode="same")
            c = lo + int(np.argmin(seg))
        cuts.append(c)
    return cuts


def compute_boxes(img: Image.Image, rows: int, cols: int,
                  snap: bool = False) -> List[Tuple[int, int, int, int]]:
    """Hitung kotak (kiri, atas, kanan, bawah) tiap panel.

    Urutan: kiri->kanan, atas->bawah. Garis potong baris dihitung di seluruh
    gambar; garis potong kolom dihitung per pita baris (panel di baris atas
    dan bawah bisa punya celah kolom yang berbeda).
    """
    w, h = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    window = SNAP_WINDOW if snap else 0.0

    row_cuts = [0] + _snap_cuts(gray.mean(axis=1), rows, window) + [h]
    boxes = []
    for r in range(rows):
        top, bottom = row_cuts[r], row_cuts[r + 1]
        band = gray[top:bottom]
        col_cuts = [0] + _snap_cuts(band.mean(axis=0), cols, window) + [w]
        for c in range(cols):
            boxes.append((col_cuts[c], top, col_cuts[c + 1], bottom))
    return boxes


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def read_multiple_list(path: Path) -> List[str]:
    """Baca ``multiple_in_one``: satu nama file per baris, abaikan baris kosong."""
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def save_preview(img: Image.Image, boxes, out_path: Path) -> None:
    """Simpan salinan gambar dengan kotak merah + nomor panel."""
    prev = img.convert("RGB")
    d = ImageDraw.Draw(prev)
    lw = max(2, min(prev.size) // 150)
    for i, (l, t, r, b) in enumerate(boxes, 1):
        d.rectangle([l, t, r - 1, b - 1], outline=(255, 0, 0), width=lw)
        d.text((l + 6, t + 6), f"P{i:02d}", fill=(255, 255, 0))
    prev.save(out_path)


def load_tabular(xlsx: Path) -> Optional[pd.DataFrame]:
    """Baca sheet 'All cases' dari app_data.xlsx. None jika tidak ada."""
    if not xlsx.exists():
        print(f"[PERINGATAN] {xlsx.name} tidak ditemukan; manifest tabular dilewati.")
        return None
    df = pd.read_excel(xlsx, sheet_name="All cases")
    # US_Number tersimpan sebagai float (882.0) dan ada yang kosong -> Int64
    df["US_Number"] = pd.to_numeric(df["US_Number"], errors="coerce").astype("Int64")
    return df


# --------------------------------------------------------------------------- #
# PROSES UTAMA
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--preview", action="store_true",
                    help="hanya buat gambar preview garis potong, tanpa output akhir")
    ap.add_argument("--snap", action="store_true",
                    help="geser garis potong ke celah gelap terdekat "
                         "(default: potong tepat di tengah)")
    ap.add_argument("--only-split", action="store_true",
                    help="jangan salin gambar yang tidak perlu dipecah")
    args = ap.parse_args()

    root: Path = args.root
    src_dir = root / "US_Pictures"
    out_dir: Path = args.out or (root / "US_Pictures_clean")
    snap = args.snap

    multi = set(read_multiple_list(root / "data/metadata/multiple_in_one"))
    missing = [n for n in multi if not (src_dir / n).exists()]
    if missing:
        print(f"[PERINGATAN] {len(missing)} file di multiple_in_one tidak ada: {missing}")

    # ---- mode preview ------------------------------------------------------
    if args.preview:
        prev_dir = out_dir / "_preview"
        prev_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(multi):
            p = src_dir / name
            if not p.exists():
                continue
            rows, cols = LAYOUTS.get(name, DEFAULT_LAYOUT)
            if (rows, cols) == (1, 1):
                print(f"lewati : {name:32s} (tidak dipotong)")
                continue
            with Image.open(p) as im:
                boxes = compute_boxes(im, rows, cols, snap)
                save_preview(im, boxes, prev_dir / f"{Path(name).stem}_preview.png")
            print(f"preview: {name:32s} layout={rows}x{cols}")
        print(f"\nSelesai. Periksa gambar di: {prev_dir}")
        return 0

    # ---- proses penuh ------------------------------------------------------
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []
    seen_views: Dict[Tuple[int, int], int] = {}  # (subject, view) -> jumlah file
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        info = parse_name(p.name)
        if info is None:
            skipped.append(p.name)
            continue

        is_multi = p.name in multi
        if not is_multi and args.only_split:
            continue

        key = (info["subject"], info["view"])
        seen_views[key] = dup = seen_views.get(key, 0) + 1

        with Image.open(p) as im:
            im.load()
            rows, cols = LAYOUTS.get(p.name, DEFAULT_LAYOUT) if is_multi else (1, 1)
            boxes = compute_boxes(im, rows, cols, snap) if (rows, cols) != (1, 1) \
                else [(0, 0, im.width, im.height)]
            n = len(boxes)
            for k, box in enumerate(boxes, 1):
                img_id = make_id(info["subject"], info["view"], k, dup)
                out_file = img_dir / f"{img_id}.png"
                im.crop(box).convert("RGB").save(out_file)
                records.append({
                    "image_id": img_id,
                    "US_Number": info["subject"],
                    "view": info["view"],
                    "panel": k,
                    "n_panels": n,
                    "label": info["label"],
                    "was_split": n > 1,
                    "orig_file": p.name,
                    "orig_width": im.width,
                    "orig_height": im.height,
                    "box_left": box[0], "box_top": box[1],
                    "box_right": box[2], "box_bottom": box[3],
                    "path": str(out_file.relative_to(out_dir)).replace("\\", "/"),
                })

    if skipped:
        print(f"[PERINGATAN] {len(skipped)} nama file tidak cocok pola, dilewati: {skipped}")

    manifest = pd.DataFrame(records).sort_values(
        ["US_Number", "view", "panel"]).reset_index(drop=True)
    manifest["US_Number"] = manifest["US_Number"].astype("Int64")
    manifest.to_csv(out_dir / "image_manifest.csv", index=False)

    # ---- gabung dengan data tabular ---------------------------------------
    tab = load_tabular(root / "app_data.xlsx")
    if tab is not None:
        merged = manifest.merge(tab, on="US_Number", how="left", indicator="_tab")
        merged["has_tabular"] = merged.pop("_tab") == "both"
        merged.to_csv(out_dir / "image_manifest_with_tabular.csv", index=False)

        # Laporan konsistensi dua arah
        img_subj = set(manifest["US_Number"].dropna())
        tab_subj = set(tab["US_Number"].dropna())
        print("\n=== LAPORAN MAPPING ===")
        print(f"Pasien dengan gambar          : {len(img_subj)}")
        print(f"Pasien dengan data tabular    : {len(tab_subj)}")
        print(f"Gambar tanpa baris tabular    : {sorted(int(x) for x in img_subj - tab_subj)}")
        print(f"Tabular tanpa gambar          : {sorted(int(x) for x in tab_subj - img_subj)}")
        print(f"Baris tabular tanpa US_Number : {int(tab['US_Number'].isna().sum())}")
        dup = tab["US_Number"].dropna()
        dup = sorted(set(dup[dup.duplicated()]))
        if dup:
            print(f"US_Number duplikat di tabular : {dup} "
                  "(gambar pasien ini akan muncul berulang di hasil join)")

    n_split = int(manifest.loc[manifest["was_split"], "orig_file"].nunique())
    print("\n=== RINGKASAN ===")
    print(f"File asli dipecah   : {n_split}")
    print(f"Total gambar output : {len(manifest)}  -> {img_dir}")
    print(f"Manifest            : {out_dir / 'image_manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
