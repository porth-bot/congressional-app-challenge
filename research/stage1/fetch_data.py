"""Pull balanced subsets of the public benchmark corpora.

ASVspoof 2019 LA (train split for training, eval split for testing, with
unseen attacks A07-A19) comes from the Hugging Face parquet mirror
Bisher/ASVspoof_2019_LA. In-the-Wild (Mueller et al. 2022) comes from the
official zip on mueller91/In-The-Wild; we read single files out of the zip
with HTTP range requests instead of downloading all 8 GB.

Writes 16 kHz mono FLAC files to data/raw/<corpus>/ and data/manifest.csv.
"""
import argparse
import csv
import io

import random
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SR = 16000


def write_audio(path, y, sr):
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if y.ndim > 1:
        y = y.mean(axis=1)
    sf.write(path, y.astype(np.float32), SR, format="FLAC")


def fetch_asvspoof(split, n_per_class, seed, rows_out):
    fs = HfFileSystem()
    path = f"datasets/Bisher/ASVspoof_2019_LA/data/{split}-00000-of-00001.parquet"
    pf = pq.ParquetFile(fs.open(path, block_size=8 << 20))
    meta = pf.read(columns=["audio_file_name", "speaker_id", "system_id", "key"]).to_pandas()
    rng = random.Random(seed)

    # Row groups hold 100 clips each, and rows are in random label order, so
    # sampling rows uniformly would touch nearly every group (the whole file).
    # Instead pick random groups and take every bona fide clip in them until we
    # have enough, then draw spoofs from those same groups, balanced by attack.
    starts, total = [], 0
    for g in range(pf.num_row_groups):
        starts.append(total)
        total += pf.metadata.row_group(g).num_rows
    order = list(range(pf.num_row_groups))
    rng.shuffle(order)
    bona, chosen_groups = [], []
    for g in order:
        lo, hi = starts[g], starts[g] + pf.metadata.row_group(g).num_rows
        rows = meta.index[(meta.index >= lo) & (meta.index < hi) & (meta["key"] == 0)].tolist()
        bona += rows
        chosen_groups.append(g)
        if len(bona) >= n_per_class:
            break
    bona = sorted(rng.sample(bona, n_per_class))
    in_groups = np.zeros(len(meta), bool)
    for g in chosen_groups:
        in_groups[starts[g]: starts[g] + pf.metadata.row_group(g).num_rows] = True
    spoof_pool = meta.index[in_groups & (meta["key"] == 1).values].tolist()
    by_sys = {}
    for i in spoof_pool:
        by_sys.setdefault(meta.at[i, "system_id"], []).append(i)
    systems = sorted(by_sys)
    spoof = []
    for k, s in enumerate(systems):
        want = n_per_class // len(systems) + (1 if k < n_per_class % len(systems) else 0)
        spoof += rng.sample(by_sys[s], min(want, len(by_sys[s])))
    picks = sorted(bona + spoof)

    starts = np.array(starts)
    groups = {}
    for r in picks:
        g = int(np.searchsorted(starts, r, side="right") - 1)
        groups.setdefault(g, []).append(r - starts[g])

    out_dir = RAW / f"asv_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"asvspoof {split}: {len(picks)} clips from {len(groups)}/{pf.num_row_groups} row groups", flush=True)
    for k, (g, offs) in enumerate(sorted(groups.items())):
        tbl = pf.read_row_group(g, columns=["audio", "audio_file_name", "speaker_id", "system_id", "key"])
        for o in offs:
            row = tbl.slice(o, 1).to_pylist()[0]
            name = row["audio_file_name"]
            dest = out_dir / f"{name}.flac"
            if not dest.exists():
                y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                write_audio(dest, y, sr)
            rows_out.append({
                "clip_id": f"asv_{split}/{name}",
                "corpus": f"asv_{split}",
                "label": "spoof" if row["key"] == 1 else "bonafide",
                "speaker": row["speaker_id"],
                "system": row["system_id"],
                "path": str(dest.relative_to(ROOT)),
            })
        print(f"  group {k + 1}/{len(groups)}", flush=True)


class RangeFile(io.RawIOBase):
    """Seekable file over HTTP range requests, so zipfile can read one member."""

    def __init__(self, url):
        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_itw(n_per_class, seed, rows_out):
    url = "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
    zf = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 16))
    names = zf.namelist()
    meta_name = next(n for n in names if n.endswith("meta.csv"))
    prefix = meta_name[: -len("meta.csv")]
    meta = list(csv.DictReader(io.TextIOWrapper(zf.open(meta_name), encoding="utf-8")))
    rng = random.Random(seed)
    picks = []
    for label in ("bona-fide", "spoof"):
        rows = [m for m in meta if m["label"] == label]
        picks += rng.sample(rows, n_per_class)
    out_dir = RAW / "itw"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"in-the-wild: {len(picks)} clips of {len(meta)}", flush=True)

    local = threading.local()

    def one(m):
        dest = out_dir / (Path(m["file"]).stem + ".flac")
        if not dest.exists():
            # one zip handle per thread (zipfile isn't thread safe), opened once,
            # so the 3 MB central directory is only fetched once per thread
            if not hasattr(local, "zf"):
                local.zf = zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 16))
            y, sr = sf.read(io.BytesIO(local.zf.read(prefix + m["file"])), dtype="float32")
            write_audio(dest, y, sr)
        return {
            "clip_id": f"itw/{Path(m['file']).stem}",
            "corpus": "itw",
            "label": "bonafide" if m["label"] == "bona-fide" else "spoof",
            "speaker": m["speaker"],
            "system": "itw",
            "path": str(dest.relative_to(ROOT)),
        }

    with ThreadPoolExecutor(12) as ex:
        for k, row in enumerate(ex.map(one, picks)):
            rows_out.append(row)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(picks)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="clips per class per corpus")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", choices=["train", "test", "itw"], nargs="*")
    args = ap.parse_args()
    rows = []
    want = args.only or ["train", "test", "itw"]
    if "train" in want:
        fetch_asvspoof("train", args.n, args.seed, rows)
    if "test" in want:
        fetch_asvspoof("test", args.n, args.seed, rows)
    if "itw" in want:
        fetch_itw(args.n, args.seed, rows)
    man = ROOT / "data" / "manifest.csv"
    old = []
    if man.exists():
        done = {r["clip_id"] for r in rows}
        old = [r for r in csv.DictReader(open(man)) if r["clip_id"] not in done]
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(old + rows)
    print("manifest rows:", len(old) + len(rows))


if __name__ == "__main__":
    sys.exit(main())
