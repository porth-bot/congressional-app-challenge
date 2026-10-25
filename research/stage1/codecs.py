"""Make a phone-quality copy of every clip, one per codec.

Each condition is a real encode/decode round trip through ffmpeg:
  g711  - landline (PSTN): 300-3400 Hz band, 8 kHz, G.711 mu-law, 64 kbps
  opus  - VoIP / app calling: Opus narrowband (SILK), 8 kHz, 8 kbps
  g723  - low-bitrate cellular-style CELP: G.723.1 at 6.3 kbps, 8 kHz
          (ffmpeg has no AMR-NB encoder; G.723.1 is the closest CELP codec it ships)

The decoded audio is resampled back to 16 kHz so every condition goes through
the same feature code. Nothing above 4 kHz survives, which is the point.
"""
import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODECS = {
    "g711": [["-af", "highpass=f=300,lowpass=f=3400", "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw", "-f", "wav"], "wav"],
    "opus": [["-ar", "8000", "-ac", "1", "-c:a", "libopus", "-b:a", "8k", "-application", "voip", "-f", "ogg"], "ogg"],
    "g723": [["-ar", "8000", "-ac", "1", "-c:a", "g723_1", "-b:a", "6300", "-f", "g723_1"], "tco"],
}


def out_path(clip_path, codec):
    p = Path(clip_path)
    return ROOT / "data" / codec / p.parent.name / p.name


def run(job):
    src, codec = job
    dest = out_path(src, codec)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    args, ext = CODECS[codec]
    with tempfile.TemporaryDirectory() as td:
        enc = Path(td) / f"enc.{ext}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / src), *args, str(enc)], check=True)
        dec_in = ["-f", "g723_1"] if codec == "g723" else []
        subprocess.run(["ffmpeg", "-v", "error", "-y", *dec_in, "-i", str(enc), "-ar", "16000", "-ac", "1",
                        "-c:a", "flac", "-sample_fmt", "s16", str(dest)], check=True)


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv")))
    jobs = [(r["path"], c) for r in rows for c in CODECS]
    with ProcessPoolExecutor(10) as ex:
        for k, _ in enumerate(ex.map(run, jobs, chunksize=16)):
            if (k + 1) % 2000 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    sys.exit(main())
