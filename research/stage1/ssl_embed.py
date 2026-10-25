"""Learned audio embedding: frozen WavLM Base+ (Microsoft, 94M parameters,
self-supervised on 94k hours of unlabelled speech; never trained on deepfakes).

Each clip is trimmed of edge silence (same as features.py), repeat-padded or
centre-cropped to 4 s, and passed through the frozen network. We keep the
time-averaged output of every layer (13 x 768). select_layer() later picks one
layer by cross-validation on the training split only.

Writes data/features/ssl_layers_<cond>.npz.
"""
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from common import CONDITIONS, ROOT, manifest
from features import clip_path

L = 64000
MODEL = "microsoft/wavlm-base-plus"


def prep(rel, cond):
    y, _ = sf.read(clip_path(rel, cond), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    y, _ = librosa.effects.trim(y, top_db=35, frame_length=400, hop_length=160)
    if len(y) < 1600:
        return None
    if len(y) < L:
        y = np.tile(y, int(np.ceil(L / len(y))))
    s = (len(y) - L) // 2
    y = y[s:s + L]
    return (y - y.mean()) / (y.std() + 1e-7)


def main():
    conds = sys.argv[1:] or CONDITIONS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL).eval().to(dev)
    man = manifest()
    for cond in conds:
        rows = man if cond != "g723" else man[man.corpus != "asv_train"]  # g723 is test-only
        ids, embs, batch, bids = [], [], [], []

        def flush():
            with torch.no_grad():
                o = model(torch.from_numpy(np.stack(batch)).to(dev), output_hidden_states=True)
                h = torch.stack(o.hidden_states, 1).mean(2).cpu().numpy().astype(np.float16)
            embs.append(h)
            ids.extend(bids)
            batch.clear()
            bids.clear()

        for k, r in enumerate(rows.itertuples()):
            y = prep(r.path, cond)
            if y is None:
                continue
            batch.append(y)
            bids.append(r.clip_id)
            if len(batch) == 32:
                flush()
            if (k + 1) % 1000 == 0:
                print(cond, k + 1, "/", len(rows), flush=True)
        if batch:
            flush()
        np.savez(ROOT / "data" / "features" / f"ssl_layers_{cond}.npz", ids=np.array(ids), emb=np.concatenate(embs))
        print(cond, "done", len(ids), flush=True)


if __name__ == "__main__":
    sys.exit(main())
