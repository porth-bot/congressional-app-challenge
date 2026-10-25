"""Download the six public Stage 2 datasets from Hugging Face (no login needed).

    ../.venv/bin/python fetch_data.py

Files land in research/data/stage2/<owner>__<name>/. menaattia/phone-scam-dataset
has no license, so we only use it as an external test set and never copy its
text into this folder or the repo.
"""

import os

from huggingface_hub import hf_hub_download

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stage2"))

# Dataset commits we downloaded and trained on (2026-10-25).
REVISIONS = {
    "BothBosu/scam-dialogue": "321b961b5ae353e19ed479b960658dcd223d5e06",
    "BothBosu/multi-agent-scam-conversation": "709db2b6c37f424c3070f29138abb33971e21ab9",
    "BothBosu/single-agent-scam-conversations": "1debe2743472927cd4ed3bf39e8b0d8234bd5e3f",
    "BothBosu/youtube-scam-conversations": "d0b94f1e2a9522af4703154c805d02d5c596110d",
    "menaattia/phone-scam-dataset": "47a62271c6d5834f0bb800b8f97a548d864a7ff6",
    "shakeleoatmeal/phone-scam-detection-synthetic": "27a1f6d0cf21dda995130d221383900b060dbcc9",
}

FILES = {
    "BothBosu/scam-dialogue": ["README.md", "scam-dialogue_all.csv",
                               "scam-dialogue_train.csv", "scam-dialogue_test.csv"],
    "BothBosu/multi-agent-scam-conversation": ["README.md", "agent_conversation_all.csv",
                                               "agent_conversation_train.csv", "agent_conversation_test.csv"],
    "BothBosu/single-agent-scam-conversations": ["README.md", "single-agent-scam-dialogue_all.csv",
                                                 "single-agent-scam-dialogue_train.csv",
                                                 "single-agent-scam-dialogue_test.csv"],
    "BothBosu/youtube-scam-conversations": ["README.md", "youtube_scam_conversations.csv"],
    "menaattia/phone-scam-dataset": ["train_dataset.csv", "test_dataset.csv"],
    "shakeleoatmeal/phone-scam-detection-synthetic": ["README.md",
                                                      "data/train-00000-of-00001.parquet",
                                                      "data/validation-00000-of-00001.parquet",
                                                      "data/test-00000-of-00001.parquet"],
}

if __name__ == "__main__":
    for repo, files in FILES.items():
        folder = os.path.join(OUT, repo.replace("/", "__"))
        for f in files:
            path = hf_hub_download(repo, f, repo_type="dataset", local_dir=folder,
                                   revision=REVISIONS[repo])
            print(path, os.path.getsize(path))
