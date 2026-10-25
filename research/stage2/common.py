"""Shared pieces for the Stage 2 scam-tactic text model.

Everything here has to match stage2_model.js exactly, because the browser
version must give the same numbers as the Python one:

  * parse_turns()      splits a transcript into (speaker, text) turns
  * normalize()        lowercases and strips a string down to a-z words
  * analyzer()         turns normalized text into unigrams + bigrams
  * prefix_caller()    the caller's words in the first f of a call

Data loading lives here too so audit.py and train.py read the same thing.
"""

import glob
import math
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "stage2"))

# ---------------------------------------------------------------------------
# Speaker turns
# ---------------------------------------------------------------------------

# Every source marks speakers with one of these four tags. "Suspect" and
# "Innocent" are used for BOTH scam and non-scam calls in the BothBosu sets,
# so the tag names are not a label, but they are dataset-specific formatting,
# and we only want the caller's words anyway.
# re.ASCII and the explicit whitespace class keep \b, \s and case folding
# identical to JavaScript's (non-unicode) regex. On the data it changes nothing.
TAG_RE = re.compile(r"\b(caller|receiver|suspect|innocent)[ \t\n\r\f\v]*:", re.IGNORECASE | re.ASCII)
ROLE = {"caller": "caller", "suspect": "caller",
        "receiver": "receiver", "innocent": "receiver"}


def parse_turns(text):
    """Return a list of (role, text) with role in {caller, receiver}."""
    turns = []
    parts = TAG_RE.split(text)
    # parts = [before, tag1, text1, tag2, text2, ...]
    if parts[0].strip():
        turns.append(("unknown", parts[0].strip()))
    for i in range(1, len(parts) - 1, 2):
        role = ROLE[parts[i].lower()]
        body = parts[i + 1].strip()
        if body:
            turns.append((role, body))
    return turns


def caller_text(text):
    return " ".join(t for r, t in parse_turns(text) if r == "caller")


def full_text(text):
    """Both speakers, with the speaker tags removed."""
    return " ".join(t for r, t in parse_turns(text))


def prefix_caller(turns, frac):
    """Caller words inside the first `frac` of the call's words.

    The call is measured in words of BOTH speakers, so "30% of the call"
    means 30% of everything said so far, cut mid-turn if needed.
    """
    words = [(r, w) for r, t in turns for w in t.split()]
    keep = int(math.ceil(frac * len(words) - 1e-9))
    return " ".join(w for r, w in words[:keep] if r == "caller")


# ---------------------------------------------------------------------------
# Text normalization and n-grams (mirrored in stage2_model.js)
# ---------------------------------------------------------------------------

APOS_RE = re.compile("['‘’ʼ`]")
DIGITS_RE = re.compile(r"[0-9]+")
NONALPHA_RE = re.compile(r"[^a-z]+")

# Generation artifacts found in audit.py: the LLM that wrote the training
# calls reused the same made-up people and placeholder companies, unevenly
# across labels ("Officer Johnson" and "David" in scams, "Karen", "Sarah",
# "Mike", "XYZ Delivery" in normal calls), and it opened scam calls with
# "Hello" and normal calls with "Hi". None of that says anything about real
# scams, so every such word becomes "name", and every greeting becomes
# "hello". Real organizations (Microsoft, Walmart, Social Security) are kept,
# because impersonating them IS a scam tactic.
NAME_TOKENS = sorted("""
    john johns johnson karen emily rachel david smith smiths smithson sarah
    thompson alex wilson mike michael james jenkins davis lee ryan doe jane
    margaret harold brown taylor fitzgerald gatsby mr mrs ms
    xyz abc anytown smithville speedy
    tyler megan brandon ashley kevin lauren justin brianna cody kayla marcus jenna
    linda steve denise gary dorothy shirley walter gloria daniels greg whitman
    morales patel tina lakes maple
""".split())
# The second block above is the name list given to the writers of the
# family-call TRAIN batch (version 2). The TEST batch used a different name
# list that is deliberately NOT here, so test names are unknown words, the
# same as a real caller's name would be.
GREETING_TOKENS = sorted("hi hey hello hiya howdy".split())
_NAMES = set(NAME_TOKENS)
_GREETINGS = set(GREETING_TOKENS)


def normalize(s, neutralize=True, names=None, greetings=None):
    """lowercase -> drop apostrophes (don't -> dont) -> digit runs become
    the word "num" -> anything that is not a-z becomes a space -> names
    become "name" and greetings become "hello" (skipped if neutralize is
    False, which only the audit uses)."""
    s = s.lower()
    s = APOS_RE.sub("", s)
    s = DIGITS_RE.sub(" num ", s)
    s = NONALPHA_RE.sub(" ", s)
    if not neutralize:
        return " ".join(s.split())
    names = _NAMES if names is None else names
    greetings = _GREETINGS if greetings is None else greetings
    out = []
    for w in s.split():
        if w in names:
            w = "name"
        elif w in greetings:
            w = "hello"
        out.append(w)
    return " ".join(out)


def analyzer(text):
    """Unigrams and bigrams of already-normalized text."""
    toks = text.split()
    return toks + [toks[i] + " " + toks[i + 1] for i in range(len(toks) - 1)]


# ---------------------------------------------------------------------------
# Rule-based tactic tagger (NOT machine learning)
# ---------------------------------------------------------------------------
# Hand-written keyword rules that give the app a plain-language "why". They
# run on normalize()d text, so patterns are lowercase words separated by
# single spaces, apostrophes are gone (dont, its, cant) and digits are "num".
# The same pattern strings are copied into model.json and compiled by
# stage2_model.js, so Python and JavaScript tag identically.

TACTICS = [
    {"id": "urgency", "label": "Rushing you or saying it is urgent",
     "patterns": [r"\b(right now|right away|immediately|immediate action|urgent|urgently|"
                  r"as soon as possible|asap|hurry|act now|act fast|act quickly|"
                  r"running out of time|time is running out|last chance|before its too late|"
                  r"only have a few minutes|need it tonight|money tonight|by the end of the day|"
                  r"within the next (num|few|one|two|an) (minutes|minute|hours|hour)|"
                  r"will be suspended|suspended today|expires today|go now)\b"]},
    {"id": "secrecy", "label": "Telling you to keep it secret",
     "patterns": [r"\b(dont tell|do not tell|dont let (anyone|anybody|them) know|"
                  r"keep (this|it) (a secret|secret|between us|confidential|private|quiet|to yourself)|"
                  r"between you and me|cant find out|tell no one|not tell anyone|"
                  r"dont (mention|discuss) (this|it))\b"]},
    {"id": "gift_cards", "label": "Asking for gift cards",
     "patterns": [r"\b(gift ?cards?|itunes cards?|google play cards?|steam cards?|"
                  r"prepaid (cards?|debit)|scratch off the back|numbers on the back of the card)\b"]},
    {"id": "wire_transfer", "label": "Asking for a wire transfer or money app",
     "patterns": [r"\b(wire (transfer|the money|money|it|the funds|funds)|western union|moneygram|"
                  r"money gram|zelle|venmo|cash ?app|money order|bank transfer|"
                  r"transfer (the )?(money|funds))\b"]},
    {"id": "crypto", "label": "Asking for Bitcoin or crypto",
     "patterns": [r"\b(bitcoin|bitcoins|crypto|cryptocurrency|btc|ethereum|tether|usdt|"
                  r"coinbase|crypto atm|bitcoin atm|wallet address)\b"]},
    {"id": "payment", "label": "Asking you to send money or pay a fee",
     "patterns": [r"\b(send (me |us )?(the )?money|send (me |us )?num|pay (a |the )?(fee|fine|deposit)|"
                  r"(processing|reactivation|release|handling|clearance|customs|activation) fee|"
                  r"fee of num|small fee|pay (it|this) (now|today)|make a payment)\b"]},
    {"id": "legal_trouble", "label": "Talking about arrest, bail, or legal trouble",
     "patterns": [r"\b(bail|arrest|arrested|jail|warrant|lawsuit|sued|legal action|"
                  r"legal consequences|prosecuted|prosecution|criminal charges|in custody|"
                  r"police|deported|court date|face charges)\b"]},
    {"id": "impersonation", "label": "Claiming to be the government, your bank, or a big company",
     "patterns": [r"\b(social security administration|internal revenue service|irs|medicare|"
                  r"government|federal|fbi|sheriff|officer|department of|treasury|"
                  r"fraud department|your bank|bank of america|wells fargo|chase bank|"
                  r"microsoft|apple support|amazon (customer )?(service|support)|"
                  r"geek squad|tech support|technical support)\b"]},
    {"id": "personal_info", "label": "Asking for personal numbers or passwords",
     "patterns": [r"\b(social security number|ssn|account number|routing number|card number|"
                  r"credit card|debit card|pin|password|passwords|date of birth|"
                  r"mothers maiden name|medicare number|security code|cvv|"
                  r"verification code|one time code|login|bank details|"
                  r"(verify|confirm) your (identity|information|details|account|social|bank|card))\b"]},
    {"id": "stay_on_line", "label": "Telling you not to hang up or call back",
     "patterns": [r"\b(stay on the (line|phone)|dont hang up|do not hang up|"
                  r"if you hang up|keep you on the line|"
                  r"(cant|cannot|unable to|not able to|not authorized to) (give|provide) (you )?(a )?"
                  r"(call ?back|callback|reference|number)|"
                  r"dont call (anyone|the bank|your bank|them|back)|no need to call)\b"]},
    {"id": "remote_access", "label": "Wanting remote access to your computer",
     "patterns": [r"\b(remote (access|desktop|control|connection|session)|remotely|teamviewer|"
                  r"anydesk|logmein|ultraviewer|"
                  r"(download|install) (this |the |our |a |an )?(software|app|program|tool)|"
                  r"access (to )?your (computer|device|pc|screen)|control of your (computer|screen)|"
                  r"share your screen|event viewer|control panel|windows key)\b"]},
]
_TACTIC_RES = [(t["id"], t["label"], [re.compile(p) for p in t["patterns"]]) for t in TACTICS]


def tag_tactics(normalized_text):
    """List of tactic ids whose rules match (in TACTICS order)."""
    return [tid for tid, _, res in _TACTIC_RES if any(r.search(normalized_text) for r in res)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

BOTHBOSU = {
    # short name: (folder/file prefix, label column)
    "scam_dialogue": ("BothBosu__scam-dialogue/scam-dialogue_", "label"),
    "single_agent": ("BothBosu__single-agent-scam-conversations/single-agent-scam-dialogue_", "labels"),
    "multi_agent": ("BothBosu__multi-agent-scam-conversation/agent_conversation_", "labels"),
}


# Family-emergency calls written for version 2 by AI (see
# RESULTS.md). Train-batch scenario clusters held out for validation; chosen
# before any model saw the data.
FAMILY_VAL_SCENARIOS = {"TS03", "TS06", "TS09", "TS11", "TN03", "TN06", "TN09", "TN11"}


def load_family():
    rows = []
    for folder in ("train", "test"):
        for path in sorted(glob.glob(os.path.join(DATA, "family", folder, "batch_*.txt"))):
            text = open(path, encoding="utf-8").read().strip()
            for block in re.split(r"\n\s*\n(?=###)", text):
                lines = [ln for ln in block.strip().split("\n") if ln.strip()]
                bid, lab, sc = re.match(r"^### id=(\S+) \| label=([01]) \| scenario=(\S+)", lines[0]).groups()
                split = folder
                if folder == "train" and sc in FAMILY_VAL_SCENARIOS:
                    split = "validation"
                rows.append({"source": "family", "split": split, "row": bid,
                             "dialogue": "\n".join(lines[1:]), "label": int(lab), "type": sc})
    return pd.DataFrame(rows)


def load_all():
    """One DataFrame with columns: source, split, row (0-based row in the
    original file), dialogue, label, type."""
    rows = []
    for name, (prefix, lab) in BOTHBOSU.items():
        for split in ("train", "test"):
            df = pd.read_csv(os.path.join(DATA, prefix + split + ".csv"))
            rows.append(pd.DataFrame({
                "source": name, "split": split, "row": df.index, "dialogue": df["dialogue"],
                "label": df[lab].astype(int), "type": df["type"]}))
    yt = pd.read_csv(os.path.join(DATA, "BothBosu__youtube-scam-conversations",
                                  "youtube_scam_conversations.csv"))
    rows.append(pd.DataFrame({"source": "youtube", "split": "all", "row": yt.index,
                              "dialogue": yt["dialogue"],
                              "label": yt["labels"].astype(int),
                              "type": yt["type"]}))
    for split in ("train", "test"):
        df = pd.read_csv(os.path.join(DATA, "menaattia__phone-scam-dataset",
                                      split + "_dataset.csv"))
        rows.append(pd.DataFrame({"source": "menaattia", "split": split, "row": df.index,
                                  "dialogue": df["dialogue"],
                                  "label": df["label"].astype(int),
                                  "type": "unknown"}))
    for split in ("train", "validation", "test"):
        f = glob.glob(os.path.join(DATA, "shakeleoatmeal__phone-scam-detection-synthetic",
                                   "data", split + "-*.parquet"))[0]
        df = pd.read_parquet(f)
        rows.append(pd.DataFrame({"source": "shakeleoatmeal", "split": split, "row": df.index,
                                  "dialogue": df["dialogue"],
                                  "label": df["label"].astype(int),
                                  "type": df["type"]}))
    rows.append(load_family())
    out = pd.concat(rows, ignore_index=True)
    out["dialogue"] = out["dialogue"].astype(str)
    out["type"] = out["type"].astype(str)
    return out
