"""Checks a hand-written family-call file.

    python family_check.py FILE N_SCAM N_NORMAL SCENARIO_IDS(comma-separated)

Format: blocks separated by a blank line. Each block starts with
    ### id=<unique id> | label=<1 scam or 0 normal> | scenario=<scenario id>
followed only by lines that start with "caller: " or "receiver: ".
The caller speaks first and has 3 to 14 turns.
"""
import re
import sys
from collections import Counter

path, n_scam, n_norm, scen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), set(sys.argv[4].split(","))
text = open(path, encoding="utf-8").read().strip()
blocks = re.split(r"\n\s*\n(?=###)", text)
errors, ids, labels, turns, per_scen = [], set(), Counter(), [], Counter()
for b in blocks:
    lines = [l for l in b.strip().split("\n") if l.strip()]
    m = re.match(r"^### id=(\S+) \| label=([01]) \| scenario=(\S+)\s*$", lines[0])
    if not m:
        errors.append(f"bad header: {lines[0][:80]}"); continue
    bid, lab, sc = m.groups()
    if bid in ids: errors.append(f"duplicate id {bid}")
    ids.add(bid)
    if sc not in scen: errors.append(f"{bid}: scenario {sc} not allowed")
    body = lines[1:]
    bad = [l for l in body if not re.match(r"^(caller|receiver): \S", l)]
    if bad: errors.append(f"{bid}: {len(bad)} lines without 'caller: ' / 'receiver: ' e.g. {bad[0][:60]}")
    if not body or not body[0].startswith("caller: "): errors.append(f"{bid}: caller must speak first")
    nc = sum(1 for l in body if l.startswith("caller: "))
    if not 3 <= nc <= 14: errors.append(f"{bid}: {nc} caller turns (need 3 to 14)")
    labels[lab] += 1; turns.append(nc); per_scen[(sc, lab)] += 1
print(f"blocks={len(blocks)} scam={labels['1']} normal={labels['0']} (want {n_scam}/{n_norm})")
print("caller turns: min", min(turns) if turns else None, "max", max(turns) if turns else None,
      "short(3-5)", sum(3 <= t <= 5 for t in turns), "medium(6-9)", sum(6 <= t <= 9 for t in turns),
      "long(10-14)", sum(10 <= t <= 14 for t in turns))
print("per scenario:", dict(sorted(per_scen.items())))
if labels["1"] != n_scam or labels["0"] != n_norm: errors.append("wrong counts")
print("OK" if not errors else "ERRORS:\n" + "\n".join(errors[:30]))
