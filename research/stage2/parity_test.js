// Checks that stage2_model.js gives the same answers as scikit-learn.
//
//   node parity_test.js
//
// Reads the cases train.py wrote to research/data/stage2/parity_cases.json
// (every in-domain test dialogue, the shakeleoatmeal and menaattia test
// splits, the youtube transcripts, the family-emergency test batch, plus 10%
// and 50% prefixes of each), and
// the demo-call scores in results.json. Fails if any probability differs by
// 1e-6 or more, or if any tactic tag differs.

"use strict";
const fs = require("fs");
const path = require("path");
const Stage2 = require("./stage2_model.js");

const here = __dirname;
Stage2.load(JSON.parse(fs.readFileSync(path.join(here, "model.json"), "utf8")));
const model = JSON.parse(fs.readFileSync(path.join(here, "model.json"), "utf8"));
const fixture = JSON.parse(fs.readFileSync(path.join(here, "..", "data", "stage2", "parity_cases.json"), "utf8"));
const logitPrior = Math.log(fixture.prior / (1 - fixture.prior));

let maxP = 0, maxLlr = 0, tagMismatch = 0, bandMismatch = 0;
const bySet = {};
const band = (p) => (p >= model.thresholds.red ? "red" : p >= model.thresholds.yellow ? "yellow" : "green");

for (const c of fixture.cases) {
  const r = c.dialogue !== undefined ? Stage2.scoreDialogue(c.dialogue) : Stage2.scoreTranscript(c.caller);
  const dp = Math.abs(r.p - c.p);
  const dl = Math.abs(r.llr - (fixture.a * c.z + (fixture.c || 0) * c.k + fixture.b - logitPrior));
  maxP = Math.max(maxP, dp);
  maxLlr = Math.max(maxLlr, dl);
  if (r.tactics.map((t) => t.id).join(",") !== c.tactics.join(",")) tagMismatch++;
  if (r.band !== band(c.p)) bandMismatch++;
  const s = (bySet[c.set] = bySet[c.set] || { n: 0, maxAbsDiffP: 0 });
  s.n++;
  s.maxAbsDiffP = Math.max(s.maxAbsDiffP, dp);
}

// Demo calls: recompute each step from the caller lines so far.
const results = JSON.parse(fs.readFileSync(path.join(here, "results.json"), "utf8"));
let demoMax = 0, demoSteps = 0;
for (const [call, variants] of Object.entries(results.demo_calls)) {
  for (const steps of Object.values(variants)) {
    const lines = [];
    for (const st of steps) {
      lines.push({ who: st.who, text: st.text });
      const r = Stage2.scoreLines(lines);
      demoMax = Math.max(demoMax, Math.abs(r.p - st.p));   // results.json rounds p to 4 decimals
      demoSteps++;
      if (r.band !== st.band) bandMismatch++;
    }
  }
}

const out = {
  cases: fixture.cases.length,
  max_abs_diff_p: maxP,
  max_abs_diff_llr: maxLlr,
  tactic_mismatches: tagMismatch,
  band_mismatches: bandMismatch,
  by_set: bySet,
  demo_steps: demoSteps,
  demo_max_abs_diff_vs_rounded_p: demoMax,
  pass: maxP < 1e-6 && tagMismatch === 0 && bandMismatch === 0 && demoMax <= 5e-5 + 1e-12,
  node: process.version,
};
fs.writeFileSync(path.join(here, "parity_result.json"), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
process.exit(out.pass ? 0 : 1);
