// Version 2.1 parity check: the browser runtime must give the same probability
// as the Python calibration script on the same caller text.
//   node parity_v21.js
var fs = require("fs"), path = require("path");
var Stage2 = require("./stage2_model.js");
Stage2.load(JSON.parse(fs.readFileSync(path.join(__dirname, "model.json"), "utf8")));
var cases = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "stage2", "parity_cases_v21.json"), "utf8"));
var worst = 0;
cases.forEach(function (c) { worst = Math.max(worst, Math.abs(Stage2.scoreTranscript(c.text).p - c.p)); });
var out = { cases: cases.length, max_abs_diff: worst, pass: worst < 1e-9 };
fs.writeFileSync(path.join(__dirname, "parity_v21.json"), JSON.stringify(out, null, 1));
console.log(out);
process.exit(out.pass ? 0 : 1);
