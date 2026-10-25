/*
 * Liveline Stage 2: scam-tactic text model.
 *
 * Scores what the caller has said so far and returns
 *   { p, llr, band, tactics }
 *   p        calibrated probability that the call is a scam (0..1)
 *   llr      log-likelihood ratio, logit(p) - logit(prior), for Bayesian fusion
 *   band     "green" | "yellow" | "red", from the thresholds in model.json
 *   tactics  [{id, label}] from hand-written keyword RULES (not machine
 *            learning), used for the plain-language "why". If model.json has
 *            a calibration coefficient c, the NUMBER of rules that fired is
 *            also one input to the calibrated score.
 *
 * The model is TF-IDF over word unigrams and bigrams of the caller's words,
 * then logistic regression, then a calibration layer (Platt scaling, plus the
 * tactic count when model.json says so). It was trained in Python
 * (train.py) and this file reproduces scikit-learn's numbers exactly;
 * parity_test.js checks that on every test dialogue.
 *
 * No libraries and no build step. Usage:
 *   Node:     var Stage2 = require("./stage2_model.js");
 *             Stage2.load(require("./model.json"));
 *   Browser:  set window.LIVELINE_STAGE2_MODEL = <contents of model.json>
 *             before this script runs (it loads itself), or call Stage2.load().
 *   Then:     Stage2.scoreTranscript("I need you to buy gift cards right now")
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.Stage2 = api;
    if (root.LIVELINE_STAGE2_MODEL) api.load(root.LIVELINE_STAGE2_MODEL);
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var M = null;          // the parsed model.json
  var wordIndex = null;  // n-gram -> column
  var charIndex = null;  // char n-gram -> column (only if the model has a char block)
  var names = null;      // words that become "name"
  var greetings = null;  // words that become "hello"
  var tacticRules = [];  // [{id, label, res: [RegExp]}]
  var logitPrior = 0;

  function toSet(list) {
    var s = Object.create(null);
    for (var i = 0; i < list.length; i++) s[list[i]] = true;
    return s;
  }

  function toIndex(list) {
    var m = new Map();
    for (var i = 0; i < list.length; i++) m.set(list[i], i);
    return m;
  }

  function load(model) {
    M = model;
    wordIndex = toIndex(M.vocab);
    charIndex = M.char_vocab ? toIndex(M.char_vocab) : null;
    names = toSet(M.tokenizer.name_tokens);
    greetings = toSet(M.tokenizer.greeting_tokens);
    tacticRules = M.tactics.map(function (t) {
      return { id: t.id, label: t.label, res: t.patterns.map(function (p) { return new RegExp(p); }) };
    });
    logitPrior = Math.log(M.prior / (1 - M.prior));
    return api;
  }

  // ---- text cleanup (same steps as normalize() in common.py) ----

  function normalize(text) {
    var s = String(text).toLowerCase();
    s = s.replace(/['‘’ʼ`]/g, "");   // don't -> dont
    s = s.replace(/[0-9]+/g, " num ");               // 555-0187 -> num num
    s = s.replace(/[^a-z]+/g, " ");                  // everything else is a space
    var words = s.split(" ");
    var out = [];
    for (var i = 0; i < words.length; i++) {
      var w = words[i];
      if (!w) continue;
      if (names[w]) w = "name";
      else if (greetings[w]) w = "hello";
      out.push(w);
    }
    return out.join(" ");
  }

  // Unigrams then bigrams, like analyzer() in common.py.
  function wordNgrams(clean) {
    var toks = clean ? clean.split(" ") : [];
    var grams = toks.slice();
    for (var i = 0; i + 1 < toks.length; i++) grams.push(toks[i] + " " + toks[i + 1]);
    return grams;
  }

  // Character 3-5 grams inside each word padded with spaces, like char_analyzer().
  function charNgrams(clean) {
    var grams = [];
    var toks = clean ? clean.split(" ") : [];
    for (var t = 0; t < toks.length; t++) {
      var w = " " + toks[t] + " ";
      for (var n = 3; n <= 5; n++) {
        for (var i = 0; i + n <= w.length; i++) grams.push(w.slice(i, i + n));
      }
    }
    return grams;
  }

  // One TF-IDF block dotted with its weights:
  // tf = 1 + ln(count), times idf, L2-normalized over the n-grams in the vocabulary.
  function blockScore(grams, index, idf, coef) {
    var counts = new Map();
    for (var i = 0; i < grams.length; i++) {
      var col = index.get(grams[i]);
      if (col === undefined) continue;   // not in the vocabulary: ignored
      counts.set(col, (counts.get(col) || 0) + 1);
    }
    var cols = [], vals = [], sumsq = 0;
    counts.forEach(function (c, col) {
      var v = (1 + Math.log(c)) * idf[col];
      cols.push(col);
      vals.push(v);
      sumsq += v * v;
    });
    if (sumsq === 0) return 0;
    var norm = Math.sqrt(sumsq), z = 0;
    for (var k = 0; k < cols.length; k++) z += (vals[k] / norm) * coef[cols[k]];
    return z;
  }

  // Raw logistic-regression score (before calibration).
  function decision(clean) {
    var z = M.intercept + blockScore(wordNgrams(clean), wordIndex, M.idf, M.coef);
    if (charIndex) z += blockScore(charNgrams(clean), charIndex, M.char_idf, M.char_coef);
    return z;
  }

  function tagTactics(clean) {
    var found = [];
    for (var i = 0; i < tacticRules.length; i++) {
      var r = tacticRules[i];
      for (var j = 0; j < r.res.length; j++) {
        if (r.res[j].test(clean)) { found.push({ id: r.id, label: r.label }); break; }
      }
    }
    return found;
  }

  function bandFor(p) {
    if (p >= M.thresholds.red) return "red";
    if (p >= M.thresholds.yellow) return "yellow";
    return "green";
  }

  // ---- public API ----

  // text: everything the CALLER has said so far (not the person answering).
  function scoreTranscript(text) {
    if (!M) throw new Error("Stage2: call load(model) first");
    var clean = normalize(text);
    var tactics = tagTactics(clean);
    // Calibrated log-odds. Version 2.1 measures the model score from an empty
    // transcript's score (u = z - z0) and lets confidence grow with the number
    // of caller words heard (n): a*u + d*u*ln(1+n) + b. Older model.json files
    // have no d or z0 and fall back to a*z + c*(tactic count) + b.
    var cal = M.calibration, z = decision(clean), s;
    if (cal.d !== undefined) {
      var u = z - cal.z0, n = clean ? clean.split(" ").length : 0;
      s = cal.a * u + cal.d * u * Math.log(1 + n) + cal.b;
    } else {
      s = cal.a * z + (cal.c || 0) * tactics.length + cal.b;
    }
    var p = 1 / (1 + Math.exp(-s));
    return { p: p, llr: s - logitPrior, band: bandFor(p), tactics: tactics };
  }

  // Pull the caller's words out of a tagged transcript such as
  // "caller: Hi ... receiver: Hello ..." (also accepts Suspect / Innocent).
  var TAG = /\b(caller|receiver|suspect|innocent)[ \t\n\r\f\v]*:/i;
  var CALLER = { caller: true, suspect: true };
  function callerText(transcript) {
    var parts = String(transcript).split(TAG);   // [before, tag, text, tag, text, ...]
    var said = [];
    for (var i = 1; i + 1 < parts.length; i += 2) {
      var body = parts[i + 1].trim();
      if (body && CALLER[parts[i].toLowerCase()]) said.push(body);
    }
    return said.join(" ");
  }

  function scoreDialogue(transcript) {
    return scoreTranscript(callerText(transcript));
  }

  // lines: [{who: "caller" | anything else, text: "..."}], as in the app's simulator.
  function scoreLines(lines) {
    var said = [];
    for (var i = 0; i < lines.length; i++) if (lines[i].who === "caller") said.push(lines[i].text);
    return scoreTranscript(said.join(" "));
  }

  var api = {
    load: load,
    scoreTranscript: scoreTranscript,
    scoreDialogue: scoreDialogue,
    scoreLines: scoreLines,
    callerText: callerText,
    normalize: normalize
  };
  return api;
});
