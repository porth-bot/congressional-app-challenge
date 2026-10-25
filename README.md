# Liveline

Liveline screens calls from unknown numbers and tells an older adult whether to trust the caller: green for seems safe, yellow for check first, red for likely scam. We built it for the 2026 Congressional App Challenge (MN-04) as Atelier Securities.

Live demo: run it locally (see below).

Everything in the demo is simulated. The calls are scripted, the scores come from fixed likelihood ratios rather than a trained model, the call history is sample data, and nothing is recorded.

## What's in the app

It uses a sidebar on a computer and a tab bar on a phone:

- **Home:** the protection switch, this week's numbers, recent calls, a setup checklist, and shortcuts.
- **Simulator:** Rose's phone next to the live analysis. Pick a scripted call, answer it, and watch the Trust Meter, signals, transcript, and math update.
- **Calls:** call history with search, filters, and a details drawer for each call.
- **Family:** members, voice samples, the safe word, and alert settings.
- **Practice:** the weekly scam drill.
- **Learn:** Signal, Scoring, Research, and Privacy.
- **Settings:** theme, Rose's screen options, and demo data.

## Running it locally

The app is one file, `atelier-securities.html`. It's a page fragment, because the artifact host adds the `<!doctype>` and `<head>` around it. To preview it the same way:

```bash
./.build-preview.sh
```

```bash
python3 -m http.server 8878 --bind 127.0.0.1
```

Then open http://127.0.0.1:8878/preview.html. `preview.html` is generated, so edit `atelier-securities.html` and rebuild.

## How it's put together

The file has three parts: CSS, markup for every screen, and one script.

- **Routing.** Hash routes (`#home`, `#simulator`, `#learn-scoring` and so on) are listed in `ROUTES`.
- **State.** Everything saves to `localStorage` under `liveline-demo-v3`. The theme choice saves under `liveline-mode`.
- **Simulator.** The scripted scenarios live in `SC`. Every 2 seconds the app multiplies the starting odds by the voice, words, and challenge likelihood ratios to get the scam risk, then maps it to green, yellow, or red.
- **Spectrogram.** `Spectro()` draws simulated speech onto a canvas, with an optional phone-line filter. It's an illustration, not measured audio.
