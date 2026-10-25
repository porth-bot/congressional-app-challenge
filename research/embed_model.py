"""Copy the trained Stage 2 language model into the single-file app.

The app is one HTML fragment, so the model's weights (stage2/model.json) and
its JavaScript runtime (stage2/stage2_model.js) are pasted in between two
marker comments. Run this after every retrain:

    .venv/bin/python embed_model.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "atelier-securities.html"
START, END = "<!-- stage2:start -->", "<!-- stage2:end -->"


def main():
    model = json.load(open(HERE / "stage2" / "model.json"))
    runtime = (HERE / "stage2" / "stage2_model.js").read_text()
    block = (f"{START}\n<script>/* Stage 2 weights, generated from research/stage2/model.json by research/embed_model.py */\n"
             f"window.LIVELINE_STAGE2_MODEL={json.dumps(model, separators=(',', ':'))};</script>\n"
             f"<script>\n{runtime}\n</script>\n{END}")
    html = APP.read_text()
    if START in html:
        a, b = html.index(START), html.index(END) + len(END)
        html = html[:a] + block + html[b:]
    else:
        # first time: put it right before the app's main script
        anchor = "\n<script>\n"
        i = html.rindex(anchor)
        html = html[:i] + "\n" + block + html[i:]
    APP.write_text(html)
    print(f"embedded {len(block) / 1024:.0f} KB; app is now {len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
