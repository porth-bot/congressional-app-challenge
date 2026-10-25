#!/bin/sh
# Wraps the artifact fragment in the same skeleton the Artifact host adds, for local preview.
cd "$(dirname "$0")"
{ printf '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}body{margin:0;font:14px system-ui}img{max-width:100%%}[hidden]{display:none!important}</style>\n'; cat atelier-securities.html; printf '\n</body></html>'; } > preview.html
