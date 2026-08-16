#!/usr/bin/env bash
# Local preview -- builds and serves the blog without pushing to GitHub.
#
#   ./preview.sh            # serve on http://localhost:4000/ and watch for edits
#   ./preview.sh 4001       # ...on a different port
#
# Jekyll lives in a conda env (this box has no system Ruby); Jekyll 3.9 is
# pinned to match what GitHub Pages actually builds with. glibc 2.17 here is
# too old for Jekyll 4's sass-embedded binaries.
#
# One-time setup, if the env is missing:
#   conda create -y -n jekyll -c conda-forge ruby=3.1 compilers
#   GEM_HOME=~/.conda/envs/jekyll/share/rubygems \
#     ~/.conda/envs/jekyll/bin/gem install --no-document \
#     jekyll:3.9.5 jekyll-feed:0.15.1 kramdown-parser-gfm
set -euo pipefail

PORT="${1:-4000}"

# Resolve which conda env has Ruby: an explicit override, then the dedicated
# `jekyll` env from the setup comment above, then whatever conda env is
# currently active (CONDA_PREFIX) as a last resort.
if [ -n "${JEKYLL_ENV_DIR:-}" ]; then
  ENV_DIR="$JEKYLL_ENV_DIR"
elif [ -x "$HOME/.conda/envs/jekyll/bin/ruby" ]; then
  ENV_DIR="$HOME/.conda/envs/jekyll"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/ruby" ]; then
  ENV_DIR="$CONDA_PREFIX"
else
  ENV_DIR="$HOME/.conda/envs/jekyll"
fi

if [ ! -x "$ENV_DIR/bin/ruby" ]; then
  echo "No Ruby at $ENV_DIR -- see the setup comment at the top of this script." >&2
  exit 1
fi

export PATH="$ENV_DIR/bin:$PATH"
export GEM_HOME="$ENV_DIR/share/rubygems"
export GEM_PATH="$GEM_HOME"

# The conda binstub has a broken shebang, so go through ruby directly.
exec ruby -e 'require "rubygems"; gem "jekyll"; load Gem.bin_path("jekyll","jekyll")' -- \
  serve --watch --host 127.0.0.1 --port "$PORT" --baseurl "" --destination .jekyll-preview
