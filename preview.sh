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

ENV_DIR="${JEKYLL_ENV_DIR:-$HOME/.conda/envs/jekyll}"
PORT="${1:-4000}"

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
