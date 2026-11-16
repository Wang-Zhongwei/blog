# Blog

Jekyll blog published by GitHub Pages at
**<https://wang-zhongwei.github.io/blog/>**.

Layouts and styles live in this repo (`_layouts/`, `assets/css/main.css`) rather
than coming from a theme gem, so nothing drifts when GitHub Pages bumps its
dependency versions.

## Local preview

```bash
./preview.sh          # http://localhost:4000/
./preview.sh 4001     # different port
```

It watches the tree, so saving a file rebuilds it — just refresh the browser.
Stop with Ctrl-C. Output goes to `.jekyll-preview/` (gitignored).

Two ways it runs, picked automatically:

- **conda Ruby** (the HPC box, setup below): plain `jekyll serve`. Math shows
  as raw TeX here, because KaTeX only runs in the deploy build.
- **Docker** (e.g. the Mac, which has no usable Ruby): builds in a `ruby:3.1`
  container and re-runs the KaTeX pass after each rebuild, so math renders
  like production. The Chinese edition is at `http://localhost:4000/zh/`.

Over Cursor/VS Code Remote-SSH the port forwards automatically, so
`localhost:4000` in the laptop browser hits the server on the HPC node. Over a
plain SSH session, forward it yourself:

```bash
ssh -L 4000:127.0.0.1:4000 <you>@coe-hpc1.sjsu.edu
```

`preview.sh` passes `--baseurl ""`, which is why the local URL has no `/blog`
prefix even though the deployed site does.

### How the toolchain was set up

This box has no system Ruby, and its glibc is 2.17 (CentOS 7) — too old for
Jekyll 4, whose `sass-embedded` ships prebuilt binaries against a newer glibc.
So: Ruby via conda, and Jekyll pinned to 3.9, which is what GitHub Pages
actually builds with anyway.

```bash
conda create -y -n jekyll -c conda-forge ruby=3.1 compilers

export GEM_HOME=~/.conda/envs/jekyll/share/rubygems
export PATH=~/.conda/envs/jekyll/bin:$PATH
gem install --no-document jekyll:3.9.5 jekyll-feed:0.15.1 \
                          kramdown-parser-gfm webrick
```

Two quirks that setup runs into, both already handled inside `preview.sh`:

- **`webrick` is a separate gem.** Ruby 3.0 dropped it from the standard
  library, and Jekyll 3.9's `serve` command still expects it.
- **The conda binstub is broken.** `share/rubygems/bin/jekyll` has a shebang
  pointing at a `ruby` that isn't there, so the script loads Jekyll through
  `ruby -e '... Gem.bin_path("jekyll","jekyll")'` instead of calling it
  directly.

Point `JEKYLL_ENV_DIR` at a different env if you move it.

## Publishing

Push to `main` and GitHub Pages rebuilds in about a minute:

```bash
git push origin main
```

`.github/workflows/pages.yml` builds with Jekyll, pre-renders math with KaTeX
(`scripts/render-math.js`) and deploys. Because this is a project page (not `<user>.github.io`), `_config.yml`
sets `baseurl: "/blog"` — leave it alone unless the repo is renamed.

Pages serves HTML with `Cache-Control: max-age=600`, so a hard refresh
(Ctrl-Shift-R) is often needed to see a fresh deploy.

## Writing

Posts are `_posts/YYYY-MM-DD-slug.md`; the filename sets the date and URL.
Front matter:

```yaml
---
layout: post
title: "A Statistical-Physics View of KL-Regularized RL"
date: 2026-08-11
tags: [statistical-mechanics, rlhf, dpo]
subtitle: "Optional — shown under the title"   # optional
image: /assets/figures/reward-kl-frontier.png   # optional, link previews
---
```

**Math.** Use `$$...$$` for both display and inline math. kramdown emits it as
`\[...\]` / `\(...\)`, and `scripts/render-math.js` pre-renders that to HTML +
MathML with KaTeX after the build — no math JavaScript ships to readers. The
script fails the deploy if any equation does not parse.

**Figures** live in `assets/figures/<post-slug>/` — only the files a post
actually shows, since everything there is published. Scripts that generate
figures, and source/vector versions (PDF), go in `experiments/<post-slug>/`.

```html
<figure id="figure-free-energy-landscape">
  <img src="{{ '/assets/figures/<post-slug>/free-energy-landscape.png' | relative_url }}" alt="...">
  <figcaption>...</figcaption>
</figure>
```

## Languages (English / 中文)

English posts live in `_posts/`; their Chinese translations live in
`_zh_posts/` **under the same filename** and are served under `/zh/`. Pairing
is by filename (slug), which drives the header's 中文/English switcher and the
`hreflang` links. Translation front matter is the English post's, with
`title`/`subtitle` translated (`lang: zh` is set by default). A post with no
translation still appears on the other language's home page, badged.

UI strings (nav, dates, reading time, tagline, author name) are in
`_data/i18n.yml`. The Chinese edition has its own feed at `/zh/feed.xml`.

A first-time visitor gets the language their browser prefers (first of en/zh
in `navigator.languages`, else English). Choosing one with the switcher is
remembered in `localStorage` and wins from then on. The redirect logic is the
first `<script>` in `_layouts/default.html`; it never fires for crawlers or
for navigation within the site.

Translation conventions: keep proper names in English (Gibbs, Boltzmann,
Lagrange, Feynman), give the English original in parentheses at a key term's
first use — 配分函数（partition function） — keep method names and acronyms
(PPO, KL, …) in English, one paragraph per line, and a space between hanzi and
Latin text or inline math.

## Design

All theme values are CSS custom properties at the top of
`assets/css/main.css`, in the `:root` block:

| Token | Controls |
|---|---|
| `--measure` | **Line width.** One value; prose, figures, tables, code, header and footer all follow it. |
| `--gutter` | Page side padding |
| `--hot` / `--cold` | Accent (amber) and its cold-side counterpart; together they form the gradient rules |
| `--bg`, `--surface`, `--ink`, `--muted`, `--rule` | Neutrals |
| `--font-display`, `--font-body`, `--font-mono` | Bodoni Moda / Source Serif 4 / IBM Plex Mono |

Light is defined on bare `:root`; dark is redefined twice — under
`@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`,
and under `:root[data-theme="dark"]` — so the header toggle wins in both
directions and the page is still correct with JavaScript disabled. The toggle
persists to `localStorage`; a first visit gets dark, the site default.

Define colors only as tokens. A color declared solely inside a media or
`[data-theme]` block won't apply when the viewer is on the default "system"
setting.

## Porting to Medium

Medium has no LaTeX. Screenshot the rendered equations from the live site, or
use [math.vercel.app](https://math.vercel.app) to generate equation images.
Prose and figures paste over directly.

## LinkedIn teaser

Draft in `linkedin-teaser.md` — post it once the blog post is live. `_config.yml`
excludes that file and this README from the build, and the layouts emit Open
Graph tags so the link preview picks up the title, description, and `image:`.
