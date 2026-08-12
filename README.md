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

Pages is configured as **Settings → Pages → Deploy from a branch → main /
(root)**. Because this is a project page (not `<user>.github.io`), `_config.yml`
sets `baseurl: "/blog"` — leave it alone unless the repo is renamed.

Pages serves HTML with `Cache-Control: max-age=600`, so a hard refresh
(Ctrl-Shift-R) is often needed to see a fresh deploy.

## Writing

Posts are `_posts/YYYY-MM-DD-slug.md`; the filename sets the date and URL.
Front matter:

```yaml
---
layout: post
title: "A Physicist's Dictionary for DPO"
date: 2026-08-11
tags: [machine-learning, statistical-mechanics]
subtitle: "Optional — shown under the title"   # optional
image: /assets/figures/real/01_reward_kl_frontier.png   # optional, link previews
---
```

**Math.** kramdown turns `$$...$$` into `\[...\]` and passes single-dollar
`$x$` through untouched, so `_includes/custom-head.html` registers both
delimiter styles with MathJax. Use `$$...$$` for display and `$...$` inline.
Raw TeX stays visible while MathJax loads, so a blocked CDN degrades to
readable source instead of a blank column.

**Figures** live in `assets/figures/` and are referenced from the site root:

```markdown
![Free energy decomposition](/blog/assets/figures/real/02_free_energy_decomposition.png)
*Italic line right after an image is styled as a caption.*
```

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
persists to `localStorage`; first visit follows the OS setting.

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
