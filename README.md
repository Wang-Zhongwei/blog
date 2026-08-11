# Blog

Jekyll blog scaffold, built remotely by GitHub Pages — no local Ruby needed.

## How to publish

1. Create a GitHub repo, either:
   - **`<your-username>.github.io`** → blog served at `https://<your-username>.github.io/`
     (leave `baseurl: ""` in `_config.yml`), or
   - **any name, e.g. `blog`** → served at `https://<your-username>.github.io/blog/`
     (set `baseurl: "/blog"` in `_config.yml`).
2. Push this directory:
   ```bash
   cd ~/blog
   git remote add origin git@github.com:<your-username>/<repo>.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Source: Deploy from a branch → main / (root)**.
   The site builds in ~1 minute.

## Writing

- The post template is `_posts/2026-08-11-a-physicists-dictionary-for-dpo.md`.
  Fill in the `✍️` markers, delete the comment blocks, rename the file if you
  change the title/date (format must stay `YYYY-MM-DD-slug.md`).
- Math: `$$...$$` for display, `$...$` inline (MathJax, wired in
  `_includes/custom-head.html`).
- Figures live in `assets/figures/` (copied from `stat-mech-dpo/figures/`,
  PNG only). Re-copy if you regenerate them.
- Set your blog `title`/`author` in `_config.yml`.

## Porting to Medium later

Medium has no LaTeX. If the post is good and you want it there too: screenshot
the rendered equations from your GitHub Pages site (or use
[math.now.sh](https://math.vercel.app) to generate equation images) and paste
them into the Medium editor. Prose and figures paste over directly.

## LinkedIn teaser

Draft in `linkedin-teaser.md` — post it once the blog post is live.
