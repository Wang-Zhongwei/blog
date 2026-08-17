source "https://rubygems.org"

# Jekyll 3.9.x matches what GitHub Pages' own builder ships, so building via
# GitHub Actions does not silently change how existing posts render.
gem "jekyll", "~> 3.9.5"

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.15"
  gem "jekyll-sitemap", "~> 1.4"
  gem "kramdown-parser-gfm", "~> 1.1"
end

gem "webrick", "~> 1.8"   # no longer bundled with Ruby 3.x, needed by `jekyll serve`

# NOTE: math is NOT rendered by a Ruby plugin. kramdown emits raw \[ ... \]
# delimiters and scripts/render-math.js pre-renders them with KaTeX (npm) after
# the build -- see package.json. That keeps this Gemfile free of anything
# outside the GitHub Pages plugin whitelist.
