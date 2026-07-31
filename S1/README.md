# S1 — Activations exist for a reason

An interactive webapp proving four neural-network first principles, each computed
(not illustrated) in pure NumPy and rendered live in the browser.

| # | Claim | Proof (measured) |
|---|-------|------------------|
| **01** | No nonlinearity ⇒ only a straight boundary; one ReLU layer fixes it | linear **53.7%** (a line) vs ReLU **100%** on concentric rings |
| **02** | Depth without activation collapses to one layer | 1-layer **54.0%** = 5-linear **54.0%**; 5 matrices multiply to one **2×1**; +ReLU → **100%** |
| **03** | Similarity emerges from next-token alone | nearest-neighbour same-category **8/8** content tokens |
| **04** | Big model memorizes tiny data; data closes the gap | train→test gap **0.37** (n=20) → **0.05** (n=2000) |

The webapp is interactive: toggle between models and the decision boundary redraws
live; **"▶ Watch it learn"** trains a 2→16→1 ReLU net from scratch in your browser and
animates the straight line morphing into a closed ring; hover the embedding plot to see
nearest neighbours; light/dark toggle.

## Reproduce / rebuild

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python experiments.py     # trains all 4, writes site/data.json (+ _verify PNGs via _verify/verify.py)
.venv/bin/python make_og.py         # social card site/og.png + favicon.svg
.venv/bin/python build_site.py      # inlines data.json into site/index.html (self-contained)
```

## Deploy (Netlify)

`site/` is a static, self-contained site (no external requests). Deploy that folder:

- **Drag-drop:** log in at netlify.com, drag the `site/` folder onto the dashboard (or
  app.netlify.com/drop) → a permanent site under your account.
- **CLI:** `npx netlify-cli deploy --dir=site --prod`

For a guaranteed LinkedIn/Twitter link preview, the `og:image` should be an **absolute**
URL once you know the domain, e.g. `https://<site>.netlify.app/og.png` — bake it in and
redeploy.

## Files

```
experiments.py      all four experiments (pure NumPy) -> site/data.json
app_template.html   webapp source (HTML/CSS/JS; /*__DATA__*/ is replaced with data.json)
build_site.py       inlines data.json -> site/index.html
make_og.py          social preview card + favicon
site/               DEPLOY THIS: index.html, og.png, favicon.svg, data.json
_verify/            offline matplotlib sanity plots (not deployed)
```
