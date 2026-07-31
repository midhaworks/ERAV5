"""Generate site/og.png (1200x630) social-preview card + a favicon."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

H = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(H, "site", "data.json")))
R = D["rings"]
BG = "#0B0D12"; INK = "#EAEEF6"; SUB = "#AAB4C6"; AMBER = "#E6A93C"
C0 = np.array([0x5B, 0x7B, 0xE8]) / 255; C1 = np.array([0xE0, 0x57, 0x4A]) / 255
PAPER = np.array([0x1A, 0x20, 0x2C]) / 255

def field(grid):
    g = np.array(grid); img = np.zeros(g.shape + (3,))
    t = (g - .5) * 2; mag = np.abs(t) ** .6
    for k in range(3):
        img[..., k] = np.where(t < 0, PAPER[k] + (C0[k] - PAPER[k]) * mag,
                               PAPER[k] + (C1[k] - PAPER[k]) * mag)
    return img

fig = plt.figure(figsize=(12, 6.3), dpi=100)
fig.patch.set_facecolor(BG)
# title band
fig.text(0.06, 0.93, "Four small experiments. Four neural-network truths.", color=INK,
         fontsize=31, fontweight="bold", va="top")
fig.text(0.06, 0.815, "Activations, nonlinear depth, embeddings, and generalization — proved in pure NumPy.",
         color=SUB, fontsize=15.5, va="top")
fig.text(0.06, 0.07, "neural-network first principles · pure-NumPy proof, live in the browser",
         color=AMBER, fontsize=13, va="top", family="monospace")

pts = np.array(R["points"])
def panel(rect, grid, title, acc):
    ax = fig.add_axes(rect)
    ax.imshow(field(grid), extent=[*R["xr"], *R["yr"]], origin="lower", aspect="auto")
    ax.contour(R["gx"], R["gy"], np.array(grid), levels=[.5], colors="white", linewidths=2)
    for c, col in [(0, C0), (1, C1)]:
        p = pts[pts[:, 2] == c]
        ax.scatter(p[:, 0], p[:, 1], s=12, color=col, edgecolors=BG, linewidths=.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2A3242")
    ax.set_title(f"{title}   ·   {acc*100:.0f}%", color=INK, fontsize=15, pad=8)

panel([0.06, 0.13, 0.40, 0.46], R["grid_linear"], "linear", R["acc_linear"])
panel([0.54, 0.13, 0.40, 0.46], R["grid_relu"], "1 ReLU layer", R["acc_relu"])
fig.text(0.50, 0.36, "→", color=AMBER, fontsize=40, ha="center", va="center", fontweight="bold")

fig.savefig(os.path.join(H, "site", "og.png"), facecolor=BG)
print("wrote site/og.png")

# tiny favicon: two squares (class colours) as an SVG file
svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
       '<rect width="32" height="32" rx="7" fill="#0B0D12"/>'
       '<circle cx="12" cy="16" r="6" fill="#5B7BE8"/>'
       '<circle cx="21" cy="16" r="6" fill="#E0574A"/></svg>')
open(os.path.join(H, "site", "favicon.svg"), "w").write(svg)
print("wrote site/favicon.svg")
