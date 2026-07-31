import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(H, "site", "data.json")))
VER = os.path.join(H, "_verify")

def boundary(ax, gx, gy, grid, pts, title, acc):
    ax.contourf(gx, gy, np.array(grid), levels=20, cmap="RdBu_r", alpha=.8, vmin=0, vmax=1)
    ax.contour(gx, gy, np.array(grid), levels=[.5], colors="k", linewidths=1.5)
    if len(pts):
        p = np.array(pts)
        ax.scatter(p[p[:,2]==0][:,0], p[p[:,2]==0][:,1], s=12, c="#1f3b8c", edgecolors="w", lw=.3)
        ax.scatter(p[p[:,2]==1][:,0], p[p[:,2]==1][:,1], s=12, c="#8c1f1f", edgecolors="w", lw=.3)
    ax.set_title(f"{title}\nacc={acc:.2f}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])

# S1-1
r = D["rings"]; fig, ax = plt.subplots(1, 2, figsize=(9, 4.3))
boundary(ax[0], r["gx"], r["gy"], r["grid_linear"], r["points"], "Linear + sigmoid", r["acc_linear"])
boundary(ax[1], r["gx"], r["gy"], r["grid_relu"], r["points"], "1 ReLU hidden layer", r["acc_relu"])
fig.tight_layout(); fig.savefig(f"{VER}/s1_1_rings.png", dpi=110); plt.close(fig)

# S1-2
d = D["depth"]; fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
boundary(ax[0], r["gx"], r["gy"], d["grid_l1"], r["points"], "1 linear layer", d["acc_l1"])
boundary(ax[1], r["gx"], r["gy"], d["grid_l5lin"], r["points"], "5 linear layers (no act)", d["acc_l5lin"])
boundary(ax[2], r["gx"], r["gy"], d["grid_l5relu"], r["points"], "5 layers + ReLU", d["acc_l5relu"])
fig.tight_layout(); fig.savefig(f"{VER}/s1_2_depth.png", dpi=110); plt.close(fig)

# S1-3
e = D["embed"]; fig, ax = plt.subplots(figsize=(6, 5))
col = {"animal":"#c0392b","verb":"#27ae60","fruit":"#8e44ad","special":"#7f8c8d"}
c = np.array(e["coords"])
for i, tok in enumerate(e["tokens"]):
    ax.scatter(c[i,0], c[i,1], s=90, c=col[e["cats"][i]])
    ax.annotate(tok, (c[i,0], c[i,1]), fontsize=9, xytext=(4,4), textcoords="offset points")
ax.set_title("Learned embeddings (PCA-2D)\nsame-category tokens cluster", fontsize=11)
fig.tight_layout(); fig.savefig(f"{VER}/s1_3_embed.png", dpi=110); plt.close(fig)

# S1-4
g = D["gen"]; fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
for i, n in enumerate(g["sizes"]):
    xs = np.linspace(*g["gxr"], 80)
    boundary(ax[i], xs.tolist(), xs.tolist(), g["grids"][i], g["train20"] if n==20 else [],
             f"n={n}", g["test_acc"][i])
ax[3].plot(g["sizes"], [1-a for a in g["train_acc"]], "o-", label="train err")
ax[3].plot(g["sizes"], [1-a for a in g["test_acc"]], "s-", label="test err")
ax[3].set_xscale("log"); ax[3].set_title("gap shrinks with data"); ax[3].legend(); ax[3].set_xticks([])
fig.tight_layout(); fig.savefig(f"{VER}/s1_4_gen.png", dpi=110); plt.close(fig)
print("saved verification PNGs")
