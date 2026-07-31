"""
S1 — Activations / depth / embeddings / generalization.
Pure-numpy experiments. Exports site/data.json (for the webapp) and a few
verification PNGs in _verify/.  Deterministic (fixed seeds).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
VER = os.path.join(HERE, "_verify")
os.makedirs(SITE, exist_ok=True)
os.makedirs(VER, exist_ok=True)


# ----------------------------------------------------------------------------
# tiny numpy NN (row-vector convention: a[N,in] @ W[in,out] + b[out])
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def he(shape, rng, relu=True):
    fan_in = shape[0]
    s = np.sqrt(2.0 / fan_in) if relu else np.sqrt(1.0 / fan_in)
    return rng.standard_normal(shape) * s


class MLP:
    """sizes=[in, h1, ..., out]; hidden activation 'relu' or 'linear';
       final layer is linear logits -> sigmoid (binary) or softmax handled outside."""
    def __init__(self, sizes, rng, hidden="relu"):
        self.sizes, self.hidden = sizes, hidden
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            self.W.append(he((sizes[i], sizes[i + 1]), rng, relu=(hidden == "relu")))
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X, cache=False):
        a = X
        acts, zs = [a], []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            zs.append(z)
            if i < len(self.W) - 1 and self.hidden == "relu":
                a = np.maximum(0.0, z)
            else:
                a = z
            acts.append(a)
        return (a, acts, zs) if cache else a

    def logits(self, X):
        return self.forward(X)

    def effective_linear(self):
        """For a no-activation net: collapse all layers into one (W_eff, b_eff)."""
        W_eff = np.eye(self.sizes[0])
        b_eff = np.zeros(self.sizes[0])
        for i in range(len(self.W)):
            b_eff = b_eff @ self.W[i] + self.b[i]
            W_eff = W_eff @ self.W[i]
        return W_eff, b_eff


def train_binary(model, X, y, epochs, lr=0.03, seed=0):
    """Full-batch Adam on BCE-with-logits. Returns loss history."""
    y = y.reshape(-1, 1).astype(float)
    mW = [np.zeros_like(w) for w in model.W]; vW = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]; vb = [np.zeros_like(b) for b in model.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for t in range(1, epochs + 1):
        out, acts, zs = model.forward(X, cache=True)
        p = sigmoid(out)
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        hist.append(float(loss))
        g = (p - y) / X.shape[0]                      # dL/dlogit
        gW, gb = [None] * len(model.W), [None] * len(model.b)
        for i in reversed(range(len(model.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = g @ model.W[i].T
                if model.hidden == "relu":
                    g = g * (zs[i - 1] > 0)
        for i in range(len(model.W)):
            mW[i] = b1 * mW[i] + (1 - b1) * gW[i]; vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
            mb[i] = b1 * mb[i] + (1 - b1) * gb[i]; vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
            mWh = mW[i] / (1 - b1 ** t); vWh = vW[i] / (1 - b2 ** t)
            mbh = mb[i] / (1 - b1 ** t); vbh = vb[i] / (1 - b2 ** t)
            model.W[i] -= lr * mWh / (np.sqrt(vWh) + eps)
            model.b[i] -= lr * mbh / (np.sqrt(vbh) + eps)
    return hist


def acc_binary(model, X, y):
    p = sigmoid(model.logits(X)).ravel()
    return float(((p > 0.5).astype(int) == y).mean())


def prob_grid(model, xr, yr, n=90):
    xs = np.linspace(*xr, n); ys = np.linspace(*yr, n)
    gx, gy = np.meshgrid(xs, ys)
    G = np.stack([gx.ravel(), gy.ravel()], axis=1)
    p = sigmoid(model.logits(G)).reshape(n, n)
    return xs.tolist(), ys.tolist(), np.round(p, 4).tolist()


# ----------------------------------------------------------------------------
def make_rings(n=300, seed=1):
    rng = np.random.default_rng(seed)
    n0 = n // 2; n1 = n - n0
    r0 = rng.normal(0.8, 0.11, n0); t0 = rng.uniform(0, 2 * np.pi, n0)
    r1 = rng.normal(2.0, 0.16, n1); t1 = rng.uniform(0, 2 * np.pi, n1)
    X0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], 1)
    X1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], 1)
    X = np.vstack([X0, X1]); y = np.array([0] * n0 + [1] * n1)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


DATA = {}

# ===== S1-1 : rings, linear vs one ReLU hidden layer =========================
X, y = make_rings(300, seed=1)
XR = (float(X[:, 0].min() - 0.4), float(X[:, 0].max() + 0.4))
YR = (float(X[:, 1].min() - 0.4), float(X[:, 1].max() + 0.4))

lin = MLP([2, 1], np.random.default_rng(0), hidden="linear")
h_lin = train_binary(lin, X, y, epochs=4000, lr=0.05)
relu = MLP([2, 16, 1], np.random.default_rng(0), hidden="relu")
h_relu = train_binary(relu, X, y, epochs=4000, lr=0.03)

gx, gy, gp_lin = prob_grid(lin, XR, YR)
_, _, gp_relu = prob_grid(relu, XR, YR)
DATA["rings"] = {
    "points": [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(X, y)],
    "xr": XR, "yr": YR, "gx": [round(v, 3) for v in gx], "gy": [round(v, 3) for v in gy],
    "grid_linear": gp_lin, "grid_relu": gp_relu,
    "acc_linear": acc_binary(lin, X, y), "acc_relu": acc_binary(relu, X, y),
    "loss_linear": [round(v, 4) for v in h_lin[::40]],
    "loss_relu": [round(v, 4) for v in h_relu[::40]],
}
print(f"S1-1 rings:  linear acc={DATA['rings']['acc_linear']:.3f}  "
      f"relu acc={DATA['rings']['acc_relu']:.3f}")

# ===== S1-2 : depth without nonlinearity =====================================
l1 = MLP([2, 1], np.random.default_rng(2), hidden="linear")
train_binary(l1, X, y, epochs=4000, lr=0.05)
l5 = MLP([2, 8, 8, 8, 8, 1], np.random.default_rng(3), hidden="linear")
train_binary(l5, X, y, epochs=6000, lr=0.02)
l5r = MLP([2, 8, 8, 8, 8, 1], np.random.default_rng(4), hidden="relu")
train_binary(l5r, X, y, epochs=6000, lr=0.01)

W_eff, b_eff = l5.effective_linear()
_, _, g1 = prob_grid(l1, XR, YR)
_, _, g5 = prob_grid(l5, XR, YR)
_, _, g5r = prob_grid(l5r, XR, YR)
DATA["depth"] = {
    "acc_l1": acc_binary(l1, X, y), "acc_l5lin": acc_binary(l5, X, y),
    "acc_l5relu": acc_binary(l5r, X, y),
    "grid_l1": g1, "grid_l5lin": g5, "grid_l5relu": g5r,
    "shapes": [list(w.shape) for w in l5.W],
    "W_eff": np.round(W_eff, 4).tolist(), "b_eff": np.round(b_eff, 4).tolist(),
    # boundary of the collapsed single matrix, to overlay on the 5-linear plot
    "boundary_dir": np.round(W_eff.ravel(), 4).tolist(),
}
print(f"S1-2 depth:  1-layer acc={DATA['depth']['acc_l1']:.3f}  "
      f"5-linear acc={DATA['depth']['acc_l5lin']:.3f}  "
      f"5-relu acc={DATA['depth']['acc_l5relu']:.3f}")
print(f"          5 stacked linear layers collapse to W_eff (shape {W_eff.shape}) = {W_eff.ravel()}")

# ===== S1-3 : embeddings from next-token only ================================
CATS = {"START": "special", "END": "special",
        "cat": "animal", "dog": "animal", "cow": "animal",
        "eat": "verb", "chase": "verb", "see": "verb",
        "apple": "fruit", "mango": "fruit"}
VOCAB = list(CATS.keys())
VI = {w: i for i, w in enumerate(VOCAB)}
animals = ["cat", "dog", "cow"]; verbs = ["eat", "chase", "see"]; fruits = ["apple", "mango"]

def gen_pairs(n_sent=4000, seed=7):
    rng = np.random.default_rng(seed); pairs = []
    for _ in range(n_sent):
        s = ["START", rng.choice(animals), rng.choice(verbs), rng.choice(fruits), "END"]
        for a, b in zip(s, s[1:]):
            pairs.append((VI[a], VI[b]))
    return np.array(pairs)

pairs = gen_pairs()
V, d = len(VOCAB), 16
rng = np.random.default_rng(0)
E = rng.standard_normal((V, d)) * 0.3          # input embedding table
Wo = rng.standard_normal((d, V)) * 0.3; bo = np.zeros(V)
# Adam state
mE = np.zeros_like(E); vE = np.zeros_like(E)
mWo = np.zeros_like(Wo); vWo = np.zeros_like(Wo)
mbo = np.zeros_like(bo); vbo = np.zeros_like(bo)
b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.02
cur, nxt = pairs[:, 0], pairs[:, 1]
N = len(pairs)
emb_loss = []
for t in range(1, 1501):
    h = E[cur]                                  # [N,d]
    logits = h @ Wo + bo
    logits -= logits.max(1, keepdims=True)
    ex = np.exp(logits); P = ex / ex.sum(1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(N), nxt] + 1e-9))
    emb_loss.append(float(loss))
    dlogits = P.copy(); dlogits[np.arange(N), nxt] -= 1; dlogits /= N
    gWo = h.T @ dlogits; gbo = dlogits.sum(0)
    gh = dlogits @ Wo.T
    gE = np.zeros_like(E); np.add.at(gE, cur, gh)
    for (par, g, m, v) in [(E, gE, mE, vE), (Wo, gWo, mWo, vWo), (bo, gbo, mbo, vbo)]:
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * g * g
        par -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)

# PCA to 2D
Ec = E - E.mean(0)
U, Sv, Vt = np.linalg.svd(Ec, full_matrices=False)
coords = Ec @ Vt[:2].T
# nearest neighbour by cosine
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
cos = En @ En.T; np.fill_diagonal(cos, -2)
nn = cos.argmax(1)
same = sum(1 for i in range(V) if CATS[VOCAB[i]] == CATS[VOCAB[nn[i]]] and CATS[VOCAB[i]] != "special")
DATA["embed"] = {
    "tokens": VOCAB, "cats": [CATS[w] for w in VOCAB],
    "coords": np.round(coords, 3).tolist(),
    "nn": [VOCAB[j] for j in nn],
    "loss": [round(v, 4) for v in emb_loss[::20]],
    "nn_same_cat": int(same), "nn_total_content": int(sum(1 for w in VOCAB if CATS[w] != "special")),
}
print(f"S1-3 embed:  nearest-neighbour same-category (content tokens): "
      f"{DATA['embed']['nn_same_cat']}/{DATA['embed']['nn_total_content']}")

# ===== S1-4 : memorization vs generalization =================================
def make_task(n, seed):
    """circular boundary (inside=1) + 15% label noise, standardized 2D."""
    rng = np.random.default_rng(seed)
    Xx = rng.uniform(-2.5, 2.5, (n, 2))
    r = np.sqrt((Xx ** 2).sum(1))
    yy = (r < 1.4).astype(int)
    flip = rng.random(n) < 0.15
    yy = np.where(flip, 1 - yy, yy)
    return Xx, yy

Xte, yte = make_task(2000, seed=999)
sizes = [20, 200, 2000]
gen = {"sizes": sizes, "train_acc": [], "test_acc": [], "train_loss": [], "test_loss": [], "grids": []}
GXR = (-2.7, 2.7)
for i, ntr in enumerate(sizes):
    Xtr, ytr = make_task(ntr, seed=100 + i)
    net = MLP([2, 256, 256, 1], np.random.default_rng(5), hidden="relu")
    train_binary(net, Xtr, ytr, epochs=3000, lr=0.005)
    ptr = sigmoid(net.logits(Xtr)).ravel(); pte = sigmoid(net.logits(Xte)).ravel()
    bce = lambda p, yy: float(-np.mean(yy * np.log(p + 1e-9) + (1 - yy) * np.log(1 - p + 1e-9)))
    gen["train_acc"].append(round(float(((ptr > .5) == ytr).mean()), 4))
    gen["test_acc"].append(round(float(((pte > .5) == yte).mean()), 4))
    gen["train_loss"].append(round(bce(ptr, ytr), 4))
    gen["test_loss"].append(round(bce(pte, yte), 4))
    _, _, gg = prob_grid(net, GXR, GXR, n=80)
    gen["grids"].append(gg)
    print(f"S1-4 gen n={ntr:4d}:  train_acc={gen['train_acc'][-1]:.3f} "
          f"test_acc={gen['test_acc'][-1]:.3f}  gap={gen['train_acc'][-1]-gen['test_acc'][-1]:.3f}")
# a small sample of train points for the smallest set (to show memorization)
Xtr20, ytr20 = make_task(20, seed=100)
gen["train20"] = [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(Xtr20, ytr20)]
gen["gxr"] = GXR
DATA["gen"] = gen

json.dump(DATA, open(os.path.join(SITE, "data.json"), "w"))
print("\nwrote site/data.json  (%.0f KB)" % (os.path.getsize(os.path.join(SITE, 'data.json')) / 1024))
