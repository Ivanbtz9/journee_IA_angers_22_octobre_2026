"""
Theoreme d'approximation universelle, idee de la preuve en 1D :
(a) un neurone sigma(w(x - c)) devient une marche quand w grandit ;
(b) une somme de N marches approche f uniformement quand N grandit.
Sortie : PNG a fond transparent, pense pour un fond de slide clair.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Palette alignee sur le CSS de la presentation -------------------------
BLEU   = "#1F6FB2"
BLEU_C = "#A9C8E3"
ROUGE  = "#C0392B"
ENCRE  = "#14141A"
GRIS   = "#5A5A63"
FILET  = "#C9C9C2"
DPI = 220

plt.rcParams.update({"font.size": 12, "text.color": ENCRE,
                     "axes.labelcolor": ENCRE, "savefig.transparent": True})


def sigma(t):
    return 1 / (1 + np.exp(-np.clip(t, -60, 60)))


def f(x):
    """Fonction cible continue sur [0, 1]."""
    return .6 * np.sin(2 * np.pi * x) + .3 * np.cos(5 * np.pi * x) + .4 * x


def g(x, k, raideur=10.0):
    """Reseau a une couche cachee de k neurones, notations du theoreme :
    g(x) = C sigma(A x + b) = sum_i c_i sigma(a_i x + b_i).
    - k - 1 marches : pente a = raideur / h, centre au milieu de chaque pas
      de la grille, hauteur c_i = f(x_i) - f(x_{i-1}) ;
    - 1 neurone de pente nulle (a = 0, b = 0) qui porte la constante f(x_0),
      puisque sigma(0) = 1/2."""
    N = k - 1
    xs = np.linspace(0, 1, N + 1)
    centres = (xs[:-1] + xs[1:]) / 2
    a = np.r_[np.full(N, raideur / (xs[1] - xs[0])), 0.]
    b = np.r_[-a[:-1] * centres, 0.]
    C = np.r_[np.diff(f(xs)), 2 * f(xs[0])]
    return sigma(np.outer(x, a) + b) @ C


x = np.linspace(0, 1, 1200)

fig = plt.figure(figsize=(10.4, 3.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=.2)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])

# ---- (a) un neurone -> une marche ------------------------------------------
xa = np.linspace(-1, 1, 600)
for w, coul, lw in [(3, BLEU_C, 2.0), (10, BLEU, 2.0), (80, ENCRE, 2.2)]:
    axa.plot(xa, sigma(w * xa), color=coul, lw=lw, label=f"$a = {w}$")
axa.axvline(0, color=FILET, lw=1, ls=(0, (4, 4)), zorder=0)
axa.set_title(r"Un neurone : $\sigma(a x + b)$", fontsize=13, pad=10)
axa.set_xlabel("$x$")
axa.set_xticks([-1, 0, 1], ["", "$-b/a$", ""])
axa.set_yticks([0, .5, 1])
axa.legend(frameon=False, fontsize=11, loc="upper left", handlelength=1.4)

# ---- (b) somme de N neurones -> approximation de f -------------------------
axb.plot(x, f(x), color=ENCRE, lw=2.6, label=r"$f \in \mathcal{C}([0,1], \mathbb{R})$", zorder=4)
for k, coul, ls, lw in [(4, BLEU_C, "-", 2.0), (12, BLEU, "-", 1.8),
                        (60, ROUGE, (0, (4, 2)), 1.8)]:
    gk = g(x, k)
    err = np.abs(f(x) - gk).max()
    axb.plot(x, gk, color=coul, lw=lw, ls=ls, zorder=5,
             label=rf"$k = {k}$ :  $\|f - g\|_\infty = {err:.2f}$")
axb.set_title(r"$g(x) = C\,\sigma(Ax + b) = \sum_{i=1}^{k} c_i\, \sigma(a_i x + b_i)$",
              fontsize=13, pad=10)
axb.set_xlabel(r"$x \in K = [0, 1]$")
axb.set_xlim(0, 1)
axb.legend(frameon=False, fontsize=11, loc="center left",
           bbox_to_anchor=(1.01, .5), handlelength=1.8)

for a in (axa, axb):
    for bord in ("top", "right"):
        a.spines[bord].set_visible(False)
    for bord in ("left", "bottom"):
        a.spines[bord].set_color(FILET)
    a.tick_params(colors=GRIS, length=3)

fig.savefig("assets/img/05_approximation_universelle.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("OK")