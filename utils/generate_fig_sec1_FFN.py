"""
Deux figures pour la section « Reseau de neurones & algebre lineaire ».

  04 : action d'une couche sur l'espace, en quatre temps
       x  ->  Wx  ->  Wx+b  ->  tanh(Wx+b)
  05 : la meme couche vue comme un produit matriciel
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# ---- Palette alignee sur le CSS de la presentation -------------------------
BLEU  = "#1F6FB2"
ROUGE = "#C0392B"
ENCRE = "#14141A"
GRIS  = "#5A5A63"
FILET = "#C9C9C2"
VERT  = "#2E7D55"

rng = np.random.default_rng(7)
DPI = 220

plt.rcParams.update({
    "font.size": 11,
    "text.color": ENCRE,
    "savefig.transparent": True,
})

# ===========================================================================
# FIGURE : lecture matricielle de la meme couche
# ===========================================================================
from matplotlib.patches import Rectangle

d_in, d_out = 6, 4
Wm = rng.normal(0, 1, (d_out, d_in))
xv = rng.normal(0, 1, (d_in, 1))
bv = rng.normal(0, 1, (d_out, 1))
av = np.tanh(Wm @ xv + bv)

carte = plt.get_cmap("RdBu_r")
norme = plt.Normalize(-2.4, 2.4)

fig, ax = plt.subplots(figsize=(13.0, 4.4))
ax.set_aspect("equal")
ax.axis("off")

CUR = {"x": 0.0}          # curseur horizontal, en unites de cellule


def matrice(M, titre, forme, ecart=1.0):
    """Dessine M en cellules unitaires, centree verticalement sur 0."""
    nl, nc = M.shape
    x0 = CUR["x"] + ecart
    y0 = -nl / 2
    for i in range(nl):
        for j in range(nc):
            ax.add_patch(Rectangle(
                (x0 + j, y0 + nl - 1 - i), 1, 1,
                facecolor=carte(norme(M[i, j])), edgecolor="white", lw=1.4))
    ax.add_patch(Rectangle((x0, y0), nc, nl, fill=False,
                           edgecolor=FILET, lw=1.6))
    ax.text(x0 + nc / 2, y0 + nl + .45, titre, ha="center", va="bottom",
            fontsize=17, color=ENCRE)
    ax.text(x0 + nc / 2, y0 - .55, forme, ha="center", va="top",
            fontsize=11, color=GRIS)
    CUR["x"] = x0 + nc
    return x0


def symbole(s, ecart=.55):
    x0 = CUR["x"] + ecart
    ax.text(x0, 0, s, ha="center", va="center", fontsize=23, color=GRIS)
    CUR["x"] = x0 + ecart


matrice(Wm, r"$W$", f"${d_out} \\times {d_in}$", ecart=0)
symbole(r"$\cdot$")
xW = matrice(xv, r"$x$", f"${d_in} \\times 1$")
symbole("+")
matrice(bv, r"$b$", f"${d_out} \\times 1$")
symbole("=")
xA = matrice(av, r"$\tanh(Wx+b)$", f"${d_out} \\times 1$")

# Mise en evidence : ligne 2 de W x colonne de x -> coordonnee 2 de la sortie
ax.add_patch(Rectangle((0, d_out / 2 - 2), d_in, 1, fill=False,
                       edgecolor=ENCRE, lw=2.2, zorder=5))
ax.add_patch(Rectangle((xW, -d_in / 2), 1, d_in, fill=False,
                       edgecolor=ENCRE, lw=2.2, zorder=5))
ax.add_patch(Rectangle((xA, d_out / 2 - 2), 1, 1, fill=False,
                       edgecolor=ENCRE, lw=2.2, zorder=5))

ax.set_xlim(-.6, CUR["x"] + .6)
ax.set_ylim(-d_in / 2 - 1.5, d_in / 2 + 1.6)
ax.text(CUR["x"] / 2, -d_in / 2 - 1.35,
        "Chaque ligne de $W$ donne un produit scalaire avec l'entrée",
        ha="center", fontsize=13, color=GRIS)

fig.tight_layout()
fig.savefig("../assets/img/05_couche_matrices.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("OK")