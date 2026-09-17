"""
Un neurone = une ligne de W : h_i = tanh(w_i . x + b_i).
Reseau 6 -> 4 coherent avec la heatmap W (4 x 6) de la slide.
Sortie : PNG a fond transparent, pense pour un fond de slide clair.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---- Palette alignee sur le CSS de la presentation -------------------------
BLEU   = "#1F6FB2"
OR     = "#E0B03C"
ENCRE  = "#14141A"
GRIS   = "#5A5A63"
FILET  = "#C9C9C2"
CMAP   = plt.get_cmap("RdBu_r")   # meme colormap que la heatmap de W
DPI = 220

plt.rcParams.update({"font.size": 11, "text.color": ENCRE,
                     "savefig.transparent": True})

rng = np.random.default_rng(3)
W = rng.normal(0, 1, size=(4, 6))
i_sel = 1                          # neurone mis en avant (2e ligne de W)
norm = TwoSlopeNorm(vcenter=0, vmin=-np.abs(W).max(), vmax=np.abs(W).max())

# ---- Positions des noeuds --------------------------------------------------
x_in, x_h = 0.0, 3.2
y_in = np.linspace(2.5, -2.5, 6)
y_h = np.linspace(1.8, -1.8, 4)

fig, ax = plt.subplots(figsize=(7.6, 4.6))

# Aretes : grises pour les neurones non selectionnes
for i, yh in enumerate(y_h):
    if i == i_sel:
        continue
    for yi in y_in:
        ax.plot([x_in, x_h], [yi, yh], color=FILET, lw=.8, zorder=1)

# Aretes du neurone selectionne : couleur et epaisseur selon le poids
for j, yi in enumerate(y_in):
    w = W[i_sel, j]
    ax.plot([x_in, x_h], [yi, y_h[i_sel]], color=GRIS, lw=.6,
            ls=(0, (3, 3)), zorder=2.5)
    ax.plot([x_in, x_h], [yi, y_h[i_sel]], color=CMAP(norm(w)),
            lw=1.8 + 2.6 * abs(w) / np.abs(W).max(), zorder=2,
            solid_capstyle="round")
    ax.text(x_in + .55, yi + (y_h[i_sel] - yi) * .17 + .12,
            f"$w_{{{i_sel+1}{j+1}}}$", fontsize=9, color=GRIS,
            ha="center", zorder=5)

# Noeuds
ax.scatter(np.full(6, x_in), y_in, s=260, c=OR, edgecolors=ENCRE,
           linewidths=.6, zorder=3)
for j, yi in enumerate(y_in):
    ax.text(x_in - .38, yi, f"$x_{j+1}$", ha="right", va="center", fontsize=12)

couleurs_h = [BLEU if i == i_sel else "#A9C8E3" for i in range(4)]
ax.scatter(np.full(4, x_h), y_h, s=[520 if i == i_sel else 300 for i in range(4)],
           c=couleurs_h, edgecolors=ENCRE, linewidths=.6, zorder=3)
for i, yh in enumerate(y_h):
    if i != i_sel:
        ax.text(x_h + .38, yh, f"$h_{i+1}$", ha="left", va="center",
                fontsize=12, color=GRIS)

# Formule du neurone selectionne, a droite du noeud
ax.text(x_h + .45, y_h[i_sel],
        rf"$h_{i_sel+1} = \sigma(\mathbf{{w}}_{i_sel+1}^\top x + b_{i_sel+1})$",
        ha="left", va="center", fontsize=13, color=ENCRE)

# Ligne w_i dessinee sous le reseau : c'est la ligne i de W
ax_w = fig.add_axes([.14, .06, .40, .07])
ax_w.imshow(W[i_sel][None, :], cmap=CMAP, norm=norm, aspect="auto")
ax_w.set_xticks(range(6), [f"$w_{{{i_sel+1}{j+1}}}$" for j in range(6)],
                fontsize=9, color=GRIS)
ax_w.set_yticks([])
ax_w.tick_params(length=0)
for s in ax_w.spines.values():
    s.set_color(ENCRE); s.set_linewidth(1.2)
fig.text(.125, .095, rf"$\mathbf{{w}}_{i_sel+1} =$", ha="right", va="center",
         fontsize=13)
fig.text(.56, .095, r"$\in \mathbb{R}^{6}$ : 2$^{e}$ ligne de $W$", ha="left",
         va="center", fontsize=11, color=GRIS)

ax.set_xlim(-1.1, 7.6)
ax.set_ylim(-2.9, 2.9)
ax.set_aspect("equal")
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=.2)
fig.savefig("./assets/img/04_neurone_affine.png", dpi=DPI)
plt.close(fig)
print("OK")