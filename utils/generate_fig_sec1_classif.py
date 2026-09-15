"""
Trois figures illustrant la chaine : espace initial -> espace latent -> probabilites.
Sortie : PNG a fond transparent, pensés pour un fond de slide clair.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

# ---- Palette alignee sur le CSS de la presentation -------------------------
BLEU   = "#1F6FB2"
ROUGE  = "#C0392B"
ENCRE  = "#14141A"
GRIS   = "#5A5A63"
FILET  = "#C9C9C2"

rng = np.random.default_rng(42)
N = 260          # points par classe
DPI = 220

plt.rcParams.update({
    "font.size": 11,
    "text.color": ENCRE,
    "axes.labelcolor": ENCRE,
    "savefig.transparent": True,
})


# ---- Fleche 3D correcte (quiver deforme les pointes en projection) ---------
class Arrow3D(FancyArrowPatch):
    def __init__(self, x0, y0, z0, dx, dy, dz, *args, **kw):
        super().__init__((0, 0), (0, 0), *args, **kw)
        self._xyz = (x0, y0, z0)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x0, y0, z0 = self._xyz
        dx, dy, dz = self._dxdydz
        xs, ys, zs = proj_transform(
            (x0, x0 + dx), (y0, y0 + dy), (z0, z0 + dz), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x0, y0, z0 = self._xyz
        dx, dy, dz = self._dxdydz
        xs, ys, zs = proj_transform(
            (x0, x0 + dx), (y0, y0 + dy), (z0, z0 + dz), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def repere(ax, longueur=2.6, labels=("", "", "")):
    """Trace un triedre discret a l'origine."""
    style = dict(mutation_scale=11, lw=1.1, arrowstyle="-|>", color=GRIS, alpha=.75)
    for dx, dy, dz in [(longueur, 0, 0), (0, longueur, 0), (0, 0, longueur)]:
        ax.add_artist(Arrow3D(0, 0, 0, dx, dy, dz, **style))


def nettoyer_3d(ax):
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))


# ===========================================================================
# Donnees : deux coquilles spheriques concentriques.
# Separables, mais par aucun hyperplan -> il faut vraiment un reseau.
# ===========================================================================
def coquille(n, rayon, epaisseur):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rayon + rng.normal(0, epaisseur, size=(n, 1))
    return v * r


X_bleu = coquille(N, 0.9, 0.22)     # classe 0, noyau interne
X_rouge = coquille(N, 2.6, 0.26)    # classe 1, coquille externe


# ===========================================================================
# FIGURE 1 : espace initial
# ===========================================================================
fig = plt.figure(figsize=(5.6, 5.2))
ax = fig.add_subplot(111, projection="3d")

# On dessine le rouge d'abord (externe) puis le bleu, pour que le noyau
# reste visible au travers de la coquille.
ax.scatter(*X_rouge.T, c=ROUGE, s=26, alpha=.30, edgecolors="none", depthshade=False)
ax.scatter(*X_bleu.T,  c=BLEU,  s=30, alpha=.95, edgecolors="none", depthshade=False)

repere(ax)
nettoyer_3d(ax)
ax.view_init(elev=16, azim=-58)
ax.set_title("Espace des données $\\mathcal{X}$\naucun plan ne sépare les deux classes",
             fontsize=12, color=ENCRE, y=.90, linespacing=1.5)
fig.subplots_adjust(left=-.10, right=1.10, bottom=-.14, top=1.02)
fig.savefig("../assets/img/01_espace_initial.png", dpi=DPI)
plt.close(fig)


# ===========================================================================
# FIGURE 2 : espace latent, apres f_theta
# ===========================================================================
# On simule le resultat de l'apprentissage : la profondeur radiale devient
# la coordonnee discriminante, les deux autres restent du bruit.
def vers_latent(X, cible):
    r = np.linalg.norm(X, axis=1)
    u = (r - 1.75) * 2.4 + rng.normal(0, .22, size=len(X))   # axe appris
    return np.column_stack([u, rng.normal(0, .72, len(X)), rng.normal(0, .72, len(X))])


Z_bleu = vers_latent(X_bleu, 0)
Z_rouge = vers_latent(X_rouge, 1)

fig = plt.figure(figsize=(5.6, 5.2))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(*Z_bleu.T,  c=BLEU,  s=28, alpha=.85, edgecolors="none", depthshade=False)
ax.scatter(*Z_rouge.T, c=ROUGE, s=28, alpha=.85, edgecolors="none", depthshade=False)

# Hyperplan separateur u = 0
yy, zz = np.meshgrid(np.linspace(-2.4, 2.4, 2), np.linspace(-2.4, 2.4, 2))
ax.plot_surface(np.zeros_like(yy), yy, zz, color=ENCRE, alpha=.07, shade=False)
for k in range(2):
    ax.plot([0, 0], [-2.4, 2.4], [zz[0, 0] if k == 0 else zz[1, 1]] * 2,
            color=FILET, lw=.9)
    ax.plot([0, 0], [yy[0, 0] if k == 0 else yy[1, 1]] * 2, [-2.4, 2.4],
            color=FILET, lw=.9)

repere(ax)
nettoyer_3d(ax)
ax.view_init(elev=16, azim=-58)
ax.set_title("Espace latent $f_\\theta(\\mathcal{X})$\nun hyperplan sépare les données ",
             fontsize=12, color=ENCRE, y=.90, linespacing=1.5)
fig.subplots_adjust(left=-.10, right=1.10, bottom=-.14, top=1.02)
fig.savefig("../assets/img/02_espace_latent.png", dpi=DPI)
plt.close(fig)


# ===========================================================================
# FIGURE 3 : logits -> probabilites par la sigmoide
# ===========================================================================
logit_bleu, logit_rouge = Z_bleu[:, 0] * 1.7, Z_rouge[:, 0] * 1.7

fig = plt.figure(figsize=(6.8, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[6, 1], wspace=.06)
ax = fig.add_subplot(gs[0, 0])
axp = fig.add_subplot(gs[0, 1], sharey=ax)

t = np.linspace(-9, 9, 400)
ax.plot(t, 1 / (1 + np.exp(-t)), color=ENCRE, lw=2.2, zorder=3)
ax.axhline(.5, color=FILET, lw=1, ls=(0, (4, 4)), zorder=1)
ax.axvline(0, color=FILET, lw=1, ls=(0, (4, 4)), zorder=1)

# Tapis des logits en bas, projection des probabilites a droite
for vals, coul in [(logit_bleu, BLEU), (logit_rouge, ROUGE)]:
    v = np.clip(vals, -8.6, 8.6)
    ax.scatter(v, np.full_like(v, -.055), c=coul, s=15, alpha=.40,
               edgecolors="none", clip_on=False, zorder=4)
    p = 1 / (1 + np.exp(-v))
    axp.scatter(rng.uniform(.12, .88, len(p)), p, c=coul, s=15, alpha=.40,
                edgecolors="none", zorder=4)

ax.annotate(r"$p_\theta(\cdot\,|\,x) = \sigma(f_\theta(x))$",
            xy=(2.6, .22), fontsize=13, color=ENCRE)

ax.set_xlabel("logit  $f_\\theta(x)[0]$", fontsize=11)
ax.set_ylabel("probabilité prédite", fontsize=11)
ax.set_xlim(-9, 9)
ax.set_ylim(-.09, 1.09)
ax.set_yticks([0, .5, 1])
ax.set_title("Des logits aux probabilités", fontsize=12, color=ENCRE, pad=12)

axp.set_xlim(0, 1)
axp.set_xticks([])
axp.tick_params(labelleft=False)
axp.set_xlabel("$\\Delta(\\mathcal{Y})$", fontsize=11, color=GRIS)

for a in (ax, axp):
    for bord in ("top", "right"):
        a.spines[bord].set_visible(False)
    for bord in ("left", "bottom"):
        a.spines[bord].set_color(FILET)
    a.tick_params(colors=GRIS, length=3)

fig.tight_layout()
fig.savefig("../assets/img/03_sigmoide.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("OK")