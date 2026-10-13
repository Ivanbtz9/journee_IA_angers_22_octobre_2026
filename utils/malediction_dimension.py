"""
Animations « malédiction de la dimension » — journée de l'IA (Angers).

Pavage uniforme de l'hypercube [-1, 1]^d avec un pas DELTA, et part des
points qui tombent dans la boule euclidienne unité B_2(0, 1).
Une vidéo mp4 + une image poster png par étape :

    1. d = 1 : boule = cube = segment
    2. d = 2 : disque dans le carré
    3. d = 3 : boule dans le cube (rotation)
    4. d = 4 : « vision temporelle » — on balaye la 4e coordonnée w,
               chaque coupe est un cube 3D qui contient une boule de rayon sqrt(1 - w^2)
    5. Bilan : rapport V_d / 2^d et nombre de points (2/DELTA + 1)^d pour d = 1..20

Usage :
    python malediction_dimension.py          # toutes les étapes
    python malediction_dimension.py 2 4      # seulement les étapes 2 et 4
"""
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------- paramètres
DELTA = 0.25                          # pas du pavage
# "sommets" : grille sur les sommets, (2/delta + 1)^d points (formule de la slide)
# "centres" : centres des petits cubes, (2/delta)^d points (proportion plus proche du rapport exact)
PAVAGE = "sommets"
FPS = 30
DOSSIER_VID = Path("../assets/videos")
DOSSIER_IMG = Path("../assets/img")

# Palette alignée sur custom.css
BLEU = "#1F6FB2"       # points dans la boule
ROUGE = "#C0392B"      # contour de la boule
GRIS = "#B9B9B2"       # points hors de la boule
TEXTE = "#14141A"
GRIS_TXT = "#5A5A63"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.edgecolor": GRIS_TXT,
    "axes.labelcolor": TEXTE,
    "xtick.color": GRIS_TXT,
    "ytick.color": GRIS_TXT,
    "figure.facecolor": "white",
})

if PAVAGE == "sommets":
    AXE = np.linspace(-1, 1, int(round(2 / DELTA)) + 1)
else:
    AXE = np.arange(-1 + DELTA / 2, 1, DELTA)
N_AXE = len(AXE)                      # 9 points par axe (sommets) ou 8 (centres)


def ratio_exact(d):
    """Rapport V_d / 2^d : volume de la boule euclidienne unité sur celui de [-1, 1]^d."""
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1) / 2 ** d


def fr(x, n=3):
    """Nombre à virgule (format français)."""
    return f"{x:.{n}f}".replace(".", ",")


def format_puissance(y, _pos):
    """Graduation de l'ordonnée sous la forme a·10^n (format français)."""
    if y == 0:
        return "0"
    n = int(math.floor(math.log10(abs(y))))
    a = y / 10 ** n
    if n < 3:
        return f"{y:.0f}"
    return rf"${a:.1f}\cdot 10^{{{n}}}$".replace(".", "{,}")


def nouvelle_figure():
    return plt.figure(figsize=(12.8, 7.2), dpi=100)   # 1280 x 720


def entete(fig, titre, sous_titre):
    fig.text(0.04, 0.93, titre, fontsize=26, fontweight="bold", color=TEXTE)
    fig.text(0.04, 0.88, sous_titre, fontsize=15, color=GRIS_TXT)


def enregistrer(fig, update, n_frames, nom, frame_poster=0):
    """Écrit la vidéo mp4 (lisible par les navigateurs) et l'image poster."""
    DOSSIER_VID.mkdir(parents=True, exist_ok=True)
    DOSSIER_IMG.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    writer = FFMpegWriter(fps=FPS, bitrate=5000,
                          extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    chemin = DOSSIER_VID / f"{nom}.mp4"
    anim.save(chemin, writer=writer)
    update(frame_poster)
    fig.savefig(DOSSIER_IMG / f"{nom}_poster.png", dpi=100)
    plt.close(fig)
    print(f"  -> {chemin}  ({n_frames / FPS:.1f} s)")


def cube_3d(ax):
    """Arêtes du cube [-1, 1]^3."""
    for a in (-1, 1):
        for b in (-1, 1):
            ax.plot([-1, 1], [a, a], [b, b], color=GRIS_TXT, lw=1)
            ax.plot([a, a], [-1, 1], [b, b], color=GRIS_TXT, lw=1)
            ax.plot([a, a], [b, b], [-1, 1], color=GRIS_TXT, lw=1)


def sphere_3d(ax, rayon):
    """Boule de rayon donné en fil de fer."""
    if rayon <= 1e-3:
        return
    u, v = np.mgrid[0:2 * np.pi:28j, 0:np.pi:16j]
    ax.plot_wireframe(rayon * np.cos(u) * np.sin(v), rayon * np.sin(u) * np.sin(v),
                      rayon * np.cos(v), color=ROUGE, lw=0.6, alpha=0.55)


def axes_3d_propres(ax, azim, elev=18):
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


# ================================================================ étape 1 : d = 1
def etape_1():
    fig = nouvelle_figure()
    entete(fig, "d = 1", f"Pavage de [-1, 1] avec un pas δ = {fr(DELTA, 2)} : la boule est le segment entier")
    ax = fig.add_axes([0.08, 0.30, 0.84, 0.40])
    info = fig.text(0.04, 0.12, "", fontsize=18, color=TEXTE)

    n_apparition, n_pause = 9 * 10, 2 * FPS

    def update(f):
        ax.cla()
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([]); ax.set_xticks([-1, 0, 1])
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        ax.plot([-1, 1], [0, 0], color=ROUGE, lw=3, alpha=0.6, zorder=1)
        k = min(N_AXE, f // 10 + 1)
        ax.scatter(AXE[:k], np.zeros(k), s=120, color=BLEU, zorder=2)
        info.set_text(f"Points dans la boule : {k} / {N_AXE}        "
                      f"Rapport exact  V₁ / 2¹ = 1")

    enregistrer(fig, update, n_apparition + n_pause, "dim_1")


# ================================================================ étape 2 : d = 2
def etape_2():
    X, Y = np.meshgrid(AXE, AXE)
    pts = np.c_[X.ravel(), Y.ravel()]
    dedans = (pts ** 2).sum(1) <= 1 + 1e-9
    N = len(pts)

    fig = nouvelle_figure()
    entete(fig, "d = 2", f"Pavage du carré [-1, 1]² avec un pas δ = {fr(DELTA, 2)} : {N_AXE}² = {N} points")
    ax = fig.add_axes([0.05, 0.06, 0.50, 0.78])
    info = fig.text(0.60, 0.55, "", fontsize=18, color=TEXTE, va="top", linespacing=1.8)

    par_frame = 1                      # un point par frame
    n_apparition, n_pause = N * par_frame, 2 * FPS

    def update(f):
        ax.cla()
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, ec=GRIS_TXT, lw=1.5))
        ax.add_patch(Circle((0, 0), 1, fc=ROUGE, alpha=0.07, ec=ROUGE, lw=2))
        k = min(N, f // par_frame + 1)
        c = np.where(dedans[:k], BLEU, GRIS)
        ax.scatter(pts[:k, 0], pts[:k, 1], s=55, c=c, zorder=3)
        n_in = int(dedans[:k].sum())
        texte = (f"Points dans la boule : {n_in} / {k}\n"
                 f"Proportion (pavage) : {fr(n_in / k)}")
        if k == N:
            texte += f"\n\nRapport exact  V₂ / 2² = π / 4 ≈ {fr(ratio_exact(2))}"
        info.set_text(texte)

    enregistrer(fig, update, n_apparition + n_pause, "dim_2")


# ================================================================ étape 3 : d = 3
def etape_3():
    X, Y, Z = np.meshgrid(AXE, AXE, AXE, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    # ordre d'apparition : couche par couche selon z
    pts = pts[np.argsort(pts[:, 2], kind="stable")]
    dedans = (pts ** 2).sum(1) <= 1 + 1e-9
    N = len(pts)
    par_couche = N_AXE ** 2

    fig = nouvelle_figure()
    entete(fig, "d = 3", f"Pavage du cube [-1, 1]³ avec un pas δ = {fr(DELTA, 2)} : {N_AXE}³ = {N} points")
    ax = fig.add_axes([0.0, 0.0, 0.62, 0.86], projection="3d")
    info = fig.text(0.62, 0.55, "", fontsize=18, color=TEXTE, va="top", linespacing=1.8)

    f_couche = 15                                   # frames par couche
    n_apparition = N_AXE * f_couche
    n_rotation = 6 * FPS
    n_pause = 1 * FPS
    azim0 = -60

    def update(f):
        ax.cla()
        if f < n_apparition:
            k = min(N, (f // f_couche + 1) * par_couche)
            azim = azim0
        else:
            k = N
            t = min(1.0, (f - n_apparition) / n_rotation)
            azim = azim0 + 360 * t
        axes_3d_propres(ax, azim)
        cube_3d(ax)
        sphere_3d(ax, 1.0)
        p, d = pts[:k], dedans[:k]
        ax.scatter(*p[~d].T, s=7, color=GRIS, alpha=0.35, depthshade=False)
        ax.scatter(*p[d].T, s=16, color=BLEU, alpha=0.95, depthshade=False)
        n_in = int(d.sum())
        texte = (f"Points dans la boule : {n_in} / {k}\n"
                 f"Proportion (pavage) : {fr(n_in / k)}")
        if k == N:
            texte += f"\n\nRapport exact  V₃ / 2³ = π / 6 ≈ {fr(ratio_exact(3))}"
        info.set_text(texte)

    enregistrer(fig, update, n_apparition + n_rotation + n_pause, "dim_3")


# ================================================================ étape 4 : d = 4
def etape_4():
    X, Y, Z = np.meshgrid(AXE, AXE, AXE, indexing="ij")
    pts3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    r2_3 = (pts3 ** 2).sum(1)
    # nombre de points de la grille 4D dans la boule, couche par couche (w fixé)
    dans_couche = np.array([int((r2_3 + w ** 2 <= 1 + 1e-9).sum()) for w in AXE])
    N_couche = len(pts3)
    N_total = N_couche * N_AXE

    fig = nouvelle_figure()
    entete(fig, "d = 4 : la 4ᵉ dimension comme le temps",
           f"À chaque instant w, la coupe de [-1, 1]⁴ est le cube [-1, 1]³ "
           f"et celle de la boule est une boule de rayon √(1 − w²)")
    ax3 = fig.add_axes([0.0, 0.0, 0.58, 0.84], projection="3d")
    axw = fig.add_axes([0.63, 0.50, 0.33, 0.30])
    info = fig.text(0.63, 0.40, "", fontsize=16, color=TEXTE, va="top", linespacing=1.8)

    n_balayage = 10 * FPS
    n_pause = 2 * FPS
    ws = np.linspace(-1, 1, n_balayage)
    profil_w = np.linspace(-1, 1, 400)

    def update(f):
        w = ws[min(f, n_balayage - 1)]
        k = int(round((w + 1) / DELTA))            # couche de la grille la plus proche
        wk = AXE[k]
        rayon = math.sqrt(max(0.0, 1 - w ** 2))

        # --- coupe 3D à l'instant w
        ax3.cla()
        axes_3d_propres(ax3, azim=-60 + 40 * min(f, n_balayage) / n_balayage)
        cube_3d(ax3)
        sphere_3d(ax3, rayon)
        d = r2_3 + wk ** 2 <= 1 + 1e-9
        ax3.scatter(*pts3[~d].T, s=7, color=GRIS, alpha=0.35, depthshade=False)
        ax3.scatter(*pts3[d].T, s=16, color=BLEU, alpha=0.95, depthshade=False)

        # --- frise temporelle : rayon de la coupe en fonction de w
        axw.cla()
        axw.fill_between(profil_w, np.sqrt(1 - profil_w ** 2), color=ROUGE, alpha=0.08)
        axw.plot(profil_w, np.sqrt(1 - profil_w ** 2), color=ROUGE, lw=2)
        axw.scatter(AXE, np.zeros(N_AXE), s=30, color=GRIS_TXT, zorder=3, clip_on=False)
        axw.scatter([wk], [0], s=90, color=BLEU, zorder=4, clip_on=False)
        axw.axvline(w, color=TEXTE, lw=1.5)
        axw.set_xlim(-1.05, 1.05); axw.set_ylim(0, 1.1)
        axw.set_xticks([-1, 0, 1]); axw.set_yticks([0, 1])
        axw.set_xlabel("w  (4ᵉ coordonnée = temps)")
        axw.set_ylabel("rayon de la coupe")
        for s in ("top", "right"):
            axw.spines[s].set_visible(False)

        # --- compteurs cumulés sur les couches déjà parcourues
        n_in = int(dans_couche[:k + 1].sum())
        n_vu = N_couche * (k + 1)
        texte = (f"Couche w = {fr(wk, 2)} :\n"
                 f"   {dans_couche[k]} / {N_couche} points dans la boule\n"
                 f"Cumul : {n_in} / {n_vu}\n"
                 f"Proportion (pavage) : {fr(n_in / n_vu)}")
        if f >= n_balayage - 1:
            texte += (f"\nTotal : {N_AXE}⁴ = {N_total} points\n"
                      f"Rapport exact  V₄ / 2⁴ = π² / 32 ≈ {fr(ratio_exact(4))}")
        info.set_text(texte)

    enregistrer(fig, update, n_balayage + n_pause, "dim_4")


# ================================================================ étape 5 : bilan
def etape_5():
    d_max = 20
    ds = np.arange(1, d_max + 1)
    ratios = np.array([ratio_exact(d) for d in ds])
    points = float(N_AXE) ** ds                    # (2/delta + 1)^d
    log10_points = ds * math.log10(N_AXE)

    # exemple d'une image 1024 x 1024 en couleur
    d_image = 3 * 1024 * 1024
    exposant_image = d_image * math.log10(N_AXE)

    fig = nouvelle_figure()
    entete(fig, "Quand la dimension grandit",
           f"La boule se vide pendant que le nombre de points du pavage explose (δ = {fr(DELTA, 2)})")
    ax1 = fig.add_axes([0.07, 0.20, 0.40, 0.58])
    ax2 = fig.add_axes([0.57, 0.20, 0.40, 0.58])
    note = fig.text(0.04, 0.05, "", fontsize=16, color=TEXTE)

    f_barre = 18
    n_apparition = d_max * f_barre
    n_pause = 3 * FPS

    def update(f):
        k = min(d_max, f // f_barre + 1)

        ax1.cla()
        ax1.bar(ds[:k], ratios[:k], color=BLEU)
        ax1.set_yscale("log"); ax1.set_ylim(1e-9, 2); ax1.set_xlim(0.3, d_max + 0.7)
        ax1.set_xticks([1, 5, 10, 15, 20])
        ax1.set_xlabel("dimension d")
        ax1.set_title(r"Part du cube occupée par la boule  $V_d\,/\,2^d$", fontsize=15, color=TEXTE)
        m, e = f"{ratios[k - 1]:.1e}".split("e")
        ax1.text(k, ratios[k - 1] * 2.5, rf"${m.replace('.', '{,}')}\cdot 10^{{{int(e)}}}$",
                 ha="center", fontsize=12, color=TEXTE)

        # Nombre de points en échelle LINÉAIRE, ordonnée recalée sur la dernière barre :
        # à chaque dimension, la nouvelle barre est N_AXE fois plus haute,
        # et toutes les précédentes s'écrasent (caractère géométrique)
        ax2.cla()
        ax2.bar(ds[:k], points[:k], color=ROUGE)
        t = min(1.0, (f % f_barre) / (f_barre * 0.6)) if f < n_apparition else 1.0
        t = 1 - (1 - t) ** 3                                  # recalage progressif
        haut_avant = points[k - 2] if k >= 2 else points[0]
        haut = haut_avant + t * (points[k - 1] - haut_avant)
        ax2.set_ylim(0, haut * 1.3); ax2.set_xlim(0.3, d_max + 0.7)
        ax2.yaxis.set_major_formatter(FuncFormatter(format_puissance))
        ax2.set_xticks([1, 5, 10, 15, 20])
        ax2.set_xlabel("dimension d")
        ax2.set_title(rf"Nombre de points du pavage  ${N_AXE}^d$", fontsize=15, color=TEXTE)
        ax2.text(k + 0.5 if k > 16 else k, points[k - 1] * 1.04, rf"${N_AXE}^{{{k}}} \approx$ " + format_puissance(points[k - 1], None),
                 ha="right" if k > 16 else "center", va="bottom", fontsize=12, color=TEXTE)
        if k >= 2:   # flèche « × N_AXE » entre les deux dernières barres
            ax2.annotate("", xy=(k - 0.2, points[k - 1] * 0.85), xytext=(k - 1, points[k - 2]),
                         arrowprops=dict(arrowstyle="->", color=TEXTE, lw=1.5,
                                         connectionstyle="arc3,rad=-0.35"))
            ax2.text(max(k - 1.7, 1.2), points[k - 1] * 0.6, rf"$\times {N_AXE}$",
                     ha="right", fontsize=14, color=TEXTE, fontweight="bold")

        for a in (ax1, ax2):
            for s in ("top", "right"):
                a.spines[s].set_visible(False)

        if f >= n_apparition:
            expo = f"{exposant_image:,.0f}".replace(",", r"\,")
            note.set_text(f"Image 1024 × 1024 en couleur : d = 3 × 1024² = {d_image:,}".replace(",", " ")
                          + rf"   →   ${N_AXE}^d \approx 10^{{{expo}}}$ points")
        else:
            note.set_text("")

    enregistrer(fig, update, n_apparition + n_pause, "dim_bilan", frame_poster=0)


# ================================================================ lancement
ETAPES = {1: etape_1, 2: etape_2, 3: etape_3, 4: etape_4, 5: etape_5}

if __name__ == "__main__":
    choix = [int(a) for a in sys.argv[1:]] or list(ETAPES)
    for i in choix:
        print(f"Étape {i}")
        ETAPES[i]()