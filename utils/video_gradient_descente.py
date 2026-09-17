"""
Descente de gradient stochastique sur un paysage de perte chaotique (2 parametres).
- SGD : theta <- theta - eta * (grad L(theta) + bruit)  -> le bruit du batch
        peut faire sortir des petits puits (ici, il atteint le minimum global)
- La camera tourne, monte/descend et zoome pendant la descente.
Sortie : video MP4 (H.264, lisible par les navigateurs) pour Reveal.js.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

# ---- Palette alignee sur le CSS de la presentation -------------------------
BLEU   = "#1F6FB2"
ROUGE  = "#C0392B"
ENCRE  = "#14141A"
GRIS   = "#5A5A63"
FILET  = "#C9C9C2"
CMAP = LinearSegmentedColormap.from_list("paysage", ["#0E3F66", BLEU, "#A9C8E3", "#F4F4F0"])

plt.rcParams.update({"font.size": 12, "text.color": ENCRE, "axes.labelcolor": ENCRE})


# ---- Paysage de perte ------------------------------------------------------
def L(u, v):
    """Cuvette large + ondulations a plusieurs frequences (nombreux minima
    locaux) + un puits global."""
    return (0.06 * (u**2 + 1.5 * v**2)
            + 0.30 * np.sin(1.4 * u) * np.cos(1.2 * v)
            + 0.15 * np.sin(2.6 * u + 0.8 * v + 1.0)
            + 0.10 * np.cos(3.1 * v - 0.5 * u)
            + 0.14 * np.sin(4.3 * u) * np.sin(3.7 * v)      # hautes frequences
            + 0.08 * np.cos(5.9 * u + 4.1 * v)
            + 0.06 * np.sin(7.3 * u - 6.1 * v + 0.4)
            - 1.30 * np.exp(-((u - 1.3)**2 + (v + 0.9)**2) / 1.4))


def grad(u, v, h=1e-4):
    return np.array([(L(u + h, v) - L(u - h, v)) / (2 * h),
                     (L(u, v + h) - L(u, v - h)) / (2 * h)])


def descente(p0, eta, T, bruit=0.0, decroissance=0.0, seed=0):
    """bruit : ecart-type du bruit de gradient (simule le mini-batch),
    decroissant en 1 / (1 + decroissance * t)."""
    rng = np.random.default_rng(seed)
    p = np.array(p0, float)
    traj = [p.copy()]
    for t in range(T):
        s = bruit / (1 + decroissance * t)
        p = p - eta * (grad(*p) + rng.normal(0, s, 2))
        traj.append(p.copy())
    return np.array(traj)


T, P0, ETA = 300, (-4.0, 3.2), .12
sgd = descente(P0, ETA, T, bruit=4.0, decroissance=.02, seed=4)
L_sgd = L(*sgd.T)

g = np.linspace(-5, 5, 130)          # maillage fin pour les hautes frequences
U, V = np.meshgrid(g, g)
Z = L(U, V)

# ---- Mise en page : surface 3D a gauche, vue de dessus + courbe a droite ---
fig = plt.figure(figsize=(12, 5.6), facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.3, 1],
                      wspace=.08, hspace=.45)
ax3 = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
axc = fig.add_subplot(gs[0, 1])
axl = fig.add_subplot(gs[1, 1])

ax3.plot_surface(U, V, Z, cmap=CMAP, rstride=1, cstride=1, lw=0,
                 antialiased=True, alpha=.95, zorder=1)
ax3.set_axis_off()
ax3.set_box_aspect((1, 1, .55), zoom=1.3)
ax3.set_title(r"Paysage de perte $\mathcal{L}(\theta_1, \theta_2)$", fontsize=14, y=1.02)

axc.contourf(U, V, Z, 40, cmap=CMAP)
axc.contour(U, V, Z, 20, colors="white", linewidths=.4, alpha=.5)
axc.set_aspect("equal")
axc.set_xticks([]); axc.set_yticks([])
axc.set_title("Vue de dessus", fontsize=12)

axl.set_xlim(0, T)
axl.set_ylim(L_sgd.min() - .1, L_sgd.max() + .1)
axl.set_xlabel("itération $t$")
axl.set_ylabel(r"$\mathcal{L}(\theta_t)$")
for bord in ("top", "right"):
    axl.spines[bord].set_visible(False)
for bord in ("left", "bottom"):
    axl.spines[bord].set_color(FILET)
axl.tick_params(colors=GRIS, length=3)

# Artistes animes
lift = .08
art = {}
for nom, coul, lab in [("sgd", ROUGE, "SGD (mini-batch)")]:
    art[nom] = dict(
        l3=ax3.plot([], [], [], color=coul, lw=1.8, zorder=10)[0],
        p3=ax3.plot([], [], [], "o", color=coul, ms=8, mec="white", mew=1, zorder=11)[0],
        lc=axc.plot([], [], color=coul, lw=1.4)[0],
        pc=axc.plot([], [], "o", color=coul, ms=6, mec="white", mew=.8)[0],
        ll=axl.plot([], [], color=coul, lw=1.8, label=lab)[0],
    )
axc.plot(*P0, "s", color="white", mec=ENCRE, ms=7, zorder=5)
texte = fig.text(.06, .06, "", fontsize=13, color=GRIS)

N_PAUSE = 45                        # images figees a la fin
# Camera : ~200 degres de rotation, elevation et zoom qui ondulent,
# avec depart et arrivee adoucis
tau = np.linspace(0, 1, T + 1 + N_PAUSE)
s_cam = (1 - np.cos(np.pi * tau)) / 2
azims = 38 + 200 * s_cam
elevs = 30 + 16 * np.sin(2 * np.pi * s_cam)
zooms = 1.05 + .2 * np.sin(np.pi * s_cam)


def maj(i):
    t = min(i, T)
    for nom, traj, pertes in [("sgd", sgd, L_sgd)]:
        a = art[nom]
        u, v, z = traj[:t + 1, 0], traj[:t + 1, 1], pertes[:t + 1] + lift
        a["l3"].set_data_3d(u, v, z)
        a["p3"].set_data_3d(u[-1:], v[-1:], z[-1:])
        a["lc"].set_data(u, v)
        a["pc"].set_data(u[-1:], v[-1:])
        a["ll"].set_data(np.arange(t + 1), pertes[:t + 1])
    ax3.view_init(elev=elevs[i], azim=azims[i])
    ax3.set_box_aspect((1, 1, .55), zoom=zooms[i])
    texte.set_text(rf"$\theta_{{t+1}} = \theta_t - \eta\, \nabla_\theta \mathcal{{L}}_{{B_t}}(\theta_t)$"
                   rf"      $t = {t}$")
    return []


anim = FuncAnimation(fig, maj, frames=T + 1 + N_PAUSE, blit=False)
anim.save("assets/videos/descente_gradient.mp4", dpi=120,
          writer=FFMpegWriter(fps=30, codec="h264",
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]))
plt.close(fig)
print("OK")