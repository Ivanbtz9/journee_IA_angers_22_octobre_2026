"""
Illustration : comment un modèle de langue autorégressif traduit une phrase.

À chaque étape, le modèle fait UNE classification multiclasse :
    « quel est le prochain token parmi tout le vocabulaire ? »
Le token choisi est ajouté au contexte, puis on recommence.
Le script produit une vidéo MP4 de ce processus.

Installation :
    pip install torch transformers matplotlib
    + ffmpeg (Linux : sudo apt install ffmpeg ; macOS : brew install ffmpeg)

Exemples :
    python llm_autoregressif_traduction.py
    python llm_autoregressif_traduction.py --phrase "Il pleut souvent à Angers."
    python llm_autoregressif_traduction.py --temperature 0.8   # tirage aléatoire (comme un chatbot)
"""
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

# Petit modèle "chatbot" (décodeur seul, ~0,5 milliard de paramètres, ~1 Go), tourne sur CPU
MODELE = "Qwen/Qwen2.5-0.5B-Instruct"
PHRASE = "Le chat dort sur le canapé."

# Couleurs de la vidéo
COULEUR_CHOISI = "#1c5cab"   # token retenu
COULEUR_AUTRES = "#9ec5f4"   # autres candidats
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
FOND = "#fcfcfb"

DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))


def lisible(tokenizer, token_id):
    """Texte d'un token, avec les espaces et retours à la ligne rendus visibles."""
    texte = tokenizer.decode([token_id])
    return texte.replace(" ", "␣").replace("\n", "↵") or "∅"


# ----------------------------------------------------------------------
# 1) Génération autorégressive, token par token
# ----------------------------------------------------------------------
def generer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.graine)

    print(f"Chargement de {args.modele} …")
    tokenizer = AutoTokenizer.from_pretrained(args.modele)
    model = AutoModelForCausalLM.from_pretrained(args.modele)
    model.eval()

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres du modèle : {nb_params / 1e9:.2f} milliard(s)")

    # La consigne, mise au format "conversation" du chatbot
    messages = [{"role": "user",
                 "content": f"Traduis en anglais. Réponds uniquement par la traduction.\n{args.phrase}"}]
    entree = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
    ids = entree["input_ids"]
    print(f"Consigne découpée en {ids.shape[1]} tokens")

    # Tokens qui signifient "fin de la réponse"
    fin = model.generation_config.eos_token_id
    fin = set(fin if isinstance(fin, list) else [fin])
    if tokenizer.eos_token_id is not None:
        fin.add(tokenizer.eos_token_id)

    print(f"\nPhrase source : {args.phrase}\n")
    genere, etapes = [], []
    for etape in range(1, args.max_etapes + 1):
        # Le modèle donne un score à CHAQUE token du vocabulaire (= une classe)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1]
        taille_vocab = logits.shape[-1]

        # Softmax : scores -> probabilités (classification multiclasse)
        probas = torch.softmax(logits.float(), dim=-1)
        top_p, top_i = probas.topk(args.top_k)

        # Choix du prochain token
        if args.temperature > 0:
            probas_t = torch.softmax(logits.float() / args.temperature, dim=-1)
            suivant = torch.multinomial(probas_t, 1)[0]
        else:
            suivant = top_i[0]  # le plus probable (décodage glouton)

        est_fin = suivant.item() in fin
        nom = lambda i: "⟨fin⟩" if i in fin else lisible(tokenizer, i)
        choisi = nom(suivant.item())
        candidats = [(nom(i.item()), p.item()) for i, p in zip(top_i, top_p)]
        if choisi not in [c[0] for c in candidats]:  # possible en mode tirage aléatoire
            candidats[-1] = (choisi, probas[suivant].item())

        contexte = tokenizer.decode(genere)
        print(f"Étape {etape:2d} | contexte : « {contexte} »")
        for n, p in candidats:
            marque = "  <== choisi" if n == choisi else ""
            print(f"           {n:>15s}  {p:6.1%}{marque}")

        etapes.append({"contexte": contexte, "candidats": candidats, "choisi": choisi,
                       "ajout": "" if est_fin else tokenizer.decode(genere + [suivant.item()])[len(contexte):],
                       "fin": est_fin})

        # Fin de phrase ? sinon on ajoute le token au contexte et on recommence
        if est_fin:
            print("\nToken de fin atteint : arrêt.")
            break
        genere.append(suivant.item())
        ids = torch.cat([ids, suivant.view(1, 1)], dim=1)

    traduction = tokenizer.decode(genere)
    print(f"\nTraduction : {traduction}")
    print(f"{len(genere)} tokens générés = {len(genere)} classifications parmi {taille_vocab:,} classes")
    return etapes, traduction, taille_vocab


# ----------------------------------------------------------------------
# 2) Vidéo du processus
# ----------------------------------------------------------------------
def video(etapes, phrase, traduction, taille_vocab, chemin, fps=25, duree_etape=1.8, aleatoire=False):
    """Pour chaque étape : les probabilités montent, le token est choisi, puis ajouté à la traduction."""
    n = max(3, round(fps * duree_etape))
    n_pousse, n_choix = round(0.45 * n), round(0.25 * n)
    images = []
    for k in range(len(etapes)):
        images += [(k, "pousse", i / (n_pousse - 1)) for i in range(n_pousse)]
        images += [(k, "choix", 1.0)] * n_choix
        images += [(k, "ajout", i / (n - n_pousse - n_choix - 1)) for i in range(n - n_pousse - n_choix)]
    images += [(len(etapes) - 1, "final", 1.0)] * (3 * fps)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150)  # 1920 x 1080
    fig.patch.set_facecolor(FOND)
    ax_txt = fig.add_axes([0.05, 0.63, 0.9, 0.33])
    ax = fig.add_axes([0.2, 0.08, 0.68, 0.42])
    vocab = f"{taille_vocab:,}".replace(",", " ")
    rendu = fig.canvas.get_renderer()

    def ecrire_traduction(contexte, ajout, alpha, curseur):
        """Contexte en noir, nouveau token en bleu juste derrière."""
        y = 0.38
        ax_txt.text(0, y, "Anglais :", fontsize=24, color=ENCRE_2, va="center")
        t = ax_txt.text(0.16, y, contexte, fontsize=30, color=ENCRE, va="center")
        bb = t.get_window_extent(renderer=rendu).transformed(ax_txt.transAxes.inverted())
        x = bb.x1 if contexte else 0.16
        if ajout:
            t2 = ax_txt.text(x, y, ajout, fontsize=30, color=COULEUR_CHOISI, va="center",
                             fontweight="bold", alpha=alpha)
            x = t2.get_window_extent(renderer=rendu).transformed(ax_txt.transAxes.inverted()).x1
        if curseur:
            ax_txt.text(x + 0.005, y, "▌", fontsize=26, color=COULEUR_CHOISI, va="center")

    def dessiner(i):
        k, phase, prog = images[i]
        e = etapes[k]
        ax_txt.clear(); ax.clear()
        ax_txt.set_axis_off()
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)

        ax_txt.text(0, 0.82, "Français :", fontsize=24, color=ENCRE_2, va="center")
        ax_txt.text(0.16, 0.82, phrase, fontsize=30, color=ENCRE, va="center")

        if phase == "final":
            ecrire_traduction(traduction, "", 1, False)
            ax.set_axis_off()
            nb = len(etapes)  # chaque étape (y compris le choix de ⟨fin⟩) est une classification
            ax.text(0.5, 0.55, f"{nb} étapes = {nb} classifications successives",
                    ha="center", fontsize=24, color=ENCRE, transform=ax.transAxes)
            ax.text(0.5, 0.35, f"chacune parmi {vocab} classes (le vocabulaire)",
                    ha="center", fontsize=20, color=ENCRE_2, transform=ax.transAxes)
            return

        ajout = e["ajout"] if phase == "ajout" else ""
        ecrire_traduction(e["contexte"], ajout, min(1, 2 * prog), phase != "ajout" or prog < 1)
        if phase == "ajout" and e["fin"]:
            ax_txt.text(0.16, 0.08, "Token ⟨fin⟩ choisi : la génération s'arrête.",
                        fontsize=18, color=COULEUR_CHOISI, va="center")

        # Barres : probabilité du prochain token
        noms = [c[0] for c in e["candidats"]][::-1]
        probas = [c[1] for c in e["candidats"]][::-1]
        lissage = 1 - (1 - prog) ** 3  # montée amortie
        hauteurs = [p * (lissage if phase == "pousse" else 1) for p in probas]
        couleurs = [COULEUR_CHOISI if (phase != "pousse" and nm == e["choisi"]) else COULEUR_AUTRES
                    for nm in noms]
        barres = ax.barh(noms, hauteurs, color=couleurs, height=0.62)
        for barre, h, nm in zip(barres, hauteurs, noms):
            texte = f"{h:.1%}"
            if phase != "pousse" and nm == e["choisi"]:
                texte += "   ← choisi" + (" (tirage)" if aleatoire else "")
            ax.text(h + 0.01, barre.get_y() + barre.get_height() / 2, texte,
                    va="center", fontsize=16, color=ENCRE if nm == e["choisi"] and phase != "pousse" else ENCRE_2)

        ax.set_facecolor(FOND)
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}" if x <= 1 else ""))
        ax.tick_params(axis="y", labelsize=20, colors=ENCRE, length=0)
        ax.tick_params(axis="x", labelsize=13, colors=ENCRE_2)
        for cote in ("top", "right", "left"):
            ax.spines[cote].set_visible(False)
        ax.spines["bottom"].set_color("#d0cfca")
        ax.grid(axis="x", color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"Étape {k + 1} : probabilité du prochain token  (top {len(noms)} sur {vocab} classes)",
                     loc="left", fontsize=17, color=ENCRE_2, pad=14)

    anim = animation.FuncAnimation(fig, dessiner, frames=len(images))
    if animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, codec="h264", bitrate=4000,
                                        extra_args=["-pix_fmt", "yuv420p"])  # lisible par les navigateurs
    else:
        print("ffmpeg introuvable : la vidéo sera enregistrée en GIF.")
        chemin = chemin.rsplit(".", 1)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
    anim.save(chemin, writer=writer, savefig_kwargs={"facecolor": FOND})
    plt.close(fig)
    print(f"Vidéo enregistrée : {chemin}  ({len(images) / fps:.0f} s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase", default=PHRASE)
    parser.add_argument("--modele", default=MODELE)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-etapes", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = on prend toujours le plus probable ; > 0 = tirage aléatoire")
    parser.add_argument("--sortie",
                    default=os.path.join(DOSSIER_SCRIPT, "..", "assets", "videos", "autoregressif.mp4"))
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--duree-etape", type=float, default=1.8, help="secondes par token")
    parser.add_argument("--graine", type=int, default=0)
    args = parser.parse_args()

    etapes, traduction, taille_vocab = generer(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.sortie)), exist_ok=True)
    video(etapes, args.phrase, traduction, taille_vocab, args.sortie,
          fps=args.fps, duree_etape=args.duree_etape, aleatoire=args.temperature > 0)


if __name__ == "__main__":
    main()