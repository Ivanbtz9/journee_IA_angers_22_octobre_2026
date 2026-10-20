#logo_aouda.py : un modèle de langue de 117 milliards de paramètres 
import os, time
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"   # terminal propre pour la démo
from vllm import LLM, SamplingParams

MODELE = f"/scratch/nautilus/users/{os.environ['USER']}/models/gpt-oss-120b"
DESSIN = open("logo_nautilus.txt").read()          # le logo ASCII affiché à la connexion

QUESTION = (f"Voici le logo ASCII du supercalculateur Nautilus :\n\n{DESSIN}\n"
            "Le mésocentre GLiCID ouvre un nouveau cluster, Aouda, riche en GPU H200. "
            "Comme Nautilus son nom vient de Jules Verne (Aouda dans Le Tour du monde en 80 jours). "
            "Dessine son logo en ASCII art dans le même style : le mot Aouda dans la même police en haut, "
            "une illustration en dessous, puis la ligne « Welcome to GLiCID HPC cluster Aouda ». "
            "80 colonnes et 25 lignes maximum. "
            "Réponds uniquement avec le dessin, sans bloc de code.")

print("Chargement de gpt-oss-120b sur le GPU ...")
t0 = time.time()
llm = LLM(model=MODELE, max_model_len=16384, enforce_eager=True)
print(f"Modèle chargé en {time.time() - t0:.0f} s\n")

params = SamplingParams(temperature=0.8, max_tokens=12000)
reponse = llm.chat([{"role": "user", "content": QUESTION}], params, use_tqdm=False,
                   chat_template_kwargs={"reasoning_effort": "high"})

for sortie in reponse[0].outputs:
    print(sortie.text.split("assistantfinal")[-1].strip(), "\n" + "=" * 80)
