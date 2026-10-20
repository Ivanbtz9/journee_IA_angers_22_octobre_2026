# requin.py : un modèle de langue de 117 milliards de paramètres dessine pour nous
import os, time
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"   # terminal propre pour la démo
from vllm import LLM, SamplingParams

MODELE = f"/scratch/nautilus/users/{os.environ['USER']}/models/gpt-oss-120b"
DESSIN = open("logo_nautilus.txt").read()          # le logo ASCII affiché à la connexion
QUESTION = (f"Voici le logo ASCII du supercalculateur Nautilus :\n\n{DESSIN}\n"
            "Dessine en ASCII art un requin qui dévore ce logo, en gardant le logo reconnaissable. "
            "Réponds uniquement avec le dessin, sans bloc de code.")

print("Chargement de gpt-oss-120b sur le GPU ...")
t0 = time.time()
llm = LLM(model=MODELE, max_model_len=8192, enforce_eager=True)
print(f"Modèle chargé en {time.time() - t0:.0f} s\n")

params = SamplingParams(temperature=0.8, max_tokens=4096)
reponse = llm.chat([{"role": "user", "content": QUESTION}], params, use_tqdm=False,
                   chat_template_kwargs={"reasoning_effort": "low"})
texte = reponse[0].outputs[0].text
print(texte.split("assistantfinal")[-1].strip())  
