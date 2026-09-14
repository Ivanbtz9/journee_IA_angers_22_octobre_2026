# Journée IA Angers — 22 octobre 2026

Présentation web interactive construite avec [Reveal.js](https://revealjs.com) 5.1.0.

## Installation

Les dépendances ne sont pas versionnées (voir `.gitignore`) :

```bash
npm install reveal.js
npm install katex
```

Cela installe `reveal.js@5.1.0` et `katex` dans `node_modules/`.

En cas de dossier `node_modules/reveal.js/plugin/` manquant, forcer la version :

```bash
npm install reveal.js@5.1.0
```

## Lancer la présentation

Un serveur HTTP local est nécessaire (la vue orateur et le chargement de
ressources externes ne fonctionnent pas en `file://`) :

```bash
python3 -m http.server 8000
```

Puis ouvrir <http://localhost:8000>.

## Raccourcis pendant la présentation

| Touche     | Effet                                    |
|------------|------------------------------------------|
| `→` / `←`  | slide suivante / précédente              |
| `↓` / `↑`  | navigation dans une sous-pile verticale  |
| `S`        | vue orateur (notes + chronomètre)        |
| `O`        | vue d'ensemble des slides                |
| `F`        | plein écran                              |
| `B`        | écran noir                               |
| `Alt+clic` | zoom sur une zone                        |
| `?`        | liste complète des raccourcis            |

## Export PDF

Ouvrir <http://localhost:8000/?print-pdf>, puis imprimer vers un PDF
(marges nulles, graphiques d'arrière-plan activés).

En ligne de commande :

```bash
npx decktape reveal http://localhost:8000 slides.pdf
```

## Structure

```
index.html      la présentation (contenu + configuration Reveal)
figures/        figures SVG/PNG appelées par les slides
node_modules/   reveal.js et katex (non versionné)
```

## Notes

- Les équations sont rendues par KaTeX, servi depuis `node_modules/katex`
  via l'option `katex: { local: ... }` du plugin math.
- Node n'est utilisé que pour récupérer les fichiers statiques ; la version
  de Node installée n'a aucune incidence sur le rendu.