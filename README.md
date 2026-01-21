# Navodila

## Kaj je v katerem folderju
- `jub-group`: FastAPI app za generiranje celotne skupine odtenkov naenkrat (API `/api/color-groups`, `/api/colorize-test`). Web UI je v `index.html`.
- `jub-group-continue`: Nadaljevalna/razširjena "group" varianta z orodjem, da lahko komentiraš in popravljaš slike, ki jih ai naredi (dodatni endpoint `/api/edit`). Web UI je v `index.html`.
- `jub-specific`: FastAPI app za generiranje enega odtenka naenkrat (API `/api/odtenki`, `/api/colorize`). Web UI je v `index.html`.
- `jub-specific-continue`: Nadaljevalna/razširjena "specific" varianta z orodjem, da lahko komentiraš in popravljaš slike, ki jih ai naredi (dodatni endpoint `/api/edit`). Web UI je v `index.html`.

## Kaj rabis v .env
Vsak app bere `GENAI_API_KEY` iz `.env` v svojem folderju.

Primer `./<folder>/.env`:
```
GENAI_API_KEY=YOUR_KEY_HERE
```

Kje dobiš ključ:
- Google AI Studio: https://aistudio.google.com/app/apikey

## Zagon (za vsak folder posebej)
1) Pojdi v izbrani folder, npr. `jub-group`.
2) Naredi virtualenv in namesti odvisnosti:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3) Nastavi `.env` z `GENAI_API_KEY`.
4) Zazeni:
```
python server.py
```

## Dostop v brskalniku
- `jub-group` in `jub-group-continue`: http://localhost:8001
- `jub-specific` in `jub-specific-continue`: http://localhost:8005