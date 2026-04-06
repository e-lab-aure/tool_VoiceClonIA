# VoiceClonIA

Studio de clonage vocal local avec consentement explicite obligatoire.
Synthetise de la parole a partir d'une voix de reference, entierement sur votre machine.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Quickstart

> Chemin le plus court pour avoir une synthese vocale fonctionnelle en moins de 5 minutes
> (hors telechargement du modele).

### 1 — Cloner et installer

```bat
git clone https://github.com/e-lab-aure/tool_VoiceClonIA.git
cd tool_VoiceClonIA
setup.bat
```

`setup.bat` cree le venv, installe PyTorch CUDA 12.8 et Chatterbox.

### 2 — Configurer

```bat
copy .env.example .env
```

Laisser `TTS_ENGINE=chatterbox` pour commencer (aucune dependance externe).

### 3 — Lancer

```bat
venv\Scripts\activate
uvicorn backend.main:app --reload
```

### 4 — Ouvrir l'interface

```
http://127.0.0.1:8000/ui
```

Creer un profil -> consentement -> uploader 10-30 s d'audio -> synthese.

---

## Quickstart Voxtral (qualite maximale)

> Voxtral-4B de Mistral AI : zero-shot multilingue, 24 kHz, voix preset sans sample.
> Necessite Podman Desktop sur Windows.

### 1 — Verifier que Podman Desktop est installe

```powershell
podman --version
```

Si absent : telecharger [Podman Desktop](https://podman.io).

### 2 — Builder l'image

```bat
podman build -t voxtral-server .
```

Prend 5 a 15 minutes (compilation vllm-omni).

### 3 — Obtenir un token HuggingFace

1. Creer un compte sur [huggingface.co](https://huggingface.co)
2. Accepter les CGU du modele : [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
3. Generer un token sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4 — Lancer le serveur Voxtral

```bat
podman run -d --network=host ^
  --device nvidia.com/gpu=all ^
  -e HF_TOKEN=hf_VOTRE_TOKEN ^
  -v voxtral-cache:/cache ^
  --name voxtral-server ^
  localhost/voxtral-server:latest
```

> **Important** : `--network=host` est obligatoire. Sans ce flag, le telechargement
> du modele echoue silencieusement (probleme de routage MTU vers le CDN XetHub de HuggingFace).

Premier demarrage : telechargement du modele (~8 Go), patienter 10-20 min selon la connexion.
Verifier que le serveur est pret :

```bat
curl http://127.0.0.1:8001/health
```

Retourne `{"status":"ok"}` quand Voxtral est pret.

### 5 — Activer Voxtral dans VoiceClonIA

Dans `.env` :

```env
TTS_ENGINE=voxtral
VOXTRAL_SERVER_URL=http://127.0.0.1:8001
```

Relancer le backend FastAPI.

### 6 — Tester la synthese directement

```bat
curl -s -X POST http://127.0.0.1:8001/v1/audio/speech ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"mistralai/Voxtral-4B-TTS-2603\",\"input\":\"Bonjour !\",\"voice\":\"fr_male\",\"response_format\":\"wav\"}" ^
  -o test.wav
```

---

## Fonctionnalites

- **3 modes de profil** — Clonage vocal, TTS direct (preset), Personnage de jeu
- **Clonage zero-shot** — 2 a 30 secondes d'audio de reference suffisent
- **TTS direct sans sample** — voix preset Voxtral sans aucun enregistrement
- **Personnages de jeu** — profils tagges avec export API pret a l'emploi
- **Consentement obligatoire** — aucune synthese sans consentement enregistre
- **Interface web integree** — studio complet sur `http://localhost:8000/ui`
- **Fine-tuning XTTS-v2** — entrainement par profil pour une fidelite maximale
- **API REST documentee** — `http://localhost:8000/docs`
- **100% local** — aucune donnee envoyee en dehors de votre machine

---

## Moteurs TTS

| Moteur | Qualite | Vitesse | Sample requis | Notes |
|--------|---------|---------|---------------|-------|
| **Chatterbox** | Bonne | Temps reel | Oui | Recommande pour commencer, Windows natif |
| **F5-TTS** | Tres bonne | Lente | Oui | Meilleure fidelite, Windows natif |
| **Voxtral-4B** | Excellente | Rapide | Non (preset) | Multilingue 9 langues, necessite Podman |
| **XTTS-v2** | Maximale | Lente | Oui | Active apres fine-tuning, 3-5h sur RTX 4090 |

---

## Prerequis

| Composant | Minimum | Recommande |
|-----------|---------|------------|
| OS | Windows 10 | Windows 11 |
| GPU | NVIDIA 8 Go VRAM | RTX 4090 (24 Go) |
| CUDA | 12.x | 12.8 |
| Python | 3.11 | 3.12 |
| Podman | — | Podman Desktop (Voxtral uniquement) |

---

## Installation complete

### Installation automatique (recommandee)

```bat
git clone https://github.com/e-lab-aure/tool_VoiceClonIA.git
cd tool_VoiceClonIA
setup.bat
```

### Installation manuelle

```bat
python -m venv venv
venv\Scripts\activate

rem PyTorch CUDA 12.8 EN PREMIER
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

rem Moteur principal
pip install chatterbox-tts

rem Backend
pip install -r backend/requirements.txt
```

### Installer F5-TTS (optionnel)

```bat
venv\Scripts\activate
pip install f5-tts
```

### Verifier l'installation

```bat
venv\Scripts\activate
python check_install.py
```

---

## Lancer VoiceClonIA

### Mode Chatterbox ou F5-TTS (sans Voxtral)

```bat
venv\Scripts\activate
uvicorn backend.main:app --reload
```

Interface : `http://127.0.0.1:8000/ui`

### Mode Voxtral (serveur Podman obligatoire)

Terminal 1 — Voxtral (si le container est arrete) :

```bat
podman start voxtral-server
```

Terminal 2 — Backend :

```bat
venv\Scripts\activate
set TTS_ENGINE=voxtral
uvicorn backend.main:app --reload
```

Ou configurer `TTS_ENGINE=voxtral` dans `.env` directement.

### Arreter proprement

```bat
rem Backend : Ctrl+C dans le terminal uvicorn
rem Voxtral :
podman stop voxtral-server
```

---

## Types de profils

### Clonage vocal

Le mode classique. Uploader 2 a 30 secondes d'audio de reference,
VoiceClonIA clone la voix et l'utilise pour toutes les syntheses.

- Moteur au choix : Chatterbox, F5-TTS ou Voxtral
- Avec Voxtral : fournir la transcription du sample ameliore la fidelite
- Maximum 30 secondes de reference utilisees (concatenation automatique)

**Conseil** : enregistrer dans un environnement silencieux, voix claire, sans echo.
10 secondes minimum pour un bon resultat, 20-30 secondes pour un excellent resultat.

### TTS direct (preset)

Utilise une voix integree de Voxtral sans aucun enregistrement.
Parfait pour prototyper rapidement ou creer un assistant vocal.

- Aucun sample a uploader
- 17 voix disponibles (FR, EN, ES, DE, IT, PT, NL)
- Synthese accessible des la validation du consentement

### Personnage de jeu

Identique au TTS direct ou au clonage, avec en plus :

- Champ **Tags** pour identifier le personnage (ex: "Garde, Boss, PNJ ennemi")
- **Carte Export** dans l'onglet Synthese : commande `curl` prete a copier
- Icone 🎮 dans la barre laterale pour identifier rapidement les personnages

```bash
# Exemple d'appel genere automatiquement pour le profil ID 5
curl -s -X POST http://127.0.0.1:8000/synthesis/5 \
  -H "Content-Type: application/json" \
  -d '{"text":"Halte, intrus !"}' \
  -o garde.wav
```

---

## Voix preset Voxtral

| Langue | Voix |
|--------|------|
| Francais | `fr_male`, `fr_female` |
| Anglais | `casual_male`, `casual_female`, `cheerful_female`, `neutral_male`, `neutral_female` |
| Espagnol | `es_male`, `es_female` |
| Allemand | `de_male`, `de_female` |
| Italien | `it_male`, `it_female` |
| Portugais | `pt_male`, `pt_female` |
| Neerlandais | `nl_male`, `nl_female` |

---

## Voxtral — Details techniques

### Architecture

Voxtral-4B tourne via **vLLM Omni** dans un container Podman (CUDA 12.8).
Il expose une API compatible OpenAI sur le port 8001.

```
[VoiceClonIA FastAPI :8000]
      |
      | HTTP POST /v1/audio/speech
      v
[Podman container — vLLM Omni :8001]
      |
      | GPU inference (RTX 4090)
      v
[Voxtral-4B-TTS-2603 (~8 Go VRAM)]
```

### Containerfile

Le `Containerfile` a la racine du projet definit l'image :

- Base : `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- Python 3.12 dans un venv isole
- PyTorch CUDA 12.8 + vllm-omni compile depuis les sources
- Utilisateur non-root (`vllm`, uid 1000)
- Volume `/cache` pour persister le modele entre les redemarrages

### Reseau : pourquoi `--network=host`

HuggingFace stocke les fichiers de Voxtral sur XetHub (`cas-bridge.xethub.hf.co`).
Le routage Podman par defaut (NAT bridge) bloque les transferts vers ce CDN
en raison d'un probleme de MTU. Le mode `--network=host` contourne ce probleme
en utilisant directement la pile reseau Windows.

### Commandes utiles

```bat
rem Statut du container
podman ps -a --filter name=voxtral-server

rem Logs en temps reel
podman logs -f voxtral-server

rem Taille du cache modele
podman exec voxtral-server sh -c "du -sh /cache/huggingface/hub"

rem Verifier le GPU depuis le container
podman exec voxtral-server nvidia-smi

rem Sante du serveur vLLM
curl http://127.0.0.1:8001/health

rem Supprimer le container (le cache modele est conserve dans le volume)
podman rm -f voxtral-server

rem Supprimer le volume (efface le modele telecharge)
podman volume rm voxtral-cache
```

### Rebuilder l'image

```bat
podman stop voxtral-server
podman rm voxtral-server
podman build -t voxtral-server .
podman run -d --network=host --device nvidia.com/gpu=all ^
  -e HF_TOKEN=hf_VOTRE_TOKEN ^
  -v voxtral-cache:/cache ^
  --name voxtral-server ^
  localhost/voxtral-server:latest
```

---

## Configuration

Copier `.env.example` en `.env` et ajuster :

```env
# Moteur TTS : "chatterbox" | "f5tts" | "voxtral"
TTS_ENGINE=chatterbox

# Voxtral — requis si TTS_ENGINE=voxtral
VOXTRAL_SERVER_URL=http://127.0.0.1:8001
VOXTRAL_DEFAULT_VOICE=neutral_male

# Serveur FastAPI
HOST=127.0.0.1
PORT=8000

# Limites audio
MAX_AUDIO_DURATION_SECONDS=300
MAX_UPLOAD_SIZE_MB=50

# Logging
LOG_LEVEL=INFO
```

---

## API REST

Documentation interactive : `http://127.0.0.1:8000/docs`

### Profils vocaux

| Methode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/voices/` | Creer un profil |
| `GET` | `/voices/` | Lister les profils |
| `GET` | `/voices/{id}` | Detail d'un profil |
| `POST` | `/voices/{id}/samples` | Uploader un sample audio |
| `DELETE` | `/voices/{id}` | Supprimer un profil |

#### Creer un profil (exemples)

Clonage vocal :
```bash
curl -X POST http://127.0.0.1:8000/voices/ \
  -H "Content-Type: application/json" \
  -d '{"name":"Ma voix","engine":"voxtral","category":"clone"}'
```

TTS direct (voix preset, sans sample) :
```bash
curl -X POST http://127.0.0.1:8000/voices/ \
  -H "Content-Type: application/json" \
  -d '{"name":"Narrateur FR","engine":"voxtral","category":"preset","preset_voice":"fr_male"}'
```

Personnage de jeu :
```bash
curl -X POST http://127.0.0.1:8000/voices/ \
  -H "Content-Type: application/json" \
  -d '{"name":"Garde_01","engine":"voxtral","category":"game","preset_voice":"fr_male","tags":"Garde, Enemi, Donjon"}'
```

### Consentement

| Methode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/consent/{id}` | Enregistrer le consentement |
| `GET` | `/consent/{id}` | Lire l'etat du consentement |
| `DELETE` | `/consent/{id}/revoke` | Revoquer le consentement |

```bash
curl -X POST http://127.0.0.1:8000/consent/1 \
  -H "Content-Type: application/json" \
  -d '{"consented_by":"Jean Dupont","accepted":true}'
```

### Synthese vocale

| Methode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/synthesis/{id}` | Generer un audio |
| `GET` | `/synthesis/{id}/outputs` | Lister les fichiers generes |
| `GET` | `/synthesis/{id}/outputs/{filename}` | Telecharger un fichier |

```bash
# Clonage vocal (depuis les samples uploades)
curl -X POST http://127.0.0.1:8000/synthesis/1 \
  -H "Content-Type: application/json" \
  -d '{"text":"Bonjour, je suis un clone vocal."}' \
  -o sortie.wav

# Voxtral avec transcription de reference (ameliore la fidelite)
curl -X POST http://127.0.0.1:8000/synthesis/1 \
  -H "Content-Type: application/json" \
  -d '{"text":"Texte a synthetiser.","ref_text":"Transcription exacte du sample uploade"}' \
  -o sortie.wav

# TTS direct avec voix preset
curl -X POST http://127.0.0.1:8000/synthesis/2 \
  -H "Content-Type: application/json" \
  -d '{"text":"Halte ! Qui va la ?","voice":"fr_male"}' \
  -o garde.wav
```

### Upload de sample

```bash
curl -X POST http://127.0.0.1:8000/voices/1/samples \
  -F "file=@mon_audio.wav"
```

### Parametres de synthese

| Champ | Type | Defaut | Description |
|-------|------|--------|-------------|
| `text` | string | requis | Texte a synthetiser (max 5000 car.) |
| `exaggeration` | float 0-1 | 0.5 | Style vocal, Chatterbox uniquement |
| `cfg_weight` | float 0-1 | 0.5 | Guidage CFG, Chatterbox uniquement |
| `voice` | string | null | Voix preset Voxtral (ex: `fr_male`) |
| `ref_text` | string | "" | Transcription du sample, Voxtral uniquement |

---

## Workflow complet — Integration jeu

Exemple : creer 3 personnages pour un jeu de role.

```bash
# 1. Creer les personnages
curl -X POST http://127.0.0.1:8000/voices/ -H "Content-Type: application/json" \
  -d '{"name":"Garde_01","engine":"voxtral","category":"game","preset_voice":"fr_male","tags":"Garde, Donjon"}'
# -> id: 1

curl -X POST http://127.0.0.1:8000/voices/ -H "Content-Type: application/json" \
  -d '{"name":"Marchande","engine":"voxtral","category":"game","preset_voice":"fr_female","tags":"PNJ, Village"}'
# -> id: 2

curl -X POST http://127.0.0.1:8000/voices/ -H "Content-Type: application/json" \
  -d '{"name":"Narrateur","engine":"voxtral","category":"game","preset_voice":"neutral_male","tags":"Narration"}'
# -> id: 3

# 2. Enregistrer les consentements (obligatoire)
for id in 1 2 3; do
  curl -X POST http://127.0.0.1:8000/consent/$id \
    -H "Content-Type: application/json" \
    -d '{"consented_by":"Prelude","accepted":true}'
done

# 3. Generer des lignes de dialogue
curl -X POST http://127.0.0.1:8000/synthesis/1 \
  -H "Content-Type: application/json" \
  -d '{"text":"Halte ! Presentez votre laissez-passer."}' -o garde_halte.wav

curl -X POST http://127.0.0.1:8000/synthesis/2 \
  -H "Content-Type: application/json" \
  -d '{"text":"Bienvenue dans ma boutique, voyageur."}' -o marchande_bienvenue.wav
```

Dans votre moteur de jeu, appeler l'API a la volee :

```python
import requests, soundfile as sf, io

def tts(profile_id: int, text: str) -> bytes:
    r = requests.post(
        f"http://127.0.0.1:8000/synthesis/{profile_id}",
        json={"text": text},
    )
    r.raise_for_status()
    filename = r.json()["output_filename"]
    wav = requests.get(f"http://127.0.0.1:8000/synthesis/{profile_id}/outputs/{filename}")
    return wav.content

audio = tts(1, "Qui ose troubler la paix de ce donjon ?")
```

---

## Structure du projet

```
tool_VoiceClonIA/
├── backend/
│   ├── main.py              # FastAPI — lifespan, routers, endpoint /ui
│   ├── api/
│   │   ├── consent.py       # POST/GET/DELETE /consent/{id}
│   │   ├── voices.py        # CRUD /voices/ + upload samples + serving WAV
│   │   ├── synthesis.py     # POST /synthesis/{id} + download outputs
│   │   └── finetune.py      # Fine-tuning XTTS-v2 (optionnel)
│   ├── core/
│   │   ├── config.py        # Variables d'environnement (.env)
│   │   ├── database.py      # SQLAlchemy + SQLite WAL + migration colonnes
│   │   └── logger.py        # Logging centralise
│   ├── models/
│   │   ├── consent.py       # ORM Consent
│   │   └── voice_profile.py # ORM VoiceProfile (category, preset_voice, tags...)
│   ├── services/
│   │   ├── audio.py         # Validation, normalisation WAV 16 kHz mono
│   │   └── tts.py           # Moteurs TTS (Chatterbox, F5-TTS, Voxtral, XTTS-v2)
│   └── requirements.txt
├── frontend/
│   └── index.html           # SPA vanilla JS/CSS — studio complet
├── Containerfile            # Image Podman pour Voxtral (CUDA 12.8 + vllm-omni)
├── .env.example             # Template de configuration
├── setup.bat                # Installation automatique Windows
└── check_install.py         # Verification de l'environnement
```

### Dossiers crees au premier lancement (exclus du depot)

```
uploads/     # Samples audio normalises, un sous-dossier par profil
outputs/     # Fichiers WAV generes, un sous-dossier par profil
data/        # Base SQLite + fichier de log
models/      # Checkpoints XTTS-v2 fine-tunes
```

---

## Securite

- Le serveur ecoute sur `127.0.0.1` uniquement — inaccessible depuis le reseau
- Le consentement est obligatoire et verifie a chaque synthese
- Les samples sont valides (format, taille, duree) avant tout traitement
- Les noms de fichiers sont strictement valides (protection contre le path traversal)
- Aucune donnee ne quitte la machine (sauf le telechargement initial du modele Voxtral)
- Le token HuggingFace ne doit jamais etre committe — utiliser uniquement `-e HF_TOKEN=...`

---

## Licence

MIT — usage personnel uniquement.
Ne jamais utiliser pour usurper l'identite d'une personne sans son consentement explicite.
