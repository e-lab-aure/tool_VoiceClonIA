# VoiceClonIA

Studio de clonage vocal local avec consentement explicite obligatoire.
Genere de la parole synthetisee a partir d'une voix de reference, entierement en local.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Fonctionnalites

- **Clonage vocal zero-shot** — quelques secondes d'audio de reference suffisent
- **Consentement obligatoire** — aucune synthese sans consentement explicite enregistre
- **Interface web integree** — studio complet accessible sur `http://localhost:8000/ui`
- **Plusieurs moteurs TTS** — Chatterbox, F5-TTS, Voxtral-4B (Mistral AI)
- **Fine-tuning XTTS-v2** — entrainement par profil pour une fidelite maximale
- **API REST documentee** — `http://localhost:8000/docs`
- **100% local** — aucune donnee envoyee en dehors de votre machine

---

## Moteurs TTS disponibles

| Moteur | Qualite | Vitesse | Clonage | Notes |
|--------|---------|---------|---------|-------|
| **Chatterbox** | Bonne | Temps reel | Zero-shot | Recommande, fonctionne sur Windows |
| **F5-TTS** | Tres bonne | Lente | Zero-shot | Fonctionne sur Windows |
| **Voxtral-4B** | Excellente | Rapide | Zero-shot | Multilingue, necessite WSL2 |
| **XTTS-v2** | Maximale | Lente | Fine-tune | Active apres entrainement |

---

## Prerequis

- **OS** : Windows 10/11
- **GPU** : NVIDIA avec CUDA 12.8 (teste sur RTX 4090)
- **Python** : 3.11 ou 3.12
- **RAM VRAM** : 8 Go minimum (24 Go recommandes pour Voxtral)

---

## Installation

### 1. Cloner le depot

```bash
git clone https://github.com/e-lab-aure/tool_VoiceClonIA.git
cd tool_VoiceClonIA
```

### 2. Installation automatique (recommandee)

Lancer le script d'installation depuis la racine du projet :

```bat
setup.bat
```

Ce script :
- Cree un environnement virtuel Python (`venv/`)
- Installe PyTorch 2.x avec support CUDA 12.8
- Installe Chatterbox TTS et toutes ses dependances
- Installe les dependances backend
- Cree le fichier `.env` depuis `.env.example`

### 3. Installation manuelle (optionnelle)

```bash
python -m venv venv
venv\Scripts\activate

# PyTorch CUDA 12.8 EN PREMIER
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Moteur principal
pip install chatterbox-tts

# Backend
pip install -r backend/requirements.txt
```

---

## Lancer VoiceClonIA

```bat
venv\Scripts\activate
uvicorn backend.main:app --reload
```

Le serveur demarre sur `http://127.0.0.1:8000`.

---

## Interface web

Ouvrez votre navigateur sur :

```
http://127.0.0.1:8000/ui
```

### Workflow en 3 etapes

```
1. Consentement  →  2. Samples  →  3. Synthese
```

**Etape 1 — Consentement**
- Cliquez sur **Nouveau profil** dans la barre laterale
- Renseignez un nom et choisissez le moteur TTS
- Lisez et validez le texte de consentement avec votre identifiant

**Etape 2 — Samples**
- Glissez-deposez vos fichiers audio de reference (WAV, MP3, FLAC, OGG, M4A)
- 10 a 30 secondes d'audio clair et sans bruit de fond
- Les fichiers sont automatiquement normalises en WAV 16 kHz mono

**Etape 3 — Synthese**
- Entrez le texte a synthetiser (5000 caracteres max)
- Cliquez sur **Generer la synthese vocale**
- Ecoutez ou telechargez le fichier WAV genere

---

## Configuration

Copiez `.env.example` en `.env` et ajustez les valeurs :

```env
# Moteur TTS actif : "chatterbox" | "f5tts" | "voxtral"
TTS_ENGINE=chatterbox

# Hote et port du serveur
HOST=127.0.0.1
PORT=8000

# Voxtral (vLLM Omni) — voir section Voxtral ci-dessous
VOXTRAL_SERVER_URL=http://127.0.0.1:8001
VOXTRAL_DEFAULT_VOICE=neutral_male
```

---

## Moteur Voxtral-4B (Mistral AI)

Voxtral est un modele TTS 4B parametres de Mistral AI offrant un clonage vocal zero-shot
multilingue de haute qualite (EN, FR, ES, DE, IT, PT, NL...).

**vLLM ne supporte pas Windows nativement — le serveur doit tourner dans WSL2.**

### Installation WSL2

```powershell
# PowerShell (administrateur) — si WSL2 n'est pas installe
wsl --install -d Ubuntu-22.04
```

### Installation vLLM dans WSL2

```bash
# Dans le terminal WSL2
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

pip install vllm vllm-omni --upgrade
```

### Lancer le serveur Voxtral

```bash
# Terminal WSL2 (a lancer AVANT VoiceClonIA)
source ~/vllm-env/bin/activate

vllm serve mistralai/Voxtral-4B-TTS-2603 \
  --omni \
  --stage-configs-path "$(python -c 'import vllm_omni,os; print(os.path.dirname(vllm_omni.__file__))')/model_executor/stage_configs/voxtral_tts.yaml" \
  --port 8001 \
  --trust-remote-code \
  --enforce-eager
```

> `--stage-configs-path` pointe vers le fichier YAML de pipeline TTS installe avec vllm-omni.
> Le `$(python -c ...)` resout automatiquement le chemin — pas besoin de le modifier.

> Le port 8001 est automatiquement accessible depuis Windows via WSL2.

### Token HuggingFace requis

Voxtral est un modele avec acces restreint. Avant de le telecharger :

1. Acceptez les conditions sur [huggingface.co/mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
2. Connectez-vous dans WSL2 :

```bash
pip install huggingface_hub
huggingface-cli login
```

### Activer Voxtral dans VoiceClonIA

Dans `.env` :
```env
TTS_ENGINE=voxtral
VOXTRAL_SERVER_URL=http://localhost:8001
```

Relancez le serveur FastAPI. Lors de la creation d'un profil, selectionnez **Voxtral-4B**.

### Voix preset disponibles

Si aucun sample n'est fourni, Voxtral utilise une voix preset :

| Langue | Voix disponibles |
|--------|-----------------|
| Anglais | `casual_male`, `casual_female`, `cheerful_female`, `neutral_male`, `neutral_female` |
| Francais | `fr_male`, `fr_female` |
| Espagnol | `es_male`, `es_female` |
| Allemand | `de_male`, `de_female` |
| Italien | `it_male`, `it_female` |
| Portugais | `pt_male`, `pt_female` |
| Neerlandais | `nl_male`, `nl_female` |

---

## API REST

Documentation interactive disponible sur `http://127.0.0.1:8000/docs`

| Methode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/voices/` | Creer un profil vocal |
| `GET` | `/voices/` | Lister les profils |
| `POST` | `/voices/{id}/samples` | Uploader un sample audio |
| `POST` | `/consent/{id}` | Enregistrer un consentement |
| `DELETE` | `/consent/{id}/revoke` | Revoquer le consentement |
| `POST` | `/synthesis/{id}` | Generer une synthese vocale |
| `GET` | `/synthesis/{id}/outputs` | Lister les fichiers generes |
| `GET` | `/synthesis/{id}/outputs/{filename}` | Telecharger un fichier |

### Exemple de synthese

```bash
curl -X POST http://127.0.0.1:8000/synthesis/1 \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, ceci est un test de clonage vocal."}'
```

Avec Voxtral et transcription de reference :

```bash
curl -X POST http://127.0.0.1:8000/synthesis/1 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, ceci est un test de clonage vocal.",
    "ref_text": "Texte prononce dans le sample audio uploade"
  }'
```

---

## Structure du projet

```
tool_VoiceClonIA/
├── backend/
│   ├── main.py              # Application FastAPI
│   ├── api/
│   │   ├── consent.py       # Gestion des consentements
│   │   ├── voices.py        # Profils vocaux + serving samples
│   │   ├── synthesis.py     # Synthese vocale
│   │   └── finetune.py      # Fine-tuning XTTS-v2
│   ├── core/
│   │   ├── config.py        # Configuration (.env)
│   │   ├── database.py      # SQLAlchemy + SQLite
│   │   └── logger.py        # Logging centralise
│   ├── models/
│   │   ├── consent.py       # ORM Consent
│   │   └── voice_profile.py # ORM VoiceProfile
│   ├── services/
│   │   ├── audio.py         # Validation, normalisation audio
│   │   └── tts.py           # Moteurs TTS (Chatterbox, F5-TTS, Voxtral, XTTS-v2)
│   └── requirements.txt
├── frontend/
│   └── index.html           # Interface web (SPA vanilla JS)
├── .env.example             # Template de configuration
├── setup.bat                # Script d'installation Windows
└── check_install.py         # Verification de l'installation
```

---

## Verifier l'installation

```bash
venv\Scripts\activate
python check_install.py
```

---

## Licence

MIT — usage personnel uniquement pour le clonage vocal.
Ne jamais utiliser pour usurper l'identite d'une personne sans son consentement explicite.
