"""
Configuration centrale de l'application.

Charge les paramètres depuis les variables d'environnement (.env).
Toutes les valeurs sensibles doivent être définies dans .env, jamais en dur.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Chargement du fichier .env situé à la racine du projet
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")


def _get_env(key: str, default: str) -> str:
    """Retourne la valeur d'une variable d'environnement ou sa valeur par défaut."""
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Retourne la valeur entière d'une variable d'environnement."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


# --- Moteur TTS ---
TTS_ENGINE: str = _get_env("TTS_ENGINE", "chatterbox")

# --- Voxtral (vLLM Omni) ---
VOXTRAL_SERVER_URL: str = _get_env("VOXTRAL_SERVER_URL", "http://127.0.0.1:8001")
VOXTRAL_DEFAULT_VOICE: str = _get_env("VOXTRAL_DEFAULT_VOICE", "neutral_male")

# --- Chemins ---
UPLOAD_DIR: Path = _ROOT / _get_env("UPLOAD_DIR", "uploads")
OUTPUT_DIR: Path = _ROOT / _get_env("OUTPUT_DIR", "outputs")
MODELS_DIR: Path = _ROOT / _get_env("MODELS_DIR", "models")
DATABASE_PATH: Path = _ROOT / _get_env("DATABASE_PATH", "data/voicelconia.db")
LOG_FILE: Path = _ROOT / _get_env("LOG_FILE", "data/voicelconia.log")

# --- Logging ---
LOG_LEVEL: str = _get_env("LOG_LEVEL", "INFO")

# --- Contraintes audio ---
MAX_AUDIO_DURATION_SECONDS: int = _get_env_int("MAX_AUDIO_DURATION_SECONDS", 300)
MAX_UPLOAD_SIZE_MB: int = _get_env_int("MAX_UPLOAD_SIZE_MB", 50)
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# --- Serveur ---
HOST: str = _get_env("HOST", "127.0.0.1")
PORT: int = _get_env_int("PORT", 8000)

# --- Formats audio acceptés ---
ALLOWED_AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
ALLOWED_AUDIO_MIME_TYPES: set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/webm",
}

# Création automatique des dossiers nécessaires au démarrage
for _directory in (UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR, DATABASE_PATH.parent):
    _directory.mkdir(parents=True, exist_ok=True)
