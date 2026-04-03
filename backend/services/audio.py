"""
Service de traitement des fichiers audio.

Responsabilités :
- Validation du format et de la taille des fichiers uploadés
- Extraction des métadonnées audio (durée, sample rate, canaux)
- Normalisation : conversion en WAV 16 kHz mono (format attendu par les moteurs TTS)
- Stockage sécurisé dans le dossier de samples du profil
"""

import hashlib
import uuid
from pathlib import Path

import soundfile as sf
import librosa
import numpy as np

from backend.core.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_AUDIO_DURATION_SECONDS,
    MAX_UPLOAD_SIZE_BYTES,
    UPLOAD_DIR,
)
from backend.core.logger import logger

# Sample rate cible pour la normalisation (standard TTS)
_TARGET_SAMPLE_RATE = 16_000


class AudioValidationError(Exception):
    """Levée lorsqu'un fichier audio ne satisfait pas les contraintes de validation."""
    pass


def validate_upload(filename: str, file_size: int) -> None:
    """
    Vérifie que le fichier uploadé respecte les contraintes de format et de taille.

    Args:
        filename: Nom original du fichier (utilisé uniquement pour l'extension).
        file_size: Taille du fichier en octets.

    Raises:
        AudioValidationError: Si le fichier est invalide.
    """
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise AudioValidationError(
            f"Format non supporté '{extension}'. "
            f"Formats acceptés : {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise AudioValidationError(
            f"Fichier trop volumineux ({file_size // (1024 * 1024)} Mo). "
            f"Maximum autorisé : {max_mb} Mo."
        )


def get_audio_metadata(file_path: Path) -> dict:
    """
    Extrait les métadonnées d'un fichier audio.

    Tente d'abord soundfile (rapide, WAV/FLAC/OGG).
    Fallback sur librosa pour les formats non supportés par libsndfile
    (M4A/AAC, MP3 VBR, WEBM…).

    Args:
        file_path: Chemin absolu vers le fichier audio.

    Returns:
        Dictionnaire contenant duration_s, sample_rate, channels, format.

    Raises:
        AudioValidationError: Si le fichier est illisible ou trop long.
    """
    duration_s: float
    sample_rate: int
    channels: int
    fmt: str

    try:
        # Chemin rapide : soundfile gère WAV, FLAC, OGG, AIFF…
        info = sf.info(str(file_path))
        duration_s = info.duration
        sample_rate = info.samplerate
        channels = info.channels
        fmt = info.format
    except Exception:
        # Chemin lent : librosa charge l'intégralité via audioread/ffmpeg
        # Supporte M4A, AAC, MP3, WEBM, etc.
        try:
            audio, sr = librosa.load(str(file_path), sr=None, mono=False)
            duration_s = float(audio.shape[-1]) / sr
            sample_rate = int(sr)
            channels = 1 if audio.ndim == 1 else int(audio.shape[0])
            fmt = file_path.suffix.lstrip(".").upper()
        except Exception as exc:
            raise AudioValidationError(
                f"Impossible de lire le fichier audio : {exc}"
            ) from exc

    if duration_s > MAX_AUDIO_DURATION_SECONDS:
        raise AudioValidationError(
            f"Durée audio trop longue ({duration_s:.1f}s). "
            f"Maximum autorisé : {MAX_AUDIO_DURATION_SECONDS}s."
        )

    if duration_s < 1.0:
        raise AudioValidationError(
            "Fichier audio trop court (minimum 1 seconde requise)."
        )

    return {
        "duration_s": round(duration_s, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "format": fmt,
    }


def normalize_audio(source_path: Path, profile_dir: Path) -> Path:
    """
    Convertit un fichier audio en WAV 16 kHz mono normalisé.

    La normalisation garantit la compatibilité avec tous les moteurs TTS.
    Le fichier source n'est pas modifié.

    Args:
        source_path: Chemin du fichier audio original.
        profile_dir: Dossier de destination (dossier du profil voix).

    Returns:
        Chemin du fichier WAV normalisé créé.

    Raises:
        AudioValidationError: Si la conversion échoue.
    """
    try:
        # Chargement avec rééchantillonnage automatique vers 16 kHz
        audio, _ = librosa.load(str(source_path), sr=_TARGET_SAMPLE_RATE, mono=True)

        # Normalisation du volume (peak normalization à -1 dBFS)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95

        # Nom de fichier unique pour éviter les collisions
        output_name = f"{uuid.uuid4().hex}.wav"
        output_path = profile_dir / output_name

        sf.write(str(output_path), audio, _TARGET_SAMPLE_RATE, subtype="PCM_16")

        logger.info(
            "Audio normalisé — source=%s durée=%.2fs sortie=%s",
            source_path.name,
            len(audio) / _TARGET_SAMPLE_RATE,
            output_path.name,
        )
        return output_path

    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError(f"Échec de la normalisation audio : {exc}") from exc


def save_upload(raw_bytes: bytes, original_filename: str, profile_id: int) -> Path:
    """
    Sauvegarde un fichier uploadé dans le dossier du profil, de façon sécurisée.

    Le nom de fichier original n'est jamais utilisé directement pour éviter
    les attaques par path traversal. Un nom UUID est généré.

    Args:
        raw_bytes: Contenu brut du fichier uploadé.
        original_filename: Nom original (utilisé uniquement pour l'extension).
        profile_id: Identifiant du profil voix cible.

    Returns:
        Chemin du fichier temporaire sauvegardé (avant normalisation).
    """
    extension = Path(original_filename).suffix.lower()
    profile_dir = UPLOAD_DIR / f"profile_{profile_id}"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Nom de fichier UUID — aucun composant du nom original n'est conservé
    safe_name = f"{uuid.uuid4().hex}{extension}"
    dest_path = profile_dir / safe_name

    dest_path.write_bytes(raw_bytes)

    # Log du hash pour traçabilité (sans exposer le contenu)
    file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
    logger.info(
        "Fichier audio uploadé — profil=%d fichier=%s taille=%d octets hash=%s",
        profile_id,
        safe_name,
        len(raw_bytes),
        file_hash,
    )

    return dest_path


def get_profile_samples(profile_id: int) -> list[Path]:
    """
    Retourne la liste des fichiers WAV normalisés d'un profil.

    Args:
        profile_id: Identifiant du profil voix.

    Returns:
        Liste triée des chemins de fichiers WAV du profil.
    """
    profile_dir = UPLOAD_DIR / f"profile_{profile_id}"
    if not profile_dir.exists():
        return []
    return sorted(profile_dir.glob("*.wav"))
