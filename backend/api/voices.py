"""
Routes API — Gestion des profils voix.

Endpoints :
    POST   /voices/                        Créer un profil voix
    GET    /voices/                        Lister tous les profils
    GET    /voices/{id}                    Détail d'un profil
    POST   /voices/{id}/samples            Uploader un fichier audio de référence
    GET    /voices/{id}/samples            Lister les samples d'un profil
    DELETE /voices/{id}/samples/{filename} Supprimer un sample
    DELETE /voices/{id}                    Supprimer un profil et ses données
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.config import OUTPUT_DIR, UPLOAD_DIR
from backend.core.database import get_db
from backend.core.logger import logger
from backend.core.utils import get_profile_or_404
from backend.models.voice_profile import ProfileStatus, VoiceProfile
from backend.services.audio import (
    AudioValidationError,
    get_audio_metadata,
    get_profile_samples,
    normalize_audio,
    save_upload,
    validate_upload,
)

router = APIRouter(prefix="/voices", tags=["Profils voix"])


# --- Schémas Pydantic ---

class VoiceProfileCreate(BaseModel):
    """Données requises pour créer un profil voix."""

    name: str = Field(..., min_length=1, max_length=255, description="Nom unique du profil")
    description: str | None = Field(None, max_length=1000, description="Description libre")
    engine: str = Field("chatterbox", description="Moteur TTS : chatterbox | f5tts | voxtral")
    category: str | None = Field(None, description="Categorie : clone | preset | game")
    preset_voice: str | None = Field(None, max_length=64, description="Voix preset Voxtral par defaut")
    tags: str | None = Field(None, max_length=255, description="Tags libres")


class VoiceProfileResponse(BaseModel):
    """Représentation publique d'un profil voix."""

    id: int
    name: str
    description: str | None
    status: ProfileStatus
    sample_count: int
    total_duration_s: float
    engine: str
    # Statut du fine-tuning XTTS-v2 : None | pending | running | done | error | cancelled
    fine_tune_status: str | None
    category: str | None
    preset_voice: str | None
    tags: str | None

    model_config = {"from_attributes": True}


class SampleInfo(BaseModel):
    """Métadonnées d'un sample audio."""

    filename: str
    size_bytes: int
    duration_s: float


class SampleUploadResponse(BaseModel):
    """Réponse après upload d'un sample audio."""

    filename: str
    duration_s: float
    sample_rate: int
    channels: int
    profile_sample_count: int
    profile_total_duration_s: float


# --- Endpoints ---

@router.post(
    "/",
    response_model=VoiceProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Créer un profil voix",
)
def create_voice_profile(
    payload: VoiceProfileCreate,
    db: Session = Depends(get_db),
) -> VoiceProfileResponse:
    """Crée un nouveau profil voix avec le statut READY."""
    # Vérifie l'unicité du nom
    existing = db.query(VoiceProfile).filter(VoiceProfile.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Un profil nommé '{payload.name}' existe déjà.",
        )

    # Valide le moteur demandé
    # xttsv2 est sélectionné automatiquement après fine-tuning — pas à la création
    allowed_engines = {"chatterbox", "f5tts", "voxtral"}
    if payload.engine not in allowed_engines:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Moteur inconnu '{payload.engine}'. Valeurs acceptées : {allowed_engines}",
        )

    # Valide la categorie si fournie
    allowed_categories = {None, "clone", "preset", "game"}
    if payload.category not in allowed_categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Categorie inconnue '{payload.category}'. Valeurs acceptees : clone, preset, game",
        )

    profile = VoiceProfile(
        name=payload.name,
        description=payload.description,
        engine=payload.engine,
        status=ProfileStatus.READY,
        category=payload.category,
        preset_voice=payload.preset_voice,
        tags=payload.tags,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)

    logger.info("Profil voix créé — id=%d name='%s'", profile.id, profile.name)
    return VoiceProfileResponse.model_validate(profile)


@router.get(
    "/",
    response_model=list[VoiceProfileResponse],
    summary="Lister les profils voix",
)
def list_voice_profiles(db: Session = Depends(get_db)) -> list[VoiceProfileResponse]:
    """Retourne la liste de tous les profils voix (actifs et révoqués)."""
    profiles = db.query(VoiceProfile).order_by(VoiceProfile.created_at.desc()).all()
    return [VoiceProfileResponse.model_validate(p) for p in profiles]


@router.get(
    "/{profile_id}",
    response_model=VoiceProfileResponse,
    summary="Détail d'un profil voix",
)
def get_voice_profile(
    profile_id: int,
    db: Session = Depends(get_db),
) -> VoiceProfileResponse:
    """Retourne les détails d'un profil voix par son identifiant."""
    profile = get_profile_or_404(profile_id, db)
    return VoiceProfileResponse.model_validate(profile)


@router.post(
    "/{profile_id}/samples",
    response_model=SampleUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Uploader un sample audio",
)
async def upload_sample(
    profile_id: int,
    file: UploadFile,
    request: Request,
    db: Session = Depends(get_db),
) -> SampleUploadResponse:
    """
    Uploade un fichier audio de référence pour un profil voix.

    Le fichier est validé, normalisé en WAV 16 kHz mono,
    puis associé au profil. Le profil doit exister et ne pas être révoqué.
    """
    profile = get_profile_or_404(profile_id, db)

    raw_bytes = await file.read()
    filename = file.filename or "upload.wav"

    # Validation format/taille
    try:
        validate_upload(filename, len(raw_bytes))
    except AudioValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Sauvegarde sécurisée du fichier brut
    try:
        raw_path = save_upload(raw_bytes, filename, profile_id)
    except Exception as exc:
        logger.error("Échec sauvegarde upload — profil=%d : %s", profile_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Échec de la sauvegarde.")

    # Extraction des métadonnées avant normalisation
    try:
        metadata = get_audio_metadata(raw_path)
    except AudioValidationError as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Normalisation en WAV 16 kHz mono
    try:
        profile_dir = raw_path.parent
        normalized_path = normalize_audio(raw_path, profile_dir)
    except AudioValidationError as exc:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    finally:
        # Suppression du fichier brut — seul le WAV normalisé est conservé
        if raw_path.exists() and raw_path != normalized_path:
            raw_path.unlink(missing_ok=True)

    # Mise à jour des métadonnées du profil
    profile.sample_count += 1
    profile.total_duration_s = round(profile.total_duration_s + metadata["duration_s"], 2)
    profile.sample_dir = f"profile_{profile_id}"

    db.commit()

    logger.info(
        "Sample ajouté — profil=%d fichier=%s durée=%.2fs total=%d samples",
        profile_id,
        normalized_path.name,
        metadata["duration_s"],
        profile.sample_count,
    )

    return SampleUploadResponse(
        filename=normalized_path.name,
        duration_s=metadata["duration_s"],
        sample_rate=metadata["sample_rate"],
        channels=metadata["channels"],
        profile_sample_count=profile.sample_count,
        profile_total_duration_s=profile.total_duration_s,
    )


@router.get(
    "/{profile_id}/samples/{filename}",
    summary="Télécharger un sample audio de référence",
    response_class=FileResponse,
)
def download_sample(
    profile_id: int,
    filename: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Sert un fichier audio de référence normalisé.

    Utilisé par le moteur Voxtral (vLLM Omni) pour accéder aux samples
    via URL localhost lors du clonage vocal.
    Le nom de fichier est validé pour prévenir les attaques par path traversal.
    """
    get_profile_or_404(profile_id, db)

    safe_filename = _safe_wav(filename)
    file_path = UPLOAD_DIR / f"profile_{profile_id}" / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample audio introuvable.",
        )

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=safe_filename,
    )


@router.get(
    "/{profile_id}/samples",
    response_model=list[SampleInfo],
    summary="Lister les samples d'un profil",
)
def list_samples(
    profile_id: int,
    db: Session = Depends(get_db),
) -> list[SampleInfo]:
    """Retourne les métadonnées des fichiers audio de référence d'un profil."""
    get_profile_or_404(profile_id, db)

    profile_dir = UPLOAD_DIR / f"profile_{profile_id}"
    if not profile_dir.exists():
        return []

    result = []
    for wav in sorted(profile_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime):
        size = wav.stat().st_size
        try:
            meta = get_audio_metadata(wav)
            dur = meta["duration_s"]
        except Exception:
            dur = 0.0
        result.append(SampleInfo(filename=wav.name, size_bytes=size, duration_s=dur))

    return result


@router.delete(
    "/{profile_id}/samples/{filename}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer un sample audio",
)
def delete_sample(
    profile_id: int,
    filename: str,
    db: Session = Depends(get_db),
) -> None:
    """
    Supprime un sample audio de référence et met à jour les compteurs du profil.

    Attention : cette opération est irréversible.
    """
    profile = get_profile_or_404(profile_id, db)

    safe_filename = _safe_wav(filename)
    file_path = UPLOAD_DIR / f"profile_{profile_id}" / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample audio introuvable.",
        )

    # Durée du sample pour mettre à jour le total
    try:
        meta = get_audio_metadata(file_path)
        duration = meta["duration_s"]
    except Exception:
        duration = 0.0

    file_path.unlink()

    profile.sample_count = max(0, profile.sample_count - 1)
    profile.total_duration_s = max(0.0, round(profile.total_duration_s - duration, 2))
    db.commit()

    logger.info(
        "Sample supprimé — profil=%d fichier=%s durée=%.2fs restant=%d samples",
        profile_id,
        safe_filename,
        duration,
        profile.sample_count,
    )


@router.delete(
    "/{profile_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer un profil voix",
)
def delete_voice_profile(
    profile_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    Supprime un profil voix et toutes ses données associées (samples, consentements).

    Attention : cette opération est irréversible.
    Les fichiers audio sur disque sont également supprimés.
    """
    profile = get_profile_or_404(profile_id, db)

    # Suppression des fichiers audio sur disque
    for base_dir in (UPLOAD_DIR, OUTPUT_DIR):
        profile_dir = base_dir / f"profile_{profile_id}"
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
            logger.info("Dossier supprimé — %s", profile_dir)

    db.delete(profile)
    db.commit()

    logger.info("Profil voix supprimé — id=%d name='%s'", profile_id, profile.name)


# --- Utilitaires internes ---

def _safe_wav(filename: str) -> str:
    """Valide un nom de fichier WAV et previent le path traversal. Retourne le nom sur."""
    safe = Path(filename).name
    if safe != filename or not safe.endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nom de fichier invalide.",
        )
    return safe
