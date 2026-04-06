"""
Routes API — Fine-tuning XTTS-v2.

Endpoints :
    POST   /finetune/{profile_id}         Lancer le fine-tuning
    GET    /finetune/{profile_id}/status  Statut et progression du job
    DELETE /finetune/{profile_id}/model   Supprimer le modèle fine-tuné
"""

from contextlib import contextmanager

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.core.database import SessionLocal, get_db
from backend.core.logger import logger
from backend.core.utils import get_profile_or_404
from backend.models.voice_profile import ProfileStatus, VoiceProfile
from backend.services.audio import get_profile_samples
from backend.services import finetune as ft

router = APIRouter(prefix="/finetune", tags=["Fine-tuning XTTS-v2"])


# --- Schémas Pydantic ---

class FineTuneStatusResponse(BaseModel):
    """Statut courant d'un job de fine-tuning."""

    profile_id: int
    status: str
    progress: float
    message: str
    error: str | None


class FineTuneStartResponse(BaseModel):
    """Réponse au lancement d'un job de fine-tuning."""

    profile_id: int
    message: str
    sample_count: int


# --- Endpoints ---

@router.post(
    "/{profile_id}",
    response_model=FineTuneStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Lancer le fine-tuning XTTS-v2 pour un profil",
)
def start_finetune(
    profile_id: int,
    db: Session = Depends(get_db),
) -> FineTuneStartResponse:
    """
    Lance le fine-tuning XTTS-v2 en arrière-plan.

    Prérequis :
    - Profil existant avec consentement actif
    - Au moins 1 sample audio de référence (10 min recommandé pour une bonne qualité)
    - CoquiTTS et openai-whisper installés

    Le fine-tuning prend plusieurs heures. Sondez GET /finetune/{id}/status pour
    suivre la progression.
    """
    profile = get_profile_or_404(profile_id, db)
    _assert_finetune_allowed(profile)

    samples = get_profile_samples(profile_id)
    if not samples:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Aucun sample audio trouvé pour ce profil.",
        )

    # Mise à jour du statut en base avant de lancer le thread
    profile.fine_tune_status = "pending"
    db.commit()

    try:
        ft.start_finetune(profile_id, samples, _db_context_factory)
    except RuntimeError as exc:
        profile.fine_tune_status = None
        db.commit()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    logger.info(
        "Fine-tuning lancé — profil=%d — %d sample(s)",
        profile_id, len(samples),
    )

    return FineTuneStartResponse(
        profile_id=profile_id,
        message="Fine-tuning lancé. Sondez /status pour suivre la progression.",
        sample_count=len(samples),
    )


@router.get(
    "/{profile_id}/status",
    response_model=FineTuneStatusResponse,
    summary="Statut et progression du fine-tuning",
)
def get_finetune_status(
    profile_id: int,
    db: Session = Depends(get_db),
) -> FineTuneStatusResponse:
    """
    Retourne le statut courant du job de fine-tuning.

    Si aucun job n'est actif en mémoire, retourne le statut persisté en base.
    """
    get_profile_or_404(profile_id, db)

    # Priorité : job actif en mémoire (plus frais)
    job_status = ft.get_status(profile_id)
    if job_status:
        return FineTuneStatusResponse(**job_status)

    # Fallback : statut persisté en base (job terminé ou serveur redémarré)
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    db_status = profile.fine_tune_status or "none"

    return FineTuneStatusResponse(
        profile_id=profile_id,
        status=db_status,
        progress=1.0 if db_status == "done" else 0.0,
        message=_status_to_message(db_status),
        error=None,
    )


@router.delete(
    "/{profile_id}/model",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer le modèle fine-tuné",
)
def delete_finetune_model(
    profile_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    Supprime le checkpoint fine-tuné sur disque et réinitialise le statut du profil.

    Attention : irréversible. Le fine-tuning devra être relancé.
    """
    profile = get_profile_or_404(profile_id, db)

    ft.delete_model(profile_id)

    profile.fine_tune_status = None
    profile.fine_tuned_model_path = None
    db.commit()

    logger.info("Modèle fine-tuné supprimé — profil=%d", profile_id)


@router.delete(
    "/{profile_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Annuler le fine-tuning en cours",
)
def cancel_finetune(
    profile_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Demande l'annulation coopérative du job de fine-tuning en cours."""
    get_profile_or_404(profile_id, db)

    if not ft.cancel_job(profile_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun job de fine-tuning actif pour ce profil.",
        )


# --- Utilitaires internes ---

def _assert_finetune_allowed(profile: VoiceProfile) -> None:
    """
    Vérifie que le fine-tuning est autorisé pour ce profil.

    Lève une HTTPException si une condition bloquante est détectée.
    """
    if profile.status == ProfileStatus.REVOKED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ce profil a été révoqué. Le fine-tuning est interdit.",
        )


@contextmanager
def _db_context_factory():
    """
    Fabrique de sessions SQLAlchemy pour le thread de fine-tuning.

    Le thread ne peut pas utiliser la session FastAPI (liée à la requête HTTP),
    il crée donc sa propre session via ce gestionnaire de contexte.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _status_to_message(status: str) -> str:
    """Convertit un statut technique en message lisible."""
    return {
        "none": "Aucun fine-tuning effectué",
        "pending": "En attente de démarrage…",
        "transcribing": "Transcription audio en cours…",
        "preparing": "Préparation du dataset…",
        "training": "Entraînement en cours…",
        "done": "Fine-tuning terminé avec succès ✓",
        "error": "Erreur lors du fine-tuning",
        "cancelled": "Fine-tuning annulé",
    }.get(status, status)
