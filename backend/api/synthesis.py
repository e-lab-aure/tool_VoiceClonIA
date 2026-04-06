"""
Routes API — Synthèse vocale.

La synthèse est conditionnée à :
1. L'existence d'un profil voix avec statut READY
2. La présence d'un consentement actif (non révoqué)
3. La disponibilité d'au moins un sample audio de référence

Endpoints :
    POST /synthesis/{profile_id}         Générer un audio à partir d'un texte
    GET  /synthesis/{profile_id}/outputs Lister les fichiers générés
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.config import OUTPUT_DIR
from backend.core.database import get_db
from backend.core.logger import logger
from backend.models.voice_profile import ProfileStatus, VoiceProfile
from backend.services.audio import get_profile_samples
from backend.services.tts import synthesize_speech

router = APIRouter(prefix="/synthesis", tags=["Synthèse vocale"])


# --- Schémas Pydantic ---

class SynthesisRequest(BaseModel):
    """
    Paramètres d'une requête de synthèse vocale.

    Tous les samples du profil sont automatiquement utilisés comme référence
    (concaténation plafonnée à 30 s) pour maximiser la fidélité du clone.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Texte à synthétiser",
    )
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Intensité du style vocal (0.0 neutre → 1.0 exagéré)",
    )
    cfg_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Poids du guidage CFG",
    )
    voice: str | None = Field(
        default=None,
        description="Voix preset Voxtral (ex: fr_male, neutral_female). Ignoré par les autres moteurs.",
    )
    ref_text: str = Field(
        default="",
        max_length=2000,
        description="Transcription de l'audio de référence. Améliore la fidélité du clonage Voxtral.",
    )


class SynthesisResponse(BaseModel):
    """Réponse après génération d'un fichier audio."""

    output_filename: str
    profile_id: int
    text_length: int
    engine: str
    download_url: str


class OutputFile(BaseModel):
    """Métadonnées d'un fichier audio généré."""

    filename: str
    size_bytes: int
    download_url: str


# --- Endpoints ---

@router.post(
    "/{profile_id}",
    response_model=SynthesisResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Synthétiser du texte avec un profil voix",
)
def synthesize(
    profile_id: int,
    payload: SynthesisRequest,
    db: Session = Depends(get_db),
) -> SynthesisResponse:
    """
    Génère un fichier audio en synthétisant le texte fourni
    avec la voix clonée du profil spécifié.

    Conditions requises :
    - Profil en statut READY
    - Consentement actif présent
    - Au moins un sample de référence disponible
    """
    profile = _get_profile_or_404(profile_id, db)

    # Vérification du statut et du consentement
    _assert_synthesis_allowed(profile)

    # Récupération de tous les samples de référence du profil
    samples = get_profile_samples(profile_id)
    if not samples:
        # Voxtral sans sample : autorisé si une voix preset est disponible
        is_voxtral = profile.engine == "voxtral"
        if not is_voxtral:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Aucun sample audio de référence trouvé pour ce profil.",
            )

    logger.info(
        "Synthèse — profil=%d — %d sample(s) de référence utilisé(s)",
        profile_id,
        len(samples),
    )

    # Voix effective : payload > preset_voice du profil > défaut Voxtral
    effective_voice = payload.voice or getattr(profile, "preset_voice", None)

    # Lancement de la synthèse avec tous les samples (concaténation interne).
    # Si un modèle fine-tuné est disponible, il est automatiquement utilisé.
    try:
        output_path = synthesize_speech(
            text=payload.text,
            profile_id=profile_id,
            reference_audios=samples,
            exaggeration=payload.exaggeration,
            cfg_weight=payload.cfg_weight,
            engine_name=profile.engine,
            fine_tuned_model_path=profile.fine_tuned_model_path,
            voice=effective_voice,
            ref_text=payload.ref_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Échec de synthèse — profil=%d : %s", profile_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La synthèse vocale a échoué. Consultez les logs pour plus de détails.",
        )

    logger.info(
        "Synthèse réussie — profil=%d sortie=%s texte=%d chars",
        profile_id,
        output_path.name,
        len(payload.text),
    )

    return SynthesisResponse(
        output_filename=output_path.name,
        profile_id=profile_id,
        text_length=len(payload.text),
        engine=profile.engine,
        download_url=f"/synthesis/{profile_id}/outputs/{output_path.name}",
    )


@router.get(
    "/{profile_id}/outputs",
    response_model=list[OutputFile],
    summary="Lister les fichiers audio générés",
)
def list_outputs(
    profile_id: int,
    db: Session = Depends(get_db),
) -> list[OutputFile]:
    """Retourne la liste des fichiers audio générés pour un profil."""
    _get_profile_or_404(profile_id, db)

    profile_output_dir = OUTPUT_DIR / f"profile_{profile_id}"
    if not profile_output_dir.exists():
        return []

    files = sorted(profile_output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)

    return [
        OutputFile(
            filename=f.name,
            size_bytes=f.stat().st_size,
            download_url=f"/synthesis/{profile_id}/outputs/{f.name}",
        )
        for f in files
    ]


@router.get(
    "/{profile_id}/outputs/{filename}",
    summary="Télécharger un fichier audio généré",
    response_class=FileResponse,
)
def download_output(
    profile_id: int,
    filename: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Télécharge un fichier audio généré.

    Le nom de fichier est validé pour prévenir les attaques par path traversal.
    """
    _get_profile_or_404(profile_id, db)

    # Validation stricte du nom de fichier — aucun composant de chemin autorisé
    safe_filename = Path(filename).name
    if safe_filename != filename or not safe_filename.endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nom de fichier invalide.",
        )

    file_path = OUTPUT_DIR / f"profile_{profile_id}" / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fichier audio introuvable.",
        )

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=safe_filename,
    )


# --- Utilitaires internes ---

def _get_profile_or_404(profile_id: int, db: Session) -> VoiceProfile:
    """Retourne un profil par son id ou lève une 404."""
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profil voix #{profile_id} introuvable.",
        )
    return profile


def _assert_synthesis_allowed(profile: VoiceProfile) -> None:
    """
    Vérifie que la synthèse est autorisée pour ce profil.

    Lève une HTTPException si une condition bloquante est détectée.
    """
    if profile.status == ProfileStatus.REVOKED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ce profil a été révoqué. La synthèse vocale est interdite.",
        )

    if profile.status == ProfileStatus.PENDING_CONSENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Consentement requis avant toute synthèse. "
                   "Enregistrez un consentement via POST /consent/{profile_id}.",
        )

    if not profile.has_active_consent:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Aucun consentement actif pour ce profil. La synthèse est bloquée.",
        )

    if profile.status == ProfileStatus.ERROR:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Ce profil est en erreur. Vérifiez les samples uploadés.",
        )
