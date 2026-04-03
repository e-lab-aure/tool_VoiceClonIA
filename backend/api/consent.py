"""
Routes API — Gestion du consentement.

Le consentement est une étape OBLIGATOIRE avant tout clonage vocal.
Sans consentement actif, la synthèse est bloquée.

Endpoints :
    POST   /consent/{profile_id}          Enregistrer un consentement
    GET    /consent/{profile_id}          Statut du consentement d'un profil
    DELETE /consent/{profile_id}/revoke   Révoquer le consentement
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.logger import logger
from backend.models.consent import Consent
from backend.models.voice_profile import ProfileStatus, VoiceProfile

router = APIRouter(prefix="/consent", tags=["Consentement"])

# Texte du consentement affiché à l'utilisateur — immuable et versionné
CONSENT_TEXT = (
    "Je certifie être la personne dont la voix est enregistrée dans les fichiers audio "
    "fournis, ou avoir obtenu l'autorisation explicite de cette personne pour procéder "
    "au clonage vocal. J'accepte que ma voix soit utilisée uniquement dans le cadre "
    "de l'application VoiceClonIA, pour un usage personnel, et sans diffusion publique "
    "non consentie. Je comprends que je peux révoquer ce consentement à tout moment."
)


# --- Schémas Pydantic ---

class ConsentRequest(BaseModel):
    """Données requises pour enregistrer un consentement."""

    consented_by: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Nom ou identifiant de la personne consentante",
    )
    accepted: bool = Field(
        ...,
        description="Doit être True pour valider le consentement",
    )


class ConsentResponse(BaseModel):
    """Représentation publique d'un consentement."""

    id: int
    voice_profile_id: int
    consented_by: str
    consent_text: str
    is_revoked: bool
    revoked_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConsentStatusResponse(BaseModel):
    """Statut synthétique du consentement pour un profil."""

    profile_id: int
    profile_name: str
    has_active_consent: bool
    consent_count: int
    latest_consent: ConsentResponse | None


# --- Endpoints ---

@router.post(
    "/{profile_id}",
    response_model=ConsentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enregistrer un consentement",
)
def record_consent(
    profile_id: int,
    payload: ConsentRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> ConsentResponse:
    """
    Enregistre le consentement explicite d'une personne pour le clonage de sa voix.

    Le champ `accepted` doit être True. Un consentement refusé n'est pas enregistré.
    L'adresse IP est collectée à des fins de traçabilité légale uniquement.
    """
    profile = _get_profile_or_404(profile_id, db)

    if profile.status == ProfileStatus.REVOKED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ce profil a été révoqué. Aucun nouveau consentement ne peut être enregistré.",
        )

    if not payload.accepted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le consentement doit être explicitement accepté (accepted=true).",
        )

    # Récupération sécurisée de l'IP (proxy-aware)
    ip_address = _extract_client_ip(request)

    consent = Consent(
        voice_profile_id=profile_id,
        consented_by=payload.consented_by,
        consent_text=CONSENT_TEXT,
        ip_address=ip_address,
        is_revoked=False,
    )
    db.add(consent)

    # Si le profil avait des samples, il passe en READY
    if profile.sample_count > 0 and profile.status == ProfileStatus.PENDING_CONSENT:
        profile.status = ProfileStatus.READY

    db.commit()
    db.refresh(consent)

    logger.info(
        "Consentement enregistré — profil=%d consent_id=%d par='%s'",
        profile_id,
        consent.id,
        payload.consented_by,
    )

    return ConsentResponse.model_validate(consent)


@router.get(
    "/{profile_id}",
    response_model=ConsentStatusResponse,
    summary="Statut du consentement",
)
def get_consent_status(
    profile_id: int,
    db: Session = Depends(get_db),
) -> ConsentStatusResponse:
    """Retourne le statut de consentement actuel d'un profil voix."""
    profile = _get_profile_or_404(profile_id, db)

    active_consents = [c for c in profile.consents if not c.is_revoked]
    latest = (
        max(profile.consents, key=lambda c: c.created_at)
        if profile.consents
        else None
    )

    return ConsentStatusResponse(
        profile_id=profile.id,
        profile_name=profile.name,
        has_active_consent=len(active_consents) > 0,
        consent_count=len(profile.consents),
        latest_consent=ConsentResponse.model_validate(latest) if latest else None,
    )


@router.delete(
    "/{profile_id}/revoke",
    status_code=status.HTTP_200_OK,
    summary="Révoquer le consentement",
)
def revoke_consent(
    profile_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Révoque tous les consentements actifs d'un profil et bloque le profil.

    Après révocation, toute nouvelle synthèse est impossible pour ce profil.
    Les données audio existantes sont conservées jusqu'à suppression manuelle du profil.
    """
    profile = _get_profile_or_404(profile_id, db)

    now = datetime.now(timezone.utc)
    revoked_count = 0

    for consent in profile.consents:
        if not consent.is_revoked:
            consent.is_revoked = True
            consent.revoked_at = now
            revoked_count += 1

    # Le profil passe en statut REVOKED — aucune synthèse n'est plus possible
    profile.status = ProfileStatus.REVOKED
    db.commit()

    logger.info(
        "Consentement révoqué — profil=%d name='%s' consents_révoqués=%d",
        profile_id,
        profile.name,
        revoked_count,
    )

    return {
        "profile_id": profile_id,
        "revoked_consents": revoked_count,
        "message": "Consentement révoqué. Le profil est désormais inactif.",
    }


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


def _extract_client_ip(request: Request) -> str | None:
    """
    Extrait l'IP cliente de façon sûre, en tenant compte des proxies.

    Utilisée uniquement pour la traçabilité — jamais exposée via API publique.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Premier IP de la chaîne = IP cliente originale
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else None
