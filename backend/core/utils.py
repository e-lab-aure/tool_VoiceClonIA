"""
Utilitaires partagés entre les routes API.
"""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.models.voice_profile import VoiceProfile


def get_profile_or_404(profile_id: int, db: Session) -> VoiceProfile:
    """Retourne un profil par son id ou lève une 404."""
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profil voix #{profile_id} introuvable.",
        )
    return profile
