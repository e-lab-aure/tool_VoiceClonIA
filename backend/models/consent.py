"""
Modèle ORM pour l'enregistrement du consentement.

Chaque profil voix doit être associé à un consentement explicite et traçable.
Sans consentement valide, aucune opération de clonage n'est autorisée.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


class Consent(Base):
    """
    Enregistrement du consentement explicite d'une personne
    à faire cloner sa voix.

    Colonnes :
        id              — Identifiant unique du consentement
        voice_profile_id— Lien vers le profil voix concerné
        consented_by    — Nom ou identifiant de la personne consentante
        consent_text    — Texte exact du consentement affiché et accepté
        ip_address      — Adresse IP au moment du consentement (traçabilité)
        is_revoked      — True si le consentement a été révoqué
        revoked_at      — Date de révocation (None si toujours actif)
        created_at      — Date d'enregistrement du consentement
    """

    __tablename__ = "consents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    voice_profile_id: Mapped[int] = mapped_column(
        ForeignKey("voice_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    consented_by: Mapped[str] = mapped_column(String(255), nullable=False)

    consent_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Stockée uniquement à des fins de traçabilité légale — jamais exposée via API
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)

    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relation inverse vers VoiceProfile
    voice_profile: Mapped["VoiceProfile"] = relationship(  # noqa: F821
        back_populates="consents"
    )

    def __repr__(self) -> str:
        status = "révoqué" if self.is_revoked else "actif"
        return f"<Consent id={self.id} profil={self.voice_profile_id} statut={status}>"
