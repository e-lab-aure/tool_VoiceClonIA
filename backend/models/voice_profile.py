"""
Modèle ORM pour les profils voix.

Un profil voix regroupe les fichiers audio d'un locuteur,
le consentement associé, et les métadonnées de clonage.
"""

from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


class ProfileStatus(str, PyEnum):
    """États du cycle de vie d'un profil voix."""

    PENDING_CONSENT = "pending_consent"  # En attente de consentement
    READY = "ready"                       # Prêt à être utilisé pour la synthèse
    PROCESSING = "processing"             # Entraînement/clonage en cours
    ERROR = "error"                       # Erreur lors du traitement
    REVOKED = "revoked"                   # Consentement révoqué — profil désactivé


class VoiceProfile(Base):
    """
    Profil voix d'un locuteur.

    Colonnes :
        id              — Identifiant unique
        name            — Nom du profil (ex. "Voix principale")
        description     — Description libre
        status          — État courant du profil (voir ProfileStatus)
        sample_dir      — Sous-dossier relatif à UPLOAD_DIR contenant les samples
        sample_count    — Nombre de fichiers audio associés
        total_duration_s— Durée totale des samples en secondes
        engine          — Moteur TTS utilisé pour ce profil
        engine_model_ref— Référence interne au modèle chargé (ex. chemin checkpoint)
        created_at      — Date de création
        updated_at      — Date de dernière modification
    """

    __tablename__ = "voice_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[ProfileStatus] = mapped_column(
        Enum(ProfileStatus),
        default=ProfileStatus.PENDING_CONSENT,
        nullable=False,
    )

    # Chemin du sous-dossier relatif à UPLOAD_DIR (ex. "profile_1")
    sample_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)

    sample_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    total_duration_s: Mapped[float] = mapped_column(default=0.0, nullable=False)

    engine: Mapped[str] = mapped_column(String(64), default="chatterbox", nullable=False)

    # Référence interne au modèle chargé — jamais exposée directement via API publique
    engine_model_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Fine-tuning XTTS-v2
    # Statut : None | "pending" | "running" | "done" | "error" | "cancelled"
    fine_tune_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Chemin vers le dossier du checkpoint fine-tuné (jamais exposé via API publique)
    fine_tuned_model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Catégorie du profil : "clone" | "preset" | "game"
    category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Voix preset par défaut (Voxtral) — utilisée si aucun sample n'est disponible
    preset_voice: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Tags libres pour l'organisation (ex : "Garde, Boss, PNJ")
    tags: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relations
    consents: Mapped[list["Consent"]] = relationship(  # noqa: F821
        back_populates="voice_profile",
        cascade="all, delete-orphan",
    )

    @property
    def has_active_consent(self) -> bool:
        """Retourne True si au moins un consentement non révoqué existe."""
        return any(not c.is_revoked for c in self.consents)

    def __repr__(self) -> str:
        return f"<VoiceProfile id={self.id} name='{self.name}' status={self.status}>"
