"""
Initialisation et gestion de la base de données SQLite via SQLAlchemy.

Fournit le moteur, la session, et la base déclarative partagés par tous les modèles.
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.core.config import DATABASE_PATH
from backend.core.logger import logger

# URL de connexion SQLite — le fichier est local, aucun credential requis
_DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# check_same_thread=False est nécessaire pour FastAPI qui utilise plusieurs threads
engine = create_engine(
    _DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Mettre à True pour logger les requêtes SQL (développement uniquement)
)


@event.listens_for(engine, "connect")
def _enable_wal_mode(dbapi_connection, _connection_record):
    """
    Active le mode WAL (Write-Ahead Logging) de SQLite pour de meilleures
    performances en lecture/écriture concurrentes.
    """
    dbapi_connection.execute("PRAGMA journal_mode=WAL")
    dbapi_connection.execute("PRAGMA foreign_keys=ON")


# Fabrique de sessions — chaque requête FastAPI obtient sa propre session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base déclarative partagée par tous les modèles ORM."""
    pass


def init_db() -> None:
    """
    Crée toutes les tables définies dans les modèles si elles n'existent pas,
    puis applique les migrations de colonnes manquantes.

    Doit être appelé une seule fois au démarrage de l'application.
    Les modèles doivent être importés avant cet appel pour être enregistrés.
    """
    # Import des modèles pour les enregistrer auprès de la Base avant création
    from backend.models import consent, voice_profile  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_add_columns()
    logger.info("Base de données initialisée — tables créées si absentes")


def _migrate_add_columns() -> None:
    """
    Ajoute les colonnes manquantes à la table voice_profiles (migration SQLite).

    SQLite ne supporte pas ALTER TABLE ADD COLUMN IF NOT EXISTS avant 3.37,
    donc on vérifie manuellement via PRAGMA table_info.
    """
    new_columns = [
        ("fine_tune_status", "VARCHAR(32)"),
        ("fine_tuned_model_path", "VARCHAR(512)"),
        ("category", "VARCHAR(32)"),
        ("preset_voice", "VARCHAR(64)"),
        ("tags", "VARCHAR(255)"),
    ]

    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(voice_profiles)"))
        existing = {row[1] for row in result}

        for col_name, col_type in new_columns:
            if col_name not in existing:
                conn.execute(
                    text(f"ALTER TABLE voice_profiles ADD COLUMN {col_name} {col_type}")
                )
                conn.commit()
                logger.info("Migration DB : colonne '%s' ajoutée à voice_profiles", col_name)


def get_db():
    """
    Dépendance FastAPI : fournit une session DB et garantit sa fermeture.

    Yields:
        Session SQLAlchemy active pour la durée de la requête.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
