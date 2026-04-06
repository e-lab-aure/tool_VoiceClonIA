"""
Initialisation du système de logging centralisé.

Format uniforme : [LEVEL] YYYY-MM-DD HH:MM:SS — context — message
Sortie vers fichier (persistant) et console (développement).
"""

import io
import logging
import sys
from logging.handlers import RotatingFileHandler

from backend.core.config import LOG_FILE, LOG_LEVEL

# Format de log uniforme conforme aux directives du projet
_LOG_FORMAT = "[%(levelname)s] %(asctime)s — %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "voicelconia") -> logging.Logger:
    """
    Crée et configure un logger nommé avec rotation de fichier.

    Args:
        name: Nom du logger (utilisé comme contexte dans les logs).

    Returns:
        Instance de logger configurée.
    """
    logger = logging.getLogger(name)

    # Évite la double initialisation si le logger est déjà configuré
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Handler fichier avec rotation (10 Mo max, 5 fichiers conservés)
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # Ne bloque pas le démarrage si le fichier de log n'est pas accessible
        print(f"[WARNING] Impossible d'initialiser le handler de log fichier : {exc}", file=sys.stderr)

    # Handler console (stdout) pour le développement
    # Force UTF-8 sur Windows (évite UnicodeEncodeError avec cp1252)
    utf8_stream = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", line_buffering=True
    )
    console_handler = logging.StreamHandler(utf8_stream)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Évite la propagation vers le logger root pour ne pas dupliquer les messages
    logger.propagate = False

    return logger


# Logger global de l'application
logger = setup_logger()
