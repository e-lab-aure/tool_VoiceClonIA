"""
Point d'entrée de l'application VoiceClonIA.

Lance le serveur FastAPI avec tous les routers et initialise la base de données.
Démarrage : uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from backend.api import consent, finetune, synthesis, voices
from backend.core.config import HOST, PORT
from backend.core.database import init_db
from backend.core.logger import logger

# Chemin vers l'interface web
_UI_FILE = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire du cycle de vie de l'application.

    Exécuté au démarrage et à l'arrêt du serveur.
    """
    logger.info("Démarrage de VoiceClonIA — initialisation de la base de données")
    init_db()
    logger.info("VoiceClonIA prêt — http://%s:%d", HOST, PORT)
    yield
    logger.info("Arrêt de VoiceClonIA")


app = FastAPI(
    title="VoiceClonIA",
    description=(
        "API de clonage vocal local. "
        "Clone une voix à partir de fichiers audio de référence "
        "avec consentement explicite obligatoire."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # Désactive les détails d'erreur en production (à configurer via env)
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restreint à localhost pour un usage local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enregistrement des routers
app.include_router(voices.router)
app.include_router(consent.router)
app.include_router(synthesis.router)
app.include_router(finetune.router)


@app.get("/", tags=["Santé"])
def health_check() -> dict:
    """Endpoint de vérification de santé du serveur."""
    return {"status": "ok", "app": "VoiceClonIA", "version": "0.1.0"}


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def serve_ui() -> HTMLResponse:
    """Sert l'interface web VoiceClonIA."""
    return HTMLResponse(content=_UI_FILE.read_text(encoding="utf-8"))
