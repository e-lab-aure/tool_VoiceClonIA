"""
Couche d'abstraction des moteurs TTS (Text-To-Speech).

Fournit une interface unifiée pour les différents moteurs supportés :
- Chatterbox (Resemble AI) — moteur par défaut, léger et temps réel
- F5-TTS — alternative haute qualité, zero-shot

Le moteur actif est sélectionné via la variable d'environnement TTS_ENGINE.
"""

import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from backend.core.config import OUTPUT_DIR, PORT, TTS_ENGINE, UPLOAD_DIR, VOXTRAL_DEFAULT_VOICE, VOXTRAL_SERVER_URL
from backend.core.logger import logger

# Durée maximale de la référence concaténée (en secondes)
_MAX_REFERENCE_DURATION_S = 30
# Sample rate attendu par les moteurs TTS (WAV normalisé)
_REFERENCE_SAMPLE_RATE = 16_000


class TTSEngine(ABC):
    """Interface abstraite commune à tous les moteurs TTS."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> Path:
        """
        Synthétise du texte en audio en clonant la voix de référence.

        Args:
            text: Texte à synthétiser.
            reference_audio: Fichier WAV 16 kHz mono de référence (voix à cloner).
            output_path: Chemin de sortie du fichier WAV généré.
            exaggeration: Intensité du style vocal (0.0 neutre → 1.0 exagéré).
            cfg_weight: Poids du guidage CFG (0.0 → 1.0, défaut 0.5).
            **kwargs: Paramètres additionnels spécifiques à chaque moteur.

        Returns:
            Chemin du fichier audio généré.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Retourne True si le moteur et ses dépendances sont disponibles."""
        ...


class ChatterboxEngine(TTSEngine):
    """
    Moteur TTS Chatterbox de Resemble AI.

    Caractéristiques :
    - Modèle 350M paramètres — rapide et léger
    - Zero-shot : pas d'entraînement requis, quelques secondes de référence suffisent
    - Tags émotionnels supportés : [laugh], [sigh], [gasp], etc.
    - Compatible Windows CUDA / CPU
    """

    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        """
        Charge le modèle Chatterbox en mémoire (lazy loading).

        Sur RTX 4090 (Ampere/Ada) avec CUDA 12.8, active les optimisations :
        - torch.compile() pour la fusion des kernels CUDA
        - float16 automatique via autocast lors de l'inférence
        """
        if self._model is not None:
            return

        try:
            import torch

            # Patch perth avant l'import de chatterbox :
            # PerthImplicitWatermarker est None quand pkg_resources n'est pas
            # accessible dans le venv (bug Python 3.12 / setuptools 72+).
            # Le watermarking est une fonctionnalité anti-piratage de Resemble AI,
            # non nécessaire pour notre usage local — DummyWatermarker est fourni
            # par perth lui-même exactement pour ce cas de fallback.
            import perth
            if perth.PerthImplicitWatermarker is None:
                from perth.dummy_watermarker import DummyWatermarker
                perth.PerthImplicitWatermarker = DummyWatermarker
                logger.warning(
                    "PerthImplicitWatermarker indisponible (pkg_resources manquant) — "
                    "DummyWatermarker utilisé à la place (aucun impact sur la qualité vocale)"
                )

            from chatterbox.tts import ChatterboxTTS

            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(
                    "GPU détecté : %s — VRAM : %.1f Go — CUDA %s",
                    gpu_name,
                    vram_gb,
                    torch.version.cuda,
                )
            else:
                device = "cpu"
                logger.warning("CUDA non disponible — inférence sur CPU (lent)")

            logger.info("Chargement de Chatterbox sur %s", device)
            self._model = ChatterboxTTS.from_pretrained(device=device)

            # Activation de TF32 pour les opérations matricielles (Ampere+)
            # Donne un gain de vitesse ~3x sur RTX 4090 sans perte de qualité perceptible
            if device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 activé pour RTX 4090 (Ampere/Ada)")

            logger.info("Chatterbox chargé avec succès")
        except ImportError as exc:
            raise RuntimeError(
                "Chatterbox n'est pas installé. "
                "Exécutez : pip install chatterbox-tts"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Échec du chargement de Chatterbox : {exc}") from exc

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> Path:
        """
        Synthétise le texte avec Chatterbox en clonant la voix de référence.

        Utilise torch.autocast(float16) sur GPU pour réduire la consommation VRAM
        et accélérer l'inférence sur RTX 4090 (~2x plus rapide qu'en float32).
        """
        import torch
        import soundfile as sf

        self._load_model()

        logger.info(
            "Synthèse Chatterbox — texte=%d chars référence=%s",
            len(text),
            reference_audio.name,
        )

        use_cuda = torch.cuda.is_available()

        # autocast float16 sur GPU : réduit la VRAM et accélère l'inférence
        # RTX 4090 : tensor cores float16 ~2x plus rapides que float32
        with torch.inference_mode():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    wav = self._model.generate(
                        text,
                        audio_prompt_path=str(reference_audio),
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                    )
            else:
                wav = self._model.generate(
                    text,
                    audio_prompt_path=str(reference_audio),
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

        # torchaudio.save() requiert torchcodec dans torchaudio >= 2.10
        # On utilise soundfile directement : plus léger et déjà installé.
        # wav est un tensor (1, N) — on le convertit en numpy (N,) pour soundfile.
        audio_np = wav.squeeze().cpu().float().numpy()
        sf.write(str(output_path), audio_np, self._model.sr, subtype="PCM_16")
        logger.info("Synthèse terminée — sortie=%s", output_path.name)
        return output_path

    def is_available(self) -> bool:
        """Vérifie si chatterbox-tts est installé."""
        try:
            import chatterbox  # noqa: F401
            return True
        except ImportError:
            return False


class F5TTSEngine(TTSEngine):
    """
    Moteur TTS F5-TTS (zero-shot, haute qualité).

    Caractéristiques :
    - Modèle flow-matching — qualité naturelle très élevée
    - Zero-shot depuis quelques secondes de référence
    - Plus lent que Chatterbox mais meilleure fidélité vocale
    """

    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        """Charge le modèle F5-TTS en mémoire (lazy loading)."""
        if self._model is not None:
            return

        try:
            from f5_tts.api import F5TTS
            self._model = F5TTS()
            logger.info("F5-TTS chargé avec succès")
        except ImportError as exc:
            raise RuntimeError(
                "F5-TTS n'est pas installé. "
                "Exécutez : pip install f5-tts"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Échec du chargement de F5-TTS : {exc}") from exc

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> Path:
        """Synthétise le texte avec F5-TTS en clonant la voix de référence."""
        self._load_model()

        logger.info(
            "Synthèse F5-TTS — texte=%d chars référence=%s",
            len(text),
            reference_audio.name,
        )

        # F5-TTS génère directement vers un fichier WAV
        self._model.infer(
            ref_file=str(reference_audio),
            ref_text="",          # Transcription auto si vide
            gen_text=text,
            file_wave=str(output_path),
            speed=1.0,
        )

        logger.info("Synthèse F5-TTS terminée — sortie=%s", output_path.name)
        return output_path

    def is_available(self) -> bool:
        """Vérifie si f5-tts est installé."""
        try:
            import f5_tts  # noqa: F401
            return True
        except ImportError:
            return False


class XTTSv2Engine(TTSEngine):
    """
    Moteur TTS XTTS-v2 (CoquiTTS) — zero-shot ou fine-tuné.

    Caractéristiques :
    - Modèle multilingue (16 langues dont le français)
    - Zero-shot depuis quelques secondes de référence
    - Supporte les checkpoints fine-tunés par profil pour une fidélité accrue
    - Plus lent que Chatterbox (~2-3x) mais meilleure qualité après fine-tuning
    """

    def __init__(self, fine_tuned_model_path: str | None = None) -> None:
        self._model = None
        # Chemin vers le checkpoint fine-tuné — None = modèle de base
        self._fine_tuned_model_path = fine_tuned_model_path

    def _load_model(self) -> None:
        """Charge le modèle XTTS-v2 en mémoire (lazy loading)."""
        if self._model is not None:
            return

        try:
            from TTS.api import TTS

            if self._fine_tuned_model_path:
                self._model = TTS(
                    model_path=self._fine_tuned_model_path,
                    gpu=True,
                )
                logger.info(
                    "XTTS-v2 fine-tuné chargé depuis %s", self._fine_tuned_model_path
                )
            else:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                logger.info("XTTS-v2 base chargé sur %s", device)

        except ImportError as exc:
            raise RuntimeError(
                "CoquiTTS n'est pas installé. Exécutez : pip install TTS"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Échec du chargement de XTTS-v2 : {exc}") from exc

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        **kwargs,
    ) -> Path:
        """
        Synthétise le texte avec XTTS-v2 en clonant la voix de référence.

        Note : exaggeration et cfg_weight sont ignorés par XTTS-v2 (API différente
        de Chatterbox). La qualité est contrôlée par le fine-tuning du modèle.
        """
        self._load_model()

        logger.info(
            "Synthèse XTTS-v2 — texte=%d chars référence=%s",
            len(text),
            reference_audio.name,
        )

        self._model.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio),
            language="fr",
            file_path=str(output_path),
        )

        logger.info("Synthèse XTTS-v2 terminée — sortie=%s", output_path.name)
        return output_path

    def is_available(self) -> bool:
        """Vérifie si CoquiTTS est installé."""
        try:
            import TTS  # noqa: F401
            return True
        except ImportError:
            return False


def _build_sample_url(reference_audio: Path) -> str:
    """
    Construit l'URL localhost du sample audio accessible par le serveur vLLM.

    Le container vLLM tourne avec --network=host, donc il peut fetcher
    http://127.0.0.1:8000. Le chemin attendu : UPLOAD_DIR/profile_{id}/{file}.wav
    """
    profile_dir = reference_audio.parent.name   # ex : "profile_1"
    profile_id  = profile_dir.removeprefix("profile_")
    return f"http://127.0.0.1:{PORT}/voices/{profile_id}/samples/{reference_audio.name}"


class VoxtralEngine(TTSEngine):
    """
    Moteur TTS Voxtral-4B (Mistral AI) via vLLM Omni.

    Caractéristiques :
    - 4B paramètres — haute qualité, multilingue (9 langues)
    - Zero-shot voice cloning via ref_audio + ref_text (3–30 s de référence)
    - 20 voix preset disponibles en fallback
    - Tourne en local sur RTX 4090 via serveur vLLM Omni (port 8001)
    - API OpenAI-compatible : POST /v1/audio/speech
    - Output : WAV 24 kHz

    Le serveur vLLM doit être démarré séparément :
        vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8001
    """

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        output_path: Path,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        voice: str | None = None,
        ref_text: str = "",
        **kwargs,
    ) -> Path:
        """
        Synthétise le texte avec Voxtral via vLLM Omni.

        Si reference_audio est fourni et accessible via URL, utilise le clonage vocal
        (task_type=Base avec ref_audio + ref_text optionnel).
        Sinon, utilise la voix preset configurée.

        Note : ref_audio dans l'API vLLM doit être une URL accessible depuis le serveur.
        La route GET /voices/{id}/samples/{filename} expose les samples pour cet usage.
        """
        payload: dict = {
            "input": text,
            "model": "mistralai/Voxtral-4B-TTS-2603",
            "response_format": "wav",
        }

        if reference_audio is not None:
            # Le clonage vocal Voxtral nécessite les poids de l'encodeur audio,
            # non disponibles dans le checkpoint open-source. Fonctionnalité à venir.
            raise RuntimeError(
                "Le clonage vocal n'est pas disponible avec le modèle open-source "
                "Voxtral-4B-TTS-2603 (encodeur audio propriétaire requis). "
                "Sélectionnez une voix preset pour synthétiser."
            )
        else:
            preset = voice or VOXTRAL_DEFAULT_VOICE
            payload["voice"] = preset
            logger.info("Voxtral voix preset — voice=%s", preset)

        logger.info(
            "Synthèse Voxtral — texte=%d chars serveur=%s",
            len(text),
            VOXTRAL_SERVER_URL,
        )

        try:
            response = httpx.post(
                f"{VOXTRAL_SERVER_URL}/v1/audio/speech",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Voxtral a retourné une erreur HTTP {exc.response.status_code} : "
                f"{exc.response.text[:600]}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Impossible de joindre le serveur Voxtral ({VOXTRAL_SERVER_URL}). "
                f"Vérifiez que vLLM Omni est démarré : "
                f"vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8001"
            ) from exc

        output_path.write_bytes(response.content)
        logger.info("Synthèse Voxtral terminée — sortie=%s", output_path.name)
        return output_path

    def is_available(self) -> bool:
        """Vérifie que le serveur vLLM Omni répond."""
        try:
            httpx.get(f"{VOXTRAL_SERVER_URL}/health", timeout=2.0)
            return True
        except Exception:
            return False


# Registre des moteurs disponibles (zero-shot)
_ENGINES: dict[str, type[TTSEngine]] = {
    "chatterbox": ChatterboxEngine,
    "f5tts": F5TTSEngine,
    "xttsv2": XTTSv2Engine,
    "voxtral": VoxtralEngine,
}

# Instance unique du moteur actif zero-shot (singleton partagé entre profils)
_active_engine: TTSEngine | None = None

# Cache des engines XTTS-v2 fine-tunés — une instance par chemin de modèle
_xtts_engines: dict[str | None, XTTSv2Engine] = {}


def get_engine(engine_name: str | None = None) -> TTSEngine:
    """
    Retourne l'instance singleton du moteur TTS zero-shot.

    Le moteur est sélectionné via engine_name ou TTS_ENGINE dans la config.
    Le chargement du modèle est différé jusqu'au premier appel à synthesize().

    Args:
        engine_name: Nom du moteur à utiliser. Si None, utilise TTS_ENGINE.

    Raises:
        ValueError: Si le moteur configuré n'est pas reconnu.
    """
    global _active_engine

    key = (engine_name or TTS_ENGINE).lower()

    # Le singleton est partagé uniquement si le moteur demandé correspond au moteur actif
    if _active_engine is not None and key == (TTS_ENGINE.lower() if engine_name is None else key):
        return _active_engine

    engine_class = _ENGINES.get(key)
    if engine_class is None:
        available = ", ".join(_ENGINES.keys())
        raise ValueError(
            f"Moteur TTS inconnu : '{key}'. "
            f"Moteurs disponibles : {available}"
        )

    logger.info("Initialisation du moteur TTS : %s", key)
    engine = engine_class()

    # Ne met en cache singleton que le moteur par défaut de la config
    if engine_name is None or key == TTS_ENGINE.lower():
        _active_engine = engine

    return engine


def get_xtts_engine(fine_tuned_model_path: str | None = None) -> XTTSv2Engine:
    """
    Retourne une instance XTTS-v2 pour le modèle donné (avec cache).

    Si fine_tuned_model_path est None, retourne le modèle XTTS-v2 de base.
    Les instances sont mises en cache par chemin pour éviter les rechargements.

    Args:
        fine_tuned_model_path: Chemin vers le checkpoint fine-tuné, ou None.

    Returns:
        Instance XTTSv2Engine prête à l'emploi (modèle non encore chargé).
    """
    if fine_tuned_model_path not in _xtts_engines:
        _xtts_engines[fine_tuned_model_path] = XTTSv2Engine(fine_tuned_model_path)
    return _xtts_engines[fine_tuned_model_path]


def _build_reference(reference_audios: list[Path]) -> Path:
    """
    Concatène plusieurs fichiers WAV de référence en un seul fichier temporaire.

    La concaténation est plafonnée à _MAX_REFERENCE_DURATION_S secondes pour
    rester dans les limites acceptables des moteurs TTS (trop long = latence
    excessive et dégradation de la cohérence temporelle).

    Args:
        reference_audios: Liste de fichiers WAV 16 kHz mono normalisés.

    Returns:
        Chemin du fichier WAV temporaire concaténé.

    Raises:
        RuntimeError: Si aucun fichier valide n'a pu être chargé.
    """
    chunks: list[np.ndarray] = []
    total_samples = 0
    max_samples = _MAX_REFERENCE_DURATION_S * _REFERENCE_SAMPLE_RATE

    for audio_path in reference_audios:
        if total_samples >= max_samples:
            logger.debug(
                "Plafond %ds atteint — %d sample(s) restant(s) ignoré(s)",
                _MAX_REFERENCE_DURATION_S,
                len(reference_audios) - reference_audios.index(audio_path),
            )
            break

        try:
            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            # Sécurité : le fichier doit déjà être à 16 kHz mono (normalisé)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Tronquer si ce chunk dépasse le plafond restant
            remaining = max_samples - total_samples
            if len(audio) > remaining:
                audio = audio[:remaining]
            chunks.append(audio)
            total_samples += len(audio)
        except Exception as exc:
            logger.warning("Impossible de charger %s — ignoré : %s", audio_path.name, exc)

    if not chunks:
        raise RuntimeError("Aucun fichier de référence valide n'a pu être chargé.")

    combined = np.concatenate(chunks)
    duration = len(combined) / _REFERENCE_SAMPLE_RATE

    # Fichier temporaire supprimé par l'appelant après la synthèse
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    sf.write(str(tmp_path), combined, _REFERENCE_SAMPLE_RATE, subtype="PCM_16")
    logger.info(
        "Référence concaténée — %d sample(s) — durée totale %.1fs → %s",
        len(chunks),
        duration,
        tmp_path.name,
    )
    return tmp_path


def synthesize_speech(
    text: str,
    profile_id: int,
    reference_audios: list[Path],
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    engine_name: str | None = None,
    fine_tuned_model_path: str | None = None,
    voice: str | None = None,
    ref_text: str = "",
) -> Path:
    """
    Point d'entrée principal pour la synthèse vocale.

    Si un modèle fine-tuné est disponible (fine_tuned_model_path), utilise XTTS-v2
    avec ce checkpoint pour une fidélité vocale maximale.
    Sinon, utilise le moteur zero-shot configuré (Chatterbox par défaut).

    Pour Voxtral : la concaténation de référence est ignorée — le premier sample
    est utilisé directement via URL localhost (accessible par le serveur vLLM).

    Args:
        text: Texte à synthétiser (max 5000 caractères).
        profile_id: Identifiant du profil voix.
        reference_audios: Liste de fichiers WAV 16 kHz mono normalisés.
        exaggeration: Intensité du style vocal (0.0–1.0, Chatterbox uniquement).
        cfg_weight: Poids du guidage (0.0–1.0, Chatterbox uniquement).
        engine_name: Moteur à utiliser. Si None, utilise TTS_ENGINE de la config.
        fine_tuned_model_path: Chemin vers le checkpoint XTTS-v2 fine-tuné, ou None.
        voice: Voix preset Voxtral (ex: "fr_male"). Ignoré par les autres moteurs.
        ref_text: Transcription de l'audio de référence. Améliore le clonage Voxtral.

    Returns:
        Chemin du fichier audio généré.

    Raises:
        ValueError: Texte vide, trop long ou liste de références vide.
        RuntimeError: Échec de la synthèse.
    """
    if not text or not text.strip():
        raise ValueError("Le texte à synthétiser ne peut pas être vide.")

    if len(text) > 5000:
        raise ValueError(
            f"Texte trop long ({len(text)} caractères). Maximum : 5000 caractères."
        )

    # Voxtral sans samples : synthese en mode preset (voix integree)
    resolved_engine = (engine_name or TTS_ENGINE).lower()
    if not reference_audios and resolved_engine != "voxtral":
        raise ValueError("Au moins un fichier de référence est requis.")

    missing = [p for p in reference_audios if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Fichier(s) de référence introuvable(s) : {', '.join(p.name for p in missing)}"
        )

    # Dossier de sortie par profil
    profile_output_dir = OUTPUT_DIR / f"profile_{profile_id}"
    profile_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = profile_output_dir / f"{uuid.uuid4().hex}.wav"

    # Sélection du moteur :
    # — Si un modèle fine-tuné est disponible → XTTS-v2 avec ce checkpoint
    # — Sinon → moteur zero-shot du profil (Chatterbox, F5-TTS ou Voxtral)
    if fine_tuned_model_path:
        engine: TTSEngine = get_xtts_engine(fine_tuned_model_path)
        logger.info("Moteur sélectionné : XTTS-v2 fine-tuné")
    else:
        engine = get_engine(engine_name)

    # Voxtral accède aux samples via URL localhost — pas de fichier temporaire
    # (vLLM doit pouvoir fetcher ref_audio, un temp file n'est pas accessible via l'API)
    # Si aucun sample : mode preset (reference_audio=None → voix integree)
    if isinstance(engine, VoxtralEngine):
        return engine.synthesize(
            text=text,
            reference_audio=reference_audios[0] if reference_audios else None,
            output_path=output_path,
            voice=voice,
            ref_text=ref_text,
        )

    # Autres moteurs : concaténation des références en fichier temporaire
    tmp_reference = _build_reference(reference_audios)

    try:
        return engine.synthesize(
            text=text,
            reference_audio=tmp_reference,
            output_path=output_path,
            exaggeration=max(0.0, min(1.0, exaggeration)),
            cfg_weight=max(0.0, min(1.0, cfg_weight)),
        )
    finally:
        # Nettoyage du fichier temporaire quoi qu'il arrive
        try:
            tmp_reference.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Impossible de supprimer la référence temporaire : %s", exc)
