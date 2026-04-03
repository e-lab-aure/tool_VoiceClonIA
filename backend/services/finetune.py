"""
Service de fine-tuning XTTS-v2.

Flux :
1. Transcription automatique des samples avec Whisper (medium)
2. Préparation du dataset au format LJSpeech (metadata.csv + wavs/)
3. Fine-tuning du modèle XTTS-v2 de base (CoquiTTS Trainer)
4. Sauvegarde du checkpoint et mise à jour du profil en base

Le fine-tuning s'exécute dans un thread dédié (non-bloquant).
Le statut est suivi via un registre en mémoire et persisté en base à la fin.
"""

import csv
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.core.config import MODELS_DIR
from backend.core.logger import logger

# Sample rate attendu par XTTS-v2 (22 050 Hz, différent du 16 kHz de Chatterbox)
_XTTS_SAMPLE_RATE = 22_050

# Registre des jobs en mémoire — perdu au redémarrage (normal, le statut DB persiste)
_jobs: dict[int, "FineTuneJob"] = {}
_lock = threading.Lock()


@dataclass
class FineTuneJob:
    """Représente un job de fine-tuning en cours d'exécution."""

    profile_id: int
    status: str = "pending"   # pending | transcribing | preparing | training | done | error | cancelled
    progress: float = 0.0     # 0.0 → 1.0
    message: str = "En attente…"
    error: str | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def start_finetune(profile_id: int, sample_paths: list[Path], db_factory) -> None:
    """
    Lance le fine-tuning XTTS-v2 en arrière-plan pour un profil donné.

    Args:
        profile_id: Identifiant du profil voix.
        sample_paths: Liste des fichiers WAV normalisés du profil.
        db_factory: Callable retournant un contexte de session SQLAlchemy.

    Raises:
        RuntimeError: Si un job est déjà en cours pour ce profil.
        ValueError: Si aucun sample n'est fourni.
    """
    if not sample_paths:
        raise ValueError("Au moins un sample est requis pour le fine-tuning.")

    with _lock:
        existing = _jobs.get(profile_id)
        if existing and existing.status in ("pending", "transcribing", "preparing", "training"):
            raise RuntimeError("Un job de fine-tuning est déjà en cours pour ce profil.")

        job = FineTuneJob(profile_id=profile_id)
        _jobs[profile_id] = job

    thread = threading.Thread(
        target=_run_finetune,
        args=(job, sample_paths, db_factory),
        daemon=True,
        name=f"finetune-{profile_id}",
    )
    job._thread = thread
    thread.start()
    logger.info("Job fine-tuning démarré — profil=%d — %d sample(s)", profile_id, len(sample_paths))


def get_status(profile_id: int) -> dict | None:
    """
    Retourne le statut courant du job de fine-tuning.

    Returns:
        Dictionnaire avec status, progress, message, error — ou None si pas de job.
    """
    job = _jobs.get(profile_id)
    if not job:
        return None
    return {
        "profile_id": job.profile_id,
        "status": job.status,
        "progress": round(job.progress, 3),
        "message": job.message,
        "error": job.error,
    }


def cancel_job(profile_id: int) -> bool:
    """
    Demande l'annulation du job en cours.

    L'annulation est coopérative : le thread vérifie _stop_event aux points clés.

    Returns:
        True si un job actif existait, False sinon.
    """
    job = _jobs.get(profile_id)
    if not job or job.status not in ("pending", "transcribing", "preparing", "training"):
        return False
    job._stop_event.set()
    logger.info("Annulation demandée — profil=%d", profile_id)
    return True


def delete_model(profile_id: int) -> bool:
    """
    Supprime le dossier du modèle fine-tuné sur disque.

    Returns:
        True si le dossier existait et a été supprimé.
    """
    model_dir = MODELS_DIR / f"profile_{profile_id}"
    if model_dir.exists():
        shutil.rmtree(model_dir)
        logger.info("Modèle fine-tuné supprimé — profil=%d", profile_id)
        return True
    return False


# ---------------------------------------------------------------------------
# Exécution du job (thread dédié)
# ---------------------------------------------------------------------------

def _run_finetune(job: FineTuneJob, sample_paths: list[Path], db_factory) -> None:
    """Point d'entrée du thread de fine-tuning."""
    try:
        # --- Étape 1 : Transcription ---
        job.status = "transcribing"
        job.message = "Transcription audio avec Whisper…"
        job.progress = 0.02

        transcripts = _transcribe(sample_paths, job)

        if job._stop_event.is_set():
            _mark_cancelled(job, profile_id=job.profile_id, db_factory=db_factory)
            return

        # --- Étape 2 : Préparation du dataset ---
        job.status = "preparing"
        job.message = "Préparation du dataset LJSpeech…"
        job.progress = 0.20

        dataset_dir = _prepare_dataset(job.profile_id, sample_paths, transcripts)

        if job._stop_event.is_set():
            _mark_cancelled(job, profile_id=job.profile_id, db_factory=db_factory)
            return

        # --- Étape 3 : Entraînement ---
        job.status = "training"
        job.message = "Fine-tuning XTTS-v2 en cours — cela peut prendre plusieurs heures…"
        job.progress = 0.25

        model_path = _train(job.profile_id, dataset_dir, job)

        if job._stop_event.is_set():
            _mark_cancelled(job, profile_id=job.profile_id, db_factory=db_factory)
            return

        # --- Terminé ---
        _persist_result(job.profile_id, str(model_path), "done", db_factory)

        job.status = "done"
        job.message = "Fine-tuning terminé avec succès ✓"
        job.progress = 1.0
        logger.info("Fine-tuning terminé — profil=%d modèle=%s", job.profile_id, model_path)

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.message = f"Erreur : {exc}"
        logger.error(
            "Erreur fine-tuning — profil=%d : %s",
            job.profile_id, exc, exc_info=True,
        )
        _persist_result(job.profile_id, None, "error", db_factory)


def _mark_cancelled(job: FineTuneJob, profile_id: int, db_factory) -> None:
    """Marque le job comme annulé et met à jour la base."""
    job.status = "cancelled"
    job.message = "Fine-tuning annulé"
    logger.info("Fine-tuning annulé — profil=%d", profile_id)
    _persist_result(profile_id, None, "cancelled", db_factory)


def _persist_result(
    profile_id: int,
    model_path: str | None,
    status: str,
    db_factory,
) -> None:
    """
    Met à jour le profil en base avec le résultat du fine-tuning.

    Ouvre une session dédiée pour le thread de fine-tuning.
    """
    try:
        from backend.models.voice_profile import VoiceProfile

        with db_factory() as db:
            profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
            if profile:
                profile.fine_tune_status = status
                if model_path:
                    profile.fine_tuned_model_path = model_path
                db.commit()
    except Exception as exc:
        logger.error(
            "Impossible de mettre à jour le profil %d en base : %s",
            profile_id, exc,
        )


# ---------------------------------------------------------------------------
# Transcription avec Whisper
# ---------------------------------------------------------------------------

def _transcribe(sample_paths: list[Path], job: FineTuneJob) -> list[str]:
    """
    Transcrit chaque sample avec Whisper (modèle medium, langue française).

    Returns:
        Liste de transcriptions dans le même ordre que sample_paths.

    Raises:
        RuntimeError: Si Whisper n'est pas installé.
    """
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Whisper n'est pas installé. Exécutez : pip install openai-whisper"
        ) from exc

    logger.info("Chargement du modèle Whisper (medium)…")
    model = whisper.load_model("medium")

    transcripts: list[str] = []
    n = len(sample_paths)

    for i, path in enumerate(sample_paths):
        if job._stop_event.is_set():
            break

        logger.info("Transcription %d/%d : %s", i + 1, n, path.name)
        result = model.transcribe(str(path), language="fr", fp16=True)
        text = result["text"].strip()
        transcripts.append(text)

        job.progress = 0.02 + (i + 1) / n * 0.16
        job.message = f"Transcription {i + 1}/{n}…"
        logger.debug("Résultat : %s → %.80s", path.name, text)

    return transcripts


# ---------------------------------------------------------------------------
# Préparation du dataset LJSpeech
# ---------------------------------------------------------------------------

def _prepare_dataset(
    profile_id: int,
    sample_paths: list[Path],
    transcripts: list[str],
) -> Path:
    """
    Prépare le dataset au format LJSpeech attendu par CoquiTTS.

    Structure créée :
        models/profile_{id}/dataset/
            wavs/          ← fichiers WAV 22 050 Hz mono
            metadata.csv   ← filename_sans_ext|transcription

    Args:
        profile_id: Identifiant du profil.
        sample_paths: Fichiers WAV source (16 kHz, normalisés).
        transcripts: Transcriptions correspondantes.

    Returns:
        Chemin du dossier dataset créé.

    Raises:
        RuntimeError: Si aucun sample valide (transcription non vide) n'est trouvé.
    """
    dataset_dir = MODELS_DIR / f"profile_{profile_id}" / "dataset"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []

    for i, (src_path, text) in enumerate(zip(sample_paths, transcripts)):
        if not text:
            logger.warning("Transcription vide pour %s — ignoré", src_path.name)
            continue

        dst_name = f"sample_{i:04d}.wav"
        dst_path = wavs_dir / dst_name

        # Chargement du WAV source (16 kHz mono normalisé)
        audio, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Rééchantillonnage vers 22 050 Hz (format attendu par XTTS-v2)
        if sr != _XTTS_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_XTTS_SAMPLE_RATE)

        sf.write(str(dst_path), audio, _XTTS_SAMPLE_RATE, subtype="PCM_16")
        # LJSpeech : nom sans extension dans metadata.csv
        rows.append((dst_name.replace(".wav", ""), text))

    if not rows:
        raise RuntimeError(
            "Aucun sample avec transcription valide. "
            "Vérifiez que les fichiers audio contiennent de la parole claire."
        )

    # Écriture du metadata.csv (séparateur pipe, sans en-tête)
    metadata_path = dataset_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for name, text in rows:
            writer.writerow([name, text])

    logger.info(
        "Dataset LJSpeech préparé — %d fichiers — %s",
        len(rows), dataset_dir,
    )
    return dataset_dir


# ---------------------------------------------------------------------------
# Localisation du modèle XTTS-v2 de base
# ---------------------------------------------------------------------------

def _get_xtts_base_checkpoint() -> Path:
    """
    Retourne le dossier du modèle XTTS-v2 téléchargé par CoquiTTS.

    Sur Windows : %USERPROFILE%\\AppData\\Local\\tts\\
    Sur Linux   : ~/.local/share/tts/

    Raises:
        RuntimeError: Si le modèle n'a pas encore été téléchargé.
    """
    # Priorité : variable d'environnement TTS_HOME
    tts_home = os.environ.get("TTS_HOME")
    if tts_home:
        base = Path(tts_home)
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "tts"
    else:
        base = Path.home() / ".local" / "share" / "tts"

    model_dir = base / "tts_models--multilingual--multi-dataset--xtts_v2"
    if not model_dir.exists():
        raise RuntimeError(
            "Modèle XTTS-v2 de base introuvable. "
            "Lancez d'abord une synthèse avec le moteur xttsv2, ou téléchargez-le manuellement :\n"
            "  python -c \"from TTS.api import TTS; "
            "TTS('tts_models/multilingual/multi-dataset/xtts_v2')\""
        )
    return model_dir


# ---------------------------------------------------------------------------
# Entraînement XTTS-v2
# ---------------------------------------------------------------------------

def _train(profile_id: int, dataset_dir: Path, job: FineTuneJob) -> Path:
    """
    Lance l'entraînement XTTS-v2 via le Trainer CoquiTTS.

    Configuration orientée "speaker adaptation" (pas full fine-tuning) :
    - 6 époques, lr=5e-6, batch=2, grad_accum=252
    - Convient pour 5–30 minutes d'audio
    - Durée estimée : 3–5h sur RTX 4090

    Args:
        profile_id: Identifiant du profil (pour isoler le dossier de sortie).
        dataset_dir: Dossier LJSpeech préparé par _prepare_dataset().
        job: Job courant (pour le suivi de progression et l'annulation).

    Returns:
        Chemin du dossier contenant le checkpoint fine-tuné.

    Raises:
        RuntimeError: Si CoquiTTS ou trainer ne sont pas installés.
    """
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts, XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from trainer import Trainer, TrainerArgs
    except ImportError as exc:
        raise RuntimeError(
            "CoquiTTS ou trainer non installé. Exécutez :\n"
            "  pip install TTS trainer"
        ) from exc

    output_dir = MODELS_DIR / f"profile_{profile_id}" / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = _get_xtts_base_checkpoint()

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=str(dataset_dir),
        language="fr",
    )

    audio_config = XttsAudioConfig(
        sample_rate=_XTTS_SAMPLE_RATE,
        dvae_sample_rate=_XTTS_SAMPLE_RATE,
        output_sample_rate=24000,
    )

    config = XttsConfig()
    config.epochs = 6
    config.output_path = str(output_dir)
    config.model_args = XttsArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
    )
    config.audio = audio_config
    config.batch_size = 2
    config.eval_batch_size = 2
    # num_loader_workers=0 requis sur Windows (pas de fork multiprocessing)
    config.num_loader_workers = 0
    config.eval_split_max_size = 256
    config.eval_split_size = 0.1
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 1000
    config.save_step = 10000
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.training_seed = 42
    config.use_phonemes = False
    config.lr = 5e-6
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = True
    config.optimizer_params = {
        "betas": [0.9, 0.96],
        "eps": 1e-8,
        "weight_decay": 1e-2,
    }
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {
        "milestones": [50000, 150000, 300000],
        "gamma": 0.5,
        "last_epoch": -1,
    }
    config.test_sentences = [
        {
            "text": "Bonjour, ceci est un test du système de clonage vocal.",
            "language": "fr",
        },
    ]

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    logger.info(
        "Entraînement XTTS-v2 — %d train / %d eval — checkpoint=%s",
        len(train_samples), len(eval_samples), checkpoint_dir,
    )

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(checkpoint_dir), use_deepspeed=False)
    model.cuda()

    # Suivi de progression : estimation linéaire sur le nombre de steps
    steps_per_epoch = max(len(train_samples) // config.batch_size, 1)
    total_steps = config.epochs * steps_per_epoch
    _step_counter: list[int] = [0]

    class _ProgressTracker:
        """Callback minimaliste pour mettre à jour la progression du job."""

        def on_train_step_end(self, trainer_obj) -> None:
            _step_counter[0] += 1
            epoch = _step_counter[0] // steps_per_epoch + 1
            raw = 0.25 + (_step_counter[0] / max(total_steps, 1)) * 0.70
            job.progress = min(0.95, raw)
            job.message = (
                f"Fine-tuning — époque {min(epoch, config.epochs)}/{config.epochs} "
                f"(step {_step_counter[0]}/{total_steps})…"
            )

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            grad_accum_steps=252,
        ),
        config,
        output_path=str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        callbacks=[_ProgressTracker()],
    )
    trainer.fit()

    logger.info("Entraînement terminé — sortie=%s", output_dir)
    return output_dir
