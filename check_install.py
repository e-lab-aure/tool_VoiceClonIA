"""
Vérification de l'installation VoiceClonIA.
Lance avec : python check_install.py
"""

import sys
import traceback

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []


def check(label, fn, verbose=False):
    try:
        result = fn()
        msg = f"  {OK}  {label}" + (f" — {result}" if result else "")
        results.append((True, msg))
        print(msg)
    except Exception as exc:
        msg = f"  {FAIL}  {label} — {exc}"
        results.append((False, msg))
        print(msg)
        if verbose:
            print("\n  ── Traceback complet ──")
            traceback.print_exc()
            print("  ───────────────────────\n")


print("\n══════════════════════════════════════════")
print("  VoiceClonIA — Vérification de l'install ")
print("══════════════════════════════════════════\n")

# --- Python ---
print("── Python")
check("Version Python", lambda: sys.version)

# --- GPU / CUDA ---
print("\n── GPU / CUDA")

def check_torch():
    import torch
    return f"version {torch.__version__}"

def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible — vérifiez votre installation NVIDIA")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{name} — {vram:.1f} Go VRAM — CUDA {torch.version.cuda}"

def check_torchaudio():
    import torchaudio
    return f"version {torchaudio.__version__}"

check("torch", check_torch)
check("CUDA disponible", check_cuda)
check("torchaudio", check_torchaudio)

# --- Deps Chatterbox ---
print("\n── Dépendances Chatterbox")

def check_numpy():
    import numpy as np
    return f"version {np.__version__}"

def check_transformers():
    import transformers
    return f"version {transformers.__version__}"

def check_diffusers():
    import diffusers
    return f"version {diffusers.__version__}"

def check_conformer():
    import conformer
    return "OK"

def check_omegaconf():
    import omegaconf
    return f"version {omegaconf.__version__}"

def check_safetensors():
    import safetensors
    return f"version {safetensors.__version__}"

def check_librosa():
    import librosa
    return f"version {librosa.__version__}"

def check_pyloudnorm():
    import pyloudnorm
    return "OK"

def check_pykakasi():
    import pykakasi
    return "OK"

def check_s3tokenizer():
    import s3tokenizer
    return "OK"

def check_resemble_perth():
    # Le package s'appelle resemble-perth mais le module peut varier
    try:
        import resemble_perth
        return "OK (resemble_perth)"
    except ImportError:
        import perth  # noqa: F401
        return "OK (perth)"

def check_spacy_pkuseg():
    import spacy_pkuseg
    return "OK"

check("numpy", check_numpy)
check("transformers", check_transformers)
check("diffusers", check_diffusers)
check("conformer", check_conformer)
check("omegaconf", check_omegaconf)
check("safetensors", check_safetensors)
check("librosa", check_librosa)
check("pyloudnorm", check_pyloudnorm)
check("pykakasi", check_pykakasi)
check("s3tokenizer", check_s3tokenizer)
check("resemble_perth", check_resemble_perth)
check("spacy_pkuseg", check_spacy_pkuseg)

# --- Chatterbox lui-même ---
print("\n── Chatterbox TTS")

def check_chatterbox_import():
    from chatterbox.tts import ChatterboxTTS
    return "import OK"

def check_chatterbox_load():
    import torch
    import perth
    # Patch DummyWatermarker si PerthImplicitWatermarker est None
    # (pkg_resources inaccessible sous Python 3.12 / setuptools 72+)
    if perth.PerthImplicitWatermarker is None:
        from perth.dummy_watermarker import DummyWatermarker
        perth.PerthImplicitWatermarker = DummyWatermarker
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)
    sr = model.sr
    return f"modèle chargé sur {device} — sample rate {sr} Hz"

check("chatterbox import", check_chatterbox_import)
print(f"  {WARN}  chatterbox chargement modèle (téléchargement au 1er lancement, peut prendre du temps)...")
check("chatterbox chargement modèle", check_chatterbox_load, verbose=True)

# --- Backend VoiceClonIA ---
print("\n── Backend FastAPI")

def check_fastapi():
    import fastapi
    return f"version {fastapi.__version__}"

def check_sqlalchemy():
    import sqlalchemy
    return f"version {sqlalchemy.__version__}"

def check_pydantic():
    import pydantic
    return f"version {pydantic.__version__}"

def check_soundfile():
    import soundfile
    return f"version {soundfile.__version__}"

check("fastapi", check_fastapi)
check("sqlalchemy", check_sqlalchemy)
check("pydantic", check_pydantic)
check("soundfile", check_soundfile)

# --- Résumé ---
total = len(results)
passed = sum(1 for ok, _ in results if ok)
failed = total - passed

print("\n══════════════════════════════════════════")
print(f"  Résultat : {passed}/{total} OK", end="")
if failed:
    print(f"  —  {failed} ÉCHEC(S)")
else:
    print("  —  Tout est bon !")
print("══════════════════════════════════════════\n")

if failed:
    print("  Packages manquants à installer :\n")
    for ok, msg in results:
        if not ok:
            pkg = msg.split("—")[0].replace(FAIL, "").strip()
            print(f"    pip install {pkg}")
    print()
    sys.exit(1)
