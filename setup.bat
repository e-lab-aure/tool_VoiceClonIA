@echo off
:: ============================================================
:: Script d'installation VoiceClonIA
:: Environnement : RTX 4090 / CUDA 12.8 / PyTorch 2.8 (cu128)
:: ============================================================

setlocal enabledelayedexpansion

echo.
echo  ================================
echo   VoiceClonIA — Installation
echo  ================================
echo.

:: Vérification Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Python n'est pas installe ou absent du PATH.
    echo Installez Python 3.11+ depuis https://python.org
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% detecte

:: Vérification CUDA
nvcc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [AVERTISSEMENT] nvcc non trouve dans le PATH — CUDA pourrait ne pas etre accessible.
) else (
    for /f "tokens=5 delims= " %%c in ('nvcc --version 2^>^&1 ^| findstr "release"') do (
        set CUDA_VER=%%c
    )
    echo [OK] CUDA detecte
)

:: Création de l'environnement virtuel
if not exist venv (
    echo.
    echo [1/6] Creation de l'environnement virtuel...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERREUR] Echec creation venv
        exit /b 1
    )
    echo [OK] venv cree
) else (
    echo [OK] venv existant detecte — supprimez-le avec rmdir /s /q venv pour repartir de zero
)

:: Activation du venv
call venv\Scripts\activate.bat

:: Mise à jour de pip + setuptools + wheel
:: OBLIGATOIRE sous Python 3.12 : pkgutil.ImpImporter a ete supprime,
:: les anciennes versions de setuptools/pkg_resources plantent au build.
echo.
echo [2/6] Mise a jour de pip, setuptools et wheel...
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    echo [ERREUR] Echec mise a jour pip/setuptools/wheel
    exit /b 1
)
echo [OK] pip, setuptools et wheel mis a jour

:: Pre-installation de numpy compatible Python 3.12
:: numpy < 1.26 ne supporte pas Python 3.12 — on force une version recente
:: avant que chatterbox-tts ne tente de builder l'ancienne via ses metadata.
echo.
echo [3/6] Pre-installation de numpy compatible Python 3.12...
pip install "numpy>=1.26.4"
if %errorlevel% neq 0 (
    echo [ERREUR] Echec installation numpy
    exit /b 1
)
echo [OK] numpy installe

:: Installation de PyTorch cu128 EN PREMIER (requis par chatterbox)
echo.
echo [4/6] Installation de PyTorch 2.x + CUDA 12.8...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo [ERREUR] Echec installation PyTorch cu128
    echo Verifiez votre connexion et que CUDA 12.8 est installe.
    exit /b 1
)
echo [OK] PyTorch installe avec support CUDA 12.8

:: Vérification que PyTorch voit bien CUDA
echo.
python -c "import torch; cuda_ok = torch.cuda.is_available(); print('[OK] CUDA disponible via PyTorch :', torch.cuda.get_device_name(0) if cuda_ok else 'NON - fallback CPU')"

:: Installation de Chatterbox
:: --no-deps contourne la contrainte numpy<1.26.0 de chatterbox-tts 0.1.6
:: (pas de wheel numpy 1.25 pour Python 3.12 — le build source plante).
:: numpy 1.26.x est entierement compatible en API avec le code 1.24-1.25.
echo.
echo [5/6] Installation de Chatterbox TTS (sans resolution de deps numpy)...
pip install chatterbox-tts --no-deps
if %errorlevel% neq 0 (
    echo [ERREUR] Echec installation chatterbox-tts
    exit /b 1
)
echo [OK] Chatterbox installe

:: ---- Dependances de Chatterbox — installes en 3 groupes ----
::
:: Groupe A : packages sans conflit torch/numpy — installation normale
echo      [A] Deps sans conflit torch/numpy...
pip install "omegaconf" "pykakasi==2.3.0" "safetensors==0.5.3" "spacy-pkuseg" "pyloudnorm" "librosa==0.11.0"
if %errorlevel% neq 0 (
    echo [AVERTISSEMENT] Groupe A : echec partiel — continuons
)

:: Groupe B : packages avec torch==2.6.0 ou numpy<1.26 dans leurs metadata
:: --no-deps evite que pip tente de downgrader torch/numpy
:: (torch 2.10.0 et numpy 1.26.4 sont fonctionnellement compatibles)
echo      [B] Deps avec contraintes torch/numpy (--no-deps)...
pip install --no-deps "conformer==0.3.2" "diffusers==0.29.0" "resemble-perth==1.0.1" "s3tokenizer" "transformers==4.46.3"
if %errorlevel% neq 0 (
    echo [AVERTISSEMENT] Groupe B : echec partiel — continuons
)

:: Groupe C : sous-deps de transformers et diffusers non couverts par --no-deps
echo      [C] Sous-deps de transformers et diffusers...
pip install "tokenizers>=0.19" "huggingface-hub>=0.23" "accelerate>=0.26" "regex" "tqdm" "filelock" "requests" "einops"
if %errorlevel% neq 0 (
    echo [AVERTISSEMENT] Groupe C : echec partiel — continuons
)

:: Installation des dépendances de base
echo.
echo [6/6] Installation des dependances backend...
pip install -r backend\requirements-base.txt
if %errorlevel% neq 0 (
    echo [ERREUR] Echec installation requirements-base.txt
    exit /b 1
)
echo [OK] Dependances backend installees

:: Création du .env si absent
if not exist .env (
    copy .env.example .env >nul
    echo [OK] Fichier .env cree depuis .env.example
) else (
    echo [OK] .env existant conserve
)

echo.
echo  ================================
echo   Installation terminee !
echo  ================================
echo.
echo  Pour demarrer le serveur :
echo    venv\Scripts\activate
echo    uvicorn backend.main:app --reload
echo.
echo  Documentation API : http://127.0.0.1:8000/docs
echo.

endlocal
