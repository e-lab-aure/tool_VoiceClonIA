@echo off
:: ============================================================
:: Script de demarrage VoiceClonIA
:: Lance le conteneur Voxtral, attend qu'il soit pret,
:: ouvre les fenetres de logs, puis demarre le serveur web.
:: ============================================================

setlocal enabledelayedexpansion

echo.
echo  ================================
echo   VoiceClonIA — Demarrage
echo  ================================
echo.

:: --- Verifier que Podman est accessible ---
podman --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Podman n'est pas installe ou absent du PATH.
    pause
    exit /b 1
)

:: --- Etat du conteneur Voxtral ---
echo [1/4] Verification du conteneur Voxtral...

podman ps --filter "name=voxtral-server" --format "{{.Status}}" 2>nul | findstr /i "Up" >nul
if %errorlevel% == 0 (
    echo [OK] Conteneur voxtral-server deja en cours d'execution
    goto :wait_health
)

:: Conteneur existe mais est stoppe ?
podman ps -a --filter "name=voxtral-server" --format "{{.Status}}" 2>nul | findstr /i "Exited\|Created\|stopped" >nul
if %errorlevel% == 0 (
    echo [INFO] Conteneur voxtral-server stoppe — redemarrage...
    podman start voxtral-server >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Conteneur redémarre
        goto :wait_health
    )
    echo [AVERTISSEMENT] Echec du redemarrage — tentative de recreation...
)

:: Conteneur inexistant — creer avec le token
echo.
echo [INFO] Conteneur voxtral-server introuvable — creation necessaire.
echo.
set /p HF_TOKEN=Entrez votre token HuggingFace (hf_...) :

if "!HF_TOKEN!" == "" (
    echo [ERREUR] Token HuggingFace requis pour telecharger Voxtral.
    pause
    exit /b 1
)

echo.
echo [INFO] Creation du conteneur voxtral-server...
podman run -d ^
    --network=host ^
    --device nvidia.com/gpu=all ^
    -e HF_TOKEN=!HF_TOKEN! ^
    -v voxtral-cache:/cache ^
    --name voxtral-server ^
    localhost/voxtral-server:latest

if %errorlevel% neq 0 (
    echo [ERREUR] Impossible de creer le conteneur voxtral-server.
    echo Verifiez que l'image localhost/voxtral-server:latest existe.
    echo Si besoin, construisez-la avec : podman build -t voxtral-server .
    pause
    exit /b 1
)
echo [OK] Conteneur cree et demarre

:: --- Attendre que Voxtral soit pret ---
:wait_health
echo.
echo [2/4] Attente de la disponibilite de Voxtral sur le port 8001...
echo [INFO] Le premier demarrage peut durer plusieurs minutes (chargement du modele).
echo.

set RETRIES=0
set MAX_RETRIES=120

:health_loop
set /a RETRIES+=1
if !RETRIES! gtr !MAX_RETRIES! (
    echo.
    echo [ERREUR] Voxtral n'a pas repondu apres !MAX_RETRIES! tentatives.
    echo Consultez les logs : podman logs voxtral-server
    pause
    exit /b 1
)

curl -sf http://127.0.0.1:8001/health >nul 2>&1
if %errorlevel% == 0 goto :health_ok

:: Affichage de la progression toutes les 5 secondes
set /a MOD=!RETRIES! %% 5
if !MOD! == 0 (
    echo [ATTENTE] !RETRIES!/!MAX_RETRIES! — Voxtral pas encore pret...
)

timeout /t 1 /nobreak >nul
goto :health_loop

:health_ok
echo [OK] Voxtral est operationnel sur http://127.0.0.1:8001

:: --- Ouvrir la fenetre de logs Voxtral ---
echo.
echo [3/4] Ouverture de la fenetre de logs Voxtral...
start "Voxtral — vLLM logs" cmd /k "echo  [Voxtral] Logs du serveur vLLM — fermez cette fenetre pour arreter le suivi && echo. && podman logs -f voxtral-server"
echo [OK] Fenetre logs Voxtral ouverte

:: Petit delai pour eviter le chevauchement des fenetres
timeout /t 1 /nobreak >nul

:: --- Verifier que le venv existe ---
if not exist "%~dp0venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel introuvable.
    echo Lancez setup.bat d'abord pour installer les dependances.
    pause
    exit /b 1
)

:: --- Ouvrir la fenetre du serveur web FastAPI ---
:: Les logs sont affiches en console ET ecrits dans backend.log (pour monitor.bat)
echo.
echo [4/4] Demarrage du serveur web VoiceClonIA...
start "VoiceClonIA — Serveur web" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && uvicorn backend.main:app --host 127.0.0.1 --port 8000 2>&1 | powershell -noprofile -command \"$input | Tee-Object -FilePath '%~dp0backend.log'\""
echo [OK] Fenetre serveur web ouverte

:: --- Resume ---
echo.
echo  ================================
echo   VoiceClonIA demarre !
echo  ================================
echo.
echo   API FastAPI  : http://127.0.0.1:8000
echo   Docs Swagger : http://127.0.0.1:8000/docs
echo   Voxtral vLLM : http://127.0.0.1:8001
echo.
echo  Deux fenetres de log ont ete ouvertes.
echo  Vous pouvez fermer cette fenetre.
echo.

endlocal
