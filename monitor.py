"""
VoiceClonIA — Monitor console
Tableau de bord temps reel : etat des serveurs, logs, redemarrage.

Usage : python monitor.py  (ou via monitor.bat)
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).resolve().parent
VENV_ACTIVATE  = PROJECT_ROOT / "venv" / "Scripts" / "activate.bat"
BACKEND_LOG    = PROJECT_ROOT / "backend.log"
CONTAINER_NAME = "voxtral-server"
BACKEND_HOST   = "127.0.0.1"
BACKEND_PORT   = 8000
VOXTRAL_HOST   = "127.0.0.1"
VOXTRAL_PORT   = 8001
REFRESH_S      = 5   # intervalle de rafraichissement auto (secondes)

# ---------------------------------------------------------------------------
# Couleurs ANSI
# ---------------------------------------------------------------------------

ESC   = "\033["
RESET = f"{ESC}0m"
BOLD  = f"{ESC}1m"
DIM   = f"{ESC}2m"
RED   = f"{ESC}91m"
GRN   = f"{ESC}92m"
YEL   = f"{ESC}93m"
BLU   = f"{ESC}94m"
CYN   = f"{ESC}96m"
WHT   = f"{ESC}97m"


def _enable_ansi() -> None:
    """Active le support ANSI sur la console Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            pass


def _clear() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


# ---------------------------------------------------------------------------
# Verifications de sante
# ---------------------------------------------------------------------------

def _check_http(host: str, port: int, path: str = "/", timeout: float = 2.0) -> tuple[bool, int | None]:
    """Retourne (ok, latence_ms) ou (False, None)."""
    try:
        import httpx
        t0 = time.perf_counter()
        r = httpx.get(f"http://{host}:{port}{path}", timeout=timeout)
        lat = int((time.perf_counter() - t0) * 1000)
        return r.status_code < 500, lat
    except Exception:
        return False, None


def _get_container_status() -> str:
    """Retourne le statut du conteneur Podman (ex: 'Up 3 hours', 'Exited', 'absent')."""
    try:
        out = subprocess.check_output(
            ["podman", "ps", "-a", "--filter", f"name={CONTAINER_NAME}",
             "--format", "{{.Status}}"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return out if out else "absent"
    except Exception:
        return "inconnu"


def _get_pid_on_port(port: int) -> str | None:
    """Retourne le PID du processus en ecoute sur le port, ou None."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and f":{port}" in parts[1] and parts[3] == "LISTENING":
                return parts[4]
    except Exception:
        pass
    return None


def _kill_pid(pid: str) -> bool:
    try:
        subprocess.run(["taskkill", "/PID", pid, "/F"],
                       capture_output=True, check=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

_LOGO = f"""{CYN}{BOLD}
  ╔══════════════════════════════════════════════════════╗
  ║          VoiceClonIA — Monitor                       ║
  ╚══════════════════════════════════════════════════════╝{RESET}"""

_SEP = f"  {DIM}{'─' * 54}{RESET}"


def _dot(ok: bool) -> str:
    return f"{GRN}●{RESET}" if ok else f"{RED}●{RESET}"


def _lbl(ok: bool, running_text: str = "RUNNING", stopped_text: str = "STOPPED") -> str:
    return f"{GRN}{running_text}{RESET}" if ok else f"{RED}{stopped_text}{RESET}"


def _container_lbl(status: str) -> str:
    sl = status.lower()
    if "up" in sl:
        return f"{GRN}{status}{RESET}"
    if sl in ("absent", "inconnu"):
        return f"{RED}{status.upper()}{RESET}"
    return f"{YEL}{status}{RESET}"


def _draw(be_ok: bool, be_lat: int | None,
          vx_ok: bool, vx_lat: int | None,
          cst: str, refreshed_at: datetime) -> None:

    be_pid = _get_pid_on_port(BACKEND_PORT)
    vx_pid = _get_pid_on_port(VOXTRAL_PORT)

    be_lat_str = f"  {DIM}{be_lat} ms{RESET}" if be_lat else ""
    vx_lat_str = f"  {DIM}{vx_lat} ms{RESET}" if vx_lat else ""
    be_pid_str = f"  {DIM}PID {be_pid}{RESET}" if be_pid else ""
    vx_pid_str = f"  {DIM}PID {vx_pid}{RESET}" if vx_pid else ""

    ts = f"{DIM}Rafraichi : {refreshed_at.strftime('%H:%M:%S')}{RESET}"

    print(_LOGO)
    print(f"  {ts}")
    print()
    print(f"  {BOLD}STATUT DES SERVEURS{RESET}")
    print(_SEP)
    print(f"  {_dot(be_ok)}  {WHT}FastAPI Backend   {RESET}  {_lbl(be_ok)}{be_lat_str}{be_pid_str}")
    print(f"       {DIM}http://127.0.0.1:{BACKEND_PORT}{RESET}")
    print(_SEP)
    print(f"  {_dot(vx_ok)}  {WHT}Voxtral vLLM 4B   {RESET}  {_lbl(vx_ok)}{vx_lat_str}{vx_pid_str}")
    print(f"       {DIM}http://127.0.0.1:{VOXTRAL_PORT}{RESET}")
    print(f"  {_dot('up' in cst.lower())}  {WHT}Container Podman  {RESET}  {_container_lbl(cst)}")
    print(_SEP)
    print()
    print(f"  {BOLD}LOGS{RESET}")
    print(f"    {WHT}[1]{RESET} Logs Backend (FastAPI)")
    print(f"    {WHT}[2]{RESET} Logs Voxtral (vLLM)")
    print(f"    {WHT}[3]{RESET} Les deux simultanement")
    print()
    print(f"  {BOLD}CONTROLE{RESET}")
    print(f"    {WHT}[4]{RESET} Redemarrer Backend")
    print(f"    {WHT}[5]{RESET} Redemarrer Voxtral (container)")
    print(f"    {WHT}[6]{RESET} Redemarrer tout le stack")
    print()
    print(_SEP)
    print(f"    {WHT}[R]{RESET} Rafraichir maintenant   {WHT}[Q]{RESET} Quitter")
    print()


# ---------------------------------------------------------------------------
# Actions sur les logs
# ---------------------------------------------------------------------------

def _open_window(title: str, command: str) -> None:
    """Ouvre une fenetre cmd independante qui reste ouverte."""
    subprocess.Popen(
        f'start "{title}" cmd /k "{command}"',
        shell=True,
    )


def _open_backend_logs() -> None:
    if not BACKEND_LOG.exists():
        BACKEND_LOG.write_text("", encoding="utf-8")

    cmd = (
        f"powershell -NoExit -Command "
        f"\"Get-Content -Path '{BACKEND_LOG}' -Wait -Tail 60\""
    )
    _open_window("VoiceClonIA — Backend logs", cmd)


def _open_voxtral_logs() -> None:
    _open_window(
        "VoiceClonIA — Voxtral logs",
        f"podman logs -f {CONTAINER_NAME}",
    )


# ---------------------------------------------------------------------------
# Actions de redemarrage
# ---------------------------------------------------------------------------

def _start_backend_window() -> None:
    """Lance uvicorn dans une fenetre dediee en ecrivant aussi dans backend.log."""
    root = str(PROJECT_ROOT)
    log  = str(BACKEND_LOG)
    inner = (
        f"cd /d {root} && "
        f"call venv\\Scripts\\activate.bat && "
        f"uvicorn backend.main:app --host {BACKEND_HOST} --port {BACKEND_PORT} 2>&1 | "
        f"powershell -noprofile -command \"$input | Tee-Object -FilePath '{log}'\""
    )
    _open_window("VoiceClonIA — Serveur web", inner)


def _restart_backend() -> None:
    print(f"\n  {YEL}Arret du backend...{RESET}")
    pid = _get_pid_on_port(BACKEND_PORT)
    if pid:
        if _kill_pid(pid):
            print(f"  {GRN}Processus PID={pid} arrete.{RESET}")
        else:
            print(f"  {YEL}Impossible d'arreter PID={pid} — le port est peut-etre libre.{RESET}")
    else:
        print(f"  {DIM}Aucun processus sur le port {BACKEND_PORT}.{RESET}")

    time.sleep(1)
    print(f"  {YEL}Demarrage du backend...{RESET}")
    _start_backend_window()
    print(f"  {GRN}Backend relance dans une nouvelle fenetre.{RESET}")


def _restart_voxtral() -> None:
    print(f"\n  {YEL}Redemarrage du container {CONTAINER_NAME}...{RESET}")
    try:
        subprocess.run(["podman", "restart", CONTAINER_NAME], check=True)
        print(f"  {GRN}Container redémarre avec succes.{RESET}")
    except subprocess.CalledProcessError as exc:
        print(f"  {RED}Erreur Podman : {exc}{RESET}")


def _restart_all() -> None:
    _restart_voxtral()
    time.sleep(1)
    _restart_backend()


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def _wait_key(timeout: float) -> str | None:
    """Attend une touche pendant `timeout` secondes. Retourne None si timeout."""
    if sys.platform != "win32":
        # Fallback UNIX (non utilise en pratique ici)
        import select
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.readline().strip().lower()
        return None

    import msvcrt
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            # getwch renvoie '\xe0' ou '\x00' pour les touches speciales — ignorer
            if ch in ("\xe0", "\x00"):
                msvcrt.getwch()  # consomme le second octet
                continue
            return ch.lower()
        time.sleep(0.05)
    return None


def main() -> None:
    _enable_ansi()

    be_ok    = be_lat = False, None
    vx_ok    = vx_lat = False, None
    cst      = "inconnu"
    last_ts  = datetime.now()

    def refresh() -> None:
        nonlocal be_ok, be_lat, vx_ok, vx_lat, cst, last_ts
        be_ok, be_lat = _check_http(BACKEND_HOST, BACKEND_PORT, "/")
        vx_ok, vx_lat = _check_http(VOXTRAL_HOST, VOXTRAL_PORT, "/health")
        cst = _get_container_status()
        last_ts = datetime.now()

    # Premier check au demarrage
    refresh()

    while True:
        _clear()
        _draw(be_ok, be_lat, vx_ok, vx_lat, cst, last_ts)

        key = _wait_key(REFRESH_S)

        if key is None:
            # Timeout → rafraichissement automatique
            refresh()
            continue

        key = key.lower()

        if key == "q":
            _clear()
            print(f"\n  {DIM}Monitor ferme. Au revoir.{RESET}\n")
            break

        elif key == "r":
            refresh()

        elif key == "1":
            _open_backend_logs()

        elif key == "2":
            _open_voxtral_logs()

        elif key == "3":
            _open_backend_logs()
            time.sleep(0.3)
            _open_voxtral_logs()

        elif key == "4":
            _clear()
            _restart_backend()
            time.sleep(3)
            refresh()

        elif key == "5":
            _clear()
            _restart_voxtral()
            time.sleep(3)
            refresh()

        elif key == "6":
            _clear()
            _restart_all()
            time.sleep(4)
            refresh()


if __name__ == "__main__":
    main()
