"""
Utilitaire tee : lit stdin et ecrit vers stdout ET vers un fichier.
Usage : commande 2>&1 | python _tee.py chemin/vers/fichier.log
"""

import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else "backend.log"

try:
    with open(log_path, "w", encoding="utf-8", buffering=1) as f:
        for line in sys.stdin:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
except KeyboardInterrupt:
    pass
