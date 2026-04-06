# ============================================================
# Voxtral-4B-TTS-2603 - vLLM Omni server
#
# Prerequis hote :
#   nvidia-container-toolkit installe et CDI genere :
#     sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
#
# Build :
#   podman build -t voxtral-server .
#
# Run (CDI - recommande) :
#   podman run --device nvidia.com/gpu=all \
#     -p 127.0.0.1:8001:8001 \
#     -e HF_TOKEN=hf_xxx \
#     -v voxtral-cache:/cache \
#     voxtral-server
#
# Run sans CDI (legacy) :
#   podman run --security-opt=label=disable \
#     --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
#     -p 127.0.0.1:8001:8001 \
#     -e HF_TOKEN=hf_xxx \
#     -v voxtral-cache:/cache \
#     voxtral-server
#
# Surcharger le modele ou les flags :
#   podman run ... voxtral-server mistralai/autre-modele --port 8001
# ============================================================

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.title="Voxtral vLLM Omni Server" \
      org.opencontainers.image.description="Voxtral-4B-TTS-2603 via vLLM Omni (CUDA 12.8)"

# ------------------------------------------------------------------ env
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # venv actif pour toutes les couches suivantes
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    # Cache HuggingFace dans le volume
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

# ------------------------------------------------------------------ dependances systeme
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        git \
        build-essential \
        cmake \
        ninja-build \
        libsndfile1 \
        sox \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ venv + pip
RUN python3.12 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------ PyTorch CUDA 12.8
# Installe en premier pour eviter les conflits de version avec vllm
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------------ vllm-omni
# Clone dans /opt pour que le chemin du stage-configs soit previsible
RUN git clone --depth=1 https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni

# vllm peut etre declare en dependance de vllm-omni mais avec une version non publiee.
# On l'installe d'abord pour garantir la version cu128.
RUN pip install vllm --no-build-isolation || true
RUN pip install setuptools_scm
RUN pip install -e /opt/vllm-omni --no-build-isolation

# ------------------------------------------------------------------ utilisateur non-root
# Le runtime n'a pas besoin des privileges root.
# UID 1000 est conventionnel et compatible avec le user namespace de Podman.
RUN useradd -m -u 1000 -s /bin/bash -d /home/vllm vllm \
    && chown -R vllm:vllm /opt/vllm-omni

USER vllm
WORKDIR /home/vllm

# ------------------------------------------------------------------ volume cache
# Le modele (~8 Go) est stocke dans ce volume pour ne pas etre re-telecharge.
# Monter avec : -v voxtral-cache:/cache
VOLUME ["/cache"]

# ------------------------------------------------------------------ reseau
# Expose uniquement l'API vLLM ; la liaison 127.0.0.1 est geree au run.
EXPOSE 8001

# ------------------------------------------------------------------ health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

# ------------------------------------------------------------------ entrypoint
# ENTRYPOINT = binaire fixe, CMD = arguments surchargeables au run.
ENTRYPOINT ["vllm", "serve"]

CMD [ \
    "mistralai/Voxtral-4B-TTS-2603", \
    "--omni", \
    "--stage-configs-path", "/opt/vllm-omni/vllm_omni/model_executor/stage_configs/voxtral_tts.yaml", \
    "--host", "0.0.0.0", \
    "--port", "8001", \
    "--trust-remote-code", \
    "--enforce-eager" \
]
