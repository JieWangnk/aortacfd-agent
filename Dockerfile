# ============================================================================
# aortacfd-agent  —  LLM agent + OpenFOAM 12 CFD pipeline
# ============================================================================
# Multi-stage build:
#   Stage 1 (of12-base):  OpenFOAM 12 + system libs
#   Stage 2 (wk-build):   Compile Windkessel boundary condition
#   Stage 3 (runtime):    Python app + compiled OF libs
#
# Build:
#   docker build -t aortacfd-agent .
#
# Run (dry-run, no API key needed):
#   docker run --rm aortacfd-agent run \
#     --case /app/external/aortacfd-app/cases_input/BPM120 \
#     --clinical-text "65yo male, aortic coarctation, HR 72, BP 140/85" \
#     --output /app/output/BPM120_demo
#
# Run (with Claude):
#   docker run --rm -e ANTHROPIC_API_KEY=sk-... aortacfd-agent run \
#     --backend anthropic --model claude-sonnet-4-20250514 \
#     --case /app/external/aortacfd-app/cases_input/BPM120 \
#     --clinical-text "65yo male, aortic coarctation" \
#     --output /app/output/BPM120_prod
#
# Run (with local Ollama on host):
#   docker run --rm --network host aortacfd-agent run \
#     --backend ollama --model qwen2.5:7b-instruct \
#     --case /app/external/aortacfd-app/cases_input/BPM120 \
#     --clinical-text "5yo with coarctation" \
#     --output /app/output/BPM120_ollama
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: OpenFOAM 12 Foundation on Ubuntu 22.04
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS of12-base

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (OpenFOAM build + Python + mesh tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget ca-certificates git gnupg \
    build-essential cmake flex libfl-dev bison zlib1g-dev \
    libboost-system-dev libboost-thread-dev \
    libopenmpi-dev openmpi-bin \
    libfftw3-dev libscotch-dev libptscotch-dev \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libgl1-mesa-glx libglu1-mesa libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install OpenFOAM 12 from openfoam.org apt repository
# Write sources list directly (avoids add-apt-repository Python/apt_pkg conflict)
RUN wget -qO- https://dl.openfoam.org/gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/openfoam.gpg \
    && echo "deb http://dl.openfoam.org/ubuntu jammy main" > /etc/apt/sources.list.d/openfoam.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends openfoam12 \
    && rm -rf /var/lib/apt/lists/*

# OpenFOAM environment for all subsequent RUN commands
ENV WM_PROJECT_DIR=/opt/openfoam12
ENV FOAM_INST_DIR=/opt
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# Stage 2: Compile Windkessel boundary condition
# ---------------------------------------------------------------------------
FROM of12-base AS wk-build

RUN source /opt/openfoam12/etc/bashrc \
    && git clone https://github.com/JieWangnk/OpenFOAM-WK.git /tmp/OpenFOAM-WK \
    && cd /tmp/OpenFOAM-WK/src/modularWKPressure \
    && wmake libso \
    && echo "Windkessel compiled: $(ls $FOAM_USER_LIBBIN/libmodularWKPressure.so)"

# ---------------------------------------------------------------------------
# Stage 3: Runtime image
# ---------------------------------------------------------------------------
FROM of12-base AS runtime

LABEL maintainer="Jie Wang <jieandwang@gmail.com>"
LABEL description="AortaCFD agent: clinical text -> patient-specific aortic CFD"
LABEL version="0.1.0"

# Copy compiled Windkessel library from build stage.
# FOAM_USER_LIBBIN = /root/OpenFOAM/-12/platforms/linux64GccDPInt32Opt/lib
# We resolve the path at build time since COPY doesn't support globs with *.
RUN source /opt/openfoam12/etc/bashrc && mkdir -p "$FOAM_USER_LIBBIN"
COPY --from=wk-build /root/OpenFOAM/ /root/OpenFOAM/
# Keep only the .so we need, drop the rest of the build tree
RUN source /opt/openfoam12/etc/bashrc \
    && ls "$FOAM_USER_LIBBIN/libmodularWKPressure.so" \
    && echo "Windkessel library OK"

# AWS CLI v2 (needed for Batch entrypoint S3 sync; lightweight, ~60MB)
RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && wget -q "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -O /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) ---
# Copy only dependency specs first for better Docker layer caching
COPY external/aortacfd-app/pyproject.toml /app/external/aortacfd-app/pyproject.toml
COPY external/aortacfd-app/src/ /app/external/aortacfd-app/src/
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/

# Install AortaCFD-app core + agent with all optional deps
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       -e /app/external/aortacfd-app \
    && python -m pip install --no-cache-dir \
       -e /app[all]

# --- Application code ---
COPY external/aortacfd-app/ /app/external/aortacfd-app/
COPY . /app/

# Smoke test: agent CLI loads, OpenFOAM available
RUN aortacfd-agent version \
    && bash -c "source /opt/openfoam12/etc/bashrc && which foamRun"

# Source OpenFOAM in every shell (for subprocess calls to foamRun, etc.)
RUN echo "source /opt/openfoam12/etc/bashrc" >> /etc/bash.bashrc \
    && echo "source /opt/openfoam12/etc/bashrc" >> /root/.bashrc

# Output directory (mount a volume here for persistence)
VOLUME ["/app/output"]

ENTRYPOINT ["bash", "-c", "source /opt/openfoam12/etc/bashrc && exec aortacfd-agent \"$@\"", "--"]
CMD ["--help"]
