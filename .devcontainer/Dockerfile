FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS base

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

ENV MINICONDA_VERSION=latest
ENV CONDA_DIR=/opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean --all -y

ENV PATH=$CONDA_DIR/bin:$PATH

# ---------------------------
# Development Stage
# ---------------------------
FROM base AS development

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean --all -y

ENV PATH=$CONDA_DIR/envs/py312/bin:$PATH
