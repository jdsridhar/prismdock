#!/usr/bin/env bash
# One-time setup: install Miniconda (if missing) and create the `docking`
# conda environment from environment.yml.
#
# Run this ONCE on a fresh machine, then use ./launch_app.sh every time after.
set -e
cd "$(dirname "$0")"

CONDA_SH=""
for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda" "/opt/miniconda3"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$base/etc/profile.d/conda.sh"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    echo "== No conda found. Installing Miniconda to \$HOME/miniconda3 =="
    TMP_INSTALLER="/tmp/miniconda_installer.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$TMP_INSTALLER"
    bash "$TMP_INSTALLER" -b -p "$HOME/miniconda3"
    rm -f "$TMP_INSTALLER"
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
fi

source "$CONDA_SH"

if conda env list | grep -q "^docking "; then
    echo "== 'docking' environment already exists. Updating from environment.yml =="
    conda env update -n docking -f environment.yml --prune
else
    echo "== Creating 'docking' environment from environment.yml (this takes a few minutes) =="
    conda env create -f environment.yml
fi

# P2Rank (apo-structure pocket prediction) is not distributed on conda, so
# it's fetched from its GitHub release directly. Preferred over fpocket for
# apo targets -- see README for the benchmark rationale -- with fpocket kept
# as an automatic fallback if this download is skipped or fails (e.g. no
# internet on the event machine, or Java missing).
P2RANK_VERSION="2.5.1"
if [ -x "vendor/p2rank/prank" ]; then
    echo "== P2Rank already installed at vendor/p2rank =="
elif ! command -v java >/dev/null 2>&1; then
    echo "== Java not found -- skipping P2Rank install (fpocket will be used instead) =="
else
    echo "== Installing P2Rank ${P2RANK_VERSION} to vendor/p2rank =="
    mkdir -p vendor
    TMP_TGZ="/tmp/p2rank_${P2RANK_VERSION}.tar.gz"
    if wget -q "https://github.com/rdk/p2rank/releases/download/${P2RANK_VERSION}/p2rank_${P2RANK_VERSION}.tar.gz" -O "$TMP_TGZ"; then
        tar xzf "$TMP_TGZ" -C vendor
        rm -f "$TMP_TGZ"
        mv "vendor/p2rank_${P2RANK_VERSION}" vendor/p2rank
        echo "== P2Rank installed =="
    else
        echo "== P2Rank download failed -- fpocket will be used instead =="
        rm -f "$TMP_TGZ"
    fi
fi

echo ""
echo "Setup complete. Launch the dashboard with:  ./launch_app.sh"

# --- Optional: DiffDock (blind-pocket sanity check) --------------------
# Not auto-installed here: it needs a CUDA GPU, a separate ~7GB conda env
# (DiffDock pins old torch/torch-geometric/numpy/scipy versions that would
# conflict with the `docking` env's own rdkit/numpy/scipy), and downloads
# (~130MB model checkpoint + ~2.5GB ESM2 language-model checkpoint on first
# run). Skip if no GPU or disk is tight -- run_dockathon.py already handles
# it being absent gracefully (FileNotFoundError -> skip, CPU consensus
# unaffected). To install, on a machine with a GPU and ~10GB free:
#
#   mkdir -p vendor && cd vendor
#   git clone --depth 1 https://github.com/gcorso/DiffDock.git
#   conda create -y -n diffdock python=3.9.18
#   conda activate diffdock
#   pip install --extra-index-url https://download.pytorch.org/whl/cu117 torch==1.13.1+cu117
#   pip install --find-links https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html \
#       torch-cluster==1.6.0+pt113cu117 torch-geometric==2.2.0 \
#       torch-scatter==2.1.0+pt113cu117 torch-sparse==0.6.16+pt113cu117 \
#       torch-spline-conv==1.2.1+pt113cu117
#   pip install e3nn==0.5.1 fair-esm "networkx==2.8.4" "pandas==1.5.1" \
#       pybind11==2.11.1 "rdkit==2022.03.3" pytorch-lightning==1.9.5 \
#       torchmetrics==0.11.0 "prody==2.4.1" pyyaml
#   # (openfold and dllogger from DiffDock's own environment.yml are NOT
#   # needed -- neither is imported anywhere in inference.py's actual code
#   # path when a protein_path/PDB is supplied, which run_dockathon.py
#   # always does; skipping them avoids openfold's fragile from-source
#   # CUDA build entirely.)
#
# Then set, before running the pipeline:
#   export DIFFDOCK_REPO=$(pwd)/vendor/DiffDock          # (from repo root)
#   # DIFFDOCK_PYTHON only needed if the env isn't at the default
#   # ~/miniconda3/envs/diffdock/bin/python3 rescore_gpu.py looks for:
#   # export DIFFDOCK_PYTHON=/path/to/envs/diffdock/bin/python3
