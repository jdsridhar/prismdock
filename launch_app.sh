#!/usr/bin/env bash
# Launch the PrismDock Streamlit dashboard.
#
# First time on this machine? You need conda installed and the `docking`
# environment created. Run ./setup_env.sh once, then use this script from then on.
set -e
cd "$(dirname "$0")"

# Find a conda installation in the usual places.
CONDA_SH=""
for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda" "/opt/miniconda3"; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$base/etc/profile.d/conda.sh"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    echo "No conda installation found on this machine (checked miniconda3/anaconda3/miniforge3/opt-conda)."
    echo "Run ./setup_env.sh first — it installs Miniconda and creates the 'docking' environment."
    exit 1
fi

source "$CONDA_SH"

if ! conda env list | grep -q "^docking "; then
    echo "The 'docking' conda environment does not exist yet."
    echo "Run ./setup_env.sh once to create it from environment.yml, then re-run this script."
    exit 1
fi

conda activate docking
streamlit run app.py --server.port 8502
