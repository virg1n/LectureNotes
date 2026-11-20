# Source this on the server:  source scripts/env.sh
# The home directory quota (40 GB) is full on the V100 node, so everything that writes
# (venv, pip/HF/uv caches, TinyTeX, figure cache, tmp) lives under $LN_ROOT on /public.
# NOTE: /public has no backups and deletes files not accessed for 30 days.
export LN_ROOT=${LN_ROOT:-/public/bsuvernev2_lecturenotes}
export VENV=${VENV:-$LN_ROOT/venv}
export HF_HOME=${HF_HOME:-$LN_ROOT/hf}
export PIP_CACHE_DIR=$LN_ROOT/.cache/pip
export UV_CACHE_DIR=$LN_ROOT/.cache/uv
export XDG_CACHE_HOME=$LN_ROOT/.cache
export LN_CACHE_DIR=$LN_ROOT/.cache/lecturenotes
export TRITON_CACHE_DIR=$LN_ROOT/.cache/triton
export VLLM_CACHE_ROOT=$LN_ROOT/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$LN_ROOT/.cache/inductor
export MPLCONFIGDIR=$LN_ROOT/.cache/mpl
export TMPDIR=$LN_ROOT/tmp
export TINYTEX_DIR=$LN_ROOT/tex
export PATH="$TINYTEX_DIR/.TinyTeX/bin/x86_64-linux:$PATH"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$LN_CACHE_DIR" "$TMPDIR" "$TINYTEX_DIR" "$MPLCONFIGDIR"
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
