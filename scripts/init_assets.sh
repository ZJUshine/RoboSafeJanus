#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Helpers
info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }

# Asset source (ASSET_SOURCE):
#   modelscope (default): private dataset zjushine/RoboSafeJanus-Asset, which holds the
#       RoboDojo Assets plus the RoboSafeJanus changes (ARX X5 official joint limits,
#       eval layouts, extra objects). Needs `modelscope login --token <token>` first.
#   hf: upstream RoboDojo assets on Hugging Face (without the RoboSafeJanus changes):
#       hf://datasets/RoboDojo-Benchmark/RoboDojo/Assets/
ASSET_SOURCE="${ASSET_SOURCE:-modelscope}"
MS_REPO_ID="${MS_REPO_ID:-zjushine/RoboSafeJanus-Asset}"
MS_REVISION="${MS_REVISION:-master}"
# Local copy of the dataset, same layout as the repo (Assets/ + ext_assets/); the repo-root
# Assets and ext_assets are symlinks into it. scripts/internal/upload_assets.py uploads it back.
ASSET_ROOT="${ASSET_ROOT:-${CURRENT_DIR}/data/RoboSafeJanus-Asset}"
# Wrapped objects (GenieSim/ArtVIP/SimReady object.usd, metadata.json) and curobo.yml in that
# dataset hold absolute paths of the machine they were built on; after download they are
# rewritten to ASSET_ROOT. The old prefix is detected from the files themselves; set
# MS_SOURCE_PREFIX only to force a specific one.
MS_SOURCE_PREFIX="${MS_SOURCE_PREFIX:-}"

HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
HF_REVISION="${HF_REVISION:-main}"
HF_REPO_URL="${HF_REPO_URL:-https://huggingface.co/datasets/${HF_REPO_ID}}"

TARGET_DIR="${CURRENT_DIR}/Assets"
EXT_TARGET_DIR="${CURRENT_DIR}/ext_assets"
ASSET_CACHE_DIR="${CURRENT_DIR}/.cache/robodojo_assets_repo"
REQUIRED_ASSET_SUBDIRS=(Robots Object Material Eval_Layout)

assets_ready() {
  if [[ ! -d "${TARGET_DIR}" ]]; then
    return 1
  fi

  local subdir
  for subdir in "${REQUIRED_ASSET_SUBDIRS[@]}"; do
    if [[ ! -d "${TARGET_DIR}/${subdir}" ]]; then
      return 1
    fi
  done
}

check_download_tools() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda env list | grep -q "^RoboDojo "; then
      info "Activating conda environment 'RoboDojo'..."
      source "$HOME/miniconda3/bin/activate" RoboDojo 2>/dev/null || conda activate RoboDojo
    fi
  fi

  if [[ "${ASSET_SOURCE}" == "modelscope" ]]; then
    if ! command -v modelscope >/dev/null 2>&1; then
      error "modelscope CLI not found. Run: pip install -U \"modelscope>=1.40\""
    fi
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    error "git not found. Please install git first."
  fi

  if ! git lfs version >/dev/null 2>&1; then
    error "git-lfs not found. Please install git-lfs first."
  fi
}

clone_asset_repo() {
  info "Cloning sparse asset repo into '${ASSET_CACHE_DIR}'..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse "${HF_REPO_URL}" "${ASSET_CACHE_DIR}"
}

archive_asset_cache() {
  local cache_dir="$1"
  local partial_dir="${cache_dir}.partial.$(date +%Y%m%d_%H%M%S)"
  warn "Moving existing asset cache to '${partial_dir}'."
  mv "${cache_dir}" "${partial_dir}"
}

detect_source_prefixes() {
  # Absolute directory prefixes that sit in front of ext_assets/ or Assets/ inside the
  # dataset (e.g. /Users/xuancun/RoboSafeJanus/ or /mnt/.../RoboSafeJanus/). Prints one
  # unique prefix per line, so assets built on different machines are all handled.
  grep -rhoI --include='*.usd' --include='*.usda' --include='*.json' --include='*.yml' --include='*.yaml' \
      -E '/[^"@[:space:]]*/(ext_assets|Assets)/' "$@" 2>/dev/null \
    | sed -E 's#/(ext_assets|Assets)/$##' \
    | sort -u || true
}

rewrite_source_prefixes() {
  local new_prefix="$1"
  shift
  local -a prefixes=()
  local prefix

  if [[ -n "${MS_SOURCE_PREFIX}" ]]; then
    prefixes=("${MS_SOURCE_PREFIX%/}/")
  else
    info "Detecting absolute asset paths baked into the dataset..."
    while IFS= read -r prefix; do
      [[ -n "${prefix}" ]] && prefixes+=("${prefix}/")
    done < <(detect_source_prefixes "$@")
  fi

  if [[ ${#prefixes[@]} -eq 0 ]]; then
    info "No absolute asset paths found; nothing to rewrite."
    return 0
  fi

  for prefix in "${prefixes[@]}"; do
    # Idempotent: paths already pointing into the cache are left alone, even when the cache
    # sits under the source prefix.
    if [[ "${prefix}" == "${new_prefix}" ]]; then
      continue
    fi
    info "Rewriting '${prefix}' -> '${new_prefix}' in asset text files..."
    grep -rlIZ --include='*.usd' --include='*.usda' --include='*.json' --include='*.yml' --include='*.yaml' \
        -F "${prefix}" "$@" \
      | OLD="${prefix}" NEW="${new_prefix}" xargs -0 -r perl -pi -e \
          'BEGIN { $r = index($ENV{NEW}, $ENV{OLD}) == 0 ? substr($ENV{NEW}, length $ENV{OLD}) : "\0" }
           s/\Q$ENV{OLD}\E(?!\Q$r\E)/$ENV{NEW}/g'
  done
}

download_assets_modelscope() {
  info "ModelScope dataset: ${MS_REPO_ID} (revision=${MS_REVISION})"
  mkdir -p "${ASSET_ROOT}"
  # Assets/ plus ext_assets/ (originals referenced by wrapped objects).
  # Re-running resumes: files already present in --local_dir are skipped.
  if ! modelscope download --dataset "${MS_REPO_ID}" --revision "${MS_REVISION}" \
      --include "Assets/*" "ext_assets/*" --local_dir "${ASSET_ROOT}"; then
    error "ModelScope download failed. The dataset is private: run 'modelscope login --token <token>' with an account that has access, then re-run this script."
  fi

  # Point the baked-in absolute paths at ASSET_ROOT, which has the same Assets/ + ext_assets/
  # layout. The source prefix is detected from the files, so it works for any build machine.
  rewrite_source_prefixes "${ASSET_ROOT}/" "${ASSET_ROOT}/Assets" "${ASSET_ROOT}/ext_assets"
  ln -s "${ASSET_ROOT}/Assets" "${TARGET_DIR}"
  link_ext_assets
}

link_ext_assets() {
  # ext_assets/ (originals of the wrapped objects, integrate/ scripts) next to Assets.
  if [[ ! -d "${ASSET_ROOT}/ext_assets" ]]; then
    return 0
  fi
  if [[ -L "${EXT_TARGET_DIR}" ]]; then
    rm "${EXT_TARGET_DIR}"
  elif [[ -e "${EXT_TARGET_DIR}" ]]; then
    warn "'${EXT_TARGET_DIR}' is a real directory; leaving it alone (assets use ${ASSET_ROOT}/ext_assets)."
    return 0
  fi
  ln -s "${ASSET_ROOT}/ext_assets" "${EXT_TARGET_DIR}"
}

download_assets_hf() {
  info "HF repo: ${HF_REPO_ID} (revision=${HF_REVISION})"
  mkdir -p "$(dirname "${ASSET_CACHE_DIR}")"

  if [[ ! -d "${ASSET_CACHE_DIR}/.git" ]]; then
    clone_asset_repo
  else
    if [[ -n "$(git -C "${ASSET_CACHE_DIR}" config --get remote.origin.promisor || true)" ]]; then
      warn "Existing cache was created as a partial clone and may hit Hugging Face promisor fetch errors."
      archive_asset_cache "${ASSET_CACHE_DIR}"
      clone_asset_repo
    else
      info "Updating sparse asset repo cache..."
      if ! git -C "${ASSET_CACHE_DIR}" fetch --depth 1 origin "${HF_REVISION}"; then
        warn "Failed to update existing asset cache."
        archive_asset_cache "${ASSET_CACHE_DIR}"
        clone_asset_repo
      fi
    fi
  fi

  git -C "${ASSET_CACHE_DIR}" sparse-checkout set Assets
  git -C "${ASSET_CACHE_DIR}" checkout "${HF_REVISION}"

  info "Pulling only Assets/** LFS objects..."
  git -C "${ASSET_CACHE_DIR}" lfs install --local >/dev/null
  git -C "${ASSET_CACHE_DIR}" lfs pull --include="Assets/**" --exclude=""

  ln -s "${ASSET_CACHE_DIR}/Assets" "${TARGET_DIR}"
}

download_assets() {
  info "Repo root: ${CURRENT_DIR}"
  info "Assets target: ${TARGET_DIR}"
  info "Asset source: ${ASSET_SOURCE}"

  if assets_ready; then
    warn "'${TARGET_DIR}' already exists, skipping..."
    return 0
  fi

  if [[ -L "${TARGET_DIR}" ]]; then
    warn "Removing stale Assets symlink '${TARGET_DIR}' -> '$(readlink "${TARGET_DIR}")'."
    rm "${TARGET_DIR}"
  elif [[ -d "${TARGET_DIR}" ]]; then
    local partial_dir="${TARGET_DIR}.partial.$(date +%Y%m%d_%H%M%S)"
    warn "'${TARGET_DIR}' exists but is incomplete."
    warn "Moving incomplete directory to '${partial_dir}' before recreating Assets."
    mv "${TARGET_DIR}" "${partial_dir}"
  fi

  case "${ASSET_SOURCE}" in
    modelscope) download_assets_modelscope ;;
    hf) download_assets_hf ;;
    *) error "Unknown ASSET_SOURCE='${ASSET_SOURCE}' (expected modelscope or hf)." ;;
  esac
}

verify_assets() {
  if [[ ! -d "${TARGET_DIR}" ]]; then
    error "Expected '${TARGET_DIR}' after download, but it was not created."
  fi

  local subdir
  for subdir in "${REQUIRED_ASSET_SUBDIRS[@]}"; do
    if [[ ! -d "${TARGET_DIR}/${subdir}" ]]; then
      error "Asset subdir '${subdir}' missing under ${TARGET_DIR}; download incomplete."
    fi
  done
}

check_download_tools
download_assets
verify_assets

info "Assets directory is ready: ${TARGET_DIR}"
