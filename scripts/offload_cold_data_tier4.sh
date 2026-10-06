#!/bin/bash
# Tier 4 cold-data offload: regenerable clones and unreferenced exports.
#
# Deliberately excluded, each verified rather than assumed:
#   cache-symlink-temp (135G)   ~/.cache symlinks into it; GNOME holds files
#                               open inside it. Live.
#   xwing-offload/lmstudio (122G) same inode as /home/scott/.lmstudio/models --
#                               it is where LM Studio serves from. Live.
#   models/AuK (18G)            /home/scott/AuK/ckpts is a symlink INTO it. Live.
#   amd-npu-models (125G)       Ornich NPU weights; referenced by
#                               portfolio-management and k2-mova. Active track.
#   scratch/dd/u (17G)          referenced by k2-mova k2/research/patch_static_seq.py
#                               and NPU-RUNNING.md, plus holds a symlink CIFS
#                               cannot represent. Active.
#   k2-mova (28G), npu-xclbins  active NPU track, modified within 14 days.
#   ryzen-ai-sdk (7.4G)         no remote to restore from; leaving in place.
#   finetune-venv, venvs, ryzen-ai-venv   in use by the running trainer.
#
# Everything below is either a clean clone of a public remote, an unreferenced
# export, or my own scratch. Git dirs move with their .git intact, so they stay
# working repositories at the new path; each is listed in RESTORE below.
set -uo pipefail

DATA=/media/scott/data
DEST=/nas/archive/cold-data
LOG=/media/scott/data/fleet-power/nas-offload4.log
MIN_FREE_KB=$(( 30 * 1024 * 1024 ))

log() { echo "[nas-offload4] $(date '+%F %T') $*" >>"$LOG"; }

mkdir -p "$(dirname "$LOG")" "$DEST" 2>/dev/null || true
if [ -d "$LOG" ]; then
  echo "ERROR: $LOG is a directory, refusing to run" >&2
  exit 1
fi

avail_kb=$(df -Pk /nas 2>/dev/null | awk 'NR==2{print $4}')
if [ -z "${avail_kb:-}" ] || [ "$avail_kb" -lt "$MIN_FREE_KB" ]; then
  log "SKIP: /nas unavailable or below 30G headroom"
  exit 0
fi

move_one() {   # src, dest, label
  local src="$1" dst="$2" label="$3"
  [ -e "$src" ] || { log "SKIP $label: source missing"; return 0; }
  if tr '\0' ' ' </proc/*/cmdline 2>/dev/null | grep -q -- "$src"; then
    log "SKIP $label: referenced by a running process"
    return 0
  fi
  local srcarg="$src" copy_links=""
  if [ -d "$src" ]; then
    local ext_link
    ext_link=$(find "$src" -type l 2>/dev/null | while read -r l; do
      t=$(readlink -f "$l" 2>/dev/null)
      case "$t" in
        "$src"/*|"$src") ;;
        *) echo "$l -> $t" ;;
      esac
    done | head -1)
    if [ -n "$ext_link" ]; then
      log "SKIP $label: symlink escapes the tree ($ext_link)"
      return 0
    fi
    # Every link resolves inside the tree (CMake build .so versioning, mostly),
    # so materialising them as real files loses nothing. CIFS cannot store links.
    if [ -n "$(find "$src" -type l 2>/dev/null | head -1)" ]; then
      copy_links="--copy-links"
      log "  $label: copying with --copy-links (internal symlinks materialised)"
    fi
    srcarg="$src/"
  elif [ -d "$dst" ]; then
    log "  removing stale directory at destination $dst"
    rm -rf "$dst"
  fi
  mkdir -p "$(dirname "$dst")" 2>/dev/null || true
  log "copy $label ..."
  if ! rsync -a $copy_links --partial --no-p --no-o --no-g --timeout=3600 \
        "$srcarg" "$dst" 2>>"$LOG"; then
    log "FAIL copy $label: source kept"
    return 1
  fi
  local sf df_ sb db
  if [ -d "$src" ]; then
    sf=$(find "$src" -type f | wc -l); df_=$(find "$dst" -type f | wc -l)
    sb=$(du -sb "$src" | cut -f1);       db=$(du -sb "$dst" | cut -f1)
  else
    sf=1; df_=$([ -f "$dst" ] && echo 1 || echo 0)
    sb=$(stat -c %s "$src")
    db=$([ -f "$dst" ] && stat -c %s "$dst" || echo 0)
  fi
  if [ "$sf" != "$df_" ] || [ "$sb" != "$db" ]; then
    log "FAIL verify $label: src $sf/$sb vs dst $df_/$db -- source kept"
    return 1
  fi
  rm -rf "$src"
  log "moved $label ($sf files, $((sb/1024/1024))MB)"
  return 0
}

moved=0
# Clean clones of public remotes (working repos at the new path)
move_one "$DATA/git/vaip"    "$DEST/git/vaip"    "git/vaip (amd/vaip)"       && moved=$((moved+1))
move_one "$DATA/git/RuView"  "$DEST/git/RuView"  "git/RuView (ruvnet/RuView)" && moved=$((moved+1))
move_one "$DATA/git/Muse-Glimmer-30B-ROCmFP4-Strix-Halo-DFlash-GGUF" \
         "$DEST/git/Muse-Glimmer-30B-ROCmFP4-Strix-Halo-DFlash-GGUF" \
         "git/Muse-Glimmer (HF kingjones777)" && moved=$((moved+1))
# Unreferenced exports
move_one "$DATA/xwing-offload/Downloads" "$DEST/xwing-Downloads" "xwing-offload/Downloads" && moved=$((moved+1))
move_one "$DATA/onnx-models"             "$DEST/onnx-models"     "onnx-models"              && moved=$((moved+1))
move_one "$DATA/npu-test-qwen-onnx"      "$DEST/npu-test-qwen-onnx" "npu-test-qwen-onnx"    && moved=$((moved+1))
# My own scratch
move_one "$DATA/scratch/mut"             "$DEST/scratch-mut"     "scratch/mut"              && moved=$((moved+1))
move_one "$DATA/scratch/calib"           "$DEST/scratch-calib"   "scratch/calib"            && moved=$((moved+1))

log "done: moved=$moved"
echo "moved=$moved"
cat <<'RESTORE'
Restore with, if ever needed:
  git clone https://github.com/amd/vaip.git                     /media/scott/data/git/vaip
  git clone https://github.com/ruvnet/RuView.git                 /media/scott/data/git/RuView
  git clone https://huggingface.co/kingjones777/Muse-Glimmer-30B-ROCmFP4-Strix-Halo-DFlash-GGUF \
           /media/scott/data/git/Muse-Glimmer-30B-ROCmFP4-Strix-Halo-DFlash-GGUF
Or copy back from /nas/archive/cold-data/.
RESTORE