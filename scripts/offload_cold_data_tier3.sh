#!/bin/bash
# Tier 3 cold-data offload: scratch working dirs and un-referenced model copies
# outside finetune-staging.
#
# Deliberately EXCLUDED after checking, because "cold" alone is not enough:
#   cache-symlink-temp (135G)  /home/scott/.cache is a symlink INTO it, and
#                              GNOME holds open files there. Live.
#   xwing-offload/lmstudio     same inode (66307:79953934) as the live
#   (122G)                     /home/scott/.lmstudio/models -- it is where LM
#                              Studio actually serves models from. Live.
#   amd-npu-models/Ornich-1.5-35B-A3B.gpt-oss-moe40 (70G)
#                              referenced by portfolio-management's
#                              end_to_end_translator.py and k2-mova's
#                              NPU-LOAD-STATUS.md. Active NPU asset.
#   data/models/flm (26G)      15 entries modified within 14 days.
#   k2-mova (28G)              36 entries modified within 14 days.
#   git/Muse-Glimmer... (14G)  benchmark GGUFs referenced by its own benchmark
#                              doc; not worth the risk for 14G.
#
# Everything below had zero references outside my own scratch tree and either
# regenerable or already-throwaway.
#
# The three rsync behaviours handled in tier 2 are applied here from the start:
#   * a trailing slash on a file source makes rsync nest the copy in a dir
#   * rsync will not create intermediate destination directories
#   * a directory where a file belongs makes the byte check compare mismatched
#     types, so clear it before copying
set -uo pipefail

DATA=/media/scott/data
DEST=/nas/archive/cold-data
LOG=/media/scott/data/fleet-power/nas-offload3.log
MIN_FREE_KB=$(( 30 * 1024 * 1024 ))

log() { echo "[nas-offload3] $(date '+%F %T') $*" >>"$LOG"; }

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
  if [ ! -e "$src" ]; then
    log "SKIP $label: source missing"
    return 2
  fi
  # Never touch anything a running process references on its cmdline.
  # /proc/*/cmdline expands to many files; shell redirection accepts one path,
  # so inspect each readable PID separately instead of using an ambiguous glob.
  local cmdline
  for cmdline in /proc/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    if tr '\0' ' ' <"$cmdline" 2>/dev/null | grep -Fq -- "$src"; then
      log "SKIP $label: referenced by a running process"
      return 2
    fi
  done
  # CIFS cannot store symlinks at all (rsync: "Operation not supported (95)").
  # Moving a directory that contains one would either fail mid-copy or, worse,
  # flatten the link into a copy of its target -- and if the link points into a
  # live model tree, deleting the source breaks whatever used it.
  if [ -d "$src" ]; then
    local nlinks
    nlinks=$(find "$src" -type l 2>/dev/null | head -1)
    if [ -n "$nlinks" ]; then
      log "SKIP $label: contains a symlink ($nlinks) that CIFS cannot represent"
      return 2
    fi
  fi
  local srcarg="$src"
  if [ -d "$src" ]; then
    srcarg="$src/"
  elif [ -d "$dst" ]; then
    log "  removing stale directory at destination $dst"
    rm -rf "$dst"
  fi
  mkdir -p "$(dirname "$dst")" 2>/dev/null || true
  log "copy $label ..."
  if ! rsync -a --partial --no-p --no-o --no-g --timeout=1800 \
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
# scratch working dirs from earlier sessions (mine, regenerable)
move_one "$DATA/scratch/attr"               "$DEST/scratch-attr"      "scratch/attr"            && moved=$((moved+1))
move_one "$DATA/scratch/git-backup-1790732515" "$DEST/git-backup-1790732515" "scratch/git-backup" && moved=$((moved+1))
# un-referenced copies of the 35B model, in three formats
move_one "$DATA/ornith-35b-moe-src"         "$DEST/ornith-35b-moe-src" "ornith-35b-moe-src"      && moved=$((moved+1))
move_one "$DATA/ornich-gguf"                "$DEST/ornich-gguf"       "ornich-gguf"             && moved=$((moved+1))

# scratch/dd is moved per-subdirectory: k2llama/ and u/ hold symlinks into the
# live K2-Horizon model, so they stay put and the guard above would skip them
# anyway. Everything else in there is direct-download scratch.
for d in "$DATA"/scratch/dd/*/; do
  sub=$(basename "$d")
  case "$sub" in
    k2llama|u) log "SKIP scratch/dd/$sub: holds symlinks into the live model tree"
              continue ;;
  esac
  move_one "$d" "$DEST/scratch-dd/$sub" "scratch/dd/$sub" && moved=$((moved+1))
done

log "done: moved=$moved"
echo "moved=$moved"
