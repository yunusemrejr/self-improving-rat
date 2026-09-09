#!/usr/bin/env bash
# Reproducible, fresh-organism evaluation. Defaults: 100k steps, five seeds.
# TAG=name BUILD_DIR=build NO_REBUILD=1 tools/bench.sh 100000 1 2 3 42 12345
# SIR_BENCH_OVERRIDES='key=value key=value' applies config ablations.
# KEEP_RUNS=/absolute/path keeps checkpoints/configs for sir_eval; otherwise
# every run is removed. Live data/checkpoints is never used.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BENCH_DIR="${BUILD_DIR:-$ROOT/build-bench}"
OUT_DIR="$ROOT/bench_results"
mkdir -p "$OUT_DIR"
TAG="${TAG:-baseline}"
STEPS="${1:-100000}"
if (( $# )); then shift; fi
if (( $# )); then SEEDS=("$@"); else SEEDS=(1 2 3 42 12345); fi
if [[ ! "$TAG" =~ ^[a-zA-Z0-9_-]+$ || ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'invalid TAG or step count' >&2; exit 2
fi
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[1-9][0-9]*$ ]]; then echo 'seeds must be positive integers' >&2; exit 2; fi
done
if [[ "${NO_REBUILD:-0}" != 1 ]]; then
  CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
  SDL_PREFIX="${SIR_SDL2_PREFIX:-$ROOT/.deps/extracted}"
  if [[ -d "$SDL_PREFIX/usr/include/SDL2" ]]; then CMAKE_ARGS+=("-DSIR_SDL2_PREFIX=$SDL_PREFIX"); fi
  cmake -S "$ROOT" -B "$BENCH_DIR" "${CMAKE_ARGS[@]}"
  cmake --build "$BENCH_DIR" -j "${BUILD_JOBS:-6}"
fi
SNAPSHOT="$(mktemp -d)"
trap 'rm -rf "$SNAPSHOT"' EXIT
# Freeze code and config so an overlapping rebuild/edit cannot mix variants.
cp "$BENCH_DIR/sir" "$SNAPSHOT/sir"
cp "$ROOT/config/default.cfg" "$SNAPSHOT/default.cfg"
CSV="$OUT_DIR/$TAG.csv"
printf 'seed,steps,cheese,training_updates,wall_seconds,user_seconds,system_seconds,max_rss_kib\n' > "$CSV"
for seed in "${SEEDS[@]}"; do
  if [[ -n "${KEEP_RUNS:-}" ]]; then
    mkdir -p "$KEEP_RUNS"
    workdir="$(mktemp -d "$KEEP_RUNS/seed-$seed-XXXXXX")"
    workdir="$(cd "$workdir" && pwd)"
  else workdir="$SNAPSHOT/seed-$seed"; mkdir -p "$workdir"; fi
  cfg="$workdir/config.cfg"
  sed -e "s|^checkpoint_dir.*|checkpoint_dir = $workdir/checkpoints|" \
      -e "s|^log_file.*|log_file = $workdir/rat.log|" "$SNAPSHOT/default.cfg" > "$cfg"
  for kv in ${SIR_BENCH_OVERRIDES:-}; do printf '%s\n' "$kv" >> "$cfg"; done
  /usr/bin/time -f '%e,%U,%S,%M' -o "$workdir/time.csv" \
    env SIR_CONFIG="$cfg" SIR_SEED="$seed" SIR_MAX_STEPS="$STEPS" \
    SIR_HEADLESS=1 SDL_VIDEODRIVER=dummy "$SNAPSHOT/sir" \
    > "$workdir/stdout.log" 2> "$workdir/stderr.log"
  line="$(rg 'shutdown: steps=' "$workdir/rat.log" | tail -1)"
  if [[ "$line" =~ steps=([0-9]+)\ cheese=([0-9]+)\ training=([0-9]+) ]]; then
    metrics="${BASH_REMATCH[1]},${BASH_REMATCH[2]},${BASH_REMATCH[3]}"
  else echo "missing shutdown metrics for seed $seed" >&2; exit 1; fi
  printf '%s,%s,%s\n' "$seed" "$metrics" "$(cat "$workdir/time.csv")" >> "$CSV"
  echo "[bench] seed=$seed $metrics ($(cat "$workdir/time.csv"))"
  if [[ -n "${KEEP_RUNS:-}" ]]; then echo "[bench] retained $cfg"; fi
done
echo "[bench] wrote $CSV"
