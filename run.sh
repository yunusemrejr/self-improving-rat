#!/usr/bin/env bash
# Self Improving Rat — launcher.
#
#   ./run.sh                 build (if needed) and run the application
#   ./run.sh --test          build and run the test suite
#   ./run.sh --clean         remove the build directory only
#   ./run.sh --sanitize      build+run with AddressSanitizer/UBSan
#   ./run.sh --install-deps  install libsdl2-dev (asks for sudo; opt-in only)
#
# Environment hooks (passed through to the app):
#   SIR_CONFIG=PATH  SIR_SEED=N  SIR_HEADLESS=1  SIR_MAX_STEPS=N
#   SIR_SCREENSHOT=path.bmp  SIR_SDL2_PREFIX=...
#
# Never touches data/checkpoints or data/logs.
set -u
cd "$(dirname "$0")"

BUILD_DIR="build"
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }

if [ "$#" -gt 0 ] && [ "$1" = "--install-deps" ]; then
  need_cmd apt-get
  need_cmd sudo
  say "Installing libsdl2-dev via apt (requires sudo)."
  sudo apt-get update
  sudo apt-get install -y libsdl2-dev
  exit $?
fi

need_cmd cmake || die "cmake not found (install with: sudo apt-get install cmake)"
need_cmd g++   || die "g++ not found (install with: sudo apt-get install g++ g++-15)"

# SDL2 detection for the build. The runtime library is usually present;
# only the development files (headers/libs) are needed to compile.
detect_sdl2() {
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists sdl2; then
    inc=$(pkg-config --cflags-only-I sdl2 2>/dev/null)
    case "$inc" in
      *-I/usr/include*) ;; # hardcoded /usr prefix: only valid if the file exists
      *) return 0 ;;
    esac
    for d in $inc; do
      p="${d#-I}"
      if [ -f "$p/SDL.h" ] || [ -f "$p/SDL2/SDL.h" ]; then return 0; fi
    done
  fi
  if [ -n "${SIR_SDL2_PREFIX:-}" ] && [ -f "$SIR_SDL2_PREFIX/usr/include/SDL2/SDL.h" ]; then
    CMAKE_ARGS+=("-DSIR_SDL2_PREFIX=$SIR_SDL2_PREFIX")
    return 0
  fi
  if [ -f ".deps/extracted/usr/include/SDL2/SDL.h" ]; then
    CMAKE_ARGS+=("-DSIR_SDL2_PREFIX=$PWD/.deps/extracted")
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# Configure. A damaged build directory is repaired by removing and
# reconfiguring once (only the regenerable build/ tree is affected).
# ---------------------------------------------------------------------------
configure() {
  if cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" "$@" >/dev/null; then
    return 0
  fi
  if [ -d "$BUILD_DIR" ]; then
    say "build directory appears damaged; removing and reconfiguring..."
    rm -rf "$BUILD_DIR"
    cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" "$@" >/dev/null
    return $?
  fi
  return 1
}

build() {
  cmake --build "$BUILD_DIR" -j"$(nproc)" "$@"
}

if [ "$#" -gt 0 ]; then
  case "$1" in
    --clean)
      rm -rf "$BUILD_DIR"
      say "removed $BUILD_DIR (data/checkpoints and data/logs untouched)"
      exit 0
      ;;
    --test)
      if ! detect_sdl2 && [ ! -d "$BUILD_DIR" ]; then
        # The test binary never links SDL, but CMake still needs SDL2 for the
        # `sir` target; resolve it (or instruct) before configuring.
        die "SDL2 development files not found. Run ./run.sh --install-deps, or set SIR_SDL2_PREFIX to an extracted libsdl2-dev tree."
      fi
      detect_sdl2 || true
      configure || die "CMake configuration failed (see errors above)"
      build --target sir_tests || die "test build failed"
      exec "$BUILD_DIR/sir_tests"
      ;;
    --sanitize)
      CMAKE_ARGS+=("-DSIR_SANITIZE=ON")
      detect_sdl2 || die "SDL2 development files not found (see --install-deps / SIR_SDL2_PREFIX)"
      configure || die "CMake configuration failed (see errors above)"
      build || die "sanitizer build failed"
      say "sanitizer build ready: $BUILD_DIR/sir"
      exec "$BUILD_DIR/sir" "${@:2}"
      ;;
    --install-deps|--help|-h)
      sed -n '1,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      # Unknown flag: treat as application argument (pass-through).
      ;;
  esac
fi

detect_sdl2 || die "SDL2 development files not found. Run ./run.sh --install-deps, or set SIR_SDL2_PREFIX to an extracted libsdl2-dev tree."
configure || die "CMake configuration failed (see errors above)"
build || die "build failed"

# exec so signals and exit codes propagate to the terminal.
exec "$BUILD_DIR/sir" "$@"
