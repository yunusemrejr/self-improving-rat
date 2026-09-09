cmake_minimum_required(VERSION 3.16)
project(self_improving_rat CXX)

# ---------------------------------------------------------------------------
# Self Improving Rat — C++20/CMake/SDL2 desktop artificial-life rat.
# Targets:
#   sir_core  : static library (utility + organism + simulation + learning +
#               persistence). No SDL dependency; used by the test binary.
#   sir       : the application (rendering + app loop + main + SDL2).
#   sir_tests : the test suite (core only, no SDL).
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Warnings. -Wall -Wextra -Wpedantic always; -Wshadow adds no noise here.
# ---------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
endif()

# ---------------------------------------------------------------------------
# Optional sanitizers: cmake -DSIR_SANITIZE=ON (or ./run.sh --sanitize).
# ---------------------------------------------------------------------------
option(SIR_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
if(SIR_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# SDL2 resolution order:
#   1) system pkg-config sdl2
#   2) SIR_SDL2_PREFIX (cmake cache var or SIR_SDL2_PREFIX env var): the
#      prefix of an extracted libsdl2-dev .deb tree (used when the dev
#      package cannot be installed system-wide, e.g. no root).
#   3) clear error with instructions.
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

set(SDL2_INCLUDE_DIRS "")
set(SDL2_LIBRARIES "")

if(NOT SDL2_LIBRARIES)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PC_SDL2 QUIET sdl2)
    if(PC_SDL2_FOUND AND PC_SDL2_INCLUDE_DIRS)
      # Only trust pkg-config when the reported include dir actually exists
      # (a stale sdl2.pc with a hardcoded /usr prefix must not win).
      set(_ok FALSE)
      foreach(_dir ${PC_SDL2_INCLUDE_DIRS})
        if(EXISTS "${_dir}/SDL.h" OR EXISTS "${_dir}/SDL2/SDL.h")
          set(_ok TRUE)
        endif()
      endforeach()
      if(_ok)
        set(SDL2_INCLUDE_DIRS ${PC_SDL2_INCLUDE_DIRS})
        set(SDL2_LIBRARIES ${PC_SDL2_LIBRARIES})
      endif()
    endif()
  endif()
endif()

if(NOT SDL2_LIBRARIES)
  set(SIR_SDL2_PREFIX "$ENV{SIR_SDL2_PREFIX}" CACHE STRING
      "Prefix of an extracted libsdl2-dev tree (contains usr/include/SDL2 and usr/lib/<arch>)")
  if(SIR_SDL2_PREFIX AND EXISTS "${SIR_SDL2_PREFIX}/usr/include/SDL2/SDL.h")
    # Detect the architecture library directory (deb layout). The dev tree
    # only ships the headers plus libSDL2.a / dangling .so symlinks; the real
    # shared object lives in the system runtime. Resolution order:
    #   1) a real shared lib inside the prefix, if present
    #   2) the system runtime libSDL2-2.0.so.0 (absolute path; same SONAME)
    set(_libdir "")
    foreach(_cand "usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
                  "usr/lib/x86_64-linux-gnu"
                  "usr/lib/aarch64-linux-gnu")
      if(EXISTS "${SIR_SDL2_PREFIX}/${_cand}/libSDL2.so" AND
         NOT IS_SYMLINK "${SIR_SDL2_PREFIX}/${_cand}/libSDL2.so")
        set(_libdir "${_cand}")
        break()
      endif()
      if(EXISTS "${SIR_SDL2_PREFIX}/${_cand}/libSDL2-2.0.so" AND
         NOT IS_SYMLINK "${SIR_SDL2_PREFIX}/${_cand}/libSDL2-2.0.so")
        set(_libdir "${_cand}")
        break()
      endif()
    endforeach()
    if(_libdir)
      set(SDL2_INCLUDE_DIRS
          "${SIR_SDL2_PREFIX}/usr/include/SDL2"
          "${SIR_SDL2_PREFIX}/usr/include"
          "${SIR_SDL2_PREFIX}/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}"
          "${SIR_SDL2_PREFIX}/usr/include/x86_64-linux-gnu"
          "${SIR_SDL2_PREFIX}/usr/include/aarch64-linux-gnu")
      set(SDL2_LIBRARIES
          "-L${SIR_SDL2_PREFIX}/${_libdir}" -lSDL2)
    else()
      # Dev-tree shared libs are dangling symlinks; link the system runtime.
      foreach(_sys "/usr/lib/x86_64-linux-gnu/libSDL2-2.0.so.0"
                    "/usr/lib/aarch64-linux-gnu/libSDL2-2.0.so.0")
        if(EXISTS "${_sys}")
          set(SDL2_INCLUDE_DIRS
              "${SIR_SDL2_PREFIX}/usr/include/SDL2"
              "${SIR_SDL2_PREFIX}/usr/include"
              "${SIR_SDL2_PREFIX}/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}"
              "${SIR_SDL2_PREFIX}/usr/include/x86_64-linux-gnu"
              "${SIR_SDL2_PREFIX}/usr/include/aarch64-linux-gnu")
          set(SDL2_LIBRARIES "${_sys}")
          break()
        endif()
      endforeach()
      if(NOT SDL2_LIBRARIES)
        message(FATAL_ERROR "SIR_SDL2_PREFIX set but no usable SDL2 shared library found "
            "in the prefix or system runtime (/usr/lib/*/libSDL2-2.0.so.0)")
      endif()
    endif()
  endif()
endif()

if(NOT SDL2_LIBRARIES)
  message(FATAL_ERROR
      "SDL2 development files not found.\n"
      "  Option 1: install them system-wide:   ./run.sh --install-deps\n"
      "  Option 2: provide an extracted dev tree:\n"
      "            cmake -DSIR_SDL2_PREFIX=/path/to/extracted\n"
      "  (The SDL2 runtime library is present system-wide; only the dev\n"
      "   headers/libs are needed for building.)")
endif()

# ---------------------------------------------------------------------------
# Core static library (no SDL).
# ---------------------------------------------------------------------------
set(SIR_CORE_SOURCES
    src/utility/config.cpp
    src/utility/logger.cpp
    src/utility/rng.cpp
    src/utility/rolling_stats.cpp
    src/utility/bitmap_font.cpp
    src/rendering/learning_notes.cpp
    src/rendering/note_typesetting.cpp
    src/organism/homeostasis.cpp
    src/organism/development.cpp
    src/organism/episodic_memory.cpp
    src/organism/novelty.cpp
    src/simulation/maze.cpp
    src/simulation/observation.cpp
    src/simulation/simulation.cpp
    src/learning/gru.cpp
    src/learning/neural_net.cpp
    src/learning/replay_buffer.cpp
    src/learning/agent.cpp
    src/persistence/checkpoint.cpp
)

add_library(sir_core STATIC ${SIR_CORE_SOURCES})
target_include_directories(sir_core PUBLIC include)
target_link_libraries(sir_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Application executable.
# ---------------------------------------------------------------------------
add_executable(sir
    src/main.cpp
    src/app/application.cpp
    src/rendering/renderer.cpp
)
target_include_directories(sir PRIVATE include ${SDL2_INCLUDE_DIRS})
target_link_libraries(sir PRIVATE sir_core ${SDL2_LIBRARIES} Threads::Threads)

# ---------------------------------------------------------------------------
# Test executable (core only; never needs SDL).
# ---------------------------------------------------------------------------
file(GLOB SIR_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
if(SIR_TEST_SOURCES)
  add_executable(sir_tests ${SIR_TEST_SOURCES})
  target_include_directories(sir_tests PRIVATE include tests)
  target_link_libraries(sir_tests PRIVATE sir_core Threads::Threads)
endif()

# ---------------------------------------------------------------------------
# Convenience targets.
# ---------------------------------------------------------------------------
add_custom_target(run COMMAND ./run.sh COMMENT "Launch via ./run.sh")

# Frozen-weight evaluation on unseen maze seeds (core only).
add_executable(sir_eval tools/evaluate.cpp)
target_link_libraries(sir_eval PRIVATE sir_core)

enable_testing()
add_test(NAME sir_tests COMMAND sir_tests)

# Exercise the actual UI without a display server or the user's saved rat.
add_executable(sir_render_tests tests/rendering/smoke.cpp src/rendering/renderer.cpp)
target_include_directories(sir_render_tests PRIVATE include ${SDL2_INCLUDE_DIRS})
target_link_libraries(sir_render_tests PRIVATE sir_core ${SDL2_LIBRARIES})
add_test(NAME sir_render_tests COMMAND sir_render_tests)
set_tests_properties(sir_render_tests PROPERTIES ENVIRONMENT "SDL_VIDEODRIVER=dummy")
