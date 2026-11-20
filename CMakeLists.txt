cmake_minimum_required(VERSION 3.24)
project(GME LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
  set(GME_C_FLAGS /W3 /O2 /arch:AVX2)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  set(GME_C_FLAGS -O3 -march=native -Wall -Wno-unused-function)
endif()

# gme_embed(<out.inc> <C variable> <sources...>): concatenates the sources and
# writes them as a C char array, for kernels compiled at run time by NVRTC.
# Re-runs automatically when a source changes.
function(gme_embed OUT VAR)
  set(_all "")
  foreach(_src ${ARGN})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_src})
    file(READ ${_src} _c)
    string(APPEND _all "${_c}\n")
  endforeach()
  file(WRITE ${CMAKE_BINARY_DIR}/gen/${OUT}.src "${_all}")
  file(READ ${CMAKE_BINARY_DIR}/gen/${OUT}.src _hex HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _arr "${_hex}")
  file(WRITE ${CMAKE_BINARY_DIR}/gen/${OUT} "static const char ${VAR}[] = {${_arr} 0x00};\n")
endfunction()

gme_embed(cosmos_kernels.inc COSMOS_KERNELS_SRC ${CMAKE_SOURCE_DIR}/src/genesis/cosmos_kernels.cu)
gme_embed(bio_kernels.inc BIO_KERNELS_SRC
  ${CMAKE_SOURCE_DIR}/src/bio/bio_types.h
  ${CMAKE_SOURCE_DIR}/src/bio/bio_core.h
  ${CMAKE_SOURCE_DIR}/src/bio/bio_kernels.cu)

# ---- life-simulation core, shared by gme_genesis and gme_biotest ----------
set(BIO_CORE_SRC
  src/bio/bio_sim.c
  src/bio/bio_cpu.c
  src/bio/bio_io.c
  src/bio/bio_test.c
)
set(COMMON_SRC
  src/rtcuda.c
  src/platform.c
  src/jobs.c
  src/image.c
)

# ---- gme_genesis: plain C. GPU kernels are compiled at run time by NVRTC,
# so this builds and runs on machines with only the NVIDIA driver (the rigs).
add_executable(gme_genesis
  src/genesis/genesis_main.c
  src/genesis/cosmos.c
  src/genesis/chem.c
  src/planetfile.c
  src/biomap.c
  src/bio/bio_env.c
  src/bio/bio_main.c
  ${BIO_CORE_SRC}
  ${COMMON_SRC}
)
target_include_directories(gme_genesis PRIVATE ${CMAKE_BINARY_DIR}/gen src src/bio)
target_compile_options(gme_genesis PRIVATE ${GME_C_FLAGS})
if(NOT WIN32)
  target_link_libraries(gme_genesis PRIVATE m pthread dl)
endif()

# ---- gme_biotest: life-simulation test suite (bio core only) ----
add_executable(gme_biotest src/bio/bio_test_main.c ${BIO_CORE_SRC} ${COMMON_SRC})
target_include_directories(gme_biotest PRIVATE ${CMAKE_BINARY_DIR}/gen src src/bio)
target_compile_options(gme_biotest PRIVATE ${GME_C_FLAGS})
if(NOT WIN32)
  target_link_libraries(gme_biotest PRIVATE m pthread dl)
endif()

# ---- gme: the real-time engine (needs the CUDA toolkit to build) ----
include(CheckLanguage)
check_language(CUDA)
option(GME_BUILD_ENGINE "Build the CUDA engine (gme)" ON)
if(GME_BUILD_ENGINE AND CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  # native window on Windows; headless (render to PNG, no display) elsewhere
  if(WIN32)
    set(GME_WINDOW_SRC src/window_win32.c)
  else()
    set(GME_WINDOW_SRC src/window_headless.c)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  add_executable(gme
    src/main.c
    src/platform.c
    src/jobs.c
    src/world.c
    src/planetfile.c
    src/biomap.c
    src/image.c
    src/gpu.cu
    src/render.cu
    src/atmo.cu
    src/gi.cu
    src/spectral.c
    ${GME_WINDOW_SRC}
  )
  target_include_directories(gme PRIVATE src)
  target_compile_options(gme PRIVATE $<$<COMPILE_LANGUAGE:C>:${GME_C_FLAGS}>)
  if(MSVC)
    target_compile_options(gme PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>)
  endif()
  target_compile_options(gme PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
  if(WIN32)
    set_target_properties(gme PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  else()
    set_target_properties(gme PROPERTIES CUDA_RUNTIME_LIBRARY Shared)   # pip CUDA runtimes ship libcudart.so only
  endif()
  if(WIN32)
    target_link_libraries(gme PRIVATE user32 gdi32 winmm)
  else()
    target_link_libraries(gme PRIVATE m pthread)
  endif()
else()
  message(STATUS "CUDA compiler not found: building gme_genesis and gme_biotest only")
endif()
