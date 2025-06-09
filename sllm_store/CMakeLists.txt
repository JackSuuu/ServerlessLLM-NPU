cmake_minimum_required(VERSION 3.20)

# Set the project name
project(storage LANGUAGES CXX)

# Set c++ 17
set(CMAKE_CXX_STANDARD 17)

option(BUILD_SLLM_TESTS "Build tests" OFF)
# Build backend options
option(USE_CANN "Enable CANN support" OFF)
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_ROCM "Enable ROCm support" OFF)
option(USE_CPU_ONLY "Build CPU-only version" OFF)
# Disable other tests
set(BUILD_TESTING OFF)

include(FetchContent)
FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG        v0.6.0
)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.13.0
)
set(FETCHCONTENT_QUIET OFF)

FetchContent_MakeAvailable(glog)
FetchContent_MakeAvailable(googletest)

# Initialize backend variables
set(BACKEND_FOUND FALSE)
set(SLLM_STORE_GPU_LANG "NONE")

# Check for explicit backend selection
if(USE_CPU_ONLY)
  message(STATUS "Building CPU-only version")
  set(BACKEND_FOUND TRUE)
  set(SLLM_STORE_GPU_LANG "CPU")
elseif(USE_CANN)
  # Check for CANN toolkit
  if(DEFINED ASCEND_TOOLKIT_HOME)
    set(CANN_TOOLKIT_ROOT ${ASCEND_TOOLKIT_HOME})
  else()
    set(CANN_TOOLKIT_ROOT $ENV{ASCEND_TOOLKIT_HOME})
  endif()
  
  if(NOT CANN_TOOLKIT_ROOT)
    set(CANN_TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
  endif()

  if(EXISTS "${CANN_TOOLKIT_ROOT}/include/acl/acl.h")
    message(STATUS "CANN found at ${CANN_TOOLKIT_ROOT}")
    set(BACKEND_FOUND TRUE)
    set(SLLM_STORE_GPU_LANG "CANN")
    add_definitions(-DUSE_CANN)
    include_directories(${CANN_TOOLKIT_ROOT}/include)
    link_directories(${CANN_TOOLKIT_ROOT}/lib64)
  else()
    message(FATAL_ERROR "CANN toolkit not found at ${CANN_TOOLKIT_ROOT}")
  endif()
elseif(USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")
    set(BACKEND_FOUND TRUE)
    set(SLLM_STORE_GPU_LANG "CUDA")
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "CUDA toolkit not found")
  endif()
elseif(USE_ROCM)
  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "HIP found")
    set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")
    set(BACKEND_FOUND TRUE)
    set(SLLM_STORE_GPU_LANG "HIP")
    enable_language(HIP)
  else()
    message(FATAL_ERROR "ROCm/HIP not found")
  endif()
else()
  # Auto-detection fallback (legacy behavior)
  find_package(CUDAToolkit QUIET)
  find_package(HIP QUIET)

  # Check for CANN toolkit
  set(CANN_TOOLKIT_ROOT $ENV{ASCEND_TOOLKIT_HOME})
  if(NOT CANN_TOOLKIT_ROOT)
    set(CANN_TOOLKIT_ROOT "/usr/local/Ascend/ascend-toolkit/latest")
  endif()

  if(EXISTS "${CANN_TOOLKIT_ROOT}/include/acl/acl.h")
    message(STATUS "CANN found at ${CANN_TOOLKIT_ROOT} (auto-detected)")
    set(BACKEND_FOUND TRUE)
    set(SLLM_STORE_GPU_LANG "CANN")
    add_definitions(-DUSE_CANN)
    include_directories(${CANN_TOOLKIT_ROOT}/include)
    link_directories(${CANN_TOOLKIT_ROOT}/lib64)
  elseif (CUDAToolkit_FOUND)
    message(STATUS "CUDA found (auto-detected)")
    set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")
    set(BACKEND_FOUND TRUE)
    set(SLLM_STORE_GPU_LANG "CUDA")
    enable_language(CUDA)
  elseif (HIP_FOUND)
    message(STATUS "HIP found (auto-detected)")
    set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100")
    set(BACKEND_FOUND TRUE)
    set(SLLM_STORE_GPU_LANG "HIP")
    enable_language(HIP)
  else()
    message(STATUS "No GPU backend found, building CPU-only version")
    set(BACKEND_FOUND TRUE)
    set(SLLM_STORE_GPU_LANG "CPU")
  endif()
endif()

if(NOT BACKEND_FOUND)
  message(FATAL_ERROR "No backend found or enabled")
endif()

# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/CMakeLists.txt
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.8" "3.9" "3.10" "3.11")

#
# Try to find python package with an executable that exactly matches
# `SLLM_STORE_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (SLLM_STORE_PYTHON_EXECUTABLE)
  find_python_from_executable(${SLLM_STORE_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set SLLM_STORE_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

# Only try to find and configure torch for GPU backends
if(NOT SLLM_STORE_GPU_LANG STREQUAL "CPU")
  #
  # Update cmake's `CMAKE_PREFIX_PATH` with torch location.
  #
  append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

  #
  # Import torch cmake configuration.
  # Torch also imports CUDA (and partially HIP) languages with some customizations,
  # so there is no need to do this explicitly with check_language/enable_language,
  # etc.
  #
  find_package(Torch REQUIRED)

  #
  # Normally `torch.utils.cpp_extension.CUDAExtension` would add
  # `libtorch_python.so` for linking against an extension. Torch's cmake
  # configuration does not include this library (presumably since the cmake
  # config is used for standalone C++ binaries that link against torch).
  # The `libtorch_python.so` library defines some of the glue code between
  # torch/python via pybind and is required by SLLM_STORE extensions for this
  # reason. So, add it by manually with `find_library` using torch's
  # installed library path.
  #
  find_library(torch_python_LIBRARY torch_python PATHS
    "${TORCH_INSTALL_PREFIX}/lib")

  #
  # Override the GPU architectures detected by cmake/torch and filter them by
  # the supported versions for the current language.
  # The final set of arches is stored in `SLLM_STORE_GPU_ARCHES`.
  #
  override_gpu_arches(SLLM_STORE_GPU_ARCHES
  ${SLLM_STORE_GPU_LANG}
  "${${SLLM_STORE_GPU_LANG}_SUPPORTED_ARCHS}")

  #
  # Query torch for additional GPU compilation flags for the given
  # `SLLM_STORE_GPU_LANG`.
  # The final set of arches is stored in `SLLM_STORE_GPU_FLAGS`.
  #
  get_torch_gpu_compiler_flags(SLLM_STORE_GPU_FLAGS ${SLLM_STORE_GPU_LANG})
else()
  # For CPU-only builds, we still need basic torch for some functionality
  # but skip GPU-specific configuration
  find_package(Torch QUIET)
  if(NOT Torch_FOUND)
    message(WARNING "Torch not found, some functionality may be limited")
  endif()
  
  # Find pybind11 for CPU-only builds
  find_package(pybind11 QUIET)
  if(NOT pybind11_FOUND)
    # Try to find pybind11 via Python
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE pybind11_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(pybind11_DIR)
      find_package(pybind11 PATHS ${pybind11_DIR})
    endif()
  endif()
  
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 is required for CPU-only builds")
  endif()
  
  set(SLLM_STORE_GPU_ARCHES "")
  set(SLLM_STORE_GPU_FLAGS "")
endif()

#
# Define extension targets
#

#
# _C extension
#

#
# Define source files used in SLLM_STORE_EXT
#

set(SLLM_STORE_EXT_SRC
  "csrc/checkpoint/aligned_buffer.cpp"
  "csrc/checkpoint/checkpoint.cpp"
  "csrc/checkpoint/checkpoint_py.cpp"
  "csrc/checkpoint/tensor_writer.cpp"
)

#
# NOTE: The following (pure CXX) source files will be excluded from the input
# of the hipify tool. This is a temporary solution to avoid the recursive
# inclusion of the hipify tool (#206).
#
# NOTE: If you are making changes to this section or adding new source files,
# please make sure to update the list of excluded files here.
#

set(SLLM_STORE_CXX_EXT_SRC
  "csrc/checkpoint/aligned_buffer.cpp"
  "csrc/checkpoint/checkpoint_py.cpp"
  "csrc/checkpoint/tensor_writer.cpp"
)

# Only define GPU extension if not CPU-only
if(NOT SLLM_STORE_GPU_LANG STREQUAL "CPU")
  define_gpu_extension_target(
    _C
    DESTINATION sllm_store
    LANGUAGE ${SLLM_STORE_GPU_LANG}
    SOURCES ${SLLM_STORE_EXT_SRC}
    CXX_SRCS ${SLLM_STORE_CXX_EXT_SRC}
    COMPILE_FLAGS ${SLLM_STORE_GPU_FLAGS}
    ARCHITECTURES ${SLLM_STORE_GPU_ARCHES}
    WITH_SOABI)
else()
  # For CPU-only build, create a simple library target
  pybind11_add_module(_C ${SLLM_STORE_EXT_SRC})
  target_link_libraries(_C PRIVATE glog::glog)
  set_target_properties(_C PROPERTIES 
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/sllm_store"
  )
endif()

#
# _checkpoint_store extension
#

# pthread
find_package(Threads REQUIRED)

#
# Define source files used in CHECKPOINT_STORE
#

if (SLLM_STORE_GPU_LANG STREQUAL "CANN")
set(CHECKPOINT_STORE_SOURCES
  "csrc/sllm_store/binary_utils.cpp"
  "csrc/sllm_store/checkpoint_store.cpp"
  "csrc/sllm_store/checkpoint_store_py.cpp"
  "csrc/sllm_store/cann_memory.cpp"
  "csrc/sllm_store/cann_memory_pool.cpp"
  "csrc/sllm_store/gpu_replica.cpp"
  "csrc/sllm_store/memory_state.cpp"
  "csrc/sllm_store/model.cpp"
  "csrc/sllm_store/pinned_memory.cpp"
  "csrc/sllm_store/pinned_memory_pool.cpp"
)
elseif(SLLM_STORE_GPU_LANG STREQUAL "CPU")
set(CHECKPOINT_STORE_SOURCES
  "csrc/sllm_store/binary_utils.cpp"
  "csrc/sllm_store/checkpoint_store.cpp"
  "csrc/sllm_store/checkpoint_store_py.cpp"
  "csrc/sllm_store/memory_state.cpp"
  "csrc/sllm_store/model.cpp"
  "csrc/sllm_store/pinned_memory.cpp"
  "csrc/sllm_store/pinned_memory_pool.cpp"
)
else()
set(CHECKPOINT_STORE_SOURCES
  "csrc/sllm_store/binary_utils.cpp"
  "csrc/sllm_store/checkpoint_store.cpp"
  "csrc/sllm_store/checkpoint_store_py.cpp"
  "csrc/sllm_store/cuda_memory.cpp"
  "csrc/sllm_store/cuda_memory_pool.cpp"
  "csrc/sllm_store/gpu_replica.cpp"
  "csrc/sllm_store/memory_state.cpp"
  "csrc/sllm_store/model.cpp"
  "csrc/sllm_store/pinned_memory.cpp"
  "csrc/sllm_store/pinned_memory_pool.cpp"
)
endif()

#
# Define pure CXX files used in CHECKPOINT_STORE (files without CUDA code)
# Used for hipify tool to only convert CUDA code
#

set(CHECKPOINT_STORE_CXX_SOURCES
  "csrc/sllm_store/binary_utils.cpp"
  "csrc/sllm_store/memory_state.cpp"
  "csrc/sllm_store/pinned_memory.cpp"
)

if (SLLM_STORE_GPU_LANG STREQUAL "CANN")
set(CHECKPOINT_STORE_LIBRARIES
  Threads::Threads
  glog::glog
  ascendcl
  acl_retr
  acl_cblas
  acl_dvpp
)
else()
set(CHECKPOINT_STORE_LIBRARIES
  Threads::Threads
  glog::glog)
endif()

# Define checkpoint store extension
if(NOT SLLM_STORE_GPU_LANG STREQUAL "CPU")
  define_gpu_extension_target(
    _checkpoint_store
    DESTINATION sllm_store
    LANGUAGE ${SLLM_STORE_GPU_LANG}
    SOURCES ${CHECKPOINT_STORE_SOURCES}
    CXX_SRCS ${CHECKPOINT_STORE_CXX_SOURCES}
    COMPILE_FLAGS ${SLLM_STORE_GPU_FLAGS}
    ARCHITECTURES ${SLLM_STORE_GPU_ARCHES}
    LIBRARIES ${CHECKPOINT_STORE_LIBRARIES}
    WITH_SOABI)
else()
  # For CPU-only build, create a simple library target
  pybind11_add_module(_checkpoint_store ${CHECKPOINT_STORE_SOURCES})
  target_link_libraries(_checkpoint_store PRIVATE ${CHECKPOINT_STORE_LIBRARIES})
  set_target_properties(_checkpoint_store PROPERTIES 
    OUTPUT_NAME "_checkpoint_store"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/sllm_store"
  )
endif()

# Enable testing if the option is ON
if(BUILD_SLLM_TESTS)
    enable_testing()

    add_subdirectory(tests/cpp)
endif()