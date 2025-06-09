# Find GTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.13.0
)
FetchContent_MakeAvailable(googletest)

# Specify the test source file
set(TEST_SOURCES test_load_to_host.cpp test_pinned_memory.cpp)

# Create an executable for the tests
add_executable(runTests ${TEST_SOURCES})

# set CUDA or HIP
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)

if (CUDAToolkit_FOUND)
  message(STATUS "CUDA found")
elseif (HIP_FOUND)
  message(STATUS "HIP found")
  target_compile_definitions(runTests PRIVATE USE_HIP)
else()
  message(FATAL_ERROR "Neither CUDA nor HIP found")
endif()

# Link the test executable with GTest and the main project
target_link_libraries(runTests GTest::gtest_main sllm_store)

# Register the test
add_test(NAME runTests COMMAND runTests)

if(USE_CANN)
  # CANN-specific test
  add_executable(test_cann_loading test_cann_loading.cpp)
  target_link_libraries(test_cann_loading
    PRIVATE
    gtest_main
    glog::glog
    Threads::Threads
    ascendcl
    acl_retr
    acl_cblas
    acl_dvpp
  )
  target_include_directories(test_cann_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/checkpoint
    ${CMAKE_SOURCE_DIR}/csrc/sllm_store
    ${ASCEND_TOOLKIT_HOME}/include
  )
  target_compile_definitions(test_cann_loading PRIVATE USE_CANN)
  
  # Add to test suite
  add_test(NAME CANNLoadingTests COMMAND test_cann_loading)
endif()
